#!/usr/bin/env python3
"""LLM-as-judge for abstraction utilization on eval JSONL outputs.

Reads a treated eval JSONL (e.g. output of examples/hrlib/40_eval.sh), groups rows
by problem, extracts injected abstraction bullets from ``input``, samples a small
number of rollouts per problem, and asks an OpenRouter model to judge whether each
injected bullet was used correctly/incorrectly/ignored/unclear and whether it was
relevant.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from openai import AsyncOpenAI
from tqdm import tqdm

try:
    from examples.hrlib.extract import _should_retry_with_fallback
except ImportError:
    from extract import _should_retry_with_fallback  # type: ignore[no-redef]

try:
    from scripts.analyze import _problem_group_key, _user_turn_from_verl_decoded_input
except Exception:
    # Keep this script robust in minimal environments where scripts.analyze imports
    # optional plotting dependencies.
    def _user_turn_from_verl_decoded_input(text: str) -> str | None:
        if "\nuser\n" not in text:
            return None
        after = text.split("\nuser\n", 1)[1]
        if "\nassistant\n" in after:
            return after.split("\nassistant\n", 1)[0].strip()
        if "\nassistant" in after:
            return after.split("\nassistant", 1)[0].strip()
        return after.strip()

    def _problem_group_key(data: dict[str, Any]) -> str:
        uid = data.get("uid")
        if uid is not None and uid != "":
            return str(uid)
        ds = data.get("data_source", "unknown")
        if isinstance(ds, (list, tuple, np.ndarray)) and len(ds) > 0:
            ds = ds[0]
        inp = str(data.get("input", ""))
        user_turn = _user_turn_from_verl_decoded_input(inp)
        if user_turn:
            return f"{ds}::user::{user_turn}"
        gts = data.get("gts")
        if gts is not None and gts != "":
            if isinstance(gts, dict):
                gts_s = json.dumps(gts, sort_keys=True, ensure_ascii=False)
            else:
                gts_s = str(gts)
            return f"{ds}::gts::{gts_s}"
        return inp


SYSTEM_PROMPT = """You are an expert math educator judging whether a model solution used injected reasoning advice.

You will receive:
1) A math problem.
2) The ground-truth answer.
3) A numbered list of injected strategies/cautions.
4) One model solution.
5) A correctness label for that solution.

For each numbered bullet, classify usage as exactly one of:
- "used_correctly"
- "used_incorrectly"
- "ignored"
- "unclear"

Also provide whether that bullet is relevant to this problem (boolean).

Return ONLY JSON in this schema:
{
  "bullets": [
    {"id": 1, "usage": "used_correctly", "relevant": true, "brief_reason": "..."}
  ]
}
"""

FENCED_JSON_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
FIRST_OBJECT_RE = re.compile(r"\{[\s\S]*\}")
BULLET_LINE_RE = re.compile(r"^- \[(strategy|caution)\]\s+(.*)$")
WHEN_LINE_RE = re.compile(r"^\s*when:\s*(.*)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_jsonl", required=True, help="Input treated eval JSONL (e.g. .../0.jsonl).")
    parser.add_argument("--out", required=True, help="Output JSONL path for parsed judge records.")
    parser.add_argument("--raw_dump_out", required=True, help="Output JSONL path for raw API response dumps.")
    parser.add_argument("--backend", default="openrouter", choices=["openrouter"], help="Inference backend.")
    parser.add_argument("--base_url", default="https://openrouter.ai/api/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--model", default="openai/gpt-oss-120b:free", help="Primary model ID (often :free).")
    parser.add_argument(
        "--fallback_model",
        default="openai/gpt-oss-120b",
        help="Paid model to call once if primary hits rate limit / quota.",
    )
    parser.add_argument(
        "--no_fallback",
        action="store_true",
        help="Disable fallback; only --model is used.",
    )
    parser.add_argument(
        "--rollouts_per_problem",
        type=int,
        default=2,
        help="How many treated rollouts to judge per problem (default: 2).",
    )
    parser.add_argument(
        "--correct_score_threshold",
        type=float,
        default=1.0,
        help="Rollout is labeled correct if score >= this threshold (default: 1.0).",
    )
    parser.add_argument("--max_concurrency", type=int, default=20, help="Max in-flight API calls.")
    parser.add_argument("--resume_from", default="", help="Optional JSONL to skip already-judged pairs.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of problems to judge (smoke tests).")
    return parser.parse_args()


def _write_jsonl(path: str, record: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prompt_hash(messages: list[dict[str, str]]) -> str:
    serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _norm_data_source(ds: Any) -> str:
    if isinstance(ds, (list, tuple, np.ndarray)) and len(ds) > 0:
        return str(ds[0])
    return str(ds) if ds is not None else "unknown"


def _to_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _score_sort_key(score: float) -> float:
    if np.isnan(score):
        return -1e30
    return score


def _extract_injected_bullets(decoded_input: str) -> list[dict[str, Any]]:
    header = "## Relevant Strategies & Cautions"
    if header not in decoded_input:
        return []

    after_header = decoded_input.split(header, 1)[1]
    stop_markers = [
        "\n\nPlease reason step by step",
        "\nuser\n",
        "\nassistant\n",
    ]
    stop_at = len(after_header)
    for marker in stop_markers:
        idx = after_header.find(marker)
        if idx >= 0:
            stop_at = min(stop_at, idx)
    block = after_header[:stop_at]

    bullets: list[dict[str, Any]] = []
    lines = block.splitlines()
    i = 0
    while i < len(lines):
        m = BULLET_LINE_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue

        btype = m.group(1).strip().lower()
        principle = " ".join(m.group(2).split())
        when_text = ""
        if i + 1 < len(lines):
            wm = WHEN_LINE_RE.match(lines[i + 1].strip())
            if wm:
                when_text = " ".join(wm.group(1).split())
                i += 1

        bullets.append(
            {
                "id": len(bullets) + 1,
                "type": btype,
                "principle": principle,
                "when_to_apply": when_text,
            }
        )
        i += 1
    return bullets


def _json_safe_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value) if value is not None else ""


def _sample_rollouts(rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if not rows or k <= 0:
        return []
    scored = [
        {
            "rollout_idx": idx,
            "row": row,
            "score": _to_score(row.get("score")),
            "tag": "",
        }
        for idx, row in enumerate(rows)
    ]
    k_eff = min(k, len(scored))

    selected_indices: list[int] = []

    best_order = sorted(
        range(len(scored)),
        key=lambda i: (-_score_sort_key(scored[i]["score"]), scored[i]["rollout_idx"]),
    )
    best_i = best_order[0]
    selected_indices.append(best_i)
    scored[best_i]["tag"] = "best"

    if k_eff >= 2:
        worst_order = sorted(
            range(len(scored)),
            key=lambda i: (_score_sort_key(scored[i]["score"]), scored[i]["rollout_idx"]),
        )
        for wi in worst_order:
            if wi not in selected_indices:
                selected_indices.append(wi)
                scored[wi]["tag"] = "worst"
                break

    if len(selected_indices) < k_eff:
        for i in range(len(scored)):
            if i in selected_indices:
                continue
            selected_indices.append(i)
            scored[i]["tag"] = f"extra_{len(selected_indices)}"
            if len(selected_indices) >= k_eff:
                break

    return [scored[i] for i in selected_indices]


def _load_grouped_eval(path: str) -> dict[str, list[dict[str, Any]]]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            by_key[_problem_group_key(data)].append(data)
    return dict(by_key)


def _build_payloads(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped = _load_grouped_eval(args.eval_jsonl)
    payloads: list[dict[str, Any]] = []

    n_groups_total = len(grouped)
    n_groups_no_bullets = 0
    n_groups_selected = 0

    for problem_key in sorted(grouped.keys()):
        rows = grouped[problem_key]
        first = rows[0]
        decoded_input = str(first.get("input", ""))
        bullets = _extract_injected_bullets(decoded_input)
        if not bullets:
            n_groups_no_bullets += 1
            continue

        sampled = _sample_rollouts(rows, args.rollouts_per_problem)
        if not sampled:
            continue

        n_groups_selected += 1
        user_problem = _user_turn_from_verl_decoded_input(decoded_input) or ""
        ground_truth = _json_safe_text(first.get("gts"))
        data_source = _norm_data_source(first.get("data_source"))

        for pick in sampled:
            score = pick["score"]
            rollout_idx = int(pick["rollout_idx"])
            pair_id = f"{problem_key}::rollout::{rollout_idx}"
            payloads.append(
                {
                    "pair_id": pair_id,
                    "problem_key": problem_key,
                    "data_source": data_source,
                    "user_problem": user_problem,
                    "ground_truth": ground_truth,
                    "injected_bullets": bullets,
                    "rollout_idx": rollout_idx,
                    "rollout_score": score,
                    "rollout_output": str(pick["row"].get("output", "")),
                    "is_rollout_correct": bool(score >= args.correct_score_threshold - 1e-9) if not np.isnan(score) else False,
                    "n_rollouts_total": len(rows),
                    "sample_tag": pick["tag"],
                }
            )

        if args.limit is not None and n_groups_selected >= args.limit:
            break

    stats = {
        "groups_total": n_groups_total,
        "groups_selected": n_groups_selected,
        "groups_no_bullets": n_groups_no_bullets,
        "pairs_total": len(payloads),
    }
    return payloads, stats


def _load_done_pair_ids(path: str) -> set[str]:
    if not path:
        return set()
    target = Path(path)
    if not target.exists():
        print(f"[warn] --resume_from file not found: {path}")
        return set()
    done: set[str] = set()
    with target.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            pair_id = obj.get("pair_id")
            if pair_id:
                done.add(str(pair_id))
                continue
            pk = obj.get("problem_key")
            ridx = obj.get("rollout_idx")
            if pk is not None and ridx is not None:
                done.add(f"{pk}::rollout::{ridx}")
    return done


def _format_bullets_for_prompt(bullets: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for b in bullets:
        lines.append(f"{b['id']}. [{b['type']}] {b['principle']}")
        lines.append(f"   when: {b['when_to_apply']}")
    return "\n".join(lines)


def build_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    correctness = "correct" if payload["is_rollout_correct"] else "incorrect"
    user_prompt = (
        "Problem:\n"
        f"{payload['user_problem']}\n\n"
        "Ground-truth final answer:\n"
        f"{payload['ground_truth']}\n\n"
        "Injected strategies/cautions (numbered):\n"
        f"{_format_bullets_for_prompt(payload['injected_bullets'])}\n\n"
        "Model solution:\n"
        f"{payload['rollout_output']}\n\n"
        "Rollout correctness label:\n"
        f"{correctness}\n\n"
        "Task:\n"
        "For each bullet ID, determine usage and relevance. "
        "Return only the required JSON object."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _candidate_json_strings(raw_text: str) -> list[str]:
    candidates = [raw_text.strip()]
    for match in FENCED_JSON_RE.findall(raw_text):
        if match:
            candidates.append(match.strip())
    first_obj = FIRST_OBJECT_RE.search(raw_text)
    if first_obj:
        candidates.append(first_obj.group(0))
    return candidates


def _normalize_usage(value: Any) -> str | None:
    text = str(value).strip().lower()
    aliases = {
        "used_correctly": "used_correctly",
        "correct": "used_correctly",
        "used": "used_correctly",
        "used_incorrectly": "used_incorrectly",
        "incorrect": "used_incorrectly",
        "misapplied": "used_incorrectly",
        "ignored": "ignored",
        "not_used": "ignored",
        "not used": "ignored",
        "unclear": "unclear",
        "unknown": "unclear",
    }
    if text in aliases:
        return aliases[text]
    if "incorrect" in text or "misappl" in text:
        return "used_incorrectly"
    if text.startswith("used"):
        return "used_correctly"
    if "ignore" in text or "not use" in text:
        return "ignored"
    if "unclear" in text:
        return "unclear"
    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "relevant"}:
        return True
    if text in {"false", "no", "n", "0", "irrelevant"}:
        return False
    return None


def parse_judge_response(raw_text: str, n_bullets: int) -> tuple[list[dict[str, Any]], bool]:
    for candidate in _candidate_json_strings(raw_text):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(obj, dict):
            items = obj.get("bullets", [])
        elif isinstance(obj, list):
            items = obj
        else:
            continue

        if not isinstance(items, list):
            continue

        parsed: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                bid = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            if bid < 1 or bid > n_bullets or bid in seen_ids:
                continue
            usage = _normalize_usage(item.get("usage"))
            relevant = _coerce_bool(item.get("relevant"))
            if usage is None or relevant is None:
                continue
            reason = " ".join(str(item.get("brief_reason", "")).strip().split())
            parsed.append(
                {
                    "id": bid,
                    "usage": usage,
                    "relevant": relevant,
                    "brief_reason": reason,
                }
            )
            seen_ids.add(bid)

        if parsed:
            parsed.sort(key=lambda x: x["id"])
            return parsed, True
    return [], False


def _extract_response_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except Exception:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                maybe_text = item.get("text")
                if maybe_text:
                    parts.append(str(maybe_text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content or "")


async def judge_one(
    client: Any,
    model: str,
    payload: dict[str, Any],
    semaphore: Any,
    *,
    fallback_model: str | None = None,
) -> dict[str, Any]:
    messages = build_messages(payload)
    prompt_hash = _prompt_hash(messages)

    primary_model = model.strip()
    fb = (fallback_model or "").strip()
    if fb == primary_model:
        fb = ""
    model_chain = [primary_model] + ([fb] if fb else [])

    primary_error = ""
    total_elapsed_sec = 0.0

    async with semaphore:
        for attempt_idx, model_id in enumerate(model_chain):
            started_at = time.perf_counter()
            try:
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=0.0,
                )
                total_elapsed_sec += round(time.perf_counter() - started_at, 3)
                raw_response = _extract_response_text(response)
                judged_bullets, parsed_ok = parse_judge_response(raw_response, len(payload["injected_bullets"]))
                return {
                    "dump": {
                        "pair_id": payload["pair_id"],
                        "problem_key": payload["problem_key"],
                        "rollout_idx": payload["rollout_idx"],
                        "rollout_score": payload["rollout_score"],
                        "is_rollout_correct": payload["is_rollout_correct"],
                        "n_injected_bullets": len(payload["injected_bullets"]),
                        "n_judged_bullets": len(judged_bullets),
                        "model": model_id,
                        "primary_model": primary_model,
                        "fallback_model": fb if fb else "",
                        "fallback_used": attempt_idx > 0,
                        "primary_error": primary_error if attempt_idx > 0 else "",
                        "prompt_hash": prompt_hash,
                        "raw_response": raw_response,
                        "parsed_ok": parsed_ok,
                        "error": "",
                        "api_elapsed_sec": total_elapsed_sec,
                        "timestamp": _utc_now(),
                    },
                    "record": {
                        "pair_id": payload["pair_id"],
                        "problem_key": payload["problem_key"],
                        "data_source": payload["data_source"],
                        "rollout_idx": payload["rollout_idx"],
                        "rollout_score": payload["rollout_score"],
                        "is_rollout_correct": payload["is_rollout_correct"],
                        "sample_tag": payload["sample_tag"],
                        "n_rollouts_total": payload["n_rollouts_total"],
                        "ground_truth": payload["ground_truth"],
                        "user_problem": payload["user_problem"],
                        "injected_bullets": payload["injected_bullets"],
                        "judged_bullets": judged_bullets,
                        "parsed_ok": parsed_ok,
                        "model": model_id,
                        "fallback_used": attempt_idx > 0,
                        "api_elapsed_sec": total_elapsed_sec,
                        "timestamp": _utc_now(),
                    },
                }
            except Exception as exc:
                total_elapsed_sec += round(time.perf_counter() - started_at, 3)
                last_error = f"{exc.__class__.__name__}: {exc}"
                if attempt_idx == 0:
                    primary_error = last_error
                can_fallback = bool(fb) and attempt_idx == 0 and len(model_chain) > 1
                if can_fallback and _should_retry_with_fallback(exc):
                    continue
                return {
                    "dump": {
                        "pair_id": payload["pair_id"],
                        "problem_key": payload["problem_key"],
                        "rollout_idx": payload["rollout_idx"],
                        "rollout_score": payload["rollout_score"],
                        "is_rollout_correct": payload["is_rollout_correct"],
                        "n_injected_bullets": len(payload["injected_bullets"]),
                        "n_judged_bullets": 0,
                        "model": model_id,
                        "primary_model": primary_model,
                        "fallback_model": fb if fb else "",
                        "fallback_used": attempt_idx > 0,
                        "primary_error": primary_error,
                        "prompt_hash": prompt_hash,
                        "raw_response": "",
                        "parsed_ok": False,
                        "error": last_error,
                        "api_elapsed_sec": total_elapsed_sec,
                        "timestamp": _utc_now(),
                    },
                    "record": {
                        "pair_id": payload["pair_id"],
                        "problem_key": payload["problem_key"],
                        "data_source": payload["data_source"],
                        "rollout_idx": payload["rollout_idx"],
                        "rollout_score": payload["rollout_score"],
                        "is_rollout_correct": payload["is_rollout_correct"],
                        "sample_tag": payload["sample_tag"],
                        "n_rollouts_total": payload["n_rollouts_total"],
                        "ground_truth": payload["ground_truth"],
                        "user_problem": payload["user_problem"],
                        "injected_bullets": payload["injected_bullets"],
                        "judged_bullets": [],
                        "parsed_ok": False,
                        "model": model_id,
                        "fallback_used": attempt_idx > 0,
                        "api_elapsed_sec": total_elapsed_sec,
                        "error": last_error,
                        "timestamp": _utc_now(),
                    },
                }


def _safe_rate(num: int, den: int) -> float:
    return float(num) / den if den else 0.0


def _print_usage_summary(title: str, usage_counter: Counter[str], relevant_true: int, total: int) -> None:
    if total <= 0:
        print(f"[summary] {title}: no judged bullets")
        return
    uc = usage_counter
    print(
        f"[summary] {title}: total_bullets={total}, "
        f"used_correctly={uc['used_correctly']} ({_safe_rate(uc['used_correctly'], total):.3f}), "
        f"used_incorrectly={uc['used_incorrectly']} ({_safe_rate(uc['used_incorrectly'], total):.3f}), "
        f"ignored={uc['ignored']} ({_safe_rate(uc['ignored'], total):.3f}), "
        f"unclear={uc['unclear']} ({_safe_rate(uc['unclear'], total):.3f}), "
        f"relevant_true={relevant_true} ({_safe_rate(relevant_true, total):.3f})"
    )


async def run_judging(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")
    if args.max_concurrency <= 0:
        raise ValueError("--max_concurrency must be a positive integer.")
    if args.rollouts_per_problem <= 0:
        raise ValueError("--rollouts_per_problem must be a positive integer.")

    payloads, stats = _build_payloads(args)
    done_pair_ids = _load_done_pair_ids(args.resume_from)
    if done_pair_ids:
        payloads = [p for p in payloads if p["pair_id"] not in done_pair_ids]

    print(
        "[info] groups_total={groups_total} groups_selected={groups_selected} "
        "groups_no_bullets={groups_no_bullets} pairs_selected={pairs_total} pairs_after_resume={pairs_after_resume}".format(
            groups_total=stats["groups_total"],
            groups_selected=stats["groups_selected"],
            groups_no_bullets=stats["groups_no_bullets"],
            pairs_total=stats["pairs_total"],
            pairs_after_resume=len(payloads),
        )
    )
    if not payloads:
        print("[done] no payloads to process.")
        return

    default_headers = {}
    if os.getenv("OPENROUTER_HTTP_REFERER"):
        default_headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER", "")
    if os.getenv("OPENROUTER_APP_TITLE"):
        default_headers["X-Title"] = os.getenv("OPENROUTER_APP_TITLE", "")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.base_url,
        default_headers=default_headers or None,
    )
    semaphore = asyncio.Semaphore(args.max_concurrency)

    fallback_model = ""
    if not args.no_fallback:
        fb = (args.fallback_model or "").strip()
        if fb and fb != (args.model or "").strip():
            fallback_model = fb
    fb_display = repr(fallback_model) if fallback_model else "(disabled)"
    print(f"[info] primary_model={args.model!r} fallback_model={fb_display}")

    tasks = [
        asyncio.create_task(
            judge_one(
                client=client,
                model=args.model,
                payload=payload,
                semaphore=semaphore,
                fallback_model=fallback_model or None,
            )
        )
        for payload in payloads
    ]

    parsed_ok = 0
    parsed_failed = 0
    fallback_used_count = 0
    api_elapsed_sum_sec = 0.0
    api_elapsed_max_sec = 0.0

    usage_all: Counter[str] = Counter()
    total_all = 0
    relevant_true_all = 0

    usage_by_correctness: dict[str, Counter[str]] = {
        "correct": Counter(),
        "incorrect": Counter(),
    }
    totals_by_correctness: Counter[str] = Counter()
    relevant_true_by_correctness: Counter[str] = Counter()

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="judge"):
        result = await fut
        dump = result["dump"]
        record = result["record"]

        _write_jsonl(args.raw_dump_out, dump)
        _write_jsonl(args.out, record)

        if dump.get("fallback_used"):
            fallback_used_count += 1
        elapsed = float(dump.get("api_elapsed_sec", 0.0))
        api_elapsed_sum_sec += elapsed
        api_elapsed_max_sec = max(api_elapsed_max_sec, elapsed)

        if dump.get("parsed_ok"):
            parsed_ok += 1
        else:
            parsed_failed += 1

        judged_bullets = record.get("judged_bullets", [])
        if isinstance(judged_bullets, list):
            correctness_key = "correct" if record.get("is_rollout_correct") else "incorrect"
            for jb in judged_bullets:
                if not isinstance(jb, dict):
                    continue
                usage = str(jb.get("usage", ""))
                relevant = bool(jb.get("relevant", False))
                usage_all[usage] += 1
                total_all += 1
                if relevant:
                    relevant_true_all += 1

                usage_by_correctness[correctness_key][usage] += 1
                totals_by_correctness[correctness_key] += 1
                if relevant:
                    relevant_true_by_correctness[correctness_key] += 1

    await client.close()

    avg_api_elapsed_sec = api_elapsed_sum_sec / len(payloads) if payloads else 0.0
    print(
        "[done] judging finished: "
        f"processed={len(payloads)}, parsed_ok={parsed_ok}, parsed_failed={parsed_failed}, "
        f"fallback_used_count={fallback_used_count}, "
        f"avg_api_elapsed_sec={avg_api_elapsed_sec:.2f}, max_api_elapsed_sec={api_elapsed_max_sec:.2f}"
    )
    _print_usage_summary(
        title="all_rollouts",
        usage_counter=usage_all,
        relevant_true=relevant_true_all,
        total=total_all,
    )
    for key in ("correct", "incorrect"):
        _print_usage_summary(
            title=f"{key}_rollouts",
            usage_counter=usage_by_correctness[key],
            relevant_true=int(relevant_true_by_correctness[key]),
            total=int(totals_by_correctness[key]),
        )


def main() -> None:
    args = parse_args()
    asyncio.run(run_judging(args))


if __name__ == "__main__":
    main()
