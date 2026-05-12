#!/usr/bin/env python3
"""LLM-as-judge for prompt-only abstraction relevance on injected parquets.

This script judges whether each injected abstraction bullet is relevant/helpful
for solving the problem, without requiring rollout outputs, correctness labels,
or eval JSONLs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

try:
    from examples.hrlib.extract import _should_retry_with_fallback
except ImportError:
    from extract import _should_retry_with_fallback  # type: ignore[no-redef]


SYSTEM_PROMPT = """You are an expert math educator judging injected reasoning advice.

You will receive:
1) A math problem statement.
2) A numbered list of injected strategy/caution bullets.

For each numbered bullet, decide:
- whether it is relevant to solving this specific problem
- how helpful it would be if followed correctly

Use exactly one helpfulness label:
- "high"
- "medium"
- "low"
- "none"

Return ONLY JSON in this schema:
{
  "bullets": [
    {"id": 1, "relevant": true, "helpfulness": "high", "brief_reason": "..."}
  ]
}
"""

FENCED_JSON_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
FIRST_OBJECT_RE = re.compile(r"\{[\s\S]*\}")
V1_BULLET_RE = re.compile(r"^- \[(strategy|caution)\]\s+(.*)$")
WHEN_LINE_RE = re.compile(r"^\s*when:\s*(.*)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^##\s+(Strategies|Cautions(?:\s*\(.*\))?)\s*$", re.IGNORECASE)
PLAIN_BULLET_RE = re.compile(r"^- (.+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", required=True, help="Injected parquet path")
    p.add_argument("--scores", default="", help="Optional *_scores.jsonl sidecar")
    p.add_argument("--out", required=True, help="Output JSONL path for parsed judge records")
    p.add_argument("--raw_dump_out", required=True, help="Output JSONL path for raw judge responses")
    p.add_argument("--backend", default="openrouter", choices=["openrouter"], help="Inference backend")
    p.add_argument("--base_url", default="https://openrouter.ai/api/v1", help="OpenAI-compatible base URL")
    p.add_argument("--model", default="openai/gpt-oss-120b:free", help="Primary model ID")
    p.add_argument(
        "--fallback_model",
        default="openai/gpt-oss-120b",
        help="Fallback model for rate limit/quota retries",
    )
    p.add_argument("--no_fallback", action="store_true", help="Disable fallback model")
    p.add_argument("--max_concurrency", type=int, default=20, help="Max in-flight API calls")
    p.add_argument("--limit", type=int, default=None, help="Max number of prompts to judge")
    p.add_argument("--resume_from", default="", help="Optional JSONL to skip already-judged prompts")
    p.add_argument(
        "--max_bullets",
        type=int,
        default=None,
        help="Optional cap on bullets per prompt (default: all injected bullets)",
    )
    return p.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_jsonl(path: str, record: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _prompt_hash(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _prompt_to_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"prompt must be list-like, got {type(value).__name__}")
    out: list[dict[str, Any]] = []
    for msg in value:
        if isinstance(msg, dict):
            out.append({str(k): v for k, v in msg.items()})
        elif hasattr(msg, "items"):
            out.append({str(k): v for k, v in msg.items()})
        else:
            raise TypeError(f"prompt item must be dict-like, got {type(msg).__name__}")
    return out


def _find_role_content(prompt: list[dict[str, Any]], role: str) -> str:
    target = role.strip().lower()
    for msg in prompt:
        if str(msg.get("role", "")).strip().lower() == target:
            return str(msg.get("content", ""))
    return ""


def _norm_data_source(ds: Any) -> str:
    if isinstance(ds, (list, tuple, np.ndarray)) and len(ds) > 0:
        return str(ds[0])
    return str(ds) if ds is not None else "unknown"


def _problem_key(data_source: str, user_problem: str) -> str:
    return f"{data_source}::user::{user_problem}"


def _extract_v1_bullets(system_text: str) -> list[dict[str, Any]]:
    header = "## Relevant Strategies & Cautions"
    if header not in system_text:
        return []

    after_header = system_text.split(header, 1)[1]
    stop_markers = [
        "\n\nPlease reason step by step",
        "\n<reference_notes>",
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
        m = V1_BULLET_RE.match(lines[i].strip())
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


def _extract_std_rag_bullets(system_text: str) -> list[dict[str, Any]]:
    start = system_text.find("<reference_notes>")
    end = system_text.find("</reference_notes>")
    if start < 0 or end < 0 or end <= start:
        return []
    block = system_text[start:end]

    bullets: list[dict[str, Any]] = []
    section_type = ""
    lines = block.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        sm = SECTION_RE.match(line)
        if sm:
            sec = sm.group(1).strip().lower()
            section_type = "caution" if sec.startswith("caution") else "strategy"
            i += 1
            continue

        bm = PLAIN_BULLET_RE.match(line)
        if bm and section_type:
            principle = " ".join(bm.group(1).split())
            when_text = ""
            if i + 1 < len(lines):
                wm = WHEN_LINE_RE.match(lines[i + 1].strip())
                if wm:
                    when_text = " ".join(wm.group(1).split())
                    i += 1
            bullets.append(
                {
                    "id": len(bullets) + 1,
                    "type": section_type,
                    "principle": principle,
                    "when_to_apply": when_text,
                }
            )
        i += 1
    return bullets


def _extract_injected_bullets(system_text: str) -> list[dict[str, Any]]:
    bullets = _extract_v1_bullets(system_text)
    if bullets:
        return bullets
    return _extract_std_rag_bullets(system_text)


def _read_scores(path: str) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not path:
        return {}, {}
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"--scores not found: {path}")

    by_idx: dict[int, dict[str, Any]] = {}
    by_key: dict[str, dict[str, Any]] = {}
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            pidx = rec.get("problem_idx")
            if isinstance(pidx, int):
                by_idx[pidx] = rec
            pkey = rec.get("problem_key")
            if pkey:
                by_key[str(pkey)] = rec
    return by_idx, by_key


def _attach_rank_scores(
    bullets: list[dict[str, Any]],
    score_record: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not score_record:
        return bullets
    top_hits = score_record.get("top_k_hits")
    if not isinstance(top_hits, list):
        return bullets

    out: list[dict[str, Any]] = []
    for idx, bullet in enumerate(bullets):
        merged = dict(bullet)
        if idx < len(top_hits) and isinstance(top_hits[idx], dict):
            hit = top_hits[idx]
            merged["rank"] = hit.get("rank", idx)
            merged["entry_id"] = hit.get("entry_id", "")
            merged["bi_rank"] = hit.get("bi_rank")
            merged["rerank_rank"] = hit.get("rerank_rank")
            merged["cosine_score"] = hit.get("cosine_score")
            merged["bi_score"] = hit.get("bi_score", hit.get("cosine_score"))
            merged["rerank_score"] = hit.get("rerank_score")
        else:
            merged["rank"] = idx
            merged["entry_id"] = ""
            merged["bi_rank"] = None
            merged["rerank_rank"] = None
            merged["cosine_score"] = None
            merged["bi_score"] = None
            merged["rerank_score"] = None
        out.append(merged)
    return out


def _build_payloads(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    df = pd.read_parquet(args.parquet)
    if "prompt" not in df.columns:
        raise KeyError(f"input parquet has no 'prompt' column: {list(df.columns)}")
    if args.limit is not None and args.limit >= 0:
        df = df.iloc[: args.limit].copy()
    else:
        df = df.copy()

    by_idx, by_key = _read_scores(args.scores)

    payloads: list[dict[str, Any]] = []
    total_rows = len(df)
    rows_no_bullets = 0
    rows_with_scores = 0

    for row_idx, (i, row) in enumerate(df.iterrows()):
        prompt = _prompt_to_list(row["prompt"])
        user_problem = _find_role_content(prompt, "user").strip()
        system_text = _find_role_content(prompt, "system")
        bullets = _extract_injected_bullets(system_text)
        if not bullets:
            rows_no_bullets += 1
            continue

        if args.max_bullets is not None and args.max_bullets > 0:
            bullets = bullets[: args.max_bullets]

        data_source = _norm_data_source(row.get("data_source", "unknown"))
        problem_key = _problem_key(data_source, user_problem)
        pair_id = f"{problem_key}::prompt"

        score_record = by_idx.get(int(i)) or by_key.get(problem_key)
        if score_record:
            rows_with_scores += 1
        bullets = _attach_rank_scores(bullets, score_record)

        payloads.append(
            {
                "pair_id": pair_id,
                "problem_idx": int(i),
                "row_idx": int(row_idx),
                "problem_key": problem_key,
                "data_source": data_source,
                "user_problem": user_problem,
                "retrieval_mode": str((score_record or {}).get("retrieval_mode", "")),
                "retrieval_mode_base": str((score_record or {}).get("retrieval_mode_base", "")),
                "top_k": int((score_record or {}).get("top_k", len(bullets))),
                "selected_query_source": str(
                    (score_record or {}).get(
                        "selected_query_source",
                        (score_record or {}).get("chosen_query_source", ""),
                    )
                ),
                "selected_query_text": str((score_record or {}).get("selected_query_text", "")),
                "rerank_model": str((score_record or {}).get("rerank_model", "")),
                "rerank_base_mode": str((score_record or {}).get("rerank_base_mode", "")),
                "rerank_candidate_k": (
                    int((score_record or {}).get("rerank_candidate_k"))
                    if (score_record or {}).get("rerank_candidate_k") is not None
                    else None
                ),
                "injected_bullets": bullets,
            }
        )

    stats = {
        "rows_total": total_rows,
        "rows_selected": len(payloads),
        "rows_no_bullets": rows_no_bullets,
        "rows_with_scores": rows_with_scores,
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
            pid = obj.get("pair_id")
            if pid:
                done.add(str(pid))
    return done


def _format_bullets_for_prompt(bullets: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for b in bullets:
        lines.append(f"{b['id']}. [{b.get('type', 'strategy')}] {b.get('principle', '')}")
        when_text = str(b.get("when_to_apply", "")).strip()
        if when_text:
            lines.append(f"   when: {when_text}")
        rank = b.get("rank")
        bi_score = b.get("bi_score")
        rerank_score = b.get("rerank_score")
        meta_bits = []
        if rank is not None:
            meta_bits.append(f"rank={rank}")
        if bi_score is not None:
            try:
                meta_bits.append(f"bi_score={float(bi_score):.4f}")
            except (TypeError, ValueError):
                pass
        if rerank_score is not None:
            try:
                meta_bits.append(f"rerank_score={float(rerank_score):.4f}")
            except (TypeError, ValueError):
                pass
        if meta_bits:
            lines.append(f"   meta: {'; '.join(meta_bits)}")
    return "\n".join(lines)


def build_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    user_prompt = (
        "Problem:\n"
        f"{payload['user_problem']}\n\n"
        "Injected strategies/cautions (numbered):\n"
        f"{_format_bullets_for_prompt(payload['injected_bullets'])}\n\n"
        "Task:\n"
        "For each bullet ID, judge relevance and helpfulness for solving this problem. "
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


def _normalize_helpfulness(value: Any) -> str | None:
    text = str(value).strip().lower()
    aliases = {
        "high": "high",
        "very_high": "high",
        "very high": "high",
        "medium": "medium",
        "med": "medium",
        "moderate": "medium",
        "low": "low",
        "slight": "low",
        "none": "none",
        "not helpful": "none",
        "irrelevant": "none",
    }
    if text in aliases:
        return aliases[text]
    if text in {"3", "2", "1", "0"}:
        return {"3": "high", "2": "medium", "1": "low", "0": "none"}[text]
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
            relevant = _coerce_bool(item.get("relevant"))
            helpfulness = _normalize_helpfulness(item.get("helpfulness"))
            if relevant is None or helpfulness is None:
                continue
            reason = " ".join(str(item.get("brief_reason", "")).strip().split())
            parsed.append(
                {
                    "id": bid,
                    "relevant": relevant,
                    "helpfulness": helpfulness,
                    "brief_reason": reason,
                }
            )
            seen_ids.add(bid)

        if parsed:
            parsed.sort(key=lambda x: x["id"])
            parsed_ok = len(parsed) == int(n_bullets)
            return parsed, parsed_ok
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
                judged_bullets, parsed_ok = parse_judge_response(
                    raw_response, len(payload["injected_bullets"])
                )
                return {
                    "dump": {
                        "pair_id": payload["pair_id"],
                        "problem_key": payload["problem_key"],
                        "problem_idx": payload["problem_idx"],
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
                        "problem_idx": payload["problem_idx"],
                        "problem_key": payload["problem_key"],
                        "data_source": payload["data_source"],
                        "retrieval_mode": payload["retrieval_mode"],
                        "retrieval_mode_base": payload["retrieval_mode_base"],
                        "top_k": payload["top_k"],
                        "rerank_model": payload["rerank_model"],
                        "rerank_base_mode": payload["rerank_base_mode"],
                        "rerank_candidate_k": payload["rerank_candidate_k"],
                        "selected_query_source": payload["selected_query_source"],
                        "selected_query_text": payload["selected_query_text"],
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
                        "problem_idx": payload["problem_idx"],
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
                        "problem_idx": payload["problem_idx"],
                        "problem_key": payload["problem_key"],
                        "data_source": payload["data_source"],
                        "retrieval_mode": payload["retrieval_mode"],
                        "retrieval_mode_base": payload["retrieval_mode_base"],
                        "top_k": payload["top_k"],
                        "rerank_model": payload["rerank_model"],
                        "rerank_base_mode": payload["rerank_base_mode"],
                        "rerank_candidate_k": payload["rerank_candidate_k"],
                        "selected_query_source": payload["selected_query_source"],
                        "selected_query_text": payload["selected_query_text"],
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


async def run_judging(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")
    if args.max_concurrency <= 0:
        raise ValueError("--max_concurrency must be a positive integer.")

    payloads, stats = _build_payloads(args)
    done_pair_ids = _load_done_pair_ids(args.resume_from)
    if done_pair_ids:
        payloads = [p for p in payloads if p["pair_id"] not in done_pair_ids]

    print(
        "[info] rows_total={rows_total} rows_selected={rows_selected} rows_no_bullets={rows_no_bullets} "
        "rows_with_scores={rows_with_scores} pairs_after_resume={pairs_after_resume}".format(
            rows_total=stats["rows_total"],
            rows_selected=stats["rows_selected"],
            rows_no_bullets=stats["rows_no_bullets"],
            rows_with_scores=stats["rows_with_scores"],
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
    total_bullets = 0
    relevant_true = 0
    helpfulness_counter: Counter[str] = Counter()

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="judge-prompts"):
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
            for jb in judged_bullets:
                if not isinstance(jb, dict):
                    continue
                total_bullets += 1
                if bool(jb.get("relevant")):
                    relevant_true += 1
                helpfulness_counter[str(jb.get("helpfulness", ""))] += 1

    await client.close()

    avg_api_elapsed_sec = api_elapsed_sum_sec / len(payloads) if payloads else 0.0
    print(
        "[done] judging finished: "
        f"processed={len(payloads)}, parsed_ok={parsed_ok}, parsed_failed={parsed_failed}, "
        f"fallback_used_count={fallback_used_count}, "
        f"avg_api_elapsed_sec={avg_api_elapsed_sec:.2f}, max_api_elapsed_sec={api_elapsed_max_sec:.2f}"
    )
    if total_bullets > 0:
        print(
            "[summary] prompt_relevance: "
            f"total_bullets={total_bullets}, relevant_true={relevant_true} "
            f"({_safe_rate(relevant_true, total_bullets):.3f}), "
            f"high={helpfulness_counter['high']} "
            f"medium={helpfulness_counter['medium']} "
            f"low={helpfulness_counter['low']} "
            f"none={helpfulness_counter['none']}"
        )


def main() -> None:
    args = parse_args()
    asyncio.run(run_judging(args))


if __name__ == "__main__":
    main()
