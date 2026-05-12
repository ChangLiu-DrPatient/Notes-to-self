#!/usr/bin/env python3
"""Utilities for Stage 0-3 abstraction extraction."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

try:
    from openai import APIStatusError, RateLimitError
except ImportError:  # pragma: no cover
    APIStatusError = type("APIStatusError", (Exception,), {})  # type: ignore[misc, assignment]
    RateLimitError = type("RateLimitError", (Exception,), {})  # type: ignore[misc, assignment]


def _should_retry_with_fallback(exc: BaseException) -> bool:
    """True when a second model may succeed (rate limits, quota, transient overload)."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError):
        code = getattr(exc, "status_code", None)
        if code in (402, 429, 503):
            return True
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return True
    if "402" in msg or "payment required" in msg or "insufficient" in msg:
        return True
    if "503" in msg or "overloaded" in msg or "unavailable" in msg:
        return True
    return False


SYSTEM_PROMPT = """You are extracting reusable mathematical reasoning abstractions from a model trace.

Your job is to read one math problem, the ground-truth final answer, one model-generated trace, and a correctness label telling you whether the trace is correct or incorrect.

Produce 1 or 2 reusable abstractions only. Each abstraction must be:
- a single sentence in "principle"
- generic and reusable across similar problems
- free of problem-specific numbers, variable names, answer values, or story details
- phrased as reasoning advice, not as a restatement of the solution

Allowed abstraction types:
- "strategy": a positive reasoning move that should be reused
- "caution": a mistake pattern or failure mode that should be avoided

Output a JSON array with 1 or 2 objects. Each object must contain exactly these keys:
- "name"
- "type"
- "principle"
- "when_to_apply"
- "domain"

Important rules:
- Do not copy phrases from the trace unless necessary.
- Do not mention the final answer explicitly.
- Do not mention concrete numbers unless they are universal constants like 0, 1, or pi.
- If the trace is correct, prefer strategy abstractions, but you may include one caution if the reasoning contains a risky pattern.
- If the trace is incorrect, prefer caution abstractions, but you may include one strategy if you can clearly infer a better reasoning move from the problem and the ground-truth answer.
- If there is only one strong abstraction, output only one.
"""

CORRECT_GUIDANCE = (
    "This trace reached the correct final answer. Focus on the key reasoning move(s) worth reusing in future "
    "problems. Prefer strategy abstractions. You may include one caution only if the trace contains a potentially "
    "dangerous shortcut that happened to work here."
)

INCORRECT_GUIDANCE = (
    "This trace did not reach the correct final answer. Focus on the main mistake pattern that should be avoided "
    "in future problems. Prefer caution abstractions. You may include one strategy only if a clearly better "
    "reasoning direction is recoverable from the problem and the ground-truth answer."
)

FENCED_JSON_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
FIRST_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


@dataclass
class Abstraction:
    name: str
    type: str
    principle: str
    when_to_apply: str
    domain: str
    source_problem_id: str
    source_difficulty: str
    source_row_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_messages(problem_text: str, ground_truth: str, trace_text: str, is_correct: bool) -> list[dict[str, str]]:
    """Build chat-completion messages with correctness-conditional guidance."""
    label = "correct" if is_correct else "incorrect"
    label_specific_guidance = CORRECT_GUIDANCE if is_correct else INCORRECT_GUIDANCE

    user_prompt = (
        "Problem:\n"
        f"{problem_text}\n\n"
        "Ground-truth final answer:\n"
        f"{ground_truth}\n\n"
        "Model trace:\n"
        f"{trace_text}\n\n"
        "Correctness label:\n"
        f"{label}\n\n"
        "Task:\n"
        "Extract 1 or 2 reusable one-sentence abstractions from this example.\n\n"
        "Additional guidance:\n"
        f"{label_specific_guidance}\n\n"
        "Return only a JSON array."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _extract_ground_truth(reward_model_obj: Any) -> str:
    if isinstance(reward_model_obj, dict):
        return str(reward_model_obj.get("ground_truth", ""))
    if isinstance(reward_model_obj, str):
        try:
            parsed = json.loads(reward_model_obj)
            if isinstance(parsed, dict):
                return str(parsed.get("ground_truth", ""))
        except json.JSONDecodeError:
            pass
    return ""


def _normalize_responses(responses_obj: Any) -> list[str]:
    if responses_obj is None:
        return []
    if isinstance(responses_obj, str):
        return [responses_obj]
    if hasattr(responses_obj, "tolist"):
        converted = responses_obj.tolist()
        if isinstance(converted, list):
            return [str(item) for item in converted]
    if isinstance(responses_obj, list):
        return [str(item) for item in responses_obj]
    if isinstance(responses_obj, tuple):
        return [str(item) for item in responses_obj]
    return [str(responses_obj)]


def _extract_problem_text(prompt_obj: Any) -> str:
    if hasattr(prompt_obj, "tolist"):
        prompt_obj = prompt_obj.tolist()
    if isinstance(prompt_obj, list):
        user_parts: list[str] = []
        for item in prompt_obj:
            if isinstance(item, dict) and item.get("role") == "user":
                user_parts.append(str(item.get("content", "")))
        if user_parts:
            return "\n\n".join(part for part in user_parts if part)
        if prompt_obj:
            return str(prompt_obj[0])
    if isinstance(prompt_obj, dict):
        if prompt_obj.get("role") == "user":
            return str(prompt_obj.get("content", ""))
        return str(prompt_obj.get("content", ""))
    return str(prompt_obj)


def _coerce_nonempty_text(value: Any) -> str:
    text = " ".join(str(value).strip().split())
    if not text:
        return ""
    if text.lower() in {"none", "nan", "null"}:
        return ""
    return text


def _extract_level(level_obj: Any, extra_info_obj: Any) -> str:
    level = _coerce_nonempty_text(level_obj)
    if level:
        return level

    if isinstance(extra_info_obj, dict):
        nested_level = _coerce_nonempty_text(extra_info_obj.get("level", ""))
        if nested_level:
            return nested_level
    if isinstance(extra_info_obj, str):
        try:
            parsed = json.loads(extra_info_obj)
            if isinstance(parsed, dict):
                nested_level = _coerce_nonempty_text(parsed.get("level", ""))
                if nested_level:
                    return nested_level
        except json.JSONDecodeError:
            pass
    return "unknown"


def iter_payloads(df: Any) -> list[dict[str, Any]]:
    """Convert labeled trace dataframe rows into extraction payloads."""
    payloads: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        responses = _normalize_responses(row.get("responses"))
        trace_text = responses[0] if responses else ""
        problem_text = _extract_problem_text(row.get("prompt"))
        ground_truth = _extract_ground_truth(row.get("reward_model"))
        is_correct = int(row.get("success_count", 0)) > 0

        source_row_index = int(idx)
        data_source = str(row.get("data_source", "?"))
        split = str(row.get("split", "train"))
        source_problem_id = f"{data_source}|{split}|{source_row_index}"
        source_difficulty = _extract_level(row.get("level"), row.get("extra_info"))

        payloads.append(
            {
                "problem_text": problem_text,
                "ground_truth": ground_truth,
                "trace_text": trace_text,
                "is_correct": is_correct,
                "source_problem_id": source_problem_id,
                "source_difficulty": source_difficulty,
                "source_row_index": source_row_index,
            }
        )
    return payloads


_SENTENCE_END_RE = re.compile(r"(?<=[.?])\s+(?=[A-Z])")


def _to_one_sentence(text: str) -> str:
    """Keep only the first sentence.

    Splits on ``.`` or ``?`` followed by whitespace and a capital letter. We
    intentionally do NOT split on ``!`` because mathematical principles often
    mention factorials (e.g., ``n!``, ``(n-1)!``) and splitting there silently
    truncated principles to unreadable fragments in Stage 0 extraction.
    """
    cleaned = " ".join(str(text).strip().split())
    if not cleaned:
        return ""
    parts = _SENTENCE_END_RE.split(cleaned, maxsplit=1)
    return parts[0].strip()


def _normalize_type(raw_type: str) -> str | None:
    value = str(raw_type).strip().lower()
    if value in {"strategy", "caution"}:
        return value
    if "strateg" in value:
        return "strategy"
    if "caution" in value or "mistake" in value or "error" in value:
        return "caution"
    return None


def _ensure_list_obj(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        abstractions = parsed.get("abstractions")
        if isinstance(abstractions, list):
            return [item for item in abstractions if isinstance(item, dict)]
    return []


def _candidate_json_strings(raw_text: str) -> list[str]:
    candidates = [raw_text]
    for match in FENCED_JSON_RE.findall(raw_text):
        if match:
            candidates.append(match.strip())
    first_array = FIRST_ARRAY_RE.search(raw_text)
    if first_array:
        candidates.append(first_array.group(0))
    return candidates


def _parse_json_like(raw_text: str) -> list[dict[str, Any]]:
    for candidate in _candidate_json_strings(raw_text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        items = _ensure_list_obj(parsed)
        if items:
            return items
    return []


def _build_abstraction(
    item: dict[str, Any],
    source_problem_id: str,
    source_difficulty: str,
    source_row_index: int,
) -> Abstraction | None:
    name = " ".join(str(item.get("name", "")).strip().split())
    kind = _normalize_type(str(item.get("type", "")))
    principle = _to_one_sentence(str(item.get("principle", "")))
    when_to_apply = " ".join(str(item.get("when_to_apply", "")).strip().split())
    domain = " ".join(str(item.get("domain", "")).strip().split())

    if not name or kind is None or not principle or not when_to_apply or not domain:
        return None
    return Abstraction(
        name=name,
        type=kind,
        principle=principle,
        when_to_apply=when_to_apply,
        domain=domain,
        source_problem_id=source_problem_id,
        source_difficulty=source_difficulty,
        source_row_index=source_row_index,
    )


def parse_response(
    raw_text: str,
    source_problem_id: str,
    source_difficulty: str,
    source_row_index: int,
) -> tuple[list[Abstraction], bool]:
    """Parse model output into at most two validated abstractions."""
    items = _parse_json_like(raw_text)
    abstractions: list[Abstraction] = []
    for item in items:
        abstraction = _build_abstraction(
            item=item,
            source_problem_id=source_problem_id,
            source_difficulty=source_difficulty,
            source_row_index=source_row_index,
        )
        if abstraction is not None:
            abstractions.append(abstraction)
        if len(abstractions) >= 2:
            break
    return abstractions, bool(abstractions)


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


def _prompt_hash(messages: list[dict[str, str]]) -> str:
    serialized = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def extract_one(
    client: Any,
    model: str,
    row_payload: dict[str, Any],
    semaphore: Any,
    *,
    fallback_model: str | None = None,
) -> dict[str, Any]:
    """Run one semaphore-guarded extraction request.

    If ``fallback_model`` is set and the primary model fails with rate limits / quota /
    overload-style errors, one retry is made with the fallback (typically paid
    ``openai/gpt-oss-120b`` after ``openai/gpt-oss-120b:free``).
    """
    messages = build_messages(
        problem_text=row_payload["problem_text"],
        ground_truth=row_payload["ground_truth"],
        trace_text=row_payload["trace_text"],
        is_correct=row_payload["is_correct"],
    )

    prompt_hash = _prompt_hash(messages)
    source_problem_id = row_payload["source_problem_id"]
    source_difficulty = row_payload["source_difficulty"]
    source_row_index = row_payload["source_row_index"]

    primary_model = model.strip()
    fb = (fallback_model or "").strip()
    if fb == primary_model:
        fb = ""
    model_chain = [primary_model]
    if fb:
        model_chain.append(fb)

    primary_error = ""
    last_error = ""
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
                abstractions, parsed_ok = parse_response(
                    raw_text=raw_response,
                    source_problem_id=source_problem_id,
                    source_difficulty=source_difficulty,
                    source_row_index=source_row_index,
                )
                return {
                    "dump": {
                        "problem_id": source_problem_id,
                        "source_row_index": source_row_index,
                        "source_difficulty": source_difficulty,
                        "is_correct": bool(row_payload["is_correct"]),
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
                    "abstractions": [item.to_dict() for item in abstractions],
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
                        "problem_id": source_problem_id,
                        "source_row_index": source_row_index,
                        "source_difficulty": source_difficulty,
                        "is_correct": bool(row_payload["is_correct"]),
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
                    "abstractions": [],
                }
