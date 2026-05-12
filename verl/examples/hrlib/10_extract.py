#!/usr/bin/env python3
"""Run abstraction extraction over labeled trace parquet."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

try:
    from examples.hrlib.extract import extract_one, iter_payloads
except ImportError:
    from extract import extract_one, iter_payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract abstractions from labeled traces via OpenRouter.")
    parser.add_argument("--traces", required=True, help="Input labeled parquet path.")
    parser.add_argument("--out", required=True, help="Output JSONL path for parsed abstractions.")
    parser.add_argument("--raw_dump_out", required=True, help="Output JSONL path for raw API responses.")
    parser.add_argument("--backend", default="openrouter", choices=["openrouter"], help="Inference backend.")
    parser.add_argument("--base_url", default="https://openrouter.ai/api/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--model", default="openai/gpt-oss-120b:free", help="Primary model ID (often :free).")
    parser.add_argument(
        "--fallback_model",
        default="openai/gpt-oss-120b",
        help="Paid model to call once if primary hits rate limit / quota (OpenRouter slug without :free).",
    )
    parser.add_argument(
        "--no_fallback",
        action="store_true",
        help="Disable fallback; only --model is used.",
    )
    parser.add_argument("--max_concurrency", type=int, default=4, help="Max in-flight API calls.")
    parser.add_argument("--resume_from", default="", help="Optional JSONL to recover completed problem IDs.")
    parser.add_argument("--limit_success", type=int, default=None, help="Limit success rows for smoke tests.")
    parser.add_argument("--limit_failure", type=int, default=None, help="Limit failure rows for smoke tests.")
    parser.add_argument("--seed", type=int, default=0, help="Reserved for future randomized sampling.")
    return parser.parse_args()


def _write_jsonl(path: str, record: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_done_ids(path: str) -> set[str]:
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
            source_problem_id = obj.get("source_problem_id") or obj.get("problem_id")
            if source_problem_id:
                done.add(str(source_problem_id))
    return done


def _slice_by_class(df: pd.DataFrame, limit_success: int | None, limit_failure: int | None) -> pd.DataFrame:
    success_df = df[df["success_count"] > 0]
    failure_df = df[df["success_count"] == 0]

    if limit_success is not None:
        success_df = success_df.head(limit_success)
    if limit_failure is not None:
        failure_df = failure_df.head(limit_failure)

    if success_df.empty and failure_df.empty:
        return df.iloc[0:0].copy()
    return pd.concat([success_df, failure_df], axis=0)


async def run_extraction(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")
    if args.max_concurrency <= 0:
        raise ValueError("--max_concurrency must be a positive integer.")

    df = pd.read_parquet(args.traces)
    required_cols = {"prompt", "reward_model", "responses", "success_count"}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in parquet: {missing}")

    selected = _slice_by_class(
        df=df,
        limit_success=args.limit_success,
        limit_failure=args.limit_failure,
    )
    payloads = iter_payloads(selected)

    done_ids = _load_done_ids(args.resume_from)
    if done_ids:
        payloads = [p for p in payloads if p["source_problem_id"] not in done_ids]

    total_success = sum(1 for p in payloads if p["is_correct"])
    total_failure = len(payloads) - total_success
    print(f"[info] selected payloads: {len(payloads)} (success={total_success}, failure={total_failure})")

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
            extract_one(
                client=client,
                model=args.model,
                row_payload=row_payload,
                semaphore=semaphore,
                fallback_model=fallback_model or None,
            )
        )
        for row_payload in payloads
    ]

    parsed_ok = 0
    parsed_failed = 0
    n_fallback_used = 0
    n_abstractions = 0
    api_elapsed_sum_sec = 0.0
    api_elapsed_max_sec = 0.0
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="extract"):
        result = await fut
        dump_record = result["dump"]
        abstractions = result["abstractions"]
        elapsed_sec = float(dump_record.get("api_elapsed_sec", 0.0))
        api_elapsed_sum_sec += elapsed_sec
        api_elapsed_max_sec = max(api_elapsed_max_sec, elapsed_sec)
        if dump_record.get("fallback_used"):
            n_fallback_used += 1

        _write_jsonl(args.raw_dump_out, dump_record)
        if dump_record.get("parsed_ok"):
            parsed_ok += 1
            for abstraction in abstractions:
                _write_jsonl(args.out, abstraction)
                n_abstractions += 1
        else:
            parsed_failed += 1

    await client.close()

    avg_api_elapsed_sec = api_elapsed_sum_sec / len(payloads) if payloads else 0.0
    print(
        "[done] extraction finished: "
        f"processed={len(payloads)}, success_rows={total_success}, failure_rows={total_failure}, "
        f"parsed_ok={parsed_ok}, parsed_failed={parsed_failed}, abstractions={n_abstractions}, "
        f"fallback_used_count={n_fallback_used}, "
        f"avg_api_elapsed_sec={avg_api_elapsed_sec:.2f}, max_api_elapsed_sec={api_elapsed_max_sec:.2f}"
    )


def main() -> None:
    args = parse_args()
    asyncio.run(run_extraction(args))


if __name__ == "__main__":
    main()
