#!/usr/bin/env python3
"""Local abstraction extraction via main_ppo validation (no API).

Subcommands:

1) ``build``: labeled trace parquet -> extraction-prompt parquet (``build_messages`` from
   ``extract.py``, same row filtering as ``10_extract.py``).
2) ``merge``: validation JSONL from ``main_ppo`` + original labeled parquet ->
   ``raw_abstractions.jsonl`` and ``raw_llm_dumps.jsonl`` (same shape as ``10_extract.py``).

Alignment uses the user-turn text + ``data_source``, matching ``25_rewrite_queries.merge``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from examples.hrlib.extract import build_messages, iter_payloads, parse_response, _prompt_hash
except ImportError:
    from extract import build_messages, iter_payloads, parse_response, _prompt_hash


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _alignment_key_user_text(text: str) -> str:
    return "".join(str(text).split())


def _extract_user_content(input_text: str) -> str:
    if not isinstance(input_text, str):
        return ""
    prefix = "user\n"
    suffix = "\nassistant\n"
    if input_text.startswith(prefix) and input_text.endswith(suffix):
        return input_text[len(prefix) : -len(suffix)]

    for sep in ("user\n\n", "user\n"):
        if sep in input_text:
            after = input_text.split(sep, 1)[1]
            for suf in ("assistant\n\n", "assistant\n", "\nassistant\n"):
                if after.endswith(suf):
                    after = after[: -len(suf)]
                    break
            return after.rstrip()

    legacy_suffix = "\nassistant\n"
    if input_text.endswith(legacy_suffix):
        body = input_text[: -len(legacy_suffix)]
        marker = "\nuser\n"
        pos = body.rfind(marker)
        if pos >= 0:
            return body[pos + len(marker) :]
        if body.startswith("user\n"):
            return body[len("user\n") :]

    return input_text


def _latest_jsonl_file(val_jsonl_dir: Path) -> Path:
    files = sorted(val_jsonl_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no .jsonl files found in {val_jsonl_dir}")
    return files[-1]


def _resolve_jsonl_source(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        return _latest_jsonl_file(path)
    raise FileNotFoundError(f"val jsonl path not found: {path}")


def _load_jsonl_entries(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    if not rows:
        raise ValueError(f"jsonl file is empty: {jsonl_path}")
    return rows


def _run_build(args: argparse.Namespace) -> int:
    if not args.input_path.exists():
        raise FileNotFoundError(f"--in not found: {args.input_path}")
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(f"--out already exists: {args.output_path} (pass --overwrite)")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_path)
    required_cols = {"prompt", "reward_model", "responses", "success_count"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in parquet: {missing}")

    total_rows = len(df)
    selected = _slice_by_class(df, args.limit_success, args.limit_failure)
    if selected.empty:
        raise ValueError("No rows selected after limit_success / limit_failure filters.")

    payloads = iter_payloads(selected)
    prompts: list[list[dict[str, str]]] = []
    for p in payloads:
        messages = build_messages(
            problem_text=p["problem_text"],
            ground_truth=p["ground_truth"],
            trace_text=p["trace_text"],
            is_correct=p["is_correct"],
        )
        prompts.append([{"role": m["role"], "content": str(m["content"])} for m in messages])

    out_df = selected.copy()
    out_df["prompt"] = prompts
    out_df.to_parquet(args.output_path, index=False)

    print(f"[self_extract/build] input           : {args.input_path}")
    print(f"[self_extract/build] output          : {args.output_path}")
    print(f"[self_extract/build] rows_in         : {total_rows}")
    print(f"[self_extract/build] rows_out        : {len(out_df)}")
    print(f"[self_extract/build] limit_success   : {args.limit_success}")
    print(f"[self_extract/build] limit_failure   : {args.limit_failure}")
    return 0


def _run_merge(args: argparse.Namespace) -> int:
    if not args.original.exists():
        raise FileNotFoundError(f"--original not found: {args.original}")

    for path, label in ((args.raw_abstractions_out, "raw_abstractions_out"), (args.raw_dump_out, "raw_dump_out")):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"--{label} already exists: {path} (pass --overwrite)")

    df = pd.read_parquet(args.original)
    required_cols = {"prompt", "reward_model", "responses", "success_count"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in parquet: {missing}")

    jsonl_path = _resolve_jsonl_source(args.val_jsonl_dir)
    entries = _load_jsonl_entries(jsonl_path)

    rewrites_by_key: dict[tuple[str, str], deque[str]] = defaultdict(deque)
    for item in entries:
        input_text = _extract_user_content(str(item.get("input", "")))
        data_source = str(item.get("data_source", ""))
        output_text = str(item.get("output", "")).strip()
        key = (_alignment_key_user_text(input_text), data_source)
        rewrites_by_key[key].append(output_text)

    args.raw_abstractions_out.parent.mkdir(parents=True, exist_ok=True)
    args.raw_dump_out.parent.mkdir(parents=True, exist_ok=True)

    model_label = (args.model_label or os.environ.get("MODEL_PATH") or "local").strip()

    if args.overwrite:
        args.raw_abstractions_out.write_text("", encoding="utf-8")
        args.raw_dump_out.write_text("", encoding="utf-8")

    matched = 0
    missing_gen = 0
    parsed_ok_n = 0
    parsed_fail_n = 0
    n_abstractions = 0

    def append_jsonl(path: Path, rec: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for _, row in df.iterrows():
        sub = pd.DataFrame([row])
        payload = iter_payloads(sub)[0]
        messages = build_messages(
            problem_text=payload["problem_text"],
            ground_truth=payload["ground_truth"],
            trace_text=payload["trace_text"],
            is_correct=payload["is_correct"],
        )
        user_content = str(messages[-1]["content"])
        data_source = str(row.get("data_source", "")) if hasattr(row, "get") else ""
        key = (_alignment_key_user_text(user_content), data_source)
        q = rewrites_by_key.get(key)

        if not q:
            missing_gen += 1
            continue

        raw_output = q.popleft().strip()
        matched += 1
        prompt_hash = _prompt_hash(messages)
        sid = payload["source_problem_id"]
        src_idx = payload["source_row_index"]
        src_diff = payload["source_difficulty"]

        abstractions, ok = parse_response(
            raw_text=raw_output,
            source_problem_id=sid,
            source_difficulty=src_diff,
            source_row_index=src_idx,
        )

        dump_record: dict[str, Any] = {
            "problem_id": sid,
            "source_row_index": src_idx,
            "source_difficulty": src_diff,
            "is_correct": bool(payload["is_correct"]),
            "model": model_label,
            "primary_model": model_label,
            "fallback_model": "",
            "fallback_used": False,
            "primary_error": "",
            "prompt_hash": prompt_hash,
            "raw_response": raw_output,
            "parsed_ok": ok,
            "error": "",
            "api_elapsed_sec": 0.0,
            "timestamp": _utc_now(),
        }

        append_jsonl(args.raw_dump_out, dump_record)

        if ok:
            parsed_ok_n += 1
            for abstraction in abstractions:
                append_jsonl(args.raw_abstractions_out, abstraction.to_dict())
                n_abstractions += 1
        else:
            parsed_fail_n += 1

    leftover = sum(len(v) for v in rewrites_by_key.values())
    if leftover > 0:
        print(
            f"[self_extract/merge] warning: {leftover} jsonl rows did not match any original parquet row "
            f"(or duplicate keys exhausted)"
        )

    meta_dir = args.raw_abstractions_out.parent / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"{args.raw_abstractions_out.stem}_self_extract_meta.json"
    meta = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "original_parquet": str(args.original),
        "val_jsonl": str(jsonl_path),
        "raw_abstractions": str(args.raw_abstractions_out),
        "raw_llm_dumps": str(args.raw_dump_out),
        "model_label": model_label,
        "total_parquet_rows": int(len(df)),
        "matched_rows": int(matched),
        "missing_generation_rows": int(missing_gen),
        "parsed_ok": int(parsed_ok_n),
        "parsed_failed": int(parsed_fail_n),
        "abstraction_lines": int(n_abstractions),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[self_extract/merge] original            : {args.original}")
    print(f"[self_extract/merge] val_jsonl           : {jsonl_path}")
    print(f"[self_extract/merge] raw_abstractions    : {args.raw_abstractions_out}")
    print(f"[self_extract/merge] raw_llm_dumps       : {args.raw_dump_out}")
    print(f"[self_extract/merge] matched_rows        : {matched} / {len(df)}")
    print(f"[self_extract/merge] missing_gen_rows    : {missing_gen}")
    print(f"[self_extract/merge] parsed_ok/fail      : {parsed_ok_n} / {parsed_fail_n}")
    print(f"[self_extract/merge] abstraction_lines   : {n_abstractions}")
    print(f"[self_extract/merge] meta                : {meta_path}")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build / merge local self-extraction parquet + val JSONL.")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build extraction-prompt parquet from labeled traces")
    b.add_argument("--in", dest="input_path", required=True, type=Path, help="labeled trace parquet")
    b.add_argument("--out", dest="output_path", required=True, type=Path, help="extraction prompt parquet")
    b.add_argument("--limit_success", type=int, default=None, help="max success rows (smoke)")
    b.add_argument("--limit_failure", type=int, default=None, help="max failure rows (smoke)")
    b.add_argument("--overwrite", action="store_true", help="allow clobbering output parquet")

    m = sub.add_parser("merge", help="merge val JSONL into raw_abstractions + raw_llm_dumps jsonl")
    m.add_argument("--original", required=True, type=Path, help="full labeled trace parquet (same schema as build --in)")
    m.add_argument(
        "--val_jsonl_dir",
        required=True,
        type=Path,
        help="validation jsonl file or directory (latest .jsonl if dir)",
    )
    m.add_argument(
        "--raw_abstractions_out",
        required=True,
        type=Path,
        help="output path for raw_abstractions.jsonl",
    )
    m.add_argument("--raw_dump_out", required=True, type=Path, help="output path for raw_llm_dumps.jsonl")
    m.add_argument(
        "--model_label",
        default="",
        help="string stored in dump records (default: MODEL_PATH env or 'local')",
    )
    m.add_argument("--overwrite", action="store_true", help="truncate output jsonl before writing")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "build":
        return _run_build(args)
    if args.command == "merge":
        return _run_merge(args)
    raise ValueError(f"unknown command: {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
