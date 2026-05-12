#!/usr/bin/env python3
"""Build and merge query-rewrite artifacts for HRLib retrieval experiments.

This script has two subcommands:

1) ``build``: convert an input prompt parquet into a rewrite-prompt parquet
   (one user instruction per row asking the model for concept keywords).
2) ``merge``: take validation JSONL generations from ``main_ppo`` and replace
   original user turns with generated rewrite text.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd

_ASY_BLOCK_RE = re.compile(r"\[asy\].*?\[/asy\]", flags=re.IGNORECASE | re.DOTALL)
_REWRITE_SYSTEM_PROMPT = (
    "You are a math tutor writing retrieval abstractions. "
    "Output exactly one line, one sentence, in this format: <Type>: <short abstraction>, "
    "where <Type> is either Strategy or Caution. "
    "For example \"Check extraneous roots\" ."
    "Do not solve the problem and do not provide a final answer."
)


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


def _find_message_content(prompt: list[dict[str, Any]], role: str) -> str:
    target = role.strip().lower()
    for msg in prompt:
        if str(msg.get("role", "")).strip().lower() == target:
            return str(msg.get("content", ""))
    raise ValueError(f"prompt has no message with role={role!r}")


def _replace_message_content(prompt: list[dict[str, Any]], role: str, content: str) -> list[dict[str, Any]]:
    target = role.strip().lower()
    out = [dict(m) for m in prompt]
    for i, msg in enumerate(out):
        if str(msg.get("role", "")).strip().lower() == target:
            out[i]["content"] = content
            return out
    raise ValueError(f"prompt has no message with role={role!r}")


def _clean_problem(problem_text: str) -> str:
    text = str(problem_text)
    text = _ASY_BLOCK_RE.sub(" ", text)
    text = " ".join(text.split())
    if len(text) > 500:
        text = text[:500].rstrip() + " ..."
    return text


def _rewrite_instruction(problem_text: str) -> str:
    text = _clean_problem(problem_text)
    return f"Problem: {text}\nAbstraction (Strategy or Caution):"


def _rewrite_prompt_messages(problem_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": _rewrite_instruction(problem_text)},
    ]


def _extract_user_content(input_text: str) -> str:
    if not isinstance(input_text, str):
        return ""
    suffix = "\nassistant\n"
    if input_text.endswith(suffix):
        body = input_text[: -len(suffix)]
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build/merge parquet files for query rewriting.")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build rewrite-prompt parquet from an input parquet")
    b.add_argument("--in", dest="input_path", required=True, type=Path, help="source parquet")
    b.add_argument("--out", dest="output_path", required=True, type=Path, help="rewrite prompt parquet")
    b.add_argument("--query_from", default="user", help="role whose content is rewritten")
    b.add_argument("--limit", type=int, default=None, help="process only first N rows")
    b.add_argument("--overwrite", action="store_true", help="allow clobbering output parquet")

    m = sub.add_parser("merge", help="merge main_ppo val JSONL generations into original parquet")
    m.add_argument("--original", required=True, type=Path, help="original parquet")
    m.add_argument(
        "--val_jsonl_dir",
        required=True,
        type=Path,
        help="directory with validation jsonl dumps (or a single jsonl file)",
    )
    m.add_argument("--out", dest="output_path", required=True, type=Path, help="rewritten parquet")
    m.add_argument("--query_from", default="user", help="role whose content gets replaced")
    m.add_argument("--overwrite", action="store_true", help="allow clobbering output parquet")

    return p.parse_args(argv)


def _run_build(args: argparse.Namespace) -> int:
    if not args.input_path.exists():
        raise FileNotFoundError(f"--in not found: {args.input_path}")
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(f"--out already exists: {args.output_path} (pass --overwrite)")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_path)
    total_rows = len(df)
    if "prompt" not in df.columns:
        raise KeyError(f"input parquet has no 'prompt' column: {list(df.columns)}")

    if args.limit is not None and args.limit >= 0:
        df = df.iloc[: args.limit].copy()
    else:
        df = df.copy()

    rewrite_prompts: list[list[dict[str, Any]]] = []
    for _, row in df.iterrows():
        prompt = _prompt_to_list(row["prompt"])
        problem_text = _find_message_content(prompt, args.query_from)
        rewrite_prompts.append(_rewrite_prompt_messages(problem_text))

    df["prompt"] = rewrite_prompts
    df.to_parquet(args.output_path, index=False)

    print(f"[rewrite/build] input      : {args.input_path}")
    print(f"[rewrite/build] output     : {args.output_path}")
    print(f"[rewrite/build] rows       : {len(df)} (from {total_rows})")
    print(f"[rewrite/build] query_from : {args.query_from}")
    return 0


def _run_merge(args: argparse.Namespace) -> int:
    if not args.original.exists():
        raise FileNotFoundError(f"--original not found: {args.original}")
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(f"--out already exists: {args.output_path} (pass --overwrite)")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    jsonl_path = _resolve_jsonl_source(args.val_jsonl_dir)
    entries = _load_jsonl_entries(jsonl_path)

    rewrites_by_key: dict[tuple[str, str], deque[str]] = defaultdict(deque)
    for item in entries:
        input_text = _extract_user_content(str(item.get("input", "")))
        data_source = str(item.get("data_source", ""))
        output_text = str(item.get("output", "")).strip()
        rewrites_by_key[(input_text, data_source)].append(output_text)

    df = pd.read_parquet(args.original).copy()
    if "prompt" not in df.columns:
        raise KeyError(f"original parquet has no 'prompt' column: {list(df.columns)}")

    rewritten_prompts: list[list[dict[str, Any]]] = []
    rewritten_rows = 0
    missing_rows = 0

    for _, row in df.iterrows():
        prompt = _prompt_to_list(row["prompt"])
        original_text = _find_message_content(prompt, args.query_from)
        instruction = _rewrite_instruction(original_text)
        data_source = str(row.get("data_source", "")) if hasattr(row, "get") else ""
        key = (instruction, data_source)
        q = rewrites_by_key.get(key)

        if q:
            rewritten_text = q.popleft().strip()
            if rewritten_text:
                rewritten_prompts.append(
                    _replace_message_content(prompt, args.query_from, rewritten_text)
                )
                rewritten_rows += 1
                continue

        rewritten_prompts.append(prompt)
        missing_rows += 1

    unmatched = sum(len(v) for v in rewrites_by_key.values())
    if unmatched > 0:
        print(
            "[rewrite/merge] warning: "
            f"{unmatched} generated rows from {jsonl_path} did not match original parquet rows"
        )

    df["prompt"] = rewritten_prompts
    df.to_parquet(args.output_path, index=False)

    meta_path = args.output_path.with_name(args.output_path.stem + "_rewrite_meta.json")
    meta = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "original_parquet": str(args.original),
        "val_jsonl": str(jsonl_path),
        "output_parquet": str(args.output_path),
        "query_from": args.query_from,
        "total_rows": int(len(df)),
        "rewritten_rows": int(rewritten_rows),
        "missing_rows": int(missing_rows),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[rewrite/merge] original   : {args.original}")
    print(f"[rewrite/merge] val_jsonl   : {jsonl_path}")
    print(f"[rewrite/merge] output      : {args.output_path}")
    print(f"[rewrite/merge] rewritten   : {rewritten_rows} / {len(df)}")
    print(f"[rewrite/merge] missing     : {missing_rows}")
    print(f"[rewrite/merge] meta        : {meta_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "build":
        return _run_build(args)
    if args.command == "merge":
        return _run_merge(args)
    raise ValueError(f"unknown command: {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
