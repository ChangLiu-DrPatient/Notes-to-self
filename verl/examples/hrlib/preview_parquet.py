#!/usr/bin/env python3
"""Quick peek at a VERL/HRLib parquet (shape, columns, sample rows).

Usage:
  python examples/hrlib/preview_parquet.py /path/to/file.parquet
  python examples/hrlib/preview_parquet.py $HOME/data/MATH-500/test_hrlib_v1.parquet

Or set PARQUET_PATH and run with no args.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

DEFAULT_PATH = os.environ.get(
    "PARQUET_PATH",
    "/home/changl9/data/MATH-500/test_hrlib_v1.parquet",
)


def _print_prompt_messages(prompt, max_content_chars: int = 2400) -> None:
    if prompt is None:
        print("  (empty)")
        return
    msgs = prompt.tolist() if hasattr(prompt, "tolist") else list(prompt)
    for i, m in enumerate(msgs):
        role = m.get("role", "?")
        content = str(m.get("content", ""))
        if len(content) > max_content_chars:
            content = content[:max_content_chars] + f"\n  ... ({len(content) - max_content_chars} more chars)"
        print(f"  --- [{i}] role={role!r} ---")
        for line in content.splitlines():
            print(f"    {line}")


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        print("Usage: python preview_parquet.py [PATH]", file=sys.stderr)
        return 1

    df = pd.read_parquet(path)
    print("path:", path)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print()
    print(df.head(3))
    print()

    if "success_count" in df.columns:
        n = len(df)
        n_zero = int((df["success_count"] == 0).sum())
        pct = 100.0 * n_zero / n if n else 0.0
        print(f"success_count==0: {n_zero} / {n} ({pct:.2f}%)")
    else:
        print("success_count: (no column — expected for raw MATH-500 / injected eval parquet)")

    if len(df) > 0:
        row_idx = 0
        print()
        print(f"prompt row[{row_idx}] (messages):")
        _print_prompt_messages(df["prompt"].iloc[row_idx])

    if "responses" in df.columns and len(df) > 0:
        print()
        print(f"responses row[{row_idx}]:")
        print(df["responses"].iloc[row_idx])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
