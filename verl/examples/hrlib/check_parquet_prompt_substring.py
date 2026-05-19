#!/usr/bin/env python3
"""Scan a parquet dataset for rows whose prompt contains a substring (default: x^4+2y^2 frac)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _prompt_as_search_text(prompt_obj: Any) -> str:
    """Flatten prompt column (list of chat dicts, etc.) into one string for substring search."""
    if hasattr(prompt_obj, "tolist"):
        prompt_obj = prompt_obj.tolist()
    if isinstance(prompt_obj, list):
        parts = []
        for item in prompt_obj:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(prompt_obj, dict):
        return str(prompt_obj.get("content", prompt_obj))
    return str(prompt_obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find parquet rows whose prompt contains a substring.")
    parser.add_argument(
        "--parquet",
        type=str,
        default=str(Path.home() / "data/math/train.parquet"),
        help="Path to parquet file",
    )
    parser.add_argument(
        "--needle",
        type=str,
        default=r"$\frac{x^4+2y^2}{6}$",
        help=r'Default: $\\frac{x^4+2y^2}{6}$ inside math mode (JSON/storage uses single backslash before frac)',
    )
    parser.add_argument("--max-report", type=int, default=20, help="Max matching row indices to print")
    args = parser.parse_args()

    path = Path(args.parquet).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    if "prompt" not in df.columns:
        raise ValueError(f"No 'prompt' column in {path}; columns={list(df.columns)}")

    needle = args.needle
    hits: list[int] = []
    for i in range(len(df)):
        text = _prompt_as_search_text(df.iloc[i]["prompt"])
        if needle in text:
            hits.append(i)

    print(f"file={path}")
    print(f"rows={len(df)} needle={needle!r}")
    print(f"matches={len(hits)}")
    if hits:
        show = hits[: args.max_report]
        print(f"first_row_indices ({len(show)} shown): {show}")
        # Optional: tiny snippet around needle for first hit
        i0 = hits[0]
        t0 = _prompt_as_search_text(df.iloc[i0]["prompt"])
        pos = t0.index(needle)
        lo = max(0, pos - 60)
        hi = min(len(t0), pos + len(needle) + 60)
        print(f"context[row {i0}]: ...{t0[lo:hi]!r}...")
    else:
        # Help debug escaping: report if relaxed fragment matches
        frag = r"\frac{x^4+2y^2}{6}"
        frag_hits = sum(frag in _prompt_as_search_text(df.iloc[i]["prompt"]) for i in range(len(df)))
        print(f"(hint) rows containing substring {frag!r}: {frag_hits}")


if __name__ == "__main__":
    main()
