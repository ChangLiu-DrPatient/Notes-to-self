#!/usr/bin/env python3
"""Re-parse raw_llm_dumps.jsonl into a fresh raw_abstractions.jsonl.

Motivation: the original `_to_one_sentence` in extract.py treated ``!`` as a
sentence terminator, which silently truncated any principle that mentioned a
factorial (e.g., ``"The prime divisors of n!"``). The raw LLM response is still
available inside `raw_llm_dumps.jsonl`, so we can regenerate the parsed
abstractions with the fixed parser without repaying API calls.

Usage (from the repo root `verl/`):

    python3 examples/hrlib/rebuild_abstractions.py \\
        --dumps /path/to/raw_llm_dumps.jsonl \\
        --out   /path/to/raw_abstractions.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from examples.hrlib.extract import parse_response
except ImportError:
    from extract import parse_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-parse raw_llm_dumps.jsonl with the current parser.")
    parser.add_argument("--dumps", required=True, help="Input raw_llm_dumps.jsonl.")
    parser.add_argument("--out", required=True, help="Output raw_abstractions.jsonl (will be overwritten).")
    parser.add_argument(
        "--keep_parse_failures_out",
        default="",
        help="Optional JSONL path to dump dump records that still fail to parse.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dumps_path = Path(args.dumps)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    failures_path = Path(args.keep_parse_failures_out) if args.keep_parse_failures_out else None
    if failures_path is not None:
        failures_path.parent.mkdir(parents=True, exist_ok=True)

    n_records = 0
    n_parsed_ok = 0
    n_parse_failed = 0
    n_abstractions = 0

    with (
        dumps_path.open("r", encoding="utf-8") as src,
        out_path.open("w", encoding="utf-8") as dst,
    ):
        failures_fp = failures_path.open("w", encoding="utf-8") if failures_path else None
        try:
            for raw_line in src:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    dump = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(dump, dict):
                    continue
                n_records += 1

                raw_response = str(dump.get("raw_response", ""))
                if not raw_response:
                    n_parse_failed += 1
                    if failures_fp is not None:
                        failures_fp.write(json.dumps(dump, ensure_ascii=False) + "\n")
                    continue

                source_problem_id = str(dump.get("problem_id", ""))
                try:
                    source_row_index = int(dump.get("source_row_index", -1))
                except (TypeError, ValueError):
                    source_row_index = -1
                source_difficulty = str(dump.get("source_difficulty", "unknown"))

                abstractions, parsed_ok = parse_response(
                    raw_text=raw_response,
                    source_problem_id=source_problem_id,
                    source_difficulty=source_difficulty,
                    source_row_index=source_row_index,
                )

                if parsed_ok:
                    n_parsed_ok += 1
                    for a in abstractions:
                        dst.write(json.dumps(a.to_dict(), ensure_ascii=False) + "\n")
                        n_abstractions += 1
                else:
                    n_parse_failed += 1
                    if failures_fp is not None:
                        failures_fp.write(json.dumps(dump, ensure_ascii=False) + "\n")
        finally:
            if failures_fp is not None:
                failures_fp.close()

    print(
        f"[rebuild] records={n_records} parsed_ok={n_parsed_ok} "
        f"parse_failed={n_parse_failed} abstractions={n_abstractions} "
        f"-> {out_path}"
    )
    if failures_path is not None:
        print(f"[rebuild] parse failures written to {failures_path}")


if __name__ == "__main__":
    main()
