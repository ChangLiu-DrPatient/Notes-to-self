#!/usr/bin/env python3
"""Run cosine-vs-relevance diagnostics for an HRLib experiment matrix.

This is a thin batch wrapper around ``examples/hrlib/cosine_relevance_plot.py``. It
expects files named like:

  test_hrlib_minilm_orig_scores.jsonl
  eval_minilm_orig/judge/judge_results.jsonl

for each requested run label.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_RUNS = (
    "minilm_orig",
    "minilm_rewrite",
    "minilm_gated",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_root", type=Path, required=True, help="score-gate experiment root")
    p.add_argument("--runs", nargs="*", default=list(DEFAULT_RUNS), help="run labels to process")
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="matrix output directory (default: <run_root>/cosine_relevance_matrix)",
    )
    p.add_argument("--bins", type=int, default=80, help="histogram bins passed to cosine_relevance_plot.py")
    p.add_argument("--skip_missing", action="store_true", help="skip missing run inputs instead of failing")
    return p.parse_args()


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100 * float(value):.1f}%"


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def _run_plot(script_path: Path, run_root: Path, out_root: Path, run: str, bins: int) -> dict[str, Any]:
    scores = run_root / f"test_hrlib_{run}_scores.jsonl"
    judge_results = run_root / f"eval_{run}" / "judge" / "judge_results.jsonl"
    run_out = out_root / run
    summary_json = run_out / "cosine_relevance_stats.json"

    cmd = [
        sys.executable,
        str(script_path),
        "--scores",
        str(scores),
        "--judge_results",
        str(judge_results),
        "--out_dir",
        str(run_out),
        "--summary_json",
        str(summary_json),
        "--bins",
        str(bins),
    ]
    subprocess.run(cmd, check=True)
    with summary_json.open(encoding="utf-8") as f:
        summary = json.load(f)
    summary["run"] = run
    return summary


def _write_matrix_report(out_root: Path, summaries: list[dict[str, Any]]) -> Path:
    rows: list[list[Any]] = []
    for summary in summaries:
        top = summary["top_k_hits"]
        rel = top["relevant"]
        irrel = top["irrelevant"]
        labeled = rel.get("n", 0) + irrel.get("n", 0)
        rows.append(
            [
                summary["run"],
                labeled,
                _fmt_pct(summary.get("labeled_positive_rate")),
                _fmt_float(rel.get("mean")),
                _fmt_float(irrel.get("mean")),
                _fmt_float(summary.get("roc_auc")),
                _fmt_float(summary.get("best_f1_threshold")),
                _fmt_pct(summary.get("best_f1_precision")),
                _fmt_pct(summary.get("best_f1_recall")),
                top.get("unmatched_hits", 0),
            ]
        )

    lines = [
        "# Cosine Vs Relevance Matrix",
        "",
        _markdown_table(
            [
                "run",
                "labeled top-k",
                "relevant rate",
                "rel mean",
                "irrel mean",
                "ROC-AUC",
                "best-F1 threshold",
                "precision",
                "recall",
                "unmatched",
            ],
            rows,
        ),
        "",
        "Per-run plots and JSON summaries are in subdirectories named by run.",
        "",
    ]

    out_path = out_root / "cosine_relevance_matrix.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> int:
    args = _parse_args()
    run_root = args.run_root.expanduser().resolve()
    out_root = (args.out_dir or (run_root / "cosine_relevance_matrix")).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).with_name("cosine_relevance_plot.py")
    summaries: list[dict[str, Any]] = []
    for run in args.runs:
        scores = run_root / f"test_hrlib_{run}_scores.jsonl"
        judge_results = run_root / f"eval_{run}" / "judge" / "judge_results.jsonl"
        missing = [str(p) for p in (scores, judge_results) if not p.exists()]
        if missing:
            if args.skip_missing:
                print(f"[skip] {run}: missing {', '.join(missing)}")
                continue
            raise FileNotFoundError(f"missing inputs for {run}: {', '.join(missing)}")
        print(f"[run] {run}")
        summaries.append(_run_plot(script_path, run_root, out_root, run, args.bins))

    combined_json = out_root / "cosine_relevance_matrix.json"
    combined_json.write_text(json.dumps({"run_root": str(run_root), "runs": summaries}, indent=2) + "\n", encoding="utf-8")
    report_path = _write_matrix_report(out_root, summaries)
    print(f"wrote: {combined_json}")
    print(f"wrote: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
