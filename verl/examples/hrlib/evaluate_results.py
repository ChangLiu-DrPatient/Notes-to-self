#!/usr/bin/env python3
"""Unified post-eval/retrieval-result CLI for HRLib experiments.

This CLI consolidates HRLib analysis utilities under one entry point while
leaving runtime evaluation (`examples/hrlib/40_eval.sh`) unchanged.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _run_python(script_name: str, args: list[str]) -> int:
    script = _script_dir() / script_name
    cmd = [sys.executable, str(script), *args]
    proc = subprocess.run(cmd)
    return proc.returncode


def _run_python_capture(script_name: str, args: list[str]) -> tuple[int, str, str]:
    script = _script_dir() / script_name
    cmd = [sys.executable, str(script), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _run_lift_if_exists(
    run_root: Path,
    out_dir: Path,
    baseline_rel: str,
    treated_rel: str,
    out_name: str,
) -> dict[str, Any]:
    baseline = run_root / baseline_rel
    treated = run_root / treated_rel
    out_prefix = out_dir / "figs" / out_name
    entry: dict[str, Any] = {
        "baseline": str(baseline),
        "treated": str(treated),
        "out_prefix": str(out_prefix),
        "status": "skipped",
    }
    if not baseline.exists() or not treated.exists():
        entry["reason"] = "missing_input"
        return entry

    code = _run_python(
        "hrlib_abstraction_lift.py",
        ["--baseline", str(baseline), "--treated", str(treated), "--out_prefix", str(out_prefix)],
    )
    entry["status"] = "ok" if code == 0 else "failed"
    entry["exit_code"] = code
    return entry


def _all_cmd(args: argparse.Namespace) -> int:
    run_root = args.run_root.expanduser().resolve()
    out_dir = (args.out_dir.expanduser().resolve() if args.out_dir else (run_root / "result_eval_all"))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "run_root": str(run_root),
        "out_dir": str(out_dir),
        "commands": {},
        "judge_summaries": {},
        "lift_jobs": [],
    }

    rp_md = out_dir / "relevance_performance.md"
    rp_json = out_dir / "relevance_performance.json"
    rp_args = ["--run_root", str(run_root), "--out_md", str(rp_md), "--out_json", str(rp_json)]
    if args.runs:
        rp_args.extend(["--runs", *args.runs])
    code = _run_python(
        "hrlib_relevance_performance_analysis.py",
        rp_args,
    )
    summary["commands"]["relevance_performance"] = {
        "exit_code": code,
        "status": "ok" if code == 0 else "failed",
        "out_md": str(rp_md),
        "out_json": str(rp_json),
    }
    if code != 0:
        (out_dir / "result_eval_all_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return code

    cosine_dir = out_dir / "cosine_relevance_matrix"
    cosine_args = ["--run_root", str(run_root), "--out_dir", str(cosine_dir), "--bins", str(args.bins)]
    if args.runs:
        cosine_args.extend(["--runs", *args.runs])
    if args.skip_missing:
        cosine_args.append("--skip_missing")
    code = _run_python("cosine_relevance_matrix.py", cosine_args)
    summary["commands"]["cosine_matrix"] = {
        "exit_code": code,
        "status": "ok" if code == 0 else "failed",
        "out_dir": str(cosine_dir),
    }
    if code != 0:
        (out_dir / "result_eval_all_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return code

    lift_jobs: list[tuple[str, str, str]] = []
    run_set = set(args.runs or [])
    emb_prefixes = sorted({run.split("_", 1)[0] for run in run_set if "_" in run})
    for emb in emb_prefixes:
        orig = f"{emb}_orig"
        rewrite = f"{emb}_rewrite"
        gated = f"{emb}_gated"
        if orig in run_set and rewrite in run_set:
            lift_jobs.append(
                (
                    f"eval_{orig}/0.jsonl",
                    f"eval_{rewrite}/0.jsonl",
                    f"lift_{emb}_rewrite_vs_orig",
                )
            )
        if orig in run_set and gated in run_set:
            lift_jobs.append(
                (
                    f"eval_{orig}/0.jsonl",
                    f"eval_{gated}/0.jsonl",
                    f"lift_{emb}_gated_vs_orig",
                )
            )

    vanilla = run_root.parent.parent / "vanilla_tokens" / "0.jsonl"
    if vanilla.exists():
        for emb in emb_prefixes:
            gated = f"{emb}_gated"
            if gated in run_set:
                lift_jobs.append(
                    (
                        str(vanilla),
                        f"eval_{gated}/0.jsonl",
                        f"lift_{emb}_gated_vs_vanilla",
                    )
                )

    for baseline_rel, treated_rel, out_name in lift_jobs:
        if baseline_rel.startswith("/"):
            baseline_path = Path(baseline_rel)
            treated_path = run_root / treated_rel
            out_prefix = out_dir / "figs" / out_name
            entry = {
                "baseline": str(baseline_path),
                "treated": str(treated_path),
                "out_prefix": str(out_prefix),
                "status": "skipped",
            }
            if baseline_path.exists() and treated_path.exists():
                code = _run_python(
                    "hrlib_abstraction_lift.py",
                    ["--baseline", str(baseline_path), "--treated", str(treated_path), "--out_prefix", str(out_prefix)],
                )
                entry["status"] = "ok" if code == 0 else "failed"
                entry["exit_code"] = code
            else:
                entry["reason"] = "missing_input"
            summary["lift_jobs"].append(entry)
        else:
            summary["lift_jobs"].append(_run_lift_if_exists(run_root, out_dir, baseline_rel, treated_rel, out_name))

    failed_lifts = [job for job in summary["lift_jobs"] if job.get("status") == "failed"]
    if failed_lifts:
        (out_dir / "result_eval_all_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return 1

    judge_dir = out_dir / "judge_summaries"
    judge_dir.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        judge_path = run_root / f"eval_{run}" / "judge" / "judge_results.jsonl"
        out_txt = judge_dir / f"{run}.txt"
        if not judge_path.exists():
            summary["judge_summaries"][run] = {"status": "skipped", "reason": "missing_input"}
            continue
        code, stdout, stderr = _run_python_capture("analyze_judge_results.py", [str(judge_path)])
        out_txt.write_text(stdout + (("\n[stderr]\n" + stderr) if stderr else ""), encoding="utf-8")
        summary["judge_summaries"][run] = {
            "status": "ok" if code == 0 else "failed",
            "exit_code": code,
            "out_txt": str(out_txt),
        }
        if code != 0:
            (out_dir / "result_eval_all_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            return code

    (out_dir / "result_eval_all_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {out_dir / 'result_eval_all_summary.json'}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in (
        "lift",
        "judge-summary",
        "cosine-plot",
        "cosine-matrix",
        "relevance-performance",
        "output-stats",
        "judge-run",
    ):
        p = sub.add_parser(name, help=f"Proxy to {name} script")
        p.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed through to the underlying script")

    all_p = sub.add_parser("all", help="Run a multifaceted analysis suite over one run root")
    all_p.add_argument("--run_root", type=Path, required=True, help="run root directory")
    all_p.add_argument("--out_dir", type=Path, default=None, help="output directory (default: <run_root>/result_eval_all)")
    all_p.add_argument("--bins", type=int, default=80, help="histogram bins for cosine diagnostics")
    all_p.add_argument("--skip_missing", action="store_true", help="skip missing inputs in cosine matrix")
    all_p.add_argument(
        "--runs",
        nargs="*",
        default=["minilm_orig", "minilm_rewrite", "minilm_gated"],
        help="run labels for judge summary collection",
    )

    return parser.parse_args()


def _strip_leading_double_dash(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def main() -> int:
    args = _parse_args()
    if args.cmd == "all":
        return _all_cmd(args)

    mapping = {
        "lift": "hrlib_abstraction_lift.py",
        "judge-summary": "analyze_judge_results.py",
        "cosine-plot": "cosine_relevance_plot.py",
        "cosine-matrix": "cosine_relevance_matrix.py",
        "relevance-performance": "hrlib_relevance_performance_analysis.py",
        "output-stats": "hrlib_token_output_stats.py",
        "judge-run": "judge_abstraction_use.py",
    }
    forwarded = _strip_leading_double_dash(args.args)
    return _run_python(mapping[args.cmd], forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
