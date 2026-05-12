#!/usr/bin/env python3
"""Output-length stats for vanilla vs treated validation JSONLs.

Combines:

1. **Character count** of ``output`` (cheap proxy; always computed).
2. **Token counts** when present on every rollout of every common problem in *both*
   JSONLs: ``response_token_count`` and ``prompt_token_count`` (from
   ``RayPPOTrainer._validate`` dumps in recent verl).

Joins:

- **Lift-style buckets** (recomputed from baseline + treated JSONLs): win, regression,
  steady_both_pass, steady_both_fail (same rules as ``hrlib_abstraction_lift.compare``).
- **LLM-judge buckets** (optional ``judge_results.jsonl``): effective relevant use vs not.

Outputs ``{out_prefix}_token_output_report.json`` and ``_token_output_report.md``.

Run from ``verl/``::

    conda activate verl
    cd verl
    python examples/hrlib/hrlib_token_output_stats.py \\
        --baseline /raid/\\$USER/eval/hrlib/stage0/vanilla_tokens/0.jsonl \\
        --treated  /raid/\\$USER/eval/hrlib/stage0/v1_top4/0.jsonl \\
        --judge    /raid/\\$USER/eval/hrlib/stage0/v1/judge_abstraction_use/judge_results.jsonl \\
        --out_prefix Figs/hrlib_stage0/lift_v1_top4
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from typing import Any, Callable, Literal

import numpy as np

try:
    from examples.hrlib.hrlib_abstraction_lift import load_grouped, summarize_problem
except ImportError:
    from hrlib_abstraction_lift import load_grouped, summarize_problem

LiftBucket = Literal["win", "regression", "steady_both_pass", "steady_both_fail"]
JudgeBucket = Literal[
    "effective_relevant_use",
    "not_effective",
    "no_relevant_bullets_judged",
    "judge_parse_failed",
    "judge_unavailable",
]


def _str_output(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        return "\n\n---\n\n".join(_str_output(x) for x in val)
    return str(val)


def rollout_char_lengths(rows: list[dict[str, Any]]) -> list[int]:
    return [len(_str_output(r.get("output"))) for r in rows]


def _numeric_field(row: dict[str, Any], key: str) -> int | None:
    v = row.get(key)
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _all_rows_have_field(rows: list[dict[str, Any]], key: str) -> bool:
    for r in rows:
        if _numeric_field(r, key) is None:
            return False
    return True


def coverage_for_common(
    base_g: dict[str, list[dict[str, Any]]],
    tre_g: dict[str, list[dict[str, Any]]],
    common: list[str],
    key: str,
) -> bool:
    for k in common:
        if k not in base_g or k not in tre_g:
            return False
        if not _all_rows_have_field(base_g[k], key):
            return False
        if not _all_rows_have_field(tre_g[k], key):
            return False
    return True


def per_problem_mean_numeric(rows: list[dict[str, Any]], key: str) -> float:
    vals = [_numeric_field(r, key) for r in rows]
    if not vals or any(v is None for v in vals):
        return float("nan")
    return float(np.mean(vals))


def per_problem_mean_chars(rows: list[dict[str, Any]]) -> float:
    lens = rollout_char_lengths(rows)
    return float(np.mean(lens)) if lens else float("nan")


def pctile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = int(round(p * (len(sorted_vals) - 1)))
    i = max(0, min(i, len(sorted_vals) - 1))
    return float(sorted_vals[i])


def summarize_lengths(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    s = sorted(values)
    return {
        "n": len(s),
        "mean": float(statistics.mean(s)),
        "median": float(statistics.median(s)),
        "p90": pctile(s, 0.9),
        "min": float(s[0]),
        "max": float(s[-1]),
    }


def lift_bucket_for(
    key: str,
    base_g: dict[str, list[dict[str, Any]]],
    tre_g: dict[str, list[dict[str, Any]]],
    *,
    score_threshold: float,
) -> LiftBucket | None:
    if key not in base_g or key not in tre_g:
        return None
    sb = summarize_problem(key, base_g[key], score_threshold=score_threshold)
    st = summarize_problem(key, tre_g[key], score_threshold=score_threshold)
    if sb.pass_at_n == 0 and st.pass_at_n == 1:
        return "win"
    if sb.pass_at_n == 1 and st.pass_at_n == 0:
        return "regression"
    if sb.pass_at_n == 1 and st.pass_at_n == 1:
        return "steady_both_pass"
    return "steady_both_fail"


def load_judge_labels(path: str | None) -> dict[str, dict[str, Any]]:
    """One row per problem_key: merge all JSONL lines for that key (any rollout)."""
    if not path or not os.path.isfile(path):
        return {}

    by_pk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pk = str(rec.get("problem_key", ""))
            if pk:
                by_pk[pk].append(rec)

    out: dict[str, dict[str, Any]] = {}
    for pk, rows in by_pk.items():
        if not rows:
            continue
        any_effective = False
        any_relevant = False
        any_misuse = False
        any_parsed = False
        for rec in rows:
            if not rec.get("parsed_ok"):
                continue
            any_parsed = True
            for b in rec.get("judged_bullets") or []:
                if not isinstance(b, dict):
                    continue
                rel = bool(b.get("relevant"))
                usage = str(b.get("usage", ""))
                if rel:
                    any_relevant = True
                if rel and usage == "used_correctly":
                    any_effective = True
                if rel and usage == "used_incorrectly":
                    any_misuse = True

        if not any_parsed:
            jbucket: JudgeBucket = "judge_parse_failed"
        elif any_effective:
            jbucket = "effective_relevant_use"
        elif not any_relevant:
            jbucket = "no_relevant_bullets_judged"
        else:
            jbucket = "not_effective"

        out[pk] = {
            "judge_bucket": jbucket,
            "any_relevant_bullet": any_relevant,
            "any_relevant_misuse": any_misuse,
            "n_judge_rows": len(rows),
        }
    return out


def cross_label(lift: LiftBucket, judge: JudgeBucket | Literal["judge_unavailable"]) -> str:
    return f"{lift}|{judge}"


def _agg_unified(
    pred: Callable[[dict[str, Any]], bool],
    records: list[dict[str, Any]],
    base_vals: list[float],
    tre_vals: list[float],
) -> dict[str, Any]:
    sub_b = [base_vals[i] for i, r in enumerate(records) if pred(r)]
    sub_t = [tre_vals[i] for i, r in enumerate(records) if pred(r)]
    sub_d = [tre_vals[i] - base_vals[i] for i, r in enumerate(records) if pred(r)]
    return {
        "n_problems": len(sub_d),
        "baseline": summarize_lengths(sub_b),
        "treated": summarize_lengths(sub_t),
        "delta_treated_minus_baseline": summarize_lengths(sub_d),
    }


def _fmt_unified_block(title: str, block: dict[str, Any], *, unit: str) -> list[str]:
    lines = [f"### {title}", ""]
    lines.append(f"- n_problems: **{block.get('n_problems', '—')}**")
    for label, key in (
        ("baseline", "baseline"),
        ("treated", "treated"),
        ("Δ(treated−baseline)", "delta_treated_minus_baseline"),
    ):
        s = block.get(key) or {}
        if not s.get("n"):
            lines.append(f"- **{label}** ({unit}): (empty)")
            continue
        lines.append(
            f"- **{label}** ({unit}): mean={s['mean']:.1f}, median={s['median']:.1f}, "
            f"p90={s['p90']:.1f}, min={s['min']:.1f}, max={s['max']:.1f}"
        )
    lines.append("")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, help="Vanilla validation JSONL.")
    p.add_argument("--treated", required=True, help="Treated (e.g. v1_dom) validation JSONL.")
    p.add_argument(
        "--judge",
        default="",
        help="Optional judge_results.jsonl from judge_abstraction_use.py.",
    )
    p.add_argument(
        "--out_prefix",
        required=True,
        help="Writes {out_prefix}_token_output_report.json and .md",
    )
    p.add_argument(
        "--score_threshold",
        type=float,
        default=1.0,
        help="Rollout considered correct if score >= this (default 1.0).",
    )
    args = p.parse_args()

    base_g = load_grouped(args.baseline)
    tre_g = load_grouped(args.treated)
    common = sorted(set(base_g) & set(tre_g))
    judge_map = load_judge_labels(args.judge.strip() or None)

    resp_tok_ok = coverage_for_common(base_g, tre_g, common, "response_token_count")
    prompt_tok_ok = coverage_for_common(base_g, tre_g, common, "prompt_token_count")

    records: list[dict[str, Any]] = []
    base_means_chars: list[float] = []
    tre_means_chars: list[float] = []
    deltas_chars: list[float] = []
    base_means_rtok: list[float] = []
    tre_means_rtok: list[float] = []
    deltas_rtok: list[float] = []
    base_means_ptok: list[float] = []
    tre_means_ptok: list[float] = []
    deltas_ptok: list[float] = []

    for key in common:
        lb = lift_bucket_for(key, base_g, tre_g, score_threshold=args.score_threshold)
        if lb is None:
            continue
        bmc = per_problem_mean_chars(base_g[key])
        tmc = per_problem_mean_chars(tre_g[key])
        base_means_chars.append(bmc)
        tre_means_chars.append(tmc)
        deltas_chars.append(tmc - bmc)

        if resp_tok_ok:
            bmr = per_problem_mean_numeric(base_g[key], "response_token_count")
            tmr = per_problem_mean_numeric(tre_g[key], "response_token_count")
            base_means_rtok.append(bmr)
            tre_means_rtok.append(tmr)
            deltas_rtok.append(tmr - bmr)
        if prompt_tok_ok:
            bmp = per_problem_mean_numeric(base_g[key], "prompt_token_count")
            tmp = per_problem_mean_numeric(tre_g[key], "prompt_token_count")
            base_means_ptok.append(bmp)
            tre_means_ptok.append(tmp)
            deltas_ptok.append(tmp - bmp)

        if key in judge_map:
            jb = str(judge_map[key]["judge_bucket"])
        else:
            jb = "judge_unavailable"

        rec: dict[str, Any] = {
            "problem_key": key,
            "lift_bucket": lb,
            "judge_bucket": jb,
            "cross": cross_label(lb, jb),  # type: ignore[arg-type]
            "baseline_mean_output_chars": bmc,
            "treated_mean_output_chars": tmc,
            "delta_treated_minus_baseline_chars": tmc - bmc,
        }
        if resp_tok_ok:
            rec["baseline_mean_response_tokens"] = bmr
            rec["treated_mean_response_tokens"] = tmr
            rec["delta_treated_minus_baseline_response_tokens"] = tmr - bmr
        if prompt_tok_ok:
            rec["baseline_mean_prompt_tokens"] = bmp
            rec["treated_mean_prompt_tokens"] = tmp
            rec["delta_treated_minus_baseline_prompt_tokens"] = tmp - bmp
        records.append(rec)

    all_base_chars: list[int] = []
    all_tre_chars: list[int] = []
    all_base_rtok: list[int] = []
    all_tre_rtok: list[int] = []
    all_base_ptok: list[int] = []
    all_tre_ptok: list[int] = []
    for key in common:
        all_base_chars.extend(rollout_char_lengths(base_g[key]))
        all_tre_chars.extend(rollout_char_lengths(tre_g[key]))
        if resp_tok_ok:
            for r in base_g[key]:
                v = _numeric_field(r, "response_token_count")
                if v is not None:
                    all_base_rtok.append(v)
            for r in tre_g[key]:
                v = _numeric_field(r, "response_token_count")
                if v is not None:
                    all_tre_rtok.append(v)
        if prompt_tok_ok:
            for r in base_g[key]:
                v = _numeric_field(r, "prompt_token_count")
                if v is not None:
                    all_base_ptok.append(v)
            for r in tre_g[key]:
                v = _numeric_field(r, "prompt_token_count")
                if v is not None:
                    all_tre_ptok.append(v)

    report: dict[str, Any] = {
        "note": (
            "Per-problem stats use the mean over rollouts for that problem. "
            "Token fields require response_token_count / prompt_token_count on every rollout "
            "of every common problem in both JSONLs; otherwise that metric is omitted."
        ),
        "length_units": {
            "output_chars": "always_computed",
            "response_tokens": "present" if resp_tok_ok else "missing_or_incomplete",
            "prompt_tokens": "present" if prompt_tok_ok else "missing_or_incomplete",
        },
        "baseline_jsonl": os.path.abspath(args.baseline),
        "treated_jsonl": os.path.abspath(args.treated),
        "judge_jsonl": os.path.abspath(args.judge) if args.judge and os.path.isfile(args.judge) else None,
        "score_threshold": args.score_threshold,
        "n_common_problems": len(common),
        "overall": {
            "per_problem_mean_output_chars": {
                "baseline": summarize_lengths(base_means_chars),
                "treated": summarize_lengths(tre_means_chars),
                "delta_treated_minus_baseline": summarize_lengths(deltas_chars),
            },
            "per_rollout_output_chars": {
                "baseline": summarize_lengths([float(x) for x in all_base_chars]),
                "treated": summarize_lengths([float(x) for x in all_tre_chars]),
            },
        },
        "judge_coverage": {
            "n_problems_with_judge_row": sum(1 for r in records if r["judge_bucket"] != "judge_unavailable"),
            "n_problems_judge_unavailable": sum(1 for r in records if r["judge_bucket"] == "judge_unavailable"),
        },
        "by_lift_bucket": {},
        "by_judge_bucket": {},
        "by_cross_lift_judge": {},
    }

    if resp_tok_ok:
        report["overall"]["per_problem_mean_response_tokens"] = {
            "baseline": summarize_lengths(base_means_rtok),
            "treated": summarize_lengths(tre_means_rtok),
            "delta_treated_minus_baseline": summarize_lengths(deltas_rtok),
        }
        report["overall"]["per_rollout_response_tokens"] = {
            "baseline": summarize_lengths([float(x) for x in all_base_rtok]),
            "treated": summarize_lengths([float(x) for x in all_tre_rtok]),
        }
    if prompt_tok_ok:
        report["overall"]["per_problem_mean_prompt_tokens"] = {
            "baseline": summarize_lengths(base_means_ptok),
            "treated": summarize_lengths(tre_means_ptok),
            "delta_treated_minus_baseline": summarize_lengths(deltas_ptok),
        }
        report["overall"]["per_rollout_prompt_tokens"] = {
            "baseline": summarize_lengths([float(x) for x in all_base_ptok]),
            "treated": summarize_lengths([float(x) for x in all_tre_ptok]),
        }

    judge_keys: list[str] = [
        "effective_relevant_use",
        "not_effective",
        "no_relevant_bullets_judged",
        "judge_parse_failed",
        "judge_unavailable",
    ]

    def fill_slices(dest: dict[str, Any], base_c: list[float], tre_c: list[float]) -> None:
        for lb in ("win", "regression", "steady_both_pass", "steady_both_fail"):
            dest.setdefault("by_lift_bucket", {})[lb] = {
                "output_chars": _agg_unified(lambda r, _lb=lb: r["lift_bucket"] == _lb, records, base_c, tre_c),
            }
        for jb in judge_keys:
            dest.setdefault("by_judge_bucket", {})[jb] = {
                "output_chars": _agg_unified(lambda r, _jb=jb: r["judge_bucket"] == _jb, records, base_c, tre_c),
            }
        cross_keys: set[str] = {r["cross"] for r in records}
        for ck in sorted(cross_keys):
            dest.setdefault("by_cross_lift_judge", {})[ck] = {
                "output_chars": _agg_unified(lambda r, _ck=ck: r["cross"] == _ck, records, base_c, tre_c),
            }

    fill_slices(report, base_means_chars, tre_means_chars)

    if resp_tok_ok:
        for lb in ("win", "regression", "steady_both_pass", "steady_both_fail"):
            report["by_lift_bucket"][lb]["response_tokens"] = _agg_unified(
                lambda r, _lb=lb: r["lift_bucket"] == _lb, records, base_means_rtok, tre_means_rtok
            )
        for jb in judge_keys:
            report["by_judge_bucket"][jb]["response_tokens"] = _agg_unified(
                lambda r, _jb=jb: r["judge_bucket"] == _jb, records, base_means_rtok, tre_means_rtok
            )
        for ck in sorted(report["by_cross_lift_judge"].keys()):
            report["by_cross_lift_judge"][ck]["response_tokens"] = _agg_unified(
                lambda r, _ck=ck: r["cross"] == _ck, records, base_means_rtok, tre_means_rtok
            )

    if prompt_tok_ok:
        for lb in ("win", "regression", "steady_both_pass", "steady_both_fail"):
            report["by_lift_bucket"][lb]["prompt_tokens"] = _agg_unified(
                lambda r, _lb=lb: r["lift_bucket"] == _lb, records, base_means_ptok, tre_means_ptok
            )
        for jb in judge_keys:
            report["by_judge_bucket"][jb]["prompt_tokens"] = _agg_unified(
                lambda r, _jb=jb: r["judge_bucket"] == _jb, records, base_means_ptok, tre_means_ptok
            )
        for ck in sorted(report["by_cross_lift_judge"].keys()):
            report["by_cross_lift_judge"][ck]["prompt_tokens"] = _agg_unified(
                lambda r, _ck=ck: r["cross"] == _ck, records, base_means_ptok, tre_means_ptok
            )

    out_dir = os.path.dirname(os.path.abspath(args.out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    json_path = f"{args.out_prefix}_token_output_report.json"
    md_path = f"{args.out_prefix}_token_output_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# HRLib token output stats (chars + optional tokens)",
        "",
        "- **output chars**: character count of decoded `output` (cheap proxy).",
        "- **response / prompt tokens**: from JSONL `response_token_count` / `prompt_token_count` when present on **all** rollouts of **all** common problems in both files.",
        "",
        f"- Baseline: `{report['baseline_jsonl']}`",
        f"- Treated: `{report['treated_jsonl']}`",
        f"- Judge (optional): `{report['judge_jsonl']}`",
        "",
        f"- Token coverage: response=`{report['length_units']['response_tokens']}`, prompt=`{report['length_units']['prompt_tokens']}`",
        "",
        "## Overall — output characters",
        "",
    ]
    om = report["overall"]["per_problem_mean_output_chars"]
    md_lines.append(
        f"- Per-problem mean rollout length: baseline mean={om['baseline']['mean']:.1f} chars, "
        f"treated mean={om['treated']['mean']:.1f} chars, "
        f"Δ mean={om['delta_treated_minus_baseline']['mean']:.1f} chars"
    )
    md_lines.append("")

    if resp_tok_ok:
        md_lines.extend(["## Overall — response tokens (non-pad)", ""])
        ort = report["overall"]["per_problem_mean_response_tokens"]
        md_lines.append(
            f"- Per-problem mean: baseline={ort['baseline']['mean']:.1f}, treated={ort['treated']['mean']:.1f}, "
            f"Δ={ort['delta_treated_minus_baseline']['mean']:.1f} tokens"
        )
        md_lines.append("")
    if prompt_tok_ok:
        md_lines.extend(["## Overall — prompt tokens (non-pad)", ""])
        opt = report["overall"]["per_problem_mean_prompt_tokens"]
        md_lines.append(
            f"- Per-problem mean: baseline={opt['baseline']['mean']:.1f}, treated={opt['treated']['mean']:.1f}, "
            f"Δ={opt['delta_treated_minus_baseline']['mean']:.1f} tokens"
        )
        md_lines.append("")

    def emit_section(title: str, metric_key: str, unit: str) -> None:
        md_lines.extend([f"## {title}", ""])
        for lb in ("win", "regression", "steady_both_pass", "steady_both_fail"):
            blk = report["by_lift_bucket"][lb][metric_key]
            md_lines.extend(_fmt_unified_block(f"lift: {lb}", blk, unit=unit))
        for jb in judge_keys:
            blk = report["by_judge_bucket"][jb][metric_key]
            md_lines.extend(_fmt_unified_block(f"judge: {jb}", blk, unit=unit))
        for ck in sorted(report["by_cross_lift_judge"].keys()):
            blk = report["by_cross_lift_judge"][ck][metric_key]
            md_lines.extend(_fmt_unified_block(ck, blk, unit=unit))

    emit_section("By bucket — output characters", "output_chars", "chars")
    if resp_tok_ok:
        emit_section("By bucket — response tokens", "response_tokens", "tokens")
    if prompt_tok_ok:
        emit_section("By bucket — prompt tokens", "prompt_tokens", "tokens")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[hrlib_token] wrote {json_path}")
    print(f"[hrlib_token] wrote {md_path}")
    if resp_tok_ok:
        ort = report["overall"]["per_problem_mean_response_tokens"]
        print(
            f"[hrlib_token] overall per-problem mean response (tokens): baseline={ort['baseline']['mean']:.1f} "
            f"treated={ort['treated']['mean']:.1f} delta={ort['delta_treated_minus_baseline']['mean']:.1f}"
        )
    if prompt_tok_ok:
        opt = report["overall"]["per_problem_mean_prompt_tokens"]
        print(
            f"[hrlib_token] overall per-problem mean prompt (tokens): baseline={opt['baseline']['mean']:.1f} "
            f"treated={opt['treated']['mean']:.1f} delta={opt['delta_treated_minus_baseline']['mean']:.1f}"
        )
    om = report["overall"]["per_problem_mean_output_chars"]
    print(
        f"[hrlib_token] overall per-problem mean output (chars): baseline={om['baseline']['mean']:.1f} "
        f"treated={om['treated']['mean']:.1f} delta={om['delta_treated_minus_baseline']['mean']:.1f}"
    )


if __name__ == "__main__":
    main()
