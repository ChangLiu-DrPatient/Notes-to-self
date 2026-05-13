#!/usr/bin/env python3
"""Fine-grained HRLib eval: problems that go from wrong → correct after abstraction injection.

Compares two VERL validation JSONLs (e.g. vanilla vs ``test_hrlib_v1``) that share the same
underlying items. Pairing uses :func:`scripts.analyze._problem_group_key` (user turn +
``data_source``), not raw ``input``.

**Pass@N** here is computed from per-rollout ``score`` fields: ``pass@N = 1`` if any of the
``N`` rollouts for that problem has score indicating correct (default: ``>= 1 - 1e-6`` for
0/1 naive math rewards). There is no ``pass@`` key in each JSONL line unless the trainer adds
one; this script defines pass@N explicitly for documentation.

Outputs:

- ``{out_prefix}_lift.json`` — machine-readable summary + full win/regression lists, including
  per-flip **treated** decoded prompt (``input`` with retrieved abstractions) and per-rollout
  **baseline** / **treated** model outputs (``output`` + ``score``), plus aggregate
  **pass@k** rates (default k=32): fraction of **baseline** problems with ≥1 correct rollout
  before vs after (``pass_at_aggregate``). Baseline problems absent from the treated JSONL are
  counted as **treated fail** (no rollouts). Plus **subset pass@k** curve for
  ``--pass_at_k_grid`` (default 1,2,4,8,16,32) under ``pass_at_aggregate.pass_at_k_subset_curve``.
- ``{out_prefix}_lift.md`` — short human-readable report + optional problem table (includes the
  same aggregate pass rates and subset pass@k table).
- **Stdout** — echoes aggregate pass@k (or pass@N when rollout counts vary) for baseline vs treated,
  plus a **pass@k curve** for k in ``--pass_at_k_grid`` (default 1,2,4,8,16,32) via the hypergeometric
  subset formula (mean over **baseline** problems of P(≥1 correct in k draws without replacement)).

Use ``--print_flip_details`` to echo prompts/outputs to stdout (truncated; full text stays in JSON).

Example::

    cd verl
    python examples/hrlib/hrlib_abstraction_lift.py \
        --baseline /raid/$USER/eval/hrlib/stage0/vanilla/0.jsonl \
        --treated  /raid/$USER/eval/hrlib/stage0/v1/0.jsonl \
        --out_prefix Figs/hrlib_stage0/lift_v1_dom
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from scripts.analyze import _problem_group_key, _user_turn_from_verl_decoded_input


@dataclass
class ProblemRollouts:
    """Aggregated rollouts for one problem (one row in the report)."""

    problem_key: str
    data_source: str
    ground_truth: Any
    user_problem: str
    n_rollouts: int
    n_correct: int
    mean_score: float
    pass_at_n: int  # 0 or 1


def _norm_data_source(ds: Any) -> str:
    if isinstance(ds, (list, tuple, np.ndarray)) and len(ds) > 0:
        return str(ds[0])
    return str(ds) if ds is not None else "unknown"


def _first_line_meta(line: dict[str, Any]) -> tuple[str, Any, str]:
    key = _problem_group_key(line)
    ds = _norm_data_source(line.get("data_source"))
    gts = line.get("gts")
    inp = str(line.get("input", ""))
    user = _user_turn_from_verl_decoded_input(inp) or ""
    return key, gts, user


def load_grouped(path: str) -> dict[str, list[dict[str, Any]]]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            k = _problem_group_key(data)
            by_key[k].append(data)
    return dict(by_key)


def _str_output(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        return "\n\n---\n\n".join(_str_output(x) for x in val)
    return str(val)


def _rollouts_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per JSONL line: model text and score (validation dumps use ``output``)."""
    out: list[dict[str, Any]] = []
    for r in rows:
        s = r.get("score")
        try:
            sf = float(s)
        except (TypeError, ValueError):
            sf = float("nan")
        out.append({"score": sf, "output": _str_output(r.get("output"))})
    return out


def _flip_prompts_and_outputs(base_rows: list[dict[str, Any]], tre_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Side-by-side rollout text for qualitative inspection (treated prompt includes HRLib injection)."""
    tre_inp = str(tre_rows[0].get("input", "")) if tre_rows else ""
    return {
        "treated_prompt": tre_inp,
        "baseline_rollouts": _rollouts_payload(base_rows),
        "treated_rollouts": _rollouts_payload(tre_rows),
    }


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n… [truncated]"


def print_flip_details_stdout(
    report: dict[str, Any],
    *,
    max_problems_per_category: int,
    max_rollouts_print: int,
    max_prompt_chars: int,
    max_output_chars: int,
) -> None:
    """Print treated prompts and baseline/treated outputs for wins and regressions."""
    for label, key in (("WINS (baseline fail → treated pass)", "wins"), ("LOSSES / REGRESSIONS (baseline pass → treated fail)", "regressions")):
        items: list[dict[str, Any]] = report.get(key, [])
        print(f"\n{'=' * 88}\n{label} — {len(items)} problems (printing up to {max_problems_per_category})\n{'=' * 88}")
        for i, item in enumerate(items[: max_problems_per_category]):
            bsum, tsum = item["baseline"], item["treated"]
            pk = str(tsum.get("problem_key", ""))
            pk_show = pk if len(pk) <= 200 else pk[:200] + "…"
            print(f"\n--- [{i + 1}] {pk_show}")
            print(f"    ground_truth: {bsum.get('ground_truth')}")
            tp = _truncate(str(item.get("treated_prompt", "")), max_prompt_chars)
            print(f"\n[TREATED PROMPT / input]\n{tp}\n")
            br = item.get("baseline_rollouts") or []
            tr = item.get("treated_rollouts") or []
            n_show = max_rollouts_print
            print(f"[BASELINE rollouts — first {min(n_show, len(br))} of {len(br)}]")
            for j, roll in enumerate(br[:n_show]):
                out = _truncate(str(roll.get("output", "")), max_output_chars)
                print(f"  ({j}) score={roll.get('score')}\n{out}\n")
            print(f"[TREATED rollouts — first {min(n_show, len(tr))} of {len(tr)}]")
            for j, roll in enumerate(tr[:n_show]):
                out = _truncate(str(roll.get("output", "")), max_output_chars)
                print(f"  ({j}) score={roll.get('score')}\n{out}\n")
        if len(items) > max_problems_per_category:
            print(f"… {len(items) - max_problems_per_category} more {key} not printed (see JSON).\n")


def _subset_pass_at_k_prob(n_rollouts: int, n_correct: int, k: int) -> float:
    """Expected pass@k for one problem: draw k distinct rollouts uniformly from n (without replacement).

    P(pass) = 1 - C(w,k_eff)/C(n,k_eff) when k_eff <= w (all k wrong), else 1 when k_eff > w.
    Here w = n - n_correct is the number of incorrect rollouts. If n_rollouts < k, uses
    k_eff = min(k, n_rollouts) (equivalent to using every available rollout when k exceeds n).
    """
    if n_rollouts <= 0 or k <= 0:
        return 0.0
    k_eff = min(k, n_rollouts)
    w = n_rollouts - n_correct
    if k_eff > w:
        return 1.0
    return 1.0 - math.comb(w, k_eff) / math.comb(n_rollouts, k_eff)


def summarize_problem(key: str, rows: list[dict[str, Any]], *, score_threshold: float) -> ProblemRollouts:
    if not rows:
        raise ValueError("empty rows")
    _, gts, user = _first_line_meta(rows[0])
    ds = _norm_data_source(rows[0].get("data_source"))
    scores = [float(r["score"]) for r in rows]
    n = len(scores)
    n_correct = sum(1 for s in scores if s >= score_threshold - 1e-9)
    pass_at_n = 1 if n_correct > 0 else 0
    return ProblemRollouts(
        problem_key=key,
        data_source=ds,
        ground_truth=gts,
        user_problem=user[:2000] if user else "",
        n_rollouts=n,
        n_correct=n_correct,
        mean_score=float(np.mean(scores)) if scores else 0.0,
        pass_at_n=pass_at_n,
    )


def _missing_treated_problem(key: str, base_rows: list[dict[str, Any]]) -> ProblemRollouts:
    """Treated JSONL has no lines for this baseline key: no rollouts, always fail."""
    if not base_rows:
        raise ValueError("empty base_rows")
    _, gts, user = _first_line_meta(base_rows[0])
    ds = _norm_data_source(base_rows[0].get("data_source"))
    return ProblemRollouts(
        problem_key=key,
        data_source=ds,
        ground_truth=gts,
        user_problem=user[:2000] if user else "",
        n_rollouts=0,
        n_correct=0,
        mean_score=0.0,
        pass_at_n=0,
    )


def _serialize_problem(p: ProblemRollouts) -> dict[str, Any]:
    d = asdict(p)
    # JSON-safe ground_truth
    if isinstance(d["ground_truth"], (dict, list)):
        d["ground_truth"] = json.dumps(d["ground_truth"], ensure_ascii=False)
    else:
        d["ground_truth"] = str(d["ground_truth"]) if d["ground_truth"] is not None else None
    return d


def compare(
    baseline_path: str,
    treated_path: str,
    *,
    score_threshold: float,
    max_examples_md: int,
    pass_at_k: int,
    pass_at_k_grid: list[int],
) -> dict[str, Any]:
    base_g = load_grouped(baseline_path)
    tre_g = load_grouped(treated_path)

    keys_b, keys_t = set(base_g), set(tre_g)
    common = sorted(keys_b & keys_t)
    only_b = sorted(keys_b - keys_t)
    only_t = sorted(keys_t - keys_b)
    baseline_keys = sorted(keys_b)
    n_baseline = len(baseline_keys)

    wins: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    regressions_no_treated_rows = 0
    steady_ok: list[str] = []
    steady_fail: list[str] = []
    baseline_pass_sum = 0
    treated_pass_sum = 0
    rollouts_b: list[int] = []
    rollouts_t: list[int] = []
    grid_sums_b = {kk: 0.0 for kk in pass_at_k_grid}
    grid_sums_t = {kk: 0.0 for kk in pass_at_k_grid}

    for key in baseline_keys:
        sb = summarize_problem(key, base_g[key], score_threshold=score_threshold)
        if key in tre_g:
            st = summarize_problem(key, tre_g[key], score_threshold=score_threshold)
            tre_rows = tre_g[key]
        else:
            st = _missing_treated_problem(key, base_g[key])
            tre_rows = []
        baseline_pass_sum += sb.pass_at_n
        treated_pass_sum += st.pass_at_n
        rollouts_b.append(sb.n_rollouts)
        rollouts_t.append(st.n_rollouts)
        for kk in pass_at_k_grid:
            grid_sums_b[kk] += _subset_pass_at_k_prob(sb.n_rollouts, sb.n_correct, kk)
            grid_sums_t[kk] += _subset_pass_at_k_prob(st.n_rollouts, st.n_correct, kk)

        if sb.pass_at_n == 0 and st.pass_at_n == 1:
            flip = _flip_prompts_and_outputs(base_g[key], tre_rows)
            wins.append(
                {
                    "baseline": _serialize_problem(sb),
                    "treated": _serialize_problem(st),
                    **flip,
                }
            )
        elif sb.pass_at_n == 1 and st.pass_at_n == 0:
            flip = _flip_prompts_and_outputs(base_g[key], tre_rows)
            regressions.append(
                {
                    "baseline": _serialize_problem(sb),
                    "treated": _serialize_problem(st),
                    **flip,
                }
            )
            if not tre_rows:
                regressions_no_treated_rows += 1
        elif sb.pass_at_n == 1 and st.pass_at_n == 1:
            steady_ok.append(key)
        else:
            steady_fail.append(key)

    rb_min = min(rollouts_b) if rollouts_b else 0
    rb_max = max(rollouts_b) if rollouts_b else 0
    rt_min = min(rollouts_t) if rollouts_t else 0
    rt_max = max(rollouts_t) if rollouts_t else 0
    consistent_k = (
        n_baseline > 0
        and rb_min == rb_max == pass_at_k
        and rt_min == rt_max == pass_at_k
    )
    baseline_rate = baseline_pass_sum / n_baseline if n_baseline else 0.0
    treated_rate = treated_pass_sum / n_baseline if n_baseline else 0.0

    pass_k_curve: list[dict[str, Any]] = []
    for kk in pass_at_k_grid:
        br_k = grid_sums_b[kk] / n_baseline if n_baseline else 0.0
        tr_k = grid_sums_t[kk] / n_baseline if n_baseline else 0.0
        pass_k_curve.append(
            {
                "k": kk,
                "baseline_pass_rate": br_k,
                "treated_pass_rate": tr_k,
                "delta_treated_minus_baseline": tr_k - br_k,
            }
        )

    report = {
        "baseline_jsonl": os.path.abspath(baseline_path),
        "treated_jsonl": os.path.abspath(treated_path),
        "score_threshold_correct": score_threshold,
        "aggregation_scope": "baseline_jsonl_keys",
        "n_problems_in_aggregate": n_baseline,
        "n_problems_baseline_only": len(only_b),
        "n_problems_treated_only": len(only_t),
        "n_problems_common": len(common),
        "pass_at_n_definition": "1 if any rollout score >= score_threshold_correct, else 0",
        "pass_at_aggregate": {
            "k": pass_at_k,
            "baseline_pass_rate": baseline_rate,
            "treated_pass_rate": treated_rate,
            "delta_treated_minus_baseline": treated_rate - baseline_rate,
            "baseline_problems_solved": baseline_pass_sum,
            "treated_problems_solved": treated_pass_sum,
            "rollouts_per_problem_baseline_min": rb_min,
            "rollouts_per_problem_baseline_max": rb_max,
            "rollouts_per_problem_treated_min": rt_min,
            "rollouts_per_problem_treated_max": rt_max,
            "all_common_problems_have_k_rollouts": consistent_k,
            "note": (
                f"Denominator is **all {n_baseline} baseline** problems; each has exactly "
                f"{pass_at_k} rollouts on both sides, so aggregate pass@{pass_at_k} is exact."
                if consistent_k
                else (
                    f"Denominator is **all {n_baseline} baseline** problems. "
                    f"{len(only_b)} lack treated JSONL rows and are scored as treated fail (0 rollouts). "
                    f"Aggregate is (# problems with ≥1 correct rollout) / n_baseline; treated rollout "
                    f"min/max include 0 when any problem is missing on the treated side."
                )
            ),
            "pass_at_k_subset_curve": pass_k_curve,
            "pass_at_k_subset_method": (
                "Per problem: P(≥1 correct among k_eff draws) = 1 - C(w,k_eff)/C(n,k_eff) for "
                "k_eff = min(k, n) and w = n - n_correct; **mean over all baseline problems**. "
                "Treated problems with n=0 contribute 0. Draws are uniform without replacement "
                "among the n empirical rollouts for that side."
            ),
        },
        "counts": {
            "wins_wrong_to_right": len(wins),
            "regressions_right_to_wrong": len(regressions),
            "regressions_with_no_treated_rows": regressions_no_treated_rows,
            "steady_both_pass": len(steady_ok),
            "steady_both_fail": len(steady_fail),
        },
        "wins": wins,
        "regressions": regressions,
        "keys_baseline_only_sample": only_b[:20],
        "keys_treated_only_sample": only_t[:20],
    }
    report["_md_fragment"] = _markdown_table(wins, max_rows=max_examples_md)
    return report


def _parse_pass_at_k_grid(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return [1, 2, 4, 8, 16, 32]
    seen: set[int] = set()
    out: list[int] = []
    for p in parts:
        v = int(p)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _markdown_table(wins: list[dict[str, Any]], *, max_rows: int) -> str:
    if not wins:
        return "_No problems with baseline pass@N=0 and treated pass@N=1._\n"
    lines = [
        "| data_source | baseline (correct/total, mean) | treated (correct/total, mean) | ground_truth (trim) |",
        "|---|---:|---:|---|",
    ]
    for w in wins[:max_rows]:
        b, t = w["baseline"], w["treated"]
        gt = str(b.get("ground_truth", ""))[:60].replace("|", "\\|")
        lines.append(
            f"| {b.get('data_source', '')} | {b['n_correct']}/{b['n_rollouts']}, {b['mean_score']:.3f} | "
            f"{t['n_correct']}/{t['n_rollouts']}, {t['mean_score']:.3f} | {gt} |"
        )
    if len(wins) > max_rows:
        lines.append(f"\n_… {len(wins) - max_rows} more wins omitted (see JSON)._")
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any], out_prefix: str) -> tuple[str, str]:
    out_dir = os.path.dirname(os.path.abspath(out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    json_path = f"{out_prefix}_lift.json"
    md_path = f"{out_prefix}_lift.md"

    payload = {k: v for k, v in report.items() if not k.startswith("_")}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    md = [
        "# HRLib abstraction lift report",
        "",
        f"- **Baseline:** `{report['baseline_jsonl']}`",
        f"- **Treated:** `{report['treated_jsonl']}`",
        f"- **Correct rollout threshold:** score ≥ {report['score_threshold_correct']}",
        "",
        "## Summary",
        "",
        f"- Aggregate over **{report.get('n_problems_in_aggregate', report['n_problems_common'])}** baseline problems "
        f"(`aggregation_scope`: {report.get('aggregation_scope', 'baseline_jsonl_keys')})",
        f"- Also in both JSONLs (common keys): **{report['n_problems_common']}**",
        f"- Wins (baseline pass@N=0 → treated pass@N=1): **{report['counts']['wins_wrong_to_right']}**",
        f"- Regressions (baseline pass@N=1 → treated pass@N=0): **{report['counts']['regressions_right_to_wrong']}** "
        f"({report['counts'].get('regressions_with_no_treated_rows', 0)} with **no** treated rows)",
        f"- Steady both pass: **{report['counts']['steady_both_pass']}**",
        f"- Steady both fail: **{report['counts']['steady_both_fail']}**",
        "",
        "### Pass@ aggregate (before vs after retrieval)",
        "",
    ]
    agg = report.get("pass_at_aggregate") or {}
    k = agg.get("k", 32)
    n_agg = report.get("n_problems_in_aggregate", report["n_problems_common"])
    md.append(
        f"- **Baseline** pass rate: **{agg.get('baseline_pass_rate', 0):.4f}** "
        f"({agg.get('baseline_problems_solved', 0)}/{n_agg} problems with ≥1 correct rollout)"
    )
    md.append(
        f"- **Treated** pass rate: **{agg.get('treated_pass_rate', 0):.4f}** "
        f"({agg.get('treated_problems_solved', 0)}/{n_agg})"
    )
    md.append(
        f"- **Δ (treated − baseline):** **{agg.get('delta_treated_minus_baseline', 0):+.4f}**"
    )
    md.append(
        f"- Rollouts per problem (baseline): {agg.get('rollouts_per_problem_baseline_min')}–"
        f"{agg.get('rollouts_per_problem_baseline_max')}; "
        f"(treated): {agg.get('rollouts_per_problem_treated_min')}–"
        f"{agg.get('rollouts_per_problem_treated_max')}"
    )
    if agg.get("all_common_problems_have_k_rollouts"):
        md.append(
            f"- Declared **pass@{k}** matches rollout count on **all** baseline problems for baseline "
            f"and treated."
        )
    else:
        md.append(
            f"- Rollout count is not uniformly **{k}**; see JSON `pass_at_aggregate.note` for interpretation."
        )
    md.append("")
    curve = agg.get("pass_at_k_subset_curve") or []
    if curve:
        md.append("### Pass@k subset curve (hypergeometric)")
        md.append("")
        md.append(
            "Mean over **baseline** problems of P(≥1 correct in **k** random draws without replacement "
            "from that problem’s **n** empirical rollouts; if **n < k**, **k_eff = min(k, n)** "
            "(missing treated rows use **n=0** → 0). "
            "See JSON `pass_at_aggregate.pass_at_k_subset_method`."
        )
        md.append("")
        md.append("| k | baseline | treated | Δ |")
        md.append("|---:|---:|---:|---:|")
        for row in curve:
            md.append(
                f"| {row['k']} | {row['baseline_pass_rate']:.4f} | {row['treated_pass_rate']:.4f} | "
                f"{row['delta_treated_minus_baseline']:+.4f} |"
            )
        md.append("")
    if report.get("n_problems_baseline_only") or report.get("n_problems_treated_only"):
        md.append("## Key alignment warnings")
        md.append("")
        md.append(
            f"- Problems only in baseline: {report['n_problems_baseline_only']} "
            f"(sample keys: {report.get('keys_baseline_only_sample', [])})"
        )
        md.append(
            f"- Problems only in treated: {report['n_problems_treated_only']} "
            f"(sample keys: {report.get('keys_treated_only_sample', [])})"
        )
        md.append("")

    md.append("## Wins (wrong → correct)")
    md.append("")
    md.append(report.get("_md_fragment", ""))
    md.append("")
    md.append(
        "Full per-problem fields (including `user_problem`, treated `treated_prompt`, and "
        "`baseline_rollouts` / `treated_rollouts` with model outputs) are in the JSON file."
    )
    md.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    return json_path, md_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("Example::")[0].strip())
    p.add_argument("--baseline", required=True, help="validation JSONL (e.g. vanilla / no HRLib)")
    p.add_argument("--treated", required=True, help="validation JSONL (e.g. injected parquet eval)")
    p.add_argument(
        "--out_prefix",
        default=os.path.join("Figs", "hrlib_stage0", "lift"),
        help="output path prefix (writes {prefix}_lift.json and {prefix}_lift.md)",
    )
    p.add_argument(
        "--score_threshold",
        type=float,
        default=1.0,
        help="rollout counted correct if score >= this (default 1.0 for 0/1 naive math)",
    )
    p.add_argument(
        "--pass_at_k",
        type=int,
        default=32,
        help=(
            "declared k for pass@k reporting (default 32). Denominator is all baseline problems; "
            "when each has exactly this many rollouts on both sides, aggregate pass@k is exact."
        ),
    )
    p.add_argument(
        "--pass_at_k_grid",
        type=str,
        default="1,2,4,8,16,32",
        help=(
            "comma-separated k values for the hypergeometric subset pass@k curve "
            "(default 1,2,4,8,16,32)"
        ),
    )
    p.add_argument(
        "--max_examples_md",
        type=int,
        default=30,
        help="max win rows in the Markdown table (JSON always lists all wins)",
    )
    p.add_argument(
        "--print_flip_details",
        action="store_true",
        help="print treated prompts and baseline/treated model outputs for wins and regressions (truncated; full in JSON)",
    )
    p.add_argument(
        "--print_max_problems",
        type=int,
        default=50,
        help="with --print_flip_details: max wins and max regressions to print",
    )
    p.add_argument(
        "--print_max_rollouts",
        type=int,
        default=8,
        help="with --print_flip_details: first N rollouts per side per problem",
    )
    p.add_argument(
        "--print_max_prompt_chars",
        type=int,
        default=12000,
        help="with --print_flip_details: truncate treated prompt after this many characters (0 = no limit)",
    )
    p.add_argument(
        "--print_max_output_chars",
        type=int,
        default=4000,
        help="with --print_flip_details: truncate each printed rollout output (0 = no limit)",
    )
    args = p.parse_args()

    k_grid = _parse_pass_at_k_grid(args.pass_at_k_grid)
    report = compare(
        args.baseline,
        args.treated,
        score_threshold=args.score_threshold,
        max_examples_md=args.max_examples_md,
        pass_at_k=args.pass_at_k,
        pass_at_k_grid=k_grid,
    )
    jp, mp = write_reports(report, args.out_prefix)
    print(f"[hrlib_abstraction_lift] wrote {jp}")
    print(f"[hrlib_abstraction_lift] wrote {mp}")
    agg = report.get("pass_at_aggregate") or {}
    k = agg.get("k", args.pass_at_k)
    br = agg.get("baseline_pass_rate", 0.0)
    tr = agg.get("treated_pass_rate", 0.0)
    dlt = agg.get("delta_treated_minus_baseline", 0.0)
    rb = f"{agg.get('rollouts_per_problem_baseline_min')}–{agg.get('rollouts_per_problem_baseline_max')}"
    rt = f"{agg.get('rollouts_per_problem_treated_min')}–{agg.get('rollouts_per_problem_treated_max')}"
    pass_label = f"pass@{k}" if agg.get("all_common_problems_have_k_rollouts") else f"pass@N (N={rb} baseline, {rt} treated)"
    print(
        f"[hrlib_abstraction_lift] {pass_label}: baseline={br:.4f} treated={tr:.4f} "
        f"(Δ {dlt:+.4f})"
    )
    print(
        f"[hrlib_abstraction_lift] wins={report['counts']['wins_wrong_to_right']} "
        f"regressions={report['counts']['regressions_right_to_wrong']}"
    )
    curve = agg.get("pass_at_k_subset_curve") or []
    if curve:
        print("[hrlib_abstraction_lift] subset pass@k (hypergeometric mean over baseline problems):")
        for row in curve:
            print(
                f"  k={row['k']}: baseline={row['baseline_pass_rate']:.4f} "
                f"treated={row['treated_pass_rate']:.4f} "
                f"(Δ {row['delta_treated_minus_baseline']:+.4f})"
            )
    if args.print_flip_details:
        print_flip_details_stdout(
            report,
            max_problems_per_category=args.print_max_problems,
            max_rollouts_print=args.print_max_rollouts,
            max_prompt_chars=args.print_max_prompt_chars,
            max_output_chars=args.print_max_output_chars,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
