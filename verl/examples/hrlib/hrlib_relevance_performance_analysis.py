#!/usr/bin/env python3
"""Analyze HRLib relevance judgments against pass@N performance.

This script expects a score-gate experiment directory with eval subdirectories such as:

    eval_minilm_orig/0.jsonl
    eval_minilm_orig/judge/judge_results.jsonl
    test_hrlib_minilm_orig_scores.jsonl

It prints a Markdown report comparing original-query retrieval, rewrite-query
retrieval, and score-gated retrieval for runs provided via ``--runs`` (default:
MiniLM ``orig/rewrite/gated``).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable


DEFAULT_RUNS = (
    "minilm_orig",
    "minilm_rewrite",
    "minilm_gated",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_root",
        required=True,
        help="Experiment root containing eval_* directories and test_hrlib_*_scores.jsonl files.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=list(DEFAULT_RUNS),
        help=f"Run labels to analyze. Default: {' '.join(DEFAULT_RUNS)}",
    )
    parser.add_argument(
        "--correct_threshold",
        type=float,
        default=1.0,
        help="Rollout score threshold counted as correct.",
    )
    parser.add_argument("--out_md", default="", help="Optional path to write the Markdown report.")
    parser.add_argument("--out_json", default="", help="Optional path to write machine-readable summary JSON.")
    return parser.parse_args()


def _norm_data_source(ds: Any) -> str:
    if isinstance(ds, (list, tuple)) and ds:
        return str(ds[0])
    return str(ds) if ds is not None else "unknown"


def _user_turn_from_input(decoded_input: str) -> str:
    if "\nuser\n" not in decoded_input:
        return decoded_input.strip()
    after = decoded_input.split("\nuser\n", 1)[1]
    if "\nassistant\n" in after:
        return after.split("\nassistant\n", 1)[0].strip()
    if "\nassistant" in after:
        return after.split("\nassistant", 1)[0].strip()
    return after.strip()


def _problem_key(row: dict[str, Any]) -> str:
    uid = row.get("uid")
    if uid is not None and uid != "":
        return str(uid)
    ds = _norm_data_source(row.get("data_source"))
    user_turn = _user_turn_from_input(str(row.get("input", "")))
    if user_turn:
        return f"{ds}::user::{user_turn}"
    gts = row.get("gts")
    if gts is not None and gts != "":
        return f"{ds}::gts::{json.dumps(gts, sort_keys=True, ensure_ascii=False)}"
    return str(row.get("input", ""))


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_eval(path: Path, correct_threshold: float) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            grouped[_problem_key(row)].append(row)

    out: dict[str, dict[str, Any]] = {}
    for key, rows in grouped.items():
        scores = [_to_float(row.get("score")) for row in rows]
        n_correct = sum(1 for score in scores if not math.isnan(score) and score >= correct_threshold - 1e-9)
        n_rollouts = len(rows)
        out[key] = {
            "n_rollouts": n_rollouts,
            "n_correct": n_correct,
            "pass_at_n": int(n_correct > 0),
            "mean_score": sum(s for s in scores if not math.isnan(s)) / n_rollouts if n_rollouts else float("nan"),
            "data_source": _norm_data_source(rows[0].get("data_source")) if rows else "unknown",
            "ground_truth": rows[0].get("gts") if rows else None,
            "user_problem": _user_turn_from_input(str(rows[0].get("input", ""))) if rows else "",
        }
    return out


def load_judge(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            judged = row.get("judged_bullets") or []
            usage = Counter(str(item.get("usage")) for item in judged)
            relevant = sum(1 for item in judged if item.get("relevant") is True)
            relevant_used_correctly = sum(
                1
                for item in judged
                if item.get("relevant") is True and item.get("usage") == "used_correctly"
            )
            relevant_used_incorrectly = sum(
                1
                for item in judged
                if item.get("relevant") is True and item.get("usage") == "used_incorrectly"
            )
            out[str(row["problem_key"])] = {
                "parsed_ok": bool(row.get("parsed_ok")),
                "n_judged": len(judged),
                "relevant": relevant,
                "used_correctly": usage["used_correctly"],
                "used_incorrectly": usage["used_incorrectly"],
                "unclear": usage["unclear"],
                "ignored": usage["ignored"],
                "relevant_used_correctly": relevant_used_correctly,
                "relevant_used_incorrectly": relevant_used_incorrectly,
                "any_relevant": int(relevant > 0),
                "any_used_correctly": int(usage["used_correctly"] > 0),
                "any_relevant_used_correctly": int(relevant_used_correctly > 0),
                "judge_rollout_correct": bool(row.get("is_rollout_correct")),
                "rollout_idx": row.get("rollout_idx"),
                "sample_tag": row.get("sample_tag", ""),
            }
    return out


def load_scores(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["problem_key"])] = {
                "chosen_query_source": row.get("chosen_query_source", ""),
                "query_gate": row.get("query_gate", ""),
                "gate_score_original": _to_float(row.get("gate_score_original")),
                "gate_score_rewrite": _to_float(row.get("gate_score_rewrite")),
                "canonical_subject": row.get("canonical_subject", ""),
                "top1_score": (
                    _to_float(row.get("top_k_hits", [{}])[0].get("cosine_score"))
                    if row.get("top_k_hits")
                    else float("nan")
                ),
            }
    return out


def load_lift_reports(run_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in sorted((run_root / "figs").glob("*lift*.json")):
        with path.open(encoding="utf-8") as f:
            out[path.name] = json.load(f)
    return out


def exact_sign_p(gain: int, loss: int) -> float:
    """Two-sided exact binomial/sign-test p-value for discordant paired counts."""
    n = gain + loss
    if n == 0:
        return 1.0
    x = min(gain, loss)
    left = sum(math.comb(n, i) for i in range(x + 1)) / (2**n)
    return min(1.0, 2.0 * left)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def f3(value: float) -> str:
    return f"{value:.3f}"


def pstr(value: float) -> str:
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def split_run(run: str) -> tuple[str, str]:
    if "_" not in run:
        return run, ""
    return tuple(run.split("_", 1))  # type: ignore[return-value]


def summarize_run(
    run: str,
    evals: dict[str, dict[str, dict[str, Any]]],
    judges: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    e = evals[run]
    j = judges[run]
    parsed_keys = [key for key, value in j.items() if value["parsed_ok"]]
    judged_bullets = sum(j[key]["n_judged"] for key in parsed_keys)
    relevant = sum(j[key]["relevant"] for key in parsed_keys)
    used_correctly = sum(j[key]["used_correctly"] for key in parsed_keys)
    used_incorrectly = sum(j[key]["used_incorrectly"] for key in parsed_keys)
    relevant_used_correctly = sum(j[key]["relevant_used_correctly"] for key in parsed_keys)
    solved = sum(item["pass_at_n"] for item in e.values())
    total_correct = sum(item["n_correct"] for item in e.values())
    total_rollouts = sum(item["n_rollouts"] for item in e.values())
    return {
        "run": run,
        "n_problems": len(e),
        "solved": solved,
        "pass_at_n": solved / len(e) if e else float("nan"),
        "total_correct": total_correct,
        "pass_at_1": total_correct / total_rollouts if total_rollouts else float("nan"),
        "rollout_correct_rate": total_correct / total_rollouts if total_rollouts else float("nan"),
        "judge_records": len(j),
        "parsed": len(parsed_keys),
        "judged_bullets": judged_bullets,
        "any_relevant": sum(j[key]["any_relevant"] for key in parsed_keys),
        "any_relevant_rate": sum(j[key]["any_relevant"] for key in parsed_keys) / len(parsed_keys),
        "relevant": relevant,
        "relevant_rate": relevant / judged_bullets if judged_bullets else float("nan"),
        "any_used_correctly": sum(j[key]["any_used_correctly"] for key in parsed_keys),
        "any_used_correctly_rate": sum(j[key]["any_used_correctly"] for key in parsed_keys) / len(parsed_keys),
        "used_correctly": used_correctly,
        "used_correctly_rate": used_correctly / judged_bullets if judged_bullets else float("nan"),
        "relevant_used_correctly": relevant_used_correctly,
        "relevant_used_correctly_rate": relevant_used_correctly / judged_bullets if judged_bullets else float("nan"),
        "used_incorrectly": used_incorrectly,
    }


def paired_relevance_compare(
    a: str,
    b: str,
    judges: dict[str, dict[str, dict[str, Any]]],
    field: str,
) -> dict[str, Any]:
    common = sorted(set(judges[a]) & set(judges[b]))
    pairs = [
        (judges[a][key][field], judges[b][key][field])
        for key in common
        if judges[a][key]["parsed_ok"] and judges[b][key]["parsed_ok"]
    ]
    if field.startswith("any_"):
        a_sum = sum(x for x, _ in pairs)
        b_sum = sum(y for _, y in pairs)
        gain = sum(1 for x, y in pairs if x == 0 and y == 1)
        loss = sum(1 for x, y in pairs if x == 1 and y == 0)
        return {
            "n": len(pairs),
            "a": a_sum,
            "b": b_sum,
            "delta": (b_sum - a_sum) / len(pairs) if pairs else float("nan"),
            "gain": gain,
            "loss": loss,
            "p": exact_sign_p(gain, loss),
        }
    diffs = [y - x for x, y in pairs]
    pos = sum(1 for diff in diffs if diff > 0)
    neg = sum(1 for diff in diffs if diff < 0)
    return {
        "n": len(pairs),
        "a_mean": statistics.mean(x for x, _ in pairs) if pairs else float("nan"),
        "b_mean": statistics.mean(y for _, y in pairs) if pairs else float("nan"),
        "delta_mean": statistics.mean(diffs) if diffs else float("nan"),
        "gain": pos,
        "loss": neg,
        "p": exact_sign_p(pos, neg),
    }


def performance_compare(
    a: str,
    b: str,
    evals: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    common = sorted(set(evals[a]) & set(evals[b]))
    a_solved = sum(evals[a][key]["pass_at_n"] for key in common)
    b_solved = sum(evals[b][key]["pass_at_n"] for key in common)
    wins = sum(1 for key in common if evals[a][key]["pass_at_n"] == 0 and evals[b][key]["pass_at_n"] == 1)
    losses = sum(1 for key in common if evals[a][key]["pass_at_n"] == 1 and evals[b][key]["pass_at_n"] == 0)
    correct_delta = sum(evals[b][key]["n_correct"] - evals[a][key]["n_correct"] for key in common)
    return {
        "n": len(common),
        "a_solved": a_solved,
        "b_solved": b_solved,
        "delta": (b_solved - a_solved) / len(common) if common else float("nan"),
        "wins": wins,
        "losses": losses,
        "p": exact_sign_p(wins, losses),
        "total_correct_delta": correct_delta,
    }


def stratify_performance_by_relevance(
    run: str,
    evals: dict[str, dict[str, dict[str, Any]]],
    judges: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    keys = sorted(set(evals[run]) & set(judges[run]))
    keys = [key for key in keys if judges[run][key]["parsed_ok"]]
    output: dict[str, Any] = {}

    def summarize_subset(subset: list[str]) -> dict[str, Any]:
        if not subset:
            return {"n": 0}
        return {
            "n": len(subset),
            "pass_at_n": sum(evals[run][key]["pass_at_n"] for key in subset) / len(subset),
            "rollout_correct_rate": sum(evals[run][key]["n_correct"] for key in subset)
            / sum(evals[run][key]["n_rollouts"] for key in subset),
            "any_relevant": sum(judges[run][key]["any_relevant"] for key in subset) / len(subset),
            "mean_relevant": statistics.mean(judges[run][key]["relevant"] for key in subset),
            "any_used_correctly": sum(judges[run][key]["any_used_correctly"] for key in subset) / len(subset),
            "mean_used_correctly": statistics.mean(judges[run][key]["used_correctly"] for key in subset),
        }

    output["any_relevant_0"] = summarize_subset([key for key in keys if judges[run][key]["any_relevant"] == 0])
    output["any_relevant_1"] = summarize_subset([key for key in keys if judges[run][key]["any_relevant"] == 1])
    output["any_used_correctly_0"] = summarize_subset(
        [key for key in keys if judges[run][key]["any_used_correctly"] == 0]
    )
    output["any_used_correctly_1"] = summarize_subset(
        [key for key in keys if judges[run][key]["any_used_correctly"] == 1]
    )
    output["rel_count_0"] = summarize_subset([key for key in keys if judges[run][key]["relevant"] == 0])
    output["rel_count_1_2"] = summarize_subset([key for key in keys if 1 <= judges[run][key]["relevant"] <= 2])
    output["rel_count_3_plus"] = summarize_subset([key for key in keys if judges[run][key]["relevant"] >= 3])
    output["pass_at_n_0"] = summarize_subset([key for key in keys if evals[run][key]["pass_at_n"] == 0])
    output["pass_at_n_1"] = summarize_subset([key for key in keys if evals[run][key]["pass_at_n"] == 1])
    return output


def gate_source_strata(
    run: str,
    evals: dict[str, dict[str, dict[str, Any]]],
    judges: dict[str, dict[str, dict[str, Any]]],
    scores: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if "gated" not in run or run not in scores:
        return {}
    out: dict[str, Any] = {}
    for source in ("original", "rewrite"):
        keys = [
            key
            for key, row in scores[run].items()
            if row.get("chosen_query_source") == source
            and key in evals[run]
            and key in judges[run]
            and judges[run][key]["parsed_ok"]
        ]
        if not keys:
            continue
        out[source] = {
            "n": len(keys),
            "pass_at_n": sum(evals[run][key]["pass_at_n"] for key in keys) / len(keys),
            "rollout_correct_rate": sum(evals[run][key]["n_correct"] for key in keys)
            / sum(evals[run][key]["n_rollouts"] for key in keys),
            "any_relevant": sum(judges[run][key]["any_relevant"] for key in keys) / len(keys),
            "mean_relevant": statistics.mean(judges[run][key]["relevant"] for key in keys),
            "any_used_correctly": sum(judges[run][key]["any_used_correctly"] for key in keys) / len(keys),
        }
    return out


def build_analysis(run_root: Path, runs: list[str], correct_threshold: float) -> dict[str, Any]:
    evals: dict[str, dict[str, dict[str, Any]]] = {}
    judges: dict[str, dict[str, dict[str, Any]]] = {}
    scores: dict[str, dict[str, dict[str, Any]]] = {}

    missing: list[str] = []
    for run in runs:
        eval_path = run_root / f"eval_{run}" / "0.jsonl"
        judge_path = run_root / f"eval_{run}" / "judge" / "judge_results.jsonl"
        score_path = run_root / f"test_hrlib_{run}_scores.jsonl"
        if not eval_path.exists():
            missing.append(str(eval_path))
            continue
        if not judge_path.exists():
            missing.append(str(judge_path))
            continue
        evals[run] = load_eval(eval_path, correct_threshold)
        judges[run] = load_judge(judge_path)
        scores[run] = load_scores(score_path)

    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    run_summary = {run: summarize_run(run, evals, judges) for run in runs}
    comparisons: dict[str, Any] = {}

    def _comparison_payload(a: str, b: str) -> dict[str, Any]:
        return {
            "performance": performance_compare(a, b, evals),
            "any_relevant": paired_relevance_compare(a, b, judges, "any_relevant"),
            "relevant_count": paired_relevance_compare(a, b, judges, "relevant"),
            "any_used_correctly": paired_relevance_compare(a, b, judges, "any_used_correctly"),
            "used_correctly_count": paired_relevance_compare(a, b, judges, "used_correctly"),
        }

    runs_by_embed: dict[str, set[str]] = defaultdict(set)
    for run in runs:
        emb, mode = split_run(run)
        if not emb or not mode:
            continue
        runs_by_embed[emb].add(mode)

    mode_pairs = (("orig", "rewrite"), ("orig", "gated"), ("rewrite", "gated"))
    for emb, modes in sorted(runs_by_embed.items()):
        for left_mode, right_mode in mode_pairs:
            if left_mode not in modes or right_mode not in modes:
                continue
            a = f"{emb}_{left_mode}"
            b = f"{emb}_{right_mode}"
            if a not in evals or b not in evals:
                continue
            comparisons[f"{a}__to__{b}"] = _comparison_payload(a, b)

    all_modes = sorted({mode for modes in runs_by_embed.values() for mode in modes})
    for mode in all_modes:
        runs_for_mode = [
            f"{emb}_{mode}"
            for emb in sorted(runs_by_embed.keys())
            if mode in runs_by_embed[emb] and f"{emb}_{mode}" in evals
        ]
        for a, b in combinations(runs_for_mode, 2):
            comparisons[f"{a}__to__{b}"] = _comparison_payload(a, b)

    return {
        "run_root": str(run_root),
        "runs": runs,
        "run_summary": run_summary,
        "comparisons": comparisons,
        "stratification": {run: stratify_performance_by_relevance(run, evals, judges) for run in runs},
        "gate_source_strata": {run: gate_source_strata(run, evals, judges, scores) for run in runs},
        "lift_reports": load_lift_reports(run_root),
    }


def render_report(analysis: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# HRLib Relevance And Performance Analysis")
    lines.append("")
    lines.append(f"- Run root: `{analysis['run_root']}`")
    lines.append("- Judge denominators below use parsed judge records for relevance/usage rates.")
    lines.append("")

    rows = []
    for run, summary in analysis["run_summary"].items():
        rows.append(
            [
                run,
                f"{summary['solved']}/{summary['n_problems']} ({pct(summary['pass_at_n'])})",
                pct(summary["pass_at_1"]),
                f"{summary['parsed']}/{summary['judge_records']}",
                pct(summary["any_relevant_rate"]),
                pct(summary["relevant_rate"]),
                pct(summary["any_used_correctly_rate"]),
                pct(summary["used_correctly_rate"]),
                summary["used_incorrectly"],
            ]
        )
    lines.append("## Run Summary")
    lines.append(markdown_table(
        [
            "run",
            "pass@N",
            "pass@1",
            "parsed",
            "any relevant",
            "relevant bullets",
            "any used correctly",
            "used correctly bullets",
            "used incorrectly bullets",
        ],
        rows,
    ))
    lines.append("")

    rows = []
    for key, comp in analysis["comparisons"].items():
        a, b = key.split("__to__")
        perf = comp["performance"]
        any_rel = comp["any_relevant"]
        rel_count = comp["relevant_count"]
        any_uc = comp["any_used_correctly"]
        uc_count = comp["used_correctly_count"]
        rows.append(
            [
                f"{a} -> {b}",
                f"{perf['a_solved']}->{perf['b_solved']} ({perf['delta']:+.3f})",
                f"{perf['wins']}/{perf['losses']}",
                pstr(perf["p"]),
                f"{any_rel['a']}->{any_rel['b']} ({any_rel['delta']:+.3f})",
                pstr(any_rel["p"]),
                f"{rel_count['a_mean']:.2f}->{rel_count['b_mean']:.2f} ({rel_count['delta_mean']:+.2f})",
                f"{any_uc['a']}->{any_uc['b']} ({any_uc['delta']:+.3f})",
                pstr(any_uc["p"]),
                f"{uc_count['a_mean']:.2f}->{uc_count['b_mean']:.2f} ({uc_count['delta_mean']:+.2f})",
            ]
        )
    lines.append("## Paired Comparisons")
    lines.append(markdown_table(
        [
            "comparison",
            "pass@N",
            "wins/losses",
            "pass p",
            "any relevant",
            "rel p",
            "mean relevant",
            "any used",
            "used p",
            "mean used",
        ],
        rows,
    ))
    lines.append("")

    rows = []
    for run, strata in analysis["stratification"].items():
        for label in ("any_relevant_0", "any_relevant_1", "any_used_correctly_0", "any_used_correctly_1", "rel_count_0", "rel_count_1_2", "rel_count_3_plus"):
            item = strata.get(label, {})
            if item.get("n", 0) == 0:
                continue
            rows.append(
                [
                    run,
                    label,
                    item["n"],
                    pct(item["pass_at_n"]),
                    pct(item["rollout_correct_rate"]),
                    pct(item["any_relevant"]),
                    f"{item['mean_relevant']:.2f}",
                    pct(item["any_used_correctly"]),
                    f"{item['mean_used_correctly']:.2f}",
                ]
            )
    lines.append("## Performance By Relevance Stratum")
    lines.append(markdown_table(
        [
            "run",
            "stratum",
            "n",
            "pass@N",
            "rollout correct",
            "any relevant",
            "mean relevant",
            "any used",
            "mean used",
        ],
        rows,
    ))
    lines.append("")

    rows = []
    for run, strata in analysis["stratification"].items():
        for label in ("pass_at_n_0", "pass_at_n_1"):
            item = strata.get(label, {})
            if item.get("n", 0) == 0:
                continue
            rows.append(
                [
                    run,
                    label,
                    item["n"],
                    pct(item["any_relevant"]),
                    f"{item['mean_relevant']:.2f}",
                    pct(item["any_used_correctly"]),
                    f"{item['mean_used_correctly']:.2f}",
                    pct(item["rollout_correct_rate"]),
                ]
            )
    lines.append("## Relevance By Performance Stratum")
    lines.append(markdown_table(
        ["run", "stratum", "n", "any relevant", "mean relevant", "any used", "mean used", "rollout correct"],
        rows,
    ))
    lines.append("")

    gate_rows = []
    for run, strata in analysis["gate_source_strata"].items():
        for source, item in strata.items():
            gate_rows.append(
                [
                    run,
                    source,
                    item["n"],
                    pct(item["pass_at_n"]),
                    pct(item["rollout_correct_rate"]),
                    pct(item["any_relevant"]),
                    f"{item['mean_relevant']:.2f}",
                    pct(item["any_used_correctly"]),
                ]
            )
    if gate_rows:
        lines.append("## Gated Runs By Selected Query Source")
        lines.append(markdown_table(
            ["run", "chosen source", "n", "pass@N", "rollout correct", "any relevant", "mean relevant", "any used"],
            gate_rows,
        ))
        lines.append("")

    if analysis["lift_reports"]:
        rows = []
        for name, report in analysis["lift_reports"].items():
            agg = report.get("pass_at_aggregate", {})
            counts = report.get("counts", {})
            rows.append(
                [
                    name,
                    f"{agg.get('baseline_problems_solved')}->{agg.get('treated_problems_solved')}",
                    f"{agg.get('delta_treated_minus_baseline', float('nan')):+.3f}",
                    counts.get("wins_wrong_to_right"),
                    counts.get("regressions_right_to_wrong"),
                ]
            )
        lines.append("## Existing Lift Reports")
        lines.append(markdown_table(["report", "solved", "delta", "wins", "regressions"], rows))
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    analysis = build_analysis(run_root, list(args.runs), args.correct_threshold)
    report = render_report(analysis)
    print(report)

    if args.out_md:
        out_md = Path(args.out_md).expanduser()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(report + "\n", encoding="utf-8")
    if args.out_json:
        out_json = Path(args.out_json).expanduser()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
