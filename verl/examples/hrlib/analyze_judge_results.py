#!/usr/bin/env python3
"""Analyze judge_results.jsonl: abstraction usage stratified by rollout correctness and relevance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


USAGE_LABELS = ["used_correctly", "used_incorrectly", "ignored", "unclear"]


def _pct(num: int, den: int) -> str:
    return f"{100.0 * num / den:.1f}%" if den else "n/a"


def _print_table(
    title: str,
    rows: list[dict[str, Any]],
    total_label: str = "TOTAL",
    *,
    tail_metrics: tuple[tuple[str, str], ...] = (("relevant", "relevant_true"),),
) -> None:
    """Print usage table; tail_metrics are (column_header, row_key_for_numerator)."""
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")

    tm_headers = "".join(f" {hdr:>9s}" for hdr, _ in tail_metrics)
    header = (
        f"{'':30s} {'count':>6s} {'used_ok':>8s} {'used_bad':>8s} "
        f"{'ignored':>8s} {'unclear':>8s}{tm_headers}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        n = r["total"]
        label = r.get("label", total_label)
        tail_cells = "".join(f" {_pct(r.get(key, 0), n):>9s}" for _, key in tail_metrics)
        print(
            f"{label:30s} {n:6d} "
            f"{_pct(r['used_correctly'], n):>8s} "
            f"{_pct(r['used_incorrectly'], n):>8s} "
            f"{_pct(r['ignored'], n):>8s} "
            f"{_pct(r['unclear'], n):>8s}"
            f"{tail_cells}"
        )


def _empty_row(label: str = "") -> dict[str, Any]:
    return {
        "label": label,
        "total": 0,
        "used_correctly": 0,
        "used_incorrectly": 0,
        "ignored": 0,
        "unclear": 0,
        "relevant_true": 0,
        "rollout_correct": 0,
    }


def _add_bullet(row: dict[str, Any], bullet: dict[str, Any], *, rollout_was_correct: bool) -> None:
    row["total"] += 1
    usage = str(bullet.get("usage", "")).strip()
    if usage in USAGE_LABELS:
        row[usage] += 1
    if bullet.get("relevant"):
        row["relevant_true"] += 1
    if rollout_was_correct:
        row["rollout_correct"] += 1


_MERGE_KEYS = ("total", "used_correctly", "used_incorrectly", "ignored", "unclear", "relevant_true", "rollout_correct")


def _merge_rows(a: dict[str, Any], b: dict[str, Any], label: str = "") -> dict[str, Any]:
    out = _empty_row(label)
    for key in _MERGE_KEYS:
        out[key] = a[key] + b[key]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("judge_results", help="Path to judge_results.jsonl")
    args = parser.parse_args()

    path = Path(args.judge_results)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        print("No records found.", file=sys.stderr)
        return 1

    n_parsed_ok = sum(1 for r in records if r.get("parsed_ok"))
    n_parsed_fail = len(records) - n_parsed_ok
    n_correct = sum(1 for r in records if r.get("is_rollout_correct"))
    n_incorrect = len(records) - n_correct

    print(f"Records: {len(records)}  (parsed_ok={n_parsed_ok}, parsed_fail={n_parsed_fail})")
    print(f"Rollouts: correct={n_correct}, incorrect={n_incorrect}")

    # ── 1. Overall usage ─────────────────────────────────────────────────
    overall = _empty_row("OVERALL")
    for r in records:
        corr = bool(r.get("is_rollout_correct"))
        for b in r.get("judged_bullets", []):
            _add_bullet(overall, b, rollout_was_correct=corr)
    _print_table("1. Overall abstraction usage", [overall])

    # ── 2. Stratified by rollout correctness ─────────────────────────────
    by_correct = _empty_row("correct_rollouts")
    by_incorrect = _empty_row("incorrect_rollouts")
    for r in records:
        corr = bool(r.get("is_rollout_correct"))
        target = by_correct if corr else by_incorrect
        for b in r.get("judged_bullets", []):
            _add_bullet(target, b, rollout_was_correct=corr)
    total_corr = _merge_rows(by_correct, by_incorrect, "TOTAL")
    _print_table("2. Stratified by rollout correctness", [by_correct, by_incorrect, total_corr])

    # ── 3. Stratified by bullet relevance ────────────────────────────────
    by_relevant = _empty_row("relevant=true")
    by_irrelevant = _empty_row("relevant=false")
    for r in records:
        corr = bool(r.get("is_rollout_correct"))
        for b in r.get("judged_bullets", []):
            target = by_relevant if b.get("relevant") else by_irrelevant
            _add_bullet(target, b, rollout_was_correct=corr)
    total_rel = _merge_rows(by_relevant, by_irrelevant, "TOTAL")
    _print_table(
        "3. Stratified by bullet relevance",
        [by_relevant, by_irrelevant, total_rel],
        tail_metrics=(("corr_roll", "rollout_correct"),),
    )

    # ── 4. Cross: correctness × relevance ────────────────────────────────
    cross = {
        (True, True): _empty_row("correct + relevant"),
        (True, False): _empty_row("correct + irrelevant"),
        (False, True): _empty_row("incorrect + relevant"),
        (False, False): _empty_row("incorrect + irrelevant"),
    }
    for r in records:
        is_correct = bool(r.get("is_rollout_correct"))
        for b in r.get("judged_bullets", []):
            is_rel = bool(b.get("relevant"))
            _add_bullet(cross[(is_correct, is_rel)], b, rollout_was_correct=is_correct)

    cross_rows = [
        cross[(True, True)],
        cross[(True, False)],
        cross[(False, True)],
        cross[(False, False)],
        _merge_rows(
            _merge_rows(cross[(True, True)], cross[(True, False)]),
            _merge_rows(cross[(False, True)], cross[(False, False)]),
            "TOTAL",
        ),
    ]
    _print_table(
        "4. Cross: rollout correctness × bullet relevance",
        cross_rows,
        tail_metrics=(
            ("relevant", "relevant_true"),
            ("corr_roll", "rollout_correct"),
        ),
    )

    # ── 5. Per-problem summary: how many relevant bullets & usage rates ──
    problem_stats: dict[str, dict[str, Any]] = {}
    for r in records:
        pk = r.get("problem_key", "?")
        if pk not in problem_stats:
            problem_stats[pk] = {
                "n_rollouts": 0,
                "n_correct": 0,
                "n_bullets": 0,
                "n_relevant": 0,
                "n_used_correctly": 0,
                "n_used_incorrectly": 0,
            }
        ps = problem_stats[pk]
        ps["n_rollouts"] += 1
        if r.get("is_rollout_correct"):
            ps["n_correct"] += 1
        for b in r.get("judged_bullets", []):
            ps["n_bullets"] += 1
            if b.get("relevant"):
                ps["n_relevant"] += 1
            usage = str(b.get("usage", ""))
            if usage == "used_correctly":
                ps["n_used_correctly"] += 1
            elif usage == "used_incorrectly":
                ps["n_used_incorrectly"] += 1

    n_problems = len(problem_stats)
    n_with_any_relevant = sum(1 for ps in problem_stats.values() if ps["n_relevant"] > 0)
    n_with_any_used_correctly = sum(1 for ps in problem_stats.values() if ps["n_used_correctly"] > 0)
    n_with_any_used_incorrectly = sum(1 for ps in problem_stats.values() if ps["n_used_incorrectly"] > 0)

    print(f"\n{'=' * 80}")
    print("  5. Per-problem summary")
    print(f"{'=' * 80}")
    print(f"  Problems judged:                         {n_problems}")
    print(f"  Problems with ≥1 relevant bullet:        {n_with_any_relevant} ({_pct(n_with_any_relevant, n_problems)})")
    print(f"  Problems with ≥1 used_correctly bullet:  {n_with_any_used_correctly} ({_pct(n_with_any_used_correctly, n_problems)})")
    print(f"  Problems with ≥1 used_incorrectly bullet:{n_with_any_used_incorrectly} ({_pct(n_with_any_used_incorrectly, n_problems)})")

    # ── 6. Bullet type (strategy vs caution) breakdown ───────────────────
    by_type: dict[str, dict[str, Any]] = {}
    for r in records:
        corr = bool(r.get("is_rollout_correct"))
        for ib, jb in zip(r.get("injected_bullets", []), r.get("judged_bullets", [])):
            btype = str(ib.get("type", "unknown")).lower()
            if btype not in by_type:
                by_type[btype] = _empty_row(btype)
            _add_bullet(by_type[btype], jb, rollout_was_correct=corr)

    type_rows = sorted(by_type.values(), key=lambda x: -x["total"])
    type_total = _empty_row("TOTAL")
    for tr in type_rows:
        type_total = _merge_rows(type_total, tr, "TOTAL")
    type_rows.append(type_total)
    _print_table("6. Stratified by bullet type (strategy vs caution)", type_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
