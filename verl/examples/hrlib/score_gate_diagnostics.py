#!/usr/bin/env python3
"""Summarize score-gated retrieval diagnostics from *_scores.jsonl.

Usage:
  python examples/hrlib/score_gate_diagnostics.py \
    --scores /path/to/test_hrlib_*_scores.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize score gate diagnostics from scores sidecar.")
    p.add_argument("--scores", type=Path, required=True, help="path to *_scores.jsonl")
    p.add_argument("--top_n", type=int, default=10, help="how many top gain/loss rows to print")
    p.add_argument(
        "--show_rows",
        action="store_true",
        help="print per-row top gain/loss details (default: only summary)",
    )
    return p.parse_args()


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fmt(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.4f}"


def _pick_str(rec: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = rec.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _infer_mode(rec: dict[str, Any]) -> str:
    mode = _pick_str(rec, "retrieval_mode").lower()
    if mode in {"orig", "rewrite", "score_gate"}:
        return mode
    gate = _pick_str(rec, "query_gate").lower()
    if gate == "score":
        return "score_gate"
    rewrite_found = bool(rec.get("query_rewrite_found", False))
    return "rewrite" if rewrite_found else "orig"


def _extract_hit_score(hit: Any) -> float | None:
    if not isinstance(hit, dict):
        return None
    return _safe_float(hit.get("bi_score", hit.get("cosine_score", hit.get("score"))))


def _extract_selected_top1(rec: dict[str, Any], selected_source: str) -> float | None:
    hits = rec.get("top_k_hits")
    if isinstance(hits, list) and hits:
        score = _extract_hit_score(hits[0])
        if score is not None:
            return score

    candidates = rec.get("candidate_hits")
    if isinstance(candidates, dict):
        source_hits = candidates.get(selected_source)
        if isinstance(source_hits, list) and source_hits:
            return _extract_hit_score(source_hits[0])
    return None


def main() -> int:
    args = _parse_args()
    if not args.scores.exists():
        raise FileNotFoundError(f"--scores not found: {args.scores}")

    rows: list[dict[str, Any]] = []
    with args.scores.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)

    if not rows:
        raise ValueError(f"scores file is empty: {args.scores}")

    chosen_top1: list[float] = []
    orig_top1: list[float] = []
    rewrite_top1: list[float] = []
    deltas: list[float] = []
    rewrite_selected = 0
    original_selected = 0
    gate_rows = 0
    rewrite_found_rows = 0

    gain_rows: list[tuple[float, int, str, str, str]] = []
    loss_rows: list[tuple[float, int, str, str, str]] = []

    for r in rows:
        chosen = _pick_str(r, "selected_query_source", "chosen_query_source").lower()
        if not chosen:
            chosen = "original"
        selected_top1 = _extract_selected_top1(r, chosen)
        if selected_top1 is not None:
            chosen_top1.append(selected_top1)

        if chosen == "rewrite":
            rewrite_selected += 1
        elif chosen == "original":
            original_selected += 1

        rewrite_found = bool(r.get("query_rewrite_found", False))
        if not rewrite_found:
            q0 = _pick_str(r, "query_text_original")
            q1 = _pick_str(r, "query_text_retrieval")
            rewrite_found = bool(q0 and q1 and q0.strip() != q1.strip())
        if rewrite_found:
            rewrite_found_rows += 1

        go = _safe_float(r.get("gate_score_original"))
        gr = _safe_float(r.get("gate_score_rewrite"))
        mode = _infer_mode(r)
        if go is not None and gr is not None:
            if mode == "score_gate":
                gate_rows += 1
            orig_top1.append(go)
            rewrite_top1.append(gr)
            delta = gr - go
            deltas.append(delta)
            idx = int(r.get("problem_idx", -1))
            q0 = _pick_str(r, "query_text_original")[:120]
            q1 = _pick_str(r, "query_text_retrieval")[:120]
            payload = (delta, idx, chosen, q0, q1)
            if delta >= 0:
                gain_rows.append(payload)
            else:
                loss_rows.append(payload)

    n = len(rows)
    print(f"scores_file: {args.scores}")
    print(f"rows: {n}")
    print(f"rewrite_found_rows: {rewrite_found_rows} ({rewrite_found_rows / n:.2%})")
    print(f"selected_rewrite_rows: {rewrite_selected} ({rewrite_selected / n:.2%})")
    print(f"selected_original_rows: {original_selected} ({original_selected / n:.2%})")
    print(f"rows_with_gate_scores: {gate_rows} ({gate_rows / n:.2%})")

    if chosen_top1:
        print(
            "chosen_top1_cosine: "
            f"mean={mean(chosen_top1):.4f} "
            f"p50={median(chosen_top1):.4f} "
            f"min={min(chosen_top1):.4f} "
            f"max={max(chosen_top1):.4f}"
        )
    else:
        print("chosen_top1_cosine: n/a")

    if orig_top1 and rewrite_top1:
        print(
            "original_top1_cosine: "
            f"mean={mean(orig_top1):.4f} "
            f"p50={median(orig_top1):.4f} "
            f"min={min(orig_top1):.4f} "
            f"max={max(orig_top1):.4f}"
        )
        print(
            "rewrite_top1_cosine: "
            f"mean={mean(rewrite_top1):.4f} "
            f"p50={median(rewrite_top1):.4f} "
            f"min={min(rewrite_top1):.4f} "
            f"max={max(rewrite_top1):.4f}"
        )
        print(
            "delta(rewrite-original): "
            f"mean={mean(deltas):.4f} "
            f"p50={median(deltas):.4f} "
            f"min={min(deltas):.4f} "
            f"max={max(deltas):.4f}"
        )
    else:
        print("gate score comparison: n/a (missing gate score fields)")

    if not args.show_rows:
        return 0

    print()
    print(f"Top {args.top_n} rewrite gains")
    gain_rows.sort(key=lambda x: x[0], reverse=True)
    for delta, idx, chosen, q0, q1 in gain_rows[: args.top_n]:
        print(f"idx={idx} delta={_fmt(delta)} chosen={chosen}")
        print(f"  orig: {q0}")
        print(f"  rw  : {q1}")

    print()
    print(f"Top {args.top_n} rewrite losses")
    loss_rows.sort(key=lambda x: x[0])
    for delta, idx, chosen, q0, q1 in loss_rows[: args.top_n]:
        print(f"idx={idx} delta={_fmt(delta)} chosen={chosen}")
        print(f"  orig: {q0}")
        print(f"  rw  : {q1}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
