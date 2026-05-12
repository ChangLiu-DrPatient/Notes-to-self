#!/usr/bin/env python3
"""Plot cosine-score diagnostics against LLM-judge relevance labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _norm_text(text: Any) -> str:
    return str(text or "").strip()


def _build_judge_lookup(
    judge_records: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build problem_key -> {by_principle, by_rank} lookup with union labels."""
    lookup: dict[str, dict[str, dict[str, Any]]] = {}

    for rec in judge_records:
        # Keep the denominator aligned with judge-based analysis scripts:
        # parse failures should be unknown labels, not false/irrelevant labels.
        if rec.get("parsed_ok") is False:
            continue

        problem_key = _norm_text(rec.get("problem_key"))
        if not problem_key:
            continue
        state = lookup.setdefault(
            problem_key,
            {"by_principle": {}, "by_rank": {}},
        )
        by_principle: dict[str, dict[str, Any]] = state["by_principle"]
        by_rank: dict[str, dict[str, Any]] = state["by_rank"]

        injected = rec.get("injected_bullets", []) or []
        judged = rec.get("judged_bullets", []) or []
        for idx, ib in enumerate(injected):
            if idx >= len(judged):
                continue
            principle = _norm_text((ib or {}).get("principle"))
            jb = judged[idx]
            relevant = bool((jb or {}).get("relevant"))
            usage = _norm_text((jb or {}).get("usage"))

            if principle:
                entry = by_principle.setdefault(
                    principle,
                    {"relevant_any": False, "usage_labels": set()},
                )
                entry["relevant_any"] = bool(entry["relevant_any"] or relevant)
                if usage:
                    entry["usage_labels"].add(usage)

            rank_entry = by_rank.setdefault(
                str(idx),
                {"relevant_any": False, "usage_labels": set()},
            )
            rank_entry["relevant_any"] = bool(rank_entry["relevant_any"] or relevant)
            if usage:
                rank_entry["usage_labels"].add(usage)

    return lookup


def _attach_labels(
    score_records: list[dict[str, Any]],
    judge_lookup: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[float], list[float], list[float], int]:
    """Attach judge labels in-place and return (rel, irrel, unknown, unmatched)."""
    rel_scores: list[float] = []
    irrel_scores: list[float] = []
    unknown_scores: list[float] = []
    unmatched_hits = 0

    for rec in score_records:
        problem_key = _norm_text(rec.get("problem_key"))
        per_problem = judge_lookup.get(problem_key, {})
        by_principle = per_problem.get("by_principle", {})
        by_rank = per_problem.get("by_rank", {})

        hits = rec.get("top_k_hits", []) or []
        for rank, hit in enumerate(hits):
            principle = _norm_text((hit or {}).get("principle"))
            matched = by_principle.get(principle)
            if matched is None:
                matched = by_rank.get(str(rank))

            score = float((hit or {}).get("cosine_score", 0.0))
            if matched is None:
                hit["is_relevant"] = None
                hit["usage_labels"] = []
                unknown_scores.append(score)
                unmatched_hits += 1
                continue

            is_relevant = bool(matched["relevant_any"])
            usage_labels = sorted(matched["usage_labels"])
            hit["is_relevant"] = is_relevant
            hit["usage_labels"] = usage_labels
            if is_relevant:
                rel_scores.append(score)
            else:
                irrel_scores.append(score)

    return rel_scores, irrel_scores, unknown_scores, unmatched_hits


def _stats_line(name: str, arr: np.ndarray) -> str:
    if arr.size == 0:
        return f"{name}: n=0"
    return (
        f"{name}: n={arr.size}, mean={arr.mean():.4f}, "
        f"median={np.median(arr):.4f}, std={arr.std(ddof=0):.4f}"
    )


def _stats_dict(arr: np.ndarray) -> dict[str, Any]:
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _best_f1_threshold(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float | None, float | None, float | None]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return None, None, None
    p = precision[:-1]
    r = recall[:-1]
    f1 = (2.0 * p * r) / np.clip(p + r, a_min=1e-12, a_max=None)
    idx = int(np.argmax(f1))
    return float(thresholds[idx]), float(p[idx]), float(r[idx])


def _auc_or_none(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _labeled_auc(
    rel_scores: np.ndarray,
    irrel_scores: np.ndarray,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    if not (rel_scores.size and irrel_scores.size):
        return None, None, None, None, None
    y_true = np.concatenate(
        [
            np.ones(rel_scores.shape[0], dtype=np.int32),
            np.zeros(irrel_scores.shape[0], dtype=np.int32),
        ]
    )
    y_score = np.concatenate([rel_scores, irrel_scores]).astype(np.float32)
    auc = _auc_or_none(y_true, y_score)
    threshold, precision_at, recall_at = _best_f1_threshold(y_true, y_score)
    return auc, threshold, precision_at, recall_at, float(y_true.mean())


def _source_breakdown(score_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for rec in score_records:
        source = _norm_text(rec.get("chosen_query_source")) or "unknown"
        state = grouped.setdefault(source, {"relevant": [], "irrelevant": [], "unknown": []})
        for hit in rec.get("top_k_hits", []) or []:
            score = float((hit or {}).get("cosine_score", 0.0))
            rel = (hit or {}).get("is_relevant")
            if rel is True:
                state["relevant"].append(score)
            elif rel is False:
                state["irrelevant"].append(score)
            else:
                state["unknown"].append(score)

    out: dict[str, dict[str, Any]] = {}
    for source, state in grouped.items():
        rel = np.asarray(state["relevant"], dtype=np.float32)
        irrel = np.asarray(state["irrelevant"], dtype=np.float32)
        unknown = np.asarray(state["unknown"], dtype=np.float32)
        auc, threshold, precision_at, recall_at, positive_rate = _labeled_auc(rel, irrel)
        out[source] = {
            "relevant": _stats_dict(rel),
            "irrelevant": _stats_dict(irrel),
            "unknown": _stats_dict(unknown),
            "roc_auc": auc,
            "best_f1_threshold": threshold,
            "best_f1_precision": precision_at,
            "best_f1_recall": recall_at,
            "labeled_positive_rate": positive_rate,
        }
    return out


def _save_aggregate_plot(
    rel_scores: np.ndarray,
    irrel_scores: np.ndarray,
    unknown_scores: np.ndarray,
    all_scores: np.ndarray,
    threshold: float | None,
    out_path: Path,
    *,
    bins: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    if rel_scores.size:
        ax.hist(rel_scores, bins=bins, alpha=0.55, color="tab:green", label="relevant")
    if irrel_scores.size:
        ax.hist(
            irrel_scores,
            bins=bins,
            alpha=0.55,
            color="tab:red",
            label="irrelevant",
        )
    if unknown_scores.size:
        ax.hist(
            unknown_scores,
            bins=bins,
            alpha=0.35,
            color="tab:gray",
            label="unknown-label",
        )
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label="best-f1 threshold")
    ax.set_title("Top-k cosine scores by judge relevance")
    ax.set_xlabel("cosine score")
    ax.set_ylabel("count")
    ax.legend()

    ax = axes[1]
    if all_scores.size:
        ax.hist(all_scores, bins=bins, alpha=0.45, color="lightgray", label="all library scores")
    if rel_scores.size:
        ax.hist(rel_scores, bins=bins, histtype="step", linewidth=1.8, color="tab:green", label="top-k relevant")
    if irrel_scores.size:
        ax.hist(irrel_scores, bins=bins, histtype="step", linewidth=1.8, color="tab:red", label="top-k irrelevant")
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label="best-f1 threshold")
    ax.set_title("Global score landscape vs selected top-k")
    ax.set_xlabel("cosine score")
    ax.set_ylabel("count")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _problem_title(rec: dict[str, Any], limit: int = 100) -> str:
    query = _norm_text(rec.get("query_text"))
    if not query:
        key = _norm_text(rec.get("problem_key"))
        marker = "::user::"
        if marker in key:
            query = key.split(marker, 1)[1]
        else:
            query = key
    query = query.replace("\n", " ")
    if len(query) > limit:
        query = query[:limit] + "..."
    subject = _norm_text(rec.get("subject")) or "unknown"
    return f"subject={subject} | {query}"


def _save_problem_plot(
    rec: dict[str, Any], out_path: Path, *, bins: int, threshold: float | None
) -> None:
    all_scores = np.asarray(rec.get("all_scores", []), dtype=np.float32)
    hits = rec.get("top_k_hits", []) or []

    fig, ax = plt.subplots(figsize=(10, 5))
    if all_scores.size:
        ax.hist(all_scores, bins=bins, alpha=0.45, color="lightgray", label="all library scores")

    used_labels: set[str] = set()
    for hit in hits:
        score = float((hit or {}).get("cosine_score", 0.0))
        rel = (hit or {}).get("is_relevant")
        if rel is True:
            color, label = "tab:green", "top-k relevant"
        elif rel is False:
            color, label = "tab:red", "top-k irrelevant"
        else:
            color, label = "tab:gray", "top-k unknown-label"
        lbl = label if label not in used_labels else None
        used_labels.add(label)
        ax.axvline(score, color=color, linewidth=2.0, alpha=0.9, label=lbl)

    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label="best-f1 threshold")

    ax.set_title(_problem_title(rec))
    ax.set_xlabel("cosine score")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", type=Path, required=True, help="path to *_scores.jsonl from 30_inject --dump_scores")
    p.add_argument("--judge_results", type=Path, required=True, help="path to judge_results.jsonl")
    p.add_argument("--out_dir", type=Path, required=True, help="output directory for plots")
    p.add_argument(
        "--summary_json",
        type=Path,
        default=None,
        help="optional path for machine-readable cosine/relevance stats (default: <out_dir>/cosine_relevance_stats.json)",
    )
    p.add_argument("--problem_idx", type=int, default=None, help="optional per-problem index to visualize")
    p.add_argument("--bins", type=int, default=80, help="histogram bins (default: 80)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.scores.exists():
        raise FileNotFoundError(f"--scores not found: {args.scores}")
    if not args.judge_results.exists():
        raise FileNotFoundError(f"--judge_results not found: {args.judge_results}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    score_records = _read_jsonl(args.scores)
    judge_records = _read_jsonl(args.judge_results)
    if not score_records:
        raise ValueError(f"no score records in {args.scores}")
    if not judge_records:
        raise ValueError(f"no judge records in {args.judge_results}")

    judge_lookup = _build_judge_lookup(judge_records)
    rel_scores_l, irrel_scores_l, unknown_scores_l, unmatched_hits = _attach_labels(
        score_records, judge_lookup
    )

    rel_scores = np.asarray(rel_scores_l, dtype=np.float32)
    irrel_scores = np.asarray(irrel_scores_l, dtype=np.float32)
    unknown_scores = np.asarray(unknown_scores_l, dtype=np.float32)
    all_scores_arrays = [np.asarray(rec.get("all_scores", []), dtype=np.float32) for rec in score_records]
    all_scores_flat = np.concatenate(all_scores_arrays) if any(arr.size for arr in all_scores_arrays) else np.asarray([])

    print(f"score_records: {len(score_records)}")
    print(f"judge_records: {len(judge_records)}")
    print(_stats_line("relevant", rel_scores))
    print(_stats_line("irrelevant", irrel_scores))
    if unknown_scores.size:
        print(_stats_line("unknown-label", unknown_scores))
    if unmatched_hits:
        print(f"unmatched_top_k_hits: {unmatched_hits}")

    threshold = None
    precision_at = None
    recall_at = None
    auc = None
    labeled_positive_rate = None
    if rel_scores.size and irrel_scores.size:
        y_true = np.concatenate(
            [
                np.ones(rel_scores.shape[0], dtype=np.int32),
                np.zeros(irrel_scores.shape[0], dtype=np.int32),
            ]
        )
        y_score = np.concatenate([rel_scores, irrel_scores]).astype(np.float32)
        auc = _auc_or_none(y_true, y_score)
        labeled_positive_rate = float(y_true.mean())
        print(f"roc_auc: {auc:.4f}")
        threshold, precision_at, recall_at = _best_f1_threshold(y_true, y_score)
        if threshold is not None:
            irrel_pruned = float(np.mean(irrel_scores < threshold) * 100.0)
            rel_lost = float(np.mean(rel_scores < threshold) * 100.0)
            print(
                "best_f1_threshold: "
                f"{threshold:.4f} (precision={precision_at:.4f}, recall={recall_at:.4f})"
            )
            print(
                f"at_threshold: prunes {irrel_pruned:.1f}% irrelevant, "
                f"loses {rel_lost:.1f}% relevant"
            )
    else:
        print("roc_auc: n/a (need both relevant and irrelevant labeled bullets)")

    aggregate_path = args.out_dir / "cosine_vs_relevance.png"
    _save_aggregate_plot(
        rel_scores,
        irrel_scores,
        unknown_scores,
        all_scores_flat,
        threshold,
        aggregate_path,
        bins=args.bins,
    )
    print(f"wrote: {aggregate_path}")

    summary_path = args.summary_json or (args.out_dir / "cosine_relevance_stats.json")
    summary = {
        "scores": str(args.scores),
        "judge_results": str(args.judge_results),
        "score_records": len(score_records),
        "judge_records": len(judge_records),
        "top_k_hits": {
            "relevant": _stats_dict(rel_scores),
            "irrelevant": _stats_dict(irrel_scores),
            "unknown": _stats_dict(unknown_scores),
            "unmatched_hits": unmatched_hits,
        },
        "all_library_scores": _stats_dict(all_scores_flat),
        "roc_auc": auc,
        "labeled_positive_rate": labeled_positive_rate,
        "best_f1_threshold": threshold,
        "best_f1_precision": precision_at,
        "best_f1_recall": recall_at,
        "source_breakdown": _source_breakdown(score_records),
        "plots": {
            "aggregate": str(aggregate_path),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote: {summary_path}")

    if args.problem_idx is not None:
        if args.problem_idx < 0 or args.problem_idx >= len(score_records):
            raise IndexError(
                f"--problem_idx {args.problem_idx} out of range [0, {len(score_records) - 1}]"
            )
        rec = score_records[args.problem_idx]
        problem_path = args.out_dir / f"cosine_problem_{args.problem_idx}.png"
        _save_problem_plot(rec, problem_path, bins=args.bins, threshold=threshold)
        print(f"wrote: {problem_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
