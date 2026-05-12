#!/usr/bin/env python3
"""Analyze prompt-only judge outputs for abstraction relevance."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _safe_rate(num: int, den: int) -> float:
    return float(num) / den if den else 0.0


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _variant_key(record: dict[str, Any]) -> str:
    mode = str(record.get("retrieval_mode", "")).strip() or "unknown"
    top_k = record.get("top_k")
    top_k_suffix = f"::topk={top_k}" if top_k is not None else ""
    if mode == "rerank":
        model = str(record.get("rerank_model", "")).strip() or "unknown_model"
        base = (
            str(record.get("rerank_base_mode", "")).strip()
            or str(record.get("retrieval_mode_base", "")).strip()
            or "unknown_base"
        )
        cand_k = record.get("rerank_candidate_k")
        cand_suffix = f"::candk={cand_k}" if cand_k is not None else ""
        return f"rerank::{model}::base={base}{top_k_suffix}{cand_suffix}"
    return f"{mode}{top_k_suffix}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _init_variant_stats() -> dict[str, Any]:
    return {
        "prompts_total": 0,
        "prompts_parsed_ok": 0,
        "prompts_any_relevant": 0,
        "prompts_with_bullets": 0,
        "bullets_total": 0,
        "bullets_relevant": 0,
        "helpfulness": Counter(),
        "by_source": defaultdict(lambda: {"bullets": 0, "relevant": 0}),
        "by_rank": defaultdict(lambda: {"bullets": 0, "relevant": 0}),
        "bi_scores_relevant": [],
        "bi_scores_irrelevant": [],
        "rerank_scores_relevant": [],
        "rerank_scores_irrelevant": [],
        "rank_shift_relevant": [],
        "rank_shift_irrelevant": [],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--judge_results",
        type=Path,
        nargs="+",
        required=True,
        help="one or more judge_prompt_relevance.py output JSONL files",
    )
    p.add_argument("--out_json", type=Path, default=None, help="optional JSON summary path")
    p.add_argument("--out_md", type=Path, default=None, help="optional markdown summary path")
    return p.parse_args()


def _format_pct(num: int, den: int) -> str:
    return f"{100.0 * _safe_rate(num, den):.1f}%"


def _build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Prompt Relevance Summary",
        "",
        f"- records_total: {summary['records_total']}",
        f"- variants: {len(summary['variants'])}",
        "",
        "| variant | prompts | parsed_ok | any_relevant_prompt | bullet_relevant | avg_relevant_bullets_per_prompt |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["variant_rows"]:
        lines.append(
            f"| {row['variant']} | {row['prompts_total']} | {row['prompts_parsed_ok']} | "
            f"{row['any_relevant_prompt_rate']} | {row['bullet_relevant_rate']} | "
            f"{row['avg_relevant_bullets_per_prompt']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    all_records: list[dict[str, Any]] = []
    for path in args.judge_results:
        if not path.exists():
            raise FileNotFoundError(f"--judge_results file not found: {path}")
        all_records.extend(_read_jsonl(path))
    if not all_records:
        raise ValueError("no judge records loaded")

    by_variant: dict[str, dict[str, Any]] = defaultdict(_init_variant_stats)

    for rec in all_records:
        key = _variant_key(rec)
        stats = by_variant[key]
        stats["prompts_total"] += 1

        parsed_ok = bool(rec.get("parsed_ok"))
        if parsed_ok:
            stats["prompts_parsed_ok"] += 1

        source = str(rec.get("selected_query_source", "")).strip() or "unknown"
        injected = rec.get("injected_bullets", [])
        judged = rec.get("judged_bullets", [])

        if isinstance(injected, list) and injected:
            stats["prompts_with_bullets"] += 1
        if not isinstance(judged, list) or not judged:
            continue

        bullet_by_id: dict[int, dict[str, Any]] = {}
        if isinstance(injected, list):
            for ib in injected:
                if not isinstance(ib, dict):
                    continue
                bid = ib.get("id")
                if isinstance(bid, int):
                    bullet_by_id[bid] = ib

        prompt_relevant = 0
        for jb in judged:
            if not isinstance(jb, dict):
                continue
            relevant = bool(jb.get("relevant", False))
            bid = jb.get("id")
            ib = bullet_by_id.get(int(bid)) if isinstance(bid, int) else {}
            if not isinstance(ib, dict):
                ib = {}

            stats["bullets_total"] += 1
            if relevant:
                stats["bullets_relevant"] += 1
                prompt_relevant += 1

            helpfulness = str(jb.get("helpfulness", "")).strip().lower()
            if helpfulness:
                stats["helpfulness"][helpfulness] += 1

            stats["by_source"][source]["bullets"] += 1
            if relevant:
                stats["by_source"][source]["relevant"] += 1

            rank = ib.get("rank")
            if isinstance(rank, int):
                stats["by_rank"][rank]["bullets"] += 1
                if relevant:
                    stats["by_rank"][rank]["relevant"] += 1

            bi_score = _safe_float(ib.get("bi_score", ib.get("cosine_score")))
            rerank_score = _safe_float(ib.get("rerank_score"))
            if bi_score is not None:
                if relevant:
                    stats["bi_scores_relevant"].append(bi_score)
                else:
                    stats["bi_scores_irrelevant"].append(bi_score)
            if rerank_score is not None:
                if relevant:
                    stats["rerank_scores_relevant"].append(rerank_score)
                else:
                    stats["rerank_scores_irrelevant"].append(rerank_score)

            bi_rank = _safe_float(ib.get("bi_rank"))
            rerank_rank = _safe_float(ib.get("rerank_rank"))
            if bi_rank is not None and rerank_rank is not None:
                delta = abs(rerank_rank - bi_rank)
                if relevant:
                    stats["rank_shift_relevant"].append(delta)
                else:
                    stats["rank_shift_irrelevant"].append(delta)

        if prompt_relevant > 0:
            stats["prompts_any_relevant"] += 1

    variant_rows: list[dict[str, Any]] = []
    for key in sorted(by_variant.keys()):
        s = by_variant[key]
        row = {
            "variant": key,
            "prompts_total": s["prompts_total"],
            "prompts_parsed_ok": s["prompts_parsed_ok"],
            "any_relevant_prompt_rate": _format_pct(s["prompts_any_relevant"], s["prompts_total"]),
            "bullet_relevant_rate": _format_pct(s["bullets_relevant"], s["bullets_total"]),
            "avg_relevant_bullets_per_prompt": (
                float(s["bullets_relevant"]) / s["prompts_total"] if s["prompts_total"] else 0.0
            ),
            "helpfulness": dict(s["helpfulness"]),
            "by_source": {
                src: {
                    "bullets": int(v["bullets"]),
                    "relevant": int(v["relevant"]),
                    "relevant_rate": _safe_rate(v["relevant"], v["bullets"]),
                }
                for src, v in sorted(s["by_source"].items())
            },
            "by_rank": {
                int(rank): {
                    "bullets": int(v["bullets"]),
                    "relevant": int(v["relevant"]),
                    "relevant_rate": _safe_rate(v["relevant"], v["bullets"]),
                }
                for rank, v in sorted(s["by_rank"].items(), key=lambda x: int(x[0]))
            },
            "bi_score_mean_relevant": (
                mean(s["bi_scores_relevant"]) if s["bi_scores_relevant"] else None
            ),
            "bi_score_mean_irrelevant": (
                mean(s["bi_scores_irrelevant"]) if s["bi_scores_irrelevant"] else None
            ),
            "rerank_score_mean_relevant": (
                mean(s["rerank_scores_relevant"]) if s["rerank_scores_relevant"] else None
            ),
            "rerank_score_mean_irrelevant": (
                mean(s["rerank_scores_irrelevant"]) if s["rerank_scores_irrelevant"] else None
            ),
            "mean_abs_rank_shift_relevant": (
                mean(s["rank_shift_relevant"]) if s["rank_shift_relevant"] else None
            ),
            "mean_abs_rank_shift_irrelevant": (
                mean(s["rank_shift_irrelevant"]) if s["rank_shift_irrelevant"] else None
            ),
        }
        variant_rows.append(row)

    print(
        "| variant | prompts | parsed_ok | any_relevant_prompt | bullet_relevant | avg_rel_bullets/prompt |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for row in variant_rows:
        print(
            f"| {row['variant']} | {row['prompts_total']} | {row['prompts_parsed_ok']} | "
            f"{row['any_relevant_prompt_rate']} | {row['bullet_relevant_rate']} | "
            f"{row['avg_relevant_bullets_per_prompt']:.3f} |"
        )

    summary = {
        "records_total": len(all_records),
        "variants": sorted(by_variant.keys()),
        "variant_rows": variant_rows,
    }

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote: {args.out_json}")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(_build_markdown(summary), encoding="utf-8")
        print(f"wrote: {args.out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
