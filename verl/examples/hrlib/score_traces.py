#!/usr/bin/env python3
"""Score sampled traces and write a labeled parquet."""

import argparse
import json
from collections.abc import Mapping, Sequence

import pandas as pd

from verl.utils.reward_score.math_reward import compute_score


def _extract_ground_truth(reward_model_obj) -> str:
    if isinstance(reward_model_obj, Mapping):
        return str(reward_model_obj.get("ground_truth", ""))
    if isinstance(reward_model_obj, str):
        try:
            parsed = json.loads(reward_model_obj)
            if isinstance(parsed, Mapping):
                return str(parsed.get("ground_truth", ""))
        except json.JSONDecodeError:
            pass
    return ""


def _normalize_responses(responses_obj) -> list[str]:
    if responses_obj is None:
        return []
    if isinstance(responses_obj, str):
        return [responses_obj]
    if hasattr(responses_obj, "tolist"):
        converted = responses_obj.tolist()
        if isinstance(converted, list):
            return [str(item) for item in converted]
    if isinstance(responses_obj, Sequence):
        return [str(item) for item in responses_obj]
    return [str(responses_obj)]


def score_row(row: pd.Series) -> tuple[list[float], int, int]:
    ground_truth = _extract_ground_truth(row.get("reward_model"))
    responses = _normalize_responses(row.get("responses"))

    scores: list[float] = []
    for response in responses:
        try:
            score = float(compute_score(response, ground_truth))
        except Exception as exc:  # defensive: one bad sample should not fail whole file
            print(f"[warn] scoring failed, fallback to 0.0: {exc}")
            score = 0.0
        scores.append(score)

    success_count = sum(score > 0 for score in scores)
    failure_count = len(scores) - success_count
    return scores, success_count, failure_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Score trace parquet with math_reward.compute_score.")
    parser.add_argument("--in_parquet", required=True, help="Input parquet from main_generation.py")
    parser.add_argument("--out_parquet", required=True, help="Output parquet with score columns")
    args = parser.parse_args()

    df = pd.read_parquet(args.in_parquet)
    if "reward_model" not in df.columns or "responses" not in df.columns:
        raise ValueError("Input parquet must contain `reward_model` and `responses` columns.")

    scores_col: list[list[float]] = []
    success_col: list[int] = []
    failure_col: list[int] = []

    for _, row in df.iterrows():
        scores, success_count, failure_count = score_row(row)
        scores_col.append(scores)
        success_col.append(success_count)
        failure_col.append(failure_count)

    df["scores"] = scores_col
    df["success_count"] = success_col
    df["failure_count"] = failure_col
    df.to_parquet(args.out_parquet)

    print(f"[done] wrote labeled traces to: {args.out_parquet}")


if __name__ == "__main__":
    main()
