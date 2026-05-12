#!/usr/bin/env python3
"""Convert trainer validation JSONL dumps into trace parquet format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict, deque
from typing import Any

import pandas as pd


def _latest_jsonl_file(val_jsonl_dir: str) -> Path:
    files = sorted(Path(val_jsonl_dir).glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No jsonl files found in {val_jsonl_dir}")
    return files[-1]


def _load_entries(jsonl_path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if not entries:
        raise ValueError(f"Validation dump {jsonl_path} is empty")
    return entries


def _group_entries(entries: list[dict[str, Any]], n_samples: int) -> tuple[list[list[str]], list[str]]:
    if len(entries) % n_samples != 0:
        raise ValueError(
            f"Validation dump line count {len(entries)} is not divisible by n_samples={n_samples}. "
            "Cannot reconstruct grouped responses."
        )

    grouped_outputs: list[list[str]] = []
    grouped_ground_truth: list[str] = []
    for start in range(0, len(entries), n_samples):
        chunk = entries[start : start + n_samples]
        grouped_outputs.append([str(item.get("output", "")) for item in chunk])
        grouped_ground_truth.append(str(chunk[0].get("gts", "")))
    return grouped_outputs, grouped_ground_truth


def _extract_user_content(input_text: str) -> str:
    if not isinstance(input_text, str):
        return ""
    prefix = "user\n"
    suffix = "\nassistant\n"
    if input_text.startswith(prefix) and input_text.endswith(suffix):
        return input_text[len(prefix) : -len(suffix)]
    return input_text


def _extract_prompt_content(prompt_obj: Any) -> str:
    # parquet prompt column stores an array containing {"role": "user", "content": "..."}.
    if hasattr(prompt_obj, "tolist"):
        prompt_obj = prompt_obj.tolist()
    if isinstance(prompt_obj, list):
        for item in prompt_obj:
            if isinstance(item, dict) and item.get("role") == "user":
                return str(item.get("content", ""))
        if prompt_obj:
            return str(prompt_obj[0])
    if isinstance(prompt_obj, dict):
        if prompt_obj.get("role") == "user":
            return str(prompt_obj.get("content", ""))
    return str(prompt_obj)


def _extract_ground_truth(reward_model_obj: Any) -> str:
    if isinstance(reward_model_obj, dict):
        return str(reward_model_obj.get("ground_truth", ""))
    if isinstance(reward_model_obj, str):
        try:
            parsed = json.loads(reward_model_obj)
            if isinstance(parsed, dict):
                return str(parsed.get("ground_truth", ""))
        except json.JSONDecodeError:
            return reward_model_obj
    return str(reward_model_obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert val JSONL to trace parquet.")
    parser.add_argument("--source_parquet", required=True, help="Original dataset parquet used as val_files")
    parser.add_argument("--val_jsonl_dir", required=True, help="Directory with trainer validation JSONL dumps")
    parser.add_argument("--n_samples", required=True, type=int, help="Number of samples per prompt")
    parser.add_argument("--out_parquet", required=True, help="Output trace parquet path")
    args = parser.parse_args()

    jsonl_path = _latest_jsonl_file(args.val_jsonl_dir)
    entries = _load_entries(jsonl_path)
    # Validate n_samples grouping invariant before alignment.
    _group_entries(entries, args.n_samples)

    df = pd.read_parquet(args.source_parquet).reset_index(drop=True)
    generation_by_key: dict[tuple[str, str, str], deque[tuple[list[str], str]]] = defaultdict(deque)
    for start in range(0, len(entries), args.n_samples):
        chunk = entries[start : start + args.n_samples]
        first = chunk[0]
        key = (
            _extract_user_content(str(first.get("input", ""))),
            str(first.get("data_source", "")),
            str(first.get("gts", "")),
        )
        generation_by_key[key].append(
            ([str(item.get("output", "")) for item in chunk], str(first.get("gts", "")))
        )

    selected_indices: list[int] = []
    aligned_outputs: list[list[str]] = []
    aligned_ground_truth: list[str] = []
    for idx, row in df.iterrows():
        key = (
            _extract_prompt_content(row.get("prompt")),
            str(row.get("data_source", "")),
            _extract_ground_truth(row.get("reward_model")),
        )
        if not generation_by_key[key]:
            continue
        outputs, ground_truth = generation_by_key[key].popleft()
        selected_indices.append(idx)
        aligned_outputs.append(outputs)
        aligned_ground_truth.append(ground_truth)

    unmatched_generation = sum(len(bucket) for bucket in generation_by_key.values())
    if unmatched_generation > 0:
        raise ValueError(
            f"Could not align {unmatched_generation * args.n_samples} generated entries back to source rows. "
            "Please verify input prompt formatting and data_source/ground_truth fields."
        )
    if not aligned_outputs:
        raise ValueError("Failed to align any generated outputs to source rows.")

    if len(df) != len(aligned_outputs):
        print(
            f"[warn] source rows ({len(df)}) != aligned outputs ({len(aligned_outputs)}). "
            "Likely due to prompt-length filtering; writing aligned subset only."
        )

    df = df.iloc[selected_indices].reset_index(drop=True)
    df["responses"] = aligned_outputs
    if "reward_model" not in df.columns:
        df["reward_model"] = [{"style": "rule", "ground_truth": gt} for gt in aligned_ground_truth]

    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet)
    print(f"[done] converted {jsonl_path} -> {args.out_parquet}")


if __name__ == "__main__":
    main()
