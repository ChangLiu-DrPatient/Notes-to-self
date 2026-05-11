from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any


_MATH_REWARD_SOURCES = {
    "lighteval/MATH",
    "DigitalLearningGmbH/MATH-lighteval",
    "HuggingFaceH4/MATH-500",
}

_MATH_DAPO_SOURCES = {
    "math",
    "math_dapo",
    "math_dapo_reasoning",
}

_PRIME_MATH_SOURCES = {
    "numina_aops_forum",
    "numina_synthetic_math",
    "numina_amc_aime",
    "numina_synthetic_amc",
    "numina_cn_k12",
    "numina_olympiads",
}

_SEARCH_R1_SOURCES = {
    "searchR1_nq",
    "searchR1_triviaqa",
    "searchR1_popqa",
    "searchR1_hotpotqa",
    "searchR1_2wikimultihopqa",
    "searchR1_musique",
    "searchR1_bamboogle",
}

_MATH_DAPO_ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"


def get_source_family(data_source: str) -> str:
    """Map a data source string to its canonical extraction/scoring family."""
    ds = str(data_source)
    if ds == "openai/gsm8k":
        return "gsm8k"
    if ds in _MATH_REWARD_SOURCES:
        return "math_reward"
    if ds in _MATH_DAPO_SOURCES or ds.startswith("aime"):
        return "math_dapo"
    if ds in _PRIME_MATH_SOURCES:
        return "prime_math"
    if ds == "hiyouga/geometry3k":
        return "geometry3k"
    if ds in _SEARCH_R1_SOURCES:
        return "search_r1"

    raise NotImplementedError(
        f"Answer extraction is not implemented for data_source={data_source!r}. "
        "This source is not currently supported by Intuitor/default_compute_score."
    )


def _extract_math_reward(solution_str: str, _: dict[str, Any] | None) -> str | None:
    from .math_reward import last_boxed_only_string, remove_boxed

    boxed = last_boxed_only_string(solution_str)
    if boxed is None:
        return None

    try:
        return remove_boxed(boxed)
    except Exception:
        return None


def _extract_math_dapo(solution_str: str, extra_info: dict[str, Any] | None) -> str | None:
    from .math_dapo import (
        is_correct_strict_box,
        last_boxed_only_string,
        normalize_final_answer,
        remove_boxed,
    )

    strict_box_verify = bool(extra_info.get("strict_box_verify", False)) if extra_info else False
    pause_tokens_index = extra_info.get("pause_tokens_index") if extra_info else None
    solution_str = solution_str[-300:]

    if strict_box_verify:
        _, pred = is_correct_strict_box(solution_str, gt="", pause_tokens_index=pause_tokens_index)
        return pred

    match = re.findall(_MATH_DAPO_ANSWER_PATTERN, solution_str)
    if match:
        return normalize_final_answer(match[-1])

    boxed = last_boxed_only_string(solution_str)
    if boxed is None:
        return None

    try:
        return normalize_final_answer(remove_boxed(boxed))
    except Exception:
        return None


def _extract_prime_math(solution_str: str, _: dict[str, Any] | None) -> str | None:
    from .prime_math import match_answer

    is_matched, extracted = match_answer(str(solution_str))
    if not is_matched:
        return None
    return extracted


def _extract_gsm8k(solution_str: str, extra_info: dict[str, Any] | None) -> str | None:
    from .gsm8k import extract_solution

    method = extra_info.get("method", "strict") if extra_info else "strict"
    return extract_solution(solution_str=solution_str, method=method)


def _extract_search_r1(solution_str: str, _: dict[str, Any] | None) -> str | None:
    from .search_r1_like_qa_em import extract_solution

    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        return None
    return answer


def _extract_geo3k(solution_str: str, extra_info: dict[str, Any] | None) -> str | None:
    use_boxed = True if extra_info is None else extra_info.get("use_boxed", True)
    if not use_boxed:
        return solution_str

    from mathruler.grader import extract_boxed_content  # type: ignore[import-not-found]

    return extract_boxed_content(solution_str)


def extract_answer(data_source: str, solution_str: str, extra_info: dict[str, Any] | None = None) -> str | None:
    """Extract one normalized answer using the same dataset-specific parser family as Intuitor/new verl scorers."""
    source_family = get_source_family(data_source)
    if source_family == "gsm8k":
        return _extract_gsm8k(solution_str, extra_info)
    if source_family == "math_reward":
        return _extract_math_reward(solution_str, extra_info)
    if source_family == "math_dapo":
        return _extract_math_dapo(solution_str, extra_info)
    if source_family == "prime_math":
        return _extract_prime_math(solution_str, extra_info)
    if source_family == "geometry3k":
        return _extract_geo3k(solution_str, extra_info)
    if source_family == "search_r1":
        return _extract_search_r1(solution_str, extra_info)
    raise RuntimeError(f"Unhandled source_family={source_family!r} for data_source={data_source!r}")


def extract_answers(
    data_source: str,
    all_outputs: Sequence[str],
    extra_info: Sequence[dict[str, Any] | None] | dict[str, Any] | None = None,
) -> list[str | None]:
    """Extract one answer per output, preserving alignment with the original outputs."""
    if extra_info is None or isinstance(extra_info, dict):
        all_extra_info = [extra_info] * len(all_outputs)
    else:
        all_extra_info = list(extra_info)
        if len(all_extra_info) != len(all_outputs):
            raise ValueError(f"len(extra_info)={len(all_extra_info)} does not match len(all_outputs)={len(all_outputs)}")

    return [extract_answer(data_source, output, sample_extra_info) for output, sample_extra_info in zip(all_outputs, all_extra_info)]


def auto_extract(
    data_source: str,
    all_outputs: Sequence[str],
    extra_info: Sequence[dict[str, Any] | None] | dict[str, Any] | None = None,
    *,
    drop_none: bool = True,
) -> list[str]:
    """Compatibility helper for old TTRL code paths.

    By default this matches EVOL's old behavior and drops failed parses before returning.
    """
    answers = extract_answers(data_source=data_source, all_outputs=all_outputs, extra_info=extra_info)
    if drop_none:
        return [answer for answer in answers if answer is not None]
    return answers


__all__ = ["auto_extract", "extract_answer", "extract_answers", "get_source_family"]
