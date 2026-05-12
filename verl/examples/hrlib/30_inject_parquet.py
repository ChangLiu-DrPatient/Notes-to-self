#!/usr/bin/env python3
"""Inject retrieved HRLib abstractions into the system turn of a prompt parquet.

Stage 0 v1 (see implementation_plan_stage0_3.md §5.2.4): for every row,

1. Read the selected role's content (default ``user``) as the query.
2. Retrieve ``top_k`` library hits (flat cosine top-k via
   :class:`hrlib.retrieve.FlatRetriever`).
3. Render them with :func:`render_hits` and prepend to the system message.
4. Write a new parquet and a sibling ``injection_meta.json`` receipt.

No row is dropped; ``prompt``/``data_source``/``reward_model``/``extra_info``
remain structurally identical apart from the edited system turn.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

_HRLIB_DIR = Path(__file__).resolve().parent
if str(_HRLIB_DIR) not in sys.path:
    sys.path.insert(0, str(_HRLIB_DIR))

from retrieve import FlatRetriever, RetrievalHit  # noqa: E402


# ---------------------------------------------------------------------------
# Subject canonicalization for query-domain prefixing
# ---------------------------------------------------------------------------

# Conservative Stage-0 mapping from MATH-500 subject buckets to the canonical
# domains used in the aggregated HRLib library.
_MATH_SUBJECT_TO_CANON: dict[str, str] = {
    "Algebra": "algebra",
    "Intermediate Algebra": "algebra",
    "Prealgebra": "algebra",
    "Number Theory": "number theory",
    "Geometry": "geometry",
    "Counting & Probability": "combinatorics",
    "Precalculus": "trigonometry",
}


def _canonicalize_subject(raw: str) -> str:
    if not raw:
        return ""
    return _MATH_SUBJECT_TO_CANON.get(raw.strip(), "")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_RENDER_HEADER = "## Relevant Strategies & Cautions"


def render_hits_v1(hits: list[RetrievalHit]) -> str:
    """v1 template: flat `[type] principle` bullet list (no meta-instruction).

    Format (stable; changing it invalidates past injected v1 parquets):

        ## Relevant Strategies & Cautions

        All strategies appear first (in retriever rank order), then all cautions
        (in retriever rank order).

        - [strategy] <principle>
          when: <when_to_apply>
        - [caution] <principle>
          ...
    """
    if not hits:
        return ""
    strategies, cautions = _bucket_hits(hits)
    ordered = [*strategies, *cautions]
    lines: list[str] = [_RENDER_HEADER, ""]
    for h in ordered:
        tag = (h.type or "").strip() or "note"
        principle = (h.principle or "").strip()
        lines.append(f"- [{tag}] {principle}".rstrip())
        when = (h.when_to_apply or "").strip()
        if when:
            lines.append(f"  when: {when}")
    return "\n".join(lines)


# Back-compat alias: `render_hits` refers to the v1 renderer.
render_hits = render_hits_v1


def _bucket_hits(
    hits: list[RetrievalHit],
) -> tuple[list[RetrievalHit], list[RetrievalHit]]:
    """Split hits into (strategies, cautions) preserving retriever rank order.

    Anything that isn't exactly ``"caution"`` (case-insensitive) bucket to
    strategies so we don't silently drop unexpected types.
    """
    strategies: list[RetrievalHit] = []
    cautions: list[RetrievalHit] = []
    for h in hits:
        t = (h.type or "").strip().lower()
        if t == "caution":
            cautions.append(h)
        else:
            strategies.append(h)
    return strategies, cautions


def _render_bullets_plain(hits: list[RetrievalHit]) -> list[str]:
    """Plain ``- <principle>\\n  when: <when>`` lines (no [type] tag).

    Used by templates that already separate hits by type into labeled sections.
    """
    out: list[str] = []
    for h in hits:
        principle = (h.principle or "").strip()
        if not principle:
            continue
        out.append(f"- {principle}")
        when = (h.when_to_apply or "").strip()
        if when:
            out.append(f"  when: {when}")
    return out


def render_hits_standard_rag(hits: list[RetrievalHit]) -> str:
    """T1+T2 merged template: "faithful-RAG" framing + asymmetric type split.

    - T1: `<reference_notes>` wrapper + permission-to-ignore + no-citation.
    - T2: labeled Strategies / Cautions sub-sections with asymmetric usage
      instructions ("adopt if it fits" vs "verify at the end").

    Gracefully collapses when a bucket is empty (drops the empty section *and*
    the usage bullet that references it). Returns ``""`` when both buckets are
    empty, which causes the injector to leave the prompt unchanged.
    """
    if not hits:
        return ""
    strategies, cautions = _bucket_hits(hits)
    if not strategies and not cautions:
        return ""

    # Intro + asymmetric usage bullets, conditional on which buckets exist.
    intro: list[str] = [
        "<reference_notes>",
        "The following notes were retrieved from a library of lessons distilled",
        "from past attempts on similar problems. They MAY OR MAY NOT apply to",
        "the current problem.",
        "",
        "How to use them:",
    ]
    if strategies:
        intro.append(
            "- STRATEGIES are candidate approaches — adopt one only if it directly fits."
        )
    if cautions:
        intro.append(
            "- CAUTIONS are common errors — before giving the final answer, verify"
        )
        intro.append("  you are not making any that apply here.")
    intro.append("- If none apply, solve the problem normally.")
    # intro.append("- Do NOT quote, list, or cite these notes in your answer.")
    intro.append("- Properly cite the retrieved notes in your answer.")

    body: list[str] = []
    if strategies:
        body.append("")
        body.append("## Strategies")
        body.extend(_render_bullets_plain(strategies))
    if cautions:
        body.append("")
        body.append("## Cautions (verify at the end)")
        body.extend(_render_bullets_plain(cautions))

    closing = ["</reference_notes>"]
    return "\n".join([*intro, *body, *closing])


RENDERERS: dict[str, Callable[[list[RetrievalHit]], str]] = {
    "v1": render_hits_v1,
    "std_rag": render_hits_standard_rag,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prompt_to_list(value: Any) -> list[dict[str, Any]]:
    """Normalize parquet-loaded prompts (ndarray/list of dicts) to a plain list."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"prompt must be list-like, got {type(value).__name__}")
    out: list[dict[str, Any]] = []
    for msg in value:
        if isinstance(msg, dict):
            out.append({str(k): v for k, v in msg.items()})
        elif hasattr(msg, "items"):
            out.append({str(k): v for k, v in msg.items()})
        else:
            raise TypeError(f"prompt item must be dict-like, got {type(msg).__name__}")
    return out


def _find_query_text(prompt: list[dict[str, Any]], role: str) -> str:
    role = role.strip().lower()
    for msg in prompt:
        if str(msg.get("role", "")).strip().lower() == role:
            return str(msg.get("content", ""))
    raise ValueError(f"prompt has no message with role={role!r}")


def _prepend_to_system(prompt: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    """Return a new prompt list with ``prefix`` prepended to the system turn."""
    if not prefix:
        return prompt
    new_prompt = [dict(m) for m in prompt]
    if new_prompt and str(new_prompt[0].get("role", "")).strip().lower() == "system":
        existing = str(new_prompt[0].get("content", ""))
        new_prompt[0]["content"] = f"{prefix}\n\n{existing}" if existing else prefix
    else:
        new_prompt = [{"role": "system", "content": prefix}, *new_prompt]
    return new_prompt


def _normalize_mapping(value: Any) -> dict[str, Any]:
    """Best-effort convert parquet cell payloads to a plain dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "items"):
        return {str(k): v for k, v in value.items()}
    return {}


def _read_subject(row: Any, field: str) -> str:
    """Read subject from row path, with optional dotted path support.

    Examples:
      - ``subject`` (top-level column)
      - ``extra_info.subject`` (nested dict-like cell)
    """
    key = str(field or "subject").strip()
    if not key:
        key = "subject"
    parts = [p.strip() for p in key.split(".") if p.strip()]
    if not parts:
        parts = ["subject"]

    # Resolve root from row first.
    try:
        cur: Any = row[parts[0]]
    except Exception:
        try:
            cur = row.get(parts[0])  # type: ignore[assignment,union-attr]
        except Exception:
            cur = None
    if cur is None:
        return ""

    # Walk nested dotted keys through dict-like values.
    for part in parts[1:]:
        mapping = _normalize_mapping(cur)
        if not mapping:
            return ""
        cur = mapping.get(part)
        if cur is None:
            return ""

    return str(cur).strip()


def _problem_key_from_row(row: Any, query_text: str) -> str:
    """Build a stable per-problem key aligned with judge/analyze scripts."""
    data_source = str(row.get("data_source", "")).strip() if hasattr(row, "get") else ""
    return f"{data_source}::user::{query_text}"


def _retrieve_hits_and_scores(
    retriever: FlatRetriever,
    *,
    query_text: str,
    top_k: int,
    canon_subject: str,
    dump_scores: bool,
) -> tuple[list[RetrievalHit], np.ndarray | None]:
    """Retrieve hits for one query with subject-aware fallback recipe behavior."""
    scores: np.ndarray | None = None
    if canon_subject:
        if dump_scores:
            hits, scores = retriever.retrieve_with_all_scores(
                query_text, k=top_k, subject=canon_subject
            )
        else:
            hits = retriever.retrieve(query_text, k=top_k, subject=canon_subject)
        return hits, scores

    # Graceful fallback: no mapped subject -> keep v1 behavior.
    fallback_recipe = retriever.query_recipe
    retriever.query_recipe = "{user_text}"
    try:
        if dump_scores:
            hits, scores = retriever.retrieve_with_all_scores(query_text, k=top_k)
        else:
            hits = retriever.retrieve(query_text, k=top_k)
    finally:
        retriever.query_recipe = fallback_recipe
    return hits, scores


def _gate_metric_score(hits: list[RetrievalHit], metric: str) -> float | None:
    """Compute the scalar gate score used to compare original vs rewritten queries."""
    if not hits:
        return None
    if metric == "top1":
        return float(hits[0].score)
    raise ValueError(f"unknown gate metric: {metric!r}")


def _pick_query_source_by_score(
    original_score: float | None,
    rewritten_score: float | None,
    *,
    margin: float,
    tie_policy: str,
) -> str:
    """Choose query source using score-only gating."""
    original = float(original_score) if original_score is not None else float("-inf")
    rewritten = float(rewritten_score) if rewritten_score is not None else float("-inf")
    delta = rewritten - original
    if delta > margin:
        return "rewrite"
    if np.isclose(delta, margin):
        return "rewrite" if tie_policy == "prefer_rewrite" else "original"
    return "original"


def _normalize_retrieval_mode(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve explicit/legacy inputs into one retrieval mode + compat query_gate."""
    explicit = str(args.retrieval_mode or "").strip().lower()
    legacy_gate = str(args.query_gate or "off").strip().lower()
    has_query_parquet = args.query_parquet is not None

    if explicit:
        if explicit not in {"orig", "rewrite", "score_gate"}:
            raise ValueError(f"unknown --retrieval_mode: {args.retrieval_mode!r}")
        if explicit in {"rewrite", "score_gate"} and not has_query_parquet:
            raise ValueError(f"--retrieval_mode={explicit} requires --query_parquet")
        if legacy_gate not in {"off", "score"}:
            raise ValueError(f"invalid --query_gate={args.query_gate!r}")
        if explicit == "score_gate":
            return explicit, "score"
        if explicit != "score_gate" and legacy_gate == "score":
            raise ValueError(
                "--query_gate=score conflicts with "
                f"--retrieval_mode={explicit}; use --retrieval_mode=score_gate"
            )
        return explicit, "off"

    if legacy_gate not in {"off", "score"}:
        raise ValueError(f"invalid --query_gate={args.query_gate!r}")
    if legacy_gate == "score":
        if not has_query_parquet:
            raise ValueError("--query_gate=score requires --query_parquet")
        return "score_gate", "score"
    if has_query_parquet:
        return "rewrite", "off"
    return "orig", "off"


def _serialize_hits(hits: list[RetrievalHit]) -> list[dict[str, Any]]:
    """Serialize retrieval hits with forward-compatible score aliases."""
    out: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits):
        score = float(hit.score)
        out.append(
            {
                "rank": int(rank),
                "entry_id": hit.entry_id,
                "type": hit.type,
                "domain": hit.domain,
                "principle": hit.principle,
                "cosine_score": score,
                "bi_score": score,
            }
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inject HRLib abstractions into a prompt parquet (Stage 0 v1)."
    )
    p.add_argument("--library", required=True, type=Path, help="library_v1_* directory")
    # 'in' is a Python keyword, so we use the long option only and store on 'input_path'.
    p.add_argument(
        "--in",
        dest="input_path",
        required=True,
        type=Path,
        help="input prompt parquet (e.g. MATH-500/test.parquet)",
    )
    p.add_argument(
        "--out",
        dest="output_path",
        required=True,
        type=Path,
        help="output parquet path; a sibling injection_meta.json is also written",
    )
    p.add_argument(
        "--query_parquet",
        type=Path,
        default=None,
        help=(
            "optional parquet whose prompt role content is used only for retrieval "
            "query text; injection still edits the --in parquet prompts"
        ),
    )
    p.add_argument(
        "--retrieval_mode",
        choices=["orig", "rewrite", "score_gate"],
        default=None,
        help=(
            "explicit retrieval mode. 'orig' uses original query text, 'rewrite' "
            "uses --query_parquet query text, and 'score_gate' compares original "
            "vs rewrite by gate score. If omitted, mode is inferred from legacy "
            "--query_gate/--query_parquet behavior."
        ),
    )
    p.add_argument("--top_k", type=int, default=6)
    p.add_argument(
        "--query_from",
        default="user",
        help="role whose content becomes the query (default: user)",
    )
    p.add_argument(
        "--query_recipe",
        default="[{subject}] {user_text}",
        help=(
            "query composition recipe for retrieval (default: '[{subject}] {user_text}'). "
            "Rows without a mapped subject automatically fall back to '{user_text}'."
        ),
    )
    p.add_argument(
        "--subject_field",
        default="subject",
        help=(
            "row field used to read subject for query prefixing; supports dotted paths "
            "like 'extra_info.subject' (default: subject)."
        ),
    )
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only process the first N rows (default: all)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="allow overwriting an existing output parquet",
    )
    p.add_argument(
        "--preview_chars",
        type=int,
        default=1200,
        help="characters of the first row's rendered prefix to print (0 to skip)",
    )
    p.add_argument(
        "--template",
        choices=sorted(RENDERERS.keys()),
        default="v1",
        help=(
            "system-prompt rendering template: "
            "'v1' (flat bullets, default, matches existing test_hrlib_v1.parquet) or "
            "'std_rag' (T1+T2: <reference_notes> wrapper + Strategies/Cautions split + "
            "permission-to-ignore)."
        ),
    )
    p.add_argument(
        "--dump_scores",
        action="store_true",
        help=(
            "write a sibling '<out_stem>_scores.jsonl' sidecar with per-problem "
            "full cosine score vectors and top-k hit details for diagnostics"
        ),
    )
    p.add_argument(
        "--query_gate",
        choices=["off", "score"],
        default="off",
        help=(
            "legacy query selection policy. 'off' means no score gate, 'score' "
            "means score gate. Prefer using --retrieval_mode for new runs."
        ),
    )
    p.add_argument(
        "--gate_metric",
        choices=["top1"],
        default="top1",
        help="scalar metric used by --query_gate=score (default: top1 cosine)",
    )
    p.add_argument(
        "--gate_margin",
        type=float,
        default=0.0,
        help=(
            "minimum rewritten-minus-original score improvement required to pick "
            "rewritten query when --query_gate=score"
        ),
    )
    p.add_argument(
        "--gate_tie_policy",
        choices=["prefer_original", "prefer_rewrite"],
        default="prefer_original",
        help="tie policy for --query_gate=score when scores are equal at margin",
    )
    return p.parse_args(argv)


def _write_meta(
    meta_path: Path,
    retriever: FlatRetriever,
    args: argparse.Namespace,
    retrieval_mode: str,
    base_mode: str,
    normalized_query_gate: str,
    injected_rows: int,
    total_rows: int,
    wall_sec: float,
    subject_counts: dict[str, Any],
    gate_stats: dict[str, Any],
) -> None:
    meta = {
        "library": str(args.library),
        "n_entries": len(retriever.entries),
        "model": retriever.model_name,
        "embedder": retriever.model_name,
        "retriever": "flat_cosine",
        "dim": retriever.expected_dim,
        "device": retriever.device,
        "top_k": args.top_k,
        "query_from": args.query_from,
        "subject_field": args.subject_field,
        "subject_map": dict(_MATH_SUBJECT_TO_CANON),
        "template": args.template,
        "query_recipe": args.query_recipe,
        "query_parquet": str(args.query_parquet) if args.query_parquet is not None else "",
        "retrieval_mode": retrieval_mode,
        "retrieval_mode_base": base_mode,
        "query_gate": normalized_query_gate,
        "query_gate_legacy_input": args.query_gate,
        "gate_metric": args.gate_metric,
        "gate_margin": float(args.gate_margin),
        "gate_tie_policy": args.gate_tie_policy,
        "composed_text_recipe": retriever.meta.get("composed_text_recipe"),
        "input_parquet": str(args.input_path),
        "output_parquet": str(args.output_path),
        "subject_counts": subject_counts,
        "gate_stats": gate_stats,
        "injected_rows": injected_rows,
        "total_rows": total_rows,
        "wall_sec": round(float(wall_sec), 3),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    retrieval_mode, normalized_query_gate = _normalize_retrieval_mode(args)
    base_mode = retrieval_mode
    effective_query_gate = "score" if base_mode == "score_gate" else "off"

    if not args.input_path.exists():
        raise FileNotFoundError(f"--in not found: {args.input_path}")
    if args.query_parquet is not None and not args.query_parquet.exists():
        raise FileNotFoundError(f"--query_parquet not found: {args.query_parquet}")
    if base_mode in {"rewrite", "score_gate"} and args.query_parquet is None:
        raise ValueError(f"--retrieval_mode={retrieval_mode} requires --query_parquet")
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"--out already exists: {args.output_path} (pass --overwrite to replace)"
        )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        renderer = RENDERERS[args.template]
    except KeyError as exc:
        raise ValueError(
            f"unknown --template {args.template!r}; valid: {sorted(RENDERERS)}"
        ) from exc

    print(f"[inject] library : {args.library}")
    print(f"[inject] input   : {args.input_path}")
    if args.query_parquet is not None:
        print(f"[inject] query_parquet: {args.query_parquet}")
    print(f"[inject] output  : {args.output_path}")
    print(f"[inject] top_k   : {args.top_k}")
    print(f"[inject] query_from: {args.query_from}")
    print(f"[inject] query_recipe: {args.query_recipe!r}")
    print(f"[inject] subject_field: {args.subject_field}")
    print(f"[inject] device  : {args.device}")
    print(f"[inject] template: {args.template}")
    print(f"[inject] dump_scores: {args.dump_scores}")
    print(f"[inject] retrieval_mode: {retrieval_mode}")
    print(
        "[inject] query_gate (legacy/effective): "
        f"{args.query_gate} -> {normalized_query_gate} (base={effective_query_gate})"
    )
    print(f"[inject] gate_metric: {args.gate_metric}")
    print(f"[inject] gate_margin: {args.gate_margin}")
    print(f"[inject] gate_tie_policy: {args.gate_tie_policy}")
    if base_mode == "orig" and args.query_parquet is not None:
        print("[inject] note: --query_parquet is ignored in retrieval_mode=orig")

    retriever = FlatRetriever(args.library, device=args.device, lazy=False)
    retriever.query_recipe = args.query_recipe
    desc = retriever.describe()
    print(
        f"[inject] retriever ready: n_entries={desc['n_entries']} "
        f"model={desc['model']} dim={desc['dim']} device={desc['device']} "
        f"query_recipe={desc['query_recipe']!r}"
    )

    df = pd.read_parquet(args.input_path)
    query_df: pd.DataFrame | None = None
    if args.query_parquet is not None:
        query_df = pd.read_parquet(args.query_parquet)
    total_rows = len(df)
    if "prompt" not in df.columns:
        raise KeyError(f"input parquet has no 'prompt' column: {list(df.columns)}")
    if query_df is not None and "prompt" not in query_df.columns:
        raise KeyError(
            f"query parquet has no 'prompt' column: {list(query_df.columns)}"
        )
    if args.limit is not None and args.limit >= 0:
        df = df.iloc[: args.limit].copy()
        if query_df is not None:
            query_df = query_df.iloc[: args.limit].copy()
    else:
        df = df.copy()
        if query_df is not None:
            query_df = query_df.copy()
    if query_df is not None and len(query_df) != len(df):
        raise ValueError(
            "query parquet row count must match input parquet row count "
            f"(after --limit). got input={len(df)} query={len(query_df)}"
        )

    t0 = time.time()
    first_prefix: str | None = None
    injected = 0
    hit_k_counts: list[int] = []
    new_prompts: list[list[dict[str, Any]]] = []
    mapped_subjects = 0
    unmapped_subjects = 0
    rows_with_subject = 0
    canonical_subject_counts: Counter[str] = Counter()
    raw_subject_counts: Counter[str] = Counter()
    gate_rows = 0
    gate_selected_original_rows = 0
    gate_selected_rewrite_rows = 0
    scores_sidecar_path: Path | None = None
    if args.dump_scores:
        scores_sidecar_path = args.output_path.with_name(args.output_path.stem + "_scores.jsonl")

    score_writer_context = (
        scores_sidecar_path.open("w", encoding="utf-8")
        if scores_sidecar_path is not None
        else nullcontext(None)
    )
    with score_writer_context as scores_f:
        for pos, (i, row) in enumerate(df.iterrows()):
            prompt = _prompt_to_list(row["prompt"])
            query_original = _find_query_text(prompt, args.query_from)
            query_retrieval = query_original
            if query_df is not None:
                query_prompt = _prompt_to_list(query_df.iloc[pos]["prompt"])
                query_retrieval = _find_query_text(query_prompt, args.query_from)
            raw_subject = _read_subject(row, args.subject_field)
            if raw_subject:
                rows_with_subject += 1
                raw_subject_counts[raw_subject] += 1
            canon_subject = _canonicalize_subject(raw_subject)

            if canon_subject:
                mapped_subjects += 1
                canonical_subject_counts[canon_subject] += 1
            elif raw_subject:
                unmapped_subjects += 1

            if base_mode == "orig":
                chosen_query_source = "original"
                chosen_query_text = query_original
            elif base_mode == "rewrite":
                chosen_query_source = "rewrite"
                chosen_query_text = query_retrieval
            else:
                chosen_query_source = "original"
                chosen_query_text = query_original
            gate_score_original: float | None = None
            gate_score_rewrite: float | None = None
            hits_original: list[RetrievalHit] | None = None
            hits_rewrite: list[RetrievalHit] | None = None
            scores_original: np.ndarray | None = None
            scores_rewrite: np.ndarray | None = None

            if base_mode == "score_gate":
                gate_rows += 1
                if query_original.strip() == query_retrieval.strip():
                    hits_original, scores_original = _retrieve_hits_and_scores(
                        retriever,
                        query_text=query_original,
                        top_k=args.top_k,
                        canon_subject=canon_subject,
                        dump_scores=args.dump_scores,
                    )
                    hits_rewrite, scores_rewrite = hits_original, scores_original
                else:
                    hits_original, scores_original = _retrieve_hits_and_scores(
                        retriever,
                        query_text=query_original,
                        top_k=args.top_k,
                        canon_subject=canon_subject,
                        dump_scores=args.dump_scores,
                    )
                    hits_rewrite, scores_rewrite = _retrieve_hits_and_scores(
                        retriever,
                        query_text=query_retrieval,
                        top_k=args.top_k,
                        canon_subject=canon_subject,
                        dump_scores=args.dump_scores,
                    )

                gate_score_original = _gate_metric_score(hits_original, args.gate_metric)
                gate_score_rewrite = _gate_metric_score(hits_rewrite, args.gate_metric)
                chosen_query_source = _pick_query_source_by_score(
                    gate_score_original,
                    gate_score_rewrite,
                    margin=float(args.gate_margin),
                    tie_policy=args.gate_tie_policy,
                )
                if chosen_query_source == "rewrite":
                    gate_selected_rewrite_rows += 1
                    chosen_query_text = query_retrieval
                    hits = hits_rewrite
                    scores = scores_rewrite
                else:
                    gate_selected_original_rows += 1
                    chosen_query_text = query_original
                    hits = hits_original
                    scores = scores_original
            else:
                hits, scores = _retrieve_hits_and_scores(
                    retriever,
                    query_text=chosen_query_text,
                    top_k=args.top_k,
                    canon_subject=canon_subject,
                    dump_scores=args.dump_scores,
                )
                if chosen_query_source == "rewrite":
                    hits_rewrite = hits
                    scores_rewrite = scores
                else:
                    hits_original = hits
                    scores_original = scores
                if chosen_query_source == "rewrite":
                    gate_selected_rewrite_rows += 1
                else:
                    gate_selected_original_rows += 1

            hit_k_counts.append(len(hits))
            prefix = renderer(hits)
            new_prompt = _prepend_to_system(prompt, prefix)
            new_prompts.append(new_prompt)
            if prefix:
                injected += 1
            if first_prefix is None:
                first_prefix = prefix

            if scores_f is not None and scores is not None:
                top_k_hits = _serialize_hits(hits)
                candidate_hits: dict[str, list[dict[str, Any]]] = {}
                if hits_original is not None:
                    candidate_hits["original"] = _serialize_hits(hits_original)
                if hits_rewrite is not None:
                    candidate_hits["rewrite"] = _serialize_hits(hits_rewrite)
                if not candidate_hits:
                    candidate_hits[chosen_query_source] = top_k_hits
                record = {
                    "problem_idx": int(i),
                    "problem_key": _problem_key_from_row(row, query_original),
                    "retrieval_mode": retrieval_mode,
                    "retrieval_mode_base": base_mode,
                    "retriever": "flat_cosine",
                    "embedder": retriever.model_name,
                    "selected_query_source": chosen_query_source,
                    "selected_query_text": chosen_query_text,
                    "candidate_hits": candidate_hits,
                    "query_text": chosen_query_text,
                    "query_text_original": query_original,
                    "query_text_retrieval": query_retrieval,
                    "query_rewrite_found": bool(
                        query_df is not None and query_original.strip() != query_retrieval.strip()
                    ),
                    "chosen_query_source": chosen_query_source,
                    "query_gate": effective_query_gate,
                    "gate_metric": args.gate_metric,
                    "gate_margin": float(args.gate_margin),
                    "gate_tie_policy": args.gate_tie_policy,
                    "gate_score_original": (
                        float(gate_score_original) if gate_score_original is not None else None
                    ),
                    "gate_score_rewrite": (
                        float(gate_score_rewrite) if gate_score_rewrite is not None else None
                    ),
                    "subject": raw_subject,
                    "canonical_subject": canon_subject,
                    "top_k": int(args.top_k),
                    "top_k_hits": top_k_hits,
                    "all_scores": [float(x) for x in np.asarray(scores, dtype=np.float32)],
                }
                scores_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Assigning a list of lists-of-dicts back to a pandas column is safe via a
    # plain Python list; pyarrow will serialize each row's list<struct> correctly.
    df["prompt"] = new_prompts
    df.to_parquet(args.output_path, index=False)
    wall = time.time() - t0

    subject_counts = {
        "rows_with_subject": rows_with_subject,
        "mapped": mapped_subjects,
        "unmapped": unmapped_subjects,
        "canonical": dict(sorted(canonical_subject_counts.items())),
        "raw": dict(raw_subject_counts.most_common()),
    }
    gate_stats = {
        "retrieval_mode": retrieval_mode,
        "retrieval_mode_base": base_mode,
        "query_gate_effective": effective_query_gate,
        "gate_rows": int(gate_rows),
        "selected_original_rows": int(gate_selected_original_rows),
        "selected_rewrite_rows": int(gate_selected_rewrite_rows),
    }

    meta_path = args.output_path.with_name(args.output_path.stem + "_injection_meta.json")
    _write_meta(
        meta_path,
        retriever,
        args,
        retrieval_mode,
        base_mode,
        effective_query_gate,
        injected,
        len(df),
        wall,
        subject_counts,
        gate_stats,
    )

    print()
    print(f"[inject] processed rows     : {len(df)} (of {total_rows} in input)")
    print(f"[inject] rows with injection: {injected}")
    if hit_k_counts:
        mean_k = sum(hit_k_counts) / len(hit_k_counts)
        print(f"[inject] avg k injected     : {mean_k:.2f} (target {args.top_k})")
    print(
        f"[inject] selected query rows : original={gate_selected_original_rows} "
        f"rewrite={gate_selected_rewrite_rows}"
    )
    print(f"[inject] wall time          : {wall:.2f}s")
    print(f"[inject] wrote parquet      : {args.output_path}")
    print(f"[inject] wrote meta         : {meta_path}")
    if scores_sidecar_path is not None:
        print(f"[inject] wrote scores       : {scores_sidecar_path}")

    if args.preview_chars > 0 and first_prefix:
        print()
        print("========== first row rendered prefix (preview) ==========")
        snippet = first_prefix[: args.preview_chars]
        print(snippet)
        if len(first_prefix) > args.preview_chars:
            print(f"... ({len(first_prefix) - args.preview_chars} more chars truncated)")
        print("=========================================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
