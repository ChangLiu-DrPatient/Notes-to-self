#!/usr/bin/env python3
"""Aggregate `raw_abstractions.jsonl` into a deduplicated library (v1, rapidfuzz text-layer).

Typical usage (from the repo root `verl/`):

    python3 examples/hrlib/20_aggregate.py \
        --raw    /path/to/raw_abstractions.jsonl \
        --out_dir /path/to/library/v1_flat/

Outputs (written under --out_dir):
    library.jsonl  — one LibraryEntry per line, sorted by (-hit_count, type, name).
    library.md     — human-readable top-N-per-type preview + run metadata.
    meta.json      — run configuration + filter / cluster counts.
    dropped.jsonl  — filtered-out raw entries with their drop reason.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

try:
    from examples.hrlib.aggregate import (
        aggregate,
        format_top_n_text,
        load_raw_abstractions,
        write_dropped_jsonl,
        write_embeddings_npy,
        write_library_jsonl,
        write_library_md,
        write_meta_json,
    )
except ImportError:
    from aggregate import (
        aggregate,
        format_top_n_text,
        load_raw_abstractions,
        write_dropped_jsonl,
        write_embeddings_npy,
        write_library_jsonl,
        write_library_md,
        write_meta_json,
    )


def _parse_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y", "t"}:
        return True
    if v in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate raw abstractions into a deduplicated library via rapidfuzz.",
    )
    parser.add_argument("--raw", required=True, help="Path to raw_abstractions.jsonl from 10_extract.py.")
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory (will contain library.jsonl / library.md / meta.json / dropped.jsonl).",
    )
    parser.add_argument(
        "--method",
        choices=("semantic", "text", "two_tier"),
        default="semantic",
        help=(
            "Clustering backend. 'semantic' (default) = sentence-transformers "
            "embeddings + cosine union-find. 'text' = rapidfuzz token_set_ratio "
            "(no GPU/embedder needed). 'two_tier' = rapidfuzz first, then semantic "
            "merge over medoids."
        ),
    )
    parser.add_argument(
        "--text_ratio",
        type=float,
        default=80.0,
        help="rapidfuzz.fuzz.token_set_ratio threshold (0-100). Used by --method text and two_tier. Default: 80.",
    )
    parser.add_argument(
        "--semantic_ratio",
        type=float,
        default=0.85,
        help="Cosine similarity threshold (0-1). Used by --method semantic and two_tier. Default: 0.85.",
    )
    parser.add_argument(
        "--embedder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformers model name. Default: all-MiniLM-L6-v2 (~80 MB, 384-d).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Embedding device: auto / cpu / cuda / cuda:N. Default: auto (cuda when available).",
    )
    parser.add_argument(
        "--embed_include_domain",
        type=_parse_bool,
        default=True,
        help="Prepend [{canonical_domain or 'unknown'}] to encoder input. Default: true.",
    )
    parser.add_argument(
        "--embed_batch_size",
        type=int,
        default=256,
        help="sentence-transformers encode batch size. Default: 256.",
    )
    parser.add_argument(
        "--write_embeddings",
        type=_parse_bool,
        default=True,
        help="Persist embeddings.npy + embeddings_meta.json next to library.jsonl. Default: true.",
    )
    parser.add_argument(
        "--filter",
        dest="filter_principles",
        type=_parse_bool,
        default=False,
        help=(
            "Apply length and leakage filters to principles. Default: false. "
            "When false, only structural type_unknown entries are dropped; "
            "every well-typed abstraction reaches the clustering stage."
        ),
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=15,
        help="Minimum normalized principle length (only used when --filter true). Default: 15.",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=240,
        help="Maximum normalized principle length (only used when --filter true). Default: 240.",
    )
    parser.add_argument(
        "--per_type",
        type=_parse_bool,
        default=True,
        help="Keep strategy / caution buckets separate during dedup. Default: true.",
    )
    parser.add_argument(
        "--top_n_preview",
        type=int,
        default=50,
        help="Rows per type in library.md preview tables. Default: 50.",
    )
    parser.add_argument(
        "--top_n_print",
        type=int,
        default=5,
        help="Top-N strategies + top-N cautions to print to stdout at the end. Default: 5.",
    )
    parser.add_argument(
        "--keep_cluster_members",
        type=_parse_bool,
        default=True,
        help="Include raw cluster_members in library.jsonl for auditing. Default: true.",
    )
    parser.add_argument(
        "--normalize_domain",
        dest="normalize_domains",
        type=_parse_bool,
        default=True,
        help=(
            "Map raw `domain` strings to canonical buckets via aggregate.normalize_domain "
            "(case fold + strip subdomain tail + alias table). Default: true. "
            "Raw labels are preserved per-cluster in `domains_seen_raw` and `cluster_members[*].domain`."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow clobbering existing files in --out_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lib_path = out_dir / "library.jsonl"
    md_path = out_dir / "library.md"
    meta_path = out_dir / "meta.json"
    dropped_path = out_dir / "dropped.jsonl"
    emb_path = out_dir / "embeddings.npy"
    emb_meta_path = out_dir / "embeddings_meta.json"

    if not args.overwrite:
        candidates = (lib_path, md_path, meta_path, dropped_path, emb_path, emb_meta_path)
        existing = [p for p in candidates if p.exists()]
        if existing:
            names = ", ".join(str(p) for p in existing)
            raise SystemExit(f"[error] refusing to clobber existing files: {names}\nPass --overwrite to proceed.")

    raw_path = Path(args.raw)
    raw_items = load_raw_abstractions(raw_path)
    print(f"[aggregate] loaded {len(raw_items)} raw abstractions from {raw_path}")

    entries, stats, dropped, entry_embeddings = aggregate(
        raw_items,
        method=args.method,
        text_ratio=args.text_ratio,
        semantic_ratio=args.semantic_ratio,
        embedder=args.embedder,
        device=args.device,
        embed_include_domain=args.embed_include_domain,
        embed_batch_size=args.embed_batch_size,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        per_type=args.per_type,
        keep_cluster_members=args.keep_cluster_members,
        filter_principles=args.filter_principles,
        normalize_domains=args.normalize_domains,
    )

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(raw_path),
        "filter_principles": args.filter_principles,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "per_type": args.per_type,
        "keep_cluster_members": args.keep_cluster_members,
        "top_n_preview": args.top_n_preview,
        "leakage_rules_active": ["digits3", "explicit_answer"] if args.filter_principles else [],
        **stats,
    }

    write_library_jsonl(entries, lib_path)
    write_dropped_jsonl(dropped, dropped_path)
    write_meta_json(meta, meta_path)
    write_library_md(entries, meta, md_path, top_n_per_type=args.top_n_preview)

    embeddings_written = False
    if args.write_embeddings and entry_embeddings is not None and entry_embeddings.size > 0:
        write_embeddings_npy(
            out_dir,
            entry_embeddings,
            model=args.embedder,
            embed_include_domain=args.embed_include_domain,
            method=args.method,
            device=stats.get("device"),
            n_entries=len(entries),
        )
        embeddings_written = True

    print(f"[aggregate] wrote library.jsonl     -> {lib_path}  ({len(entries)} entries)")
    print(f"[aggregate] wrote library.md        -> {md_path}   (top {args.top_n_preview} per type)")
    print(f"[aggregate] wrote meta.json         -> {meta_path}")
    print(f"[aggregate] wrote dropped.jsonl     -> {dropped_path}  ({len(dropped)} dropped)")
    if embeddings_written:
        print(
            f"[aggregate] wrote embeddings.npy    -> {emb_path}  "
            f"(shape={entry_embeddings.shape}, dim={stats.get('embed_dim')}, model={args.embedder})"
        )
        print(f"[aggregate] wrote embeddings_meta.json -> {emb_meta_path}")
    elif args.write_embeddings:
        print(
            f"[aggregate] embeddings.npy skipped (method={args.method}; "
            f"text mode does not produce embeddings)"
        )
    print(
        "[aggregate] summary: "
        f"method={args.method}, "
        f"filter_principles={args.filter_principles}, "
        f"n_raw={stats['n_raw']}, n_after_filter={stats['n_after_filter']}, n_final={stats['n_final']}, "
        f"per_type={stats['per_type_counts']}, "
        f"avg_cluster={stats['avg_cluster_size']:.2f}, median_cluster={stats['median_cluster_size']:.1f}"
    )
    if args.method != "text":
        print(
            "[aggregate] embedder: "
            f"name={args.embedder}, device={stats.get('device')}, dim={stats.get('embed_dim')}, "
            f"include_domain={args.embed_include_domain}, semantic_ratio={args.semantic_ratio}"
        )
    if args.method in {"text", "two_tier"}:
        print(f"[aggregate] text_ratio={args.text_ratio}")
    print(
        "[aggregate] domains: "
        f"normalize_domains={stats['normalize_domains']}, "
        f"raw_distinct={stats['n_domains_raw_distinct']}, "
        f"canonical_distinct={stats['n_domains_canonical_distinct']}"
    )
    if stats["dropped"]:
        print(f"[aggregate] dropped breakdown: {stats['dropped']}")

    if args.top_n_print > 0:
        print(format_top_n_text(entries, top_n=args.top_n_print))


if __name__ == "__main__":
    main()
