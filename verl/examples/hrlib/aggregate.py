#!/usr/bin/env python3
"""Aggregate raw abstractions into a deduplicated library (semantic-first, v1).

Pipeline (see implementation_plan_stage0_3.md §5.2.3):
  normalize principle + canonical domain -> optional length / leakage filter ->
  bucket by type -> cluster (semantic by default; rapidfuzz "text" or two-tier
  also available) -> medoid per cluster -> aggregate hit_count / source ids /
  domains / difficulties -> sort by hit_count.

Produces:
  library.jsonl       (canonical, one LibraryEntry per line)
  library.md          (human-readable top-N per type)
  meta.json           (run config + counts + cluster stats + embedder metadata)
  dropped.jsonl       (auditable filter reasons; empty when --filter false)
  embeddings.npy      (float32, L2-normalized, ordered to match library.jsonl)
  embeddings_meta.json (model/revision/dim/composed-text recipe sidecar)
"""

from __future__ import annotations

import functools
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from rapidfuzz import fuzz, process

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class LibraryEntry:
    entry_id: str
    type: str
    name: str
    principle: str
    when_to_apply: str
    domain: str
    hit_count: int
    num_unique_problems: int
    source_problem_ids: list[str] = field(default_factory=list)
    source_row_indices: list[int] = field(default_factory=list)
    source_difficulties: dict[str, int] = field(default_factory=dict)
    domains_seen: dict[str, int] = field(default_factory=dict)
    domains_seen_raw: dict[str, int] = field(default_factory=dict)
    cluster_members: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Normalization and filters
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"[\s\.,;:!\?\-\"'`]+$")
_LEAKAGE_DIGITS3 = re.compile(r"\d{3,}")
_LEAKAGE_ANSWER = re.compile(r"\\boxed|answer\s+is|=\s*-?\d")

# Unicode dash variants we need to fold into a plain "-" before splitting on
# en/em-dash subdomain separators (e.g., "Number theory – modular arithmetic").
_UNICODE_DASHES = {
    "\u2010": "-",  # HYPHEN
    "\u2011": "-",  # NON-BREAKING HYPHEN
    "\u2012": "-",  # FIGURE DASH
    "\u2013": "-",  # EN DASH
    "\u2014": "-",  # EM DASH
    "\u2015": "-",  # HORIZONTAL BAR
}

# Order matters: more specific aliases first. Each tuple is (substring, canonical).
# We match against the casefolded, dash-and-joiner-stripped string, so case is
# irrelevant. The first matching needle wins.
_DOMAIN_ALIASES: list[tuple[str, str]] = [
    ("linear algebra", "linear algebra"),
    ("matrix", "linear algebra"),
    ("matrice", "linear algebra"),
    ("vector", "linear algebra"),
    ("number theor", "number theory"),
    ("modular", "number theory"),
    ("divisor", "number theory"),
    ("number bases", "number theory"),
    ("base conversion", "number theory"),
    ("diophant", "number theory"),
    ("complex number", "complex numbers"),
    ("trigonom", "trigonometry"),
    ("geometr", "geometry"),
    ("combinator", "combinatorics"),
    ("probab", "probability"),
    ("calculus", "calculus"),
    ("inequal", "inequalities"),
    ("optim", "optimization"),
    ("statistic", "statistics"),
    ("set theory", "set theory"),
    ("graph theory", "graph theory"),
    ("logic", "logic"),
    ("function", "functions"),
    ("arithmetic", "arithmetic"),
    ("algebra", "algebra"),
]


def normalize_principle(text: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation."""
    t = str(text).strip().lower()
    t = _WS.sub(" ", t)
    t = _TRAILING_PUNCT.sub("", t)
    return t


def normalize_domain(text: str) -> str:
    """Map a raw `domain` string to a canonical bucket.

    Pipeline:
      1. Replace unicode dashes with ``-``.
      2. Split off any subdomain qualifier (` - `, `:`, `/`, `,`); take the head.
      3. Lowercase, replace ``_`` with space, collapse whitespace.
      4. Match against a hand-tuned alias table (``_DOMAIN_ALIASES``); the first
         matching needle wins.
      5. If nothing matches, return the cleaned-but-unmapped string so the long
         tail stays inspectable rather than being silently merged into "other".

    Empty / falsy input returns ``""``. This function is intentionally simple;
    when the alias table outgrows ~50 entries we should move it to a YAML file.
    """
    raw = str(text or "").strip()
    if not raw:
        return ""
    s = raw
    for ch, repl in _UNICODE_DASHES.items():
        if ch in s:
            s = s.replace(ch, repl)
    for sep in (" - ", ":", "/", ","):
        if sep in s:
            s = s.split(sep, 1)[0]
            break
    s = s.lower().replace("_", " ").strip()
    s = _WS.sub(" ", s)
    if not s:
        return ""
    for needle, canon in _DOMAIN_ALIASES:
        if needle in s:
            return canon
    return s


def check_filter(principle_norm: str, *, min_chars: int, max_chars: int) -> str | None:
    """Return a drop reason string if filtered, else None."""
    n = len(principle_norm)
    if n < min_chars:
        return "too_short"
    if n > max_chars:
        return "too_long"
    if _LEAKAGE_DIGITS3.search(principle_norm):
        return "leakage_digits3"
    if _LEAKAGE_ANSWER.search(principle_norm):
        return "leakage_answer"
    return None


# ---------------------------------------------------------------------------
# Union-find (disjoint-set)
# ---------------------------------------------------------------------------


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

# Recipe used both at clustering time and recorded into embeddings_meta.json so
# downstream retrieval can re-encode user queries with the exact same envelope.
COMPOSED_TEXT_RECIPE = (
    "[{canonical_domain or 'unknown'}] {principle}"
    "[ [WHEN] {when_to_apply}]   # the [WHEN] block is omitted when when_to_apply is empty"
)

# Recipe the retriever uses to compose the encoder input for a USER query at
# retrieval time. v1 keeps it minimal (naked problem text). When Stage 1's
# hierarchical retriever needs a domain signal, flip this to a templated string
# (e.g. "[{subject}] {user_text}") and the retriever will assert compatibility
# against whatever was baked into a given library.
QUERY_RECIPE_V1 = "{user_text}"


def _compose_embed_text(item: dict[str, Any], *, include_domain: bool) -> str:
    """Build the Option B composed string for a filtered abstraction item.

    Format:  ``[{canonical_domain or 'unknown'}] {principle} [WHEN] {when_to_apply}``

    The ``[WHEN]`` block is appended only when ``when_to_apply`` is non-empty so
    the encoder doesn't learn a "no trigger" sentinel. Whitespace is collapsed
    but case is preserved (sentence encoders are case-aware).
    """
    principle = " ".join(str(item.get("principle", "")).split()).strip()
    when_to_apply = " ".join(str(item.get("when_to_apply", "")).split()).strip()
    parts: list[str] = []
    if include_domain:
        canon = str(item.get("_domain_canon", "")).strip() or "unknown"
        parts.append(f"[{canon}]")
    parts.append(principle)
    if when_to_apply:
        parts.append("[WHEN]")
        parts.append(when_to_apply)
    return " ".join(parts).strip()


def _resolve_device(request: str) -> str:
    """Resolve a device string. ``auto`` / empty -> cuda when available, else cpu."""
    req = (request or "auto").strip().lower()
    if req in {"cpu"}:
        return "cpu"
    try:
        import torch  # noqa: WPS433 (local import to keep startup light)
    except ImportError:
        return "cpu"
    if req in {"auto", ""}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda"):
        return req if torch.cuda.is_available() else "cpu"
    return req


@functools.lru_cache(maxsize=4)
def _load_embedder(name: str, device: str):
    """Load + cache a SentenceTransformer. Lazy-imports so CPU-only or
    rapidfuzz-only callers don't pay the import cost."""
    from sentence_transformers import SentenceTransformer  # noqa: WPS433

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model = SentenceTransformer(name, device=device)
    return model


def _embed_texts(
    texts: list[str],
    *,
    embedder_name: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Encode ``texts`` and return a float32 L2-normalized array of shape (N, D)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = _load_embedder(embedder_name, device)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    arr = np.asarray(vecs, dtype=np.float32)
    return np.ascontiguousarray(arr)


def _semantic_pairs_above(
    embeddings: np.ndarray,
    threshold: float,
    chunk: int = 2000,
) -> list[tuple[int, int]]:
    """Return all (i, j) with i < j and cos(i, j) >= threshold.

    Uses chunked matmul so peak memory stays at O(chunk * N * 4 bytes) instead
    of the full N^2 cosine matrix.
    """
    n = int(embeddings.shape[0])
    if n < 2:
        return []
    pairs: list[tuple[int, int]] = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sub = embeddings[start:end] @ embeddings.T  # (chunk, n)
        rows, cols = np.where(sub >= threshold)
        for r, c in zip(rows.tolist(), cols.tolist()):
            i = start + int(r)
            j = int(c)
            if i < j:
                pairs.append((i, j))
    return pairs


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _medoid_from_sim(members: list[int], sim_local: np.ndarray) -> int:
    """Argmax of the row-sums on a (k x k) intra-cluster similarity matrix."""
    if len(members) <= 1:
        return members[0]
    row_sums = sim_local.sum(axis=1)
    return members[int(np.argmax(row_sums))]


def _text_cluster(norms: list[str], threshold: float) -> list[tuple[list[int], int]]:
    """rapidfuzz token_set_ratio clustering (legacy `method=text` path)."""
    n = len(norms)
    if n == 0:
        return []
    if n == 1:
        return [([0], 0)]

    sim = process.cdist(
        norms,
        norms,
        scorer=fuzz.token_set_ratio,
        workers=-1,
        dtype=np.uint8,
    )

    dsu = DSU(n)
    thr = int(round(threshold))
    rows, cols = np.where(sim >= thr)
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i < j:
            dsu.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(dsu.find(i), []).append(i)

    clusters: list[tuple[list[int], int]] = []
    for members in groups.values():
        sub = sim[np.ix_(members, members)]
        clusters.append((members, _medoid_from_sim(members, sub)))
    return clusters


def _semantic_cluster(
    items: list[dict[str, Any]],
    *,
    threshold: float,
    embedder_name: str,
    device: str,
    embed_include_domain: bool,
    embed_batch_size: int,
) -> tuple[list[tuple[list[int], int]], np.ndarray]:
    """Embed-then-union-find clustering.

    Returns ``(clusters, embeddings)`` where ``embeddings`` is shape
    ``(len(items), D)``, float32, L2-normalized, ordered to match ``items``.
    """
    n = len(items)
    if n == 0:
        return [], np.zeros((0, 0), dtype=np.float32)

    texts = [_compose_embed_text(it, include_domain=embed_include_domain) for it in items]
    emb = _embed_texts(
        texts,
        embedder_name=embedder_name,
        device=device,
        batch_size=embed_batch_size,
    )
    if n == 1:
        return [([0], 0)], emb

    pairs = _semantic_pairs_above(emb, float(threshold))
    dsu = DSU(n)
    for i, j in pairs:
        dsu.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(dsu.find(i), []).append(i)

    clusters: list[tuple[list[int], int]] = []
    for members in groups.values():
        if len(members) == 1:
            clusters.append((members, members[0]))
            continue
        sub_emb = emb[members]
        sub_sim = sub_emb @ sub_emb.T
        clusters.append((members, _medoid_from_sim(members, sub_sim)))
    return clusters, emb


def _two_tier_cluster(
    items: list[dict[str, Any]],
    *,
    text_ratio: float,
    semantic_threshold: float,
    embedder_name: str,
    device: str,
    embed_include_domain: bool,
    embed_batch_size: int,
) -> tuple[list[tuple[list[int], int]], np.ndarray]:
    """rapidfuzz pre-merge, then semantic merge over the resulting medoids.

    Returns ``(clusters, embeddings_per_item)`` so the caller can still pull a
    single embedding per output cluster (the post-merge medoid's row).
    """
    n = len(items)
    if n == 0:
        return [], np.zeros((0, 0), dtype=np.float32)

    # Stage 1: rapidfuzz over normalized principles.
    norms = [str(it.get("_norm", "")) for it in items]
    text_clusters = _text_cluster(norms, threshold=text_ratio)

    # Stage 2: encode every item once (we'll need every row for the final
    # per-cluster embedding lookup; we also reuse the medoid rows for merging).
    texts = [_compose_embed_text(it, include_domain=embed_include_domain) for it in items]
    emb = _embed_texts(
        texts,
        embedder_name=embedder_name,
        device=device,
        batch_size=embed_batch_size,
    )

    if len(text_clusters) <= 1:
        # Single cluster -> nothing to semantic-merge.
        return text_clusters, emb

    medoid_indices = [med for (_, med) in text_clusters]
    medoid_emb = emb[medoid_indices]
    pairs = _semantic_pairs_above(medoid_emb, float(semantic_threshold))

    dsu = DSU(len(text_clusters))
    for i, j in pairs:
        dsu.union(i, j)

    super_groups: dict[int, list[int]] = {}
    for ci in range(len(text_clusters)):
        super_groups.setdefault(dsu.find(ci), []).append(ci)

    merged: list[tuple[list[int], int]] = []
    for sub_cluster_ids in super_groups.values():
        members: list[int] = []
        for ci in sub_cluster_ids:
            members.extend(text_clusters[ci][0])
        if len(members) == 1:
            merged.append((members, members[0]))
            continue
        sub_emb = emb[members]
        sub_sim = sub_emb @ sub_emb.T
        merged.append((members, _medoid_from_sim(members, sub_sim)))
    return merged, emb


def _cluster_bucket(
    items: list[dict[str, Any]],
    *,
    method: str,
    text_ratio: float,
    semantic_ratio: float,
    embedder_name: str,
    device: str,
    embed_include_domain: bool,
    embed_batch_size: int,
) -> tuple[list[tuple[list[int], int]], np.ndarray | None]:
    """Dispatch to the requested clustering backend.

    Returns ``(clusters, embeddings_per_item_or_None)``. ``method='text'`` does
    not produce embeddings; the other two backends return per-item embeddings
    so the caller can pluck out a per-output-cluster medoid embedding.
    """
    if not items:
        return [], None
    if method == "text":
        norms = [str(it.get("_norm", "")) for it in items]
        return _text_cluster(norms, threshold=text_ratio), None
    if method == "semantic":
        return _semantic_cluster(
            items,
            threshold=semantic_ratio,
            embedder_name=embedder_name,
            device=device,
            embed_include_domain=embed_include_domain,
            embed_batch_size=embed_batch_size,
        )
    if method == "two_tier":
        return _two_tier_cluster(
            items,
            text_ratio=text_ratio,
            semantic_threshold=semantic_ratio,
            embedder_name=embedder_name,
            device=device,
            embed_include_domain=embed_include_domain,
            embed_batch_size=embed_batch_size,
        )
    raise ValueError(f"unknown clustering method: {method!r}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _top_item(counter: Counter[str]) -> str:
    """Most common key, alphabetical tiebreaker, empty string on empty counter."""
    if not counter:
        return ""
    top_count = max(counter.values())
    candidates = sorted(k for k, v in counter.items() if v == top_count)
    return candidates[0]


def load_raw_abstractions(path: str | Path) -> list[dict[str, Any]]:
    """Load newline-delimited Abstraction dicts emitted by 10_extract.py."""
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                items.append(obj)
    return items


def aggregate(
    raw_items: list[dict[str, Any]],
    *,
    method: str = "semantic",
    text_ratio: float = 80.0,
    semantic_ratio: float = 0.85,
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "auto",
    embed_include_domain: bool = True,
    embed_batch_size: int = 256,
    min_chars: int = 15,
    max_chars: int = 240,
    per_type: bool = True,
    keep_cluster_members: bool = True,
    filter_principles: bool = False,
    normalize_domains: bool = True,
) -> tuple[list[LibraryEntry], dict[str, Any], list[dict[str, Any]], np.ndarray | None]:
    """Cluster raw abstractions into LibraryEntry objects.

    Parameters
    ----------
    method : {"semantic", "text", "two_tier"}, default "semantic"
        ``semantic``  — sentence-transformers embeddings + cosine union-find at
        ``semantic_ratio``. ``text`` — rapidfuzz token_set_ratio union-find at
        ``text_ratio`` (the v0 behavior; no GPU/embedder needed).
        ``two_tier`` — rapidfuzz first, then semantic merge over the medoids.
    semantic_ratio : float, default 0.88
        Cosine similarity threshold for semantic union-find. Used by
        ``semantic`` and ``two_tier``.
    embedder : str
        sentence-transformers model name. Default: MiniLM-L6-v2 (~80 MB, 384-d).
    device : {"auto", "cpu", "cuda", "cuda:N"}, default "auto"
        Embedding device. ``auto`` falls back to CPU when CUDA isn't available.
    embed_include_domain : bool, default True
        Prepend ``[{canonical_domain or 'unknown'}]`` to the encoder input so
        same-wording-but-different-domain principles drift apart in the space.
    embed_batch_size : int, default 256
        Encoder batch size. 64-256 is the typical sweet spot on a single GPU.
    filter_principles : bool, default False
        When True, drop principles outside ``[min_chars, max_chars]`` and apply
        the leakage regexes. When False (default for v1), only the structural
        ``type_unknown`` filter runs; every well-typed abstraction reaches the
        clustering stage. Use ``meta.json`` to track which mode was used.

    Returns
    -------
    entries : list[LibraryEntry]
        Sorted by (-hit_count, type, name.lower()).
    stats : dict
        Counts plus cluster-size summary. Caller adds run metadata for meta.json.
    dropped : list[dict]
        {"reason": str, "abstraction": dict} for each filtered-out raw item.
    entry_embeddings : np.ndarray | None
        Float32 array of shape ``(len(entries), D)`` aligned to ``entries``
        (one row per output cluster, taken from that cluster's medoid). ``None``
        when ``method='text'`` (no embedder was invoked).
    """
    if method not in {"semantic", "text", "two_tier"}:
        raise ValueError(f"unknown method: {method!r} (choose semantic / text / two_tier)")
    resolved_device = _resolve_device(device)
    n_raw = len(raw_items)

    drop_counts: Counter[str] = Counter()
    dropped: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    raw_domain_counter: Counter[str] = Counter()
    canon_domain_counter: Counter[str] = Counter()

    for item in raw_items:
        raw_type = str(item.get("type", "")).strip().lower()
        raw_domain = str(item.get("domain", "")).strip()
        canon_domain = normalize_domain(raw_domain) if normalize_domains else raw_domain
        if raw_domain:
            raw_domain_counter[raw_domain] += 1
        if canon_domain:
            canon_domain_counter[canon_domain] += 1

        if raw_type not in {"strategy", "caution"}:
            drop_counts["type_unknown"] += 1
            dropped.append({"reason": "type_unknown", "abstraction": item})
            continue
        principle = str(item.get("principle", ""))
        norm = normalize_principle(principle)
        if filter_principles:
            reason = check_filter(norm, min_chars=min_chars, max_chars=max_chars)
            if reason is not None:
                drop_counts[reason] += 1
                dropped.append({"reason": reason, "abstraction": item})
                continue
        filtered.append(
            {
                **item,
                "_norm": norm,
                "_type": raw_type,
                "_domain_canon": canon_domain,
                "_domain_raw": raw_domain,
            }
        )

    n_filtered = len(filtered)

    if per_type:
        buckets: dict[str, list[int]] = {"strategy": [], "caution": []}
        for idx, it in enumerate(filtered):
            buckets.setdefault(it["_type"], []).append(idx)
    else:
        buckets = {"all": list(range(n_filtered))}

    pending: list[tuple[LibraryEntry, np.ndarray | None]] = []
    per_type_counts: Counter[str] = Counter()
    cluster_sizes: list[int] = []
    embed_dim: int | None = None

    for _bucket_key, bucket_indices in buckets.items():
        if not bucket_indices:
            continue
        bucket_items = [filtered[i] for i in bucket_indices]
        clusters, bucket_emb = _cluster_bucket(
            bucket_items,
            method=method,
            text_ratio=text_ratio,
            semantic_ratio=semantic_ratio,
            embedder_name=embedder,
            device=resolved_device,
            embed_include_domain=embed_include_domain,
            embed_batch_size=embed_batch_size,
        )
        if bucket_emb is not None and bucket_emb.size > 0 and embed_dim is None:
            embed_dim = int(bucket_emb.shape[1])
        for members_local, medoid_local in clusters:
            cluster_size = len(members_local)
            cluster_sizes.append(cluster_size)
            members = [filtered[bucket_indices[i]] for i in members_local]
            medoid = filtered[bucket_indices[medoid_local]]
            medoid_embedding = (
                bucket_emb[medoid_local]
                if bucket_emb is not None and bucket_emb.size > 0
                else None
            )

            domain_counter: Counter[str] = Counter()
            domain_raw_counter: Counter[str] = Counter()
            difficulty_counter: Counter[str] = Counter()
            unique_problem_ids: set[str] = set()
            row_index_set: set[int] = set()
            for m in members:
                canon = str(m.get("_domain_canon", "")).strip()
                raw_d = str(m.get("_domain_raw", "")).strip()
                if canon:
                    domain_counter[canon] += 1
                if raw_d:
                    domain_raw_counter[raw_d] += 1
                difficulty_counter[str(m.get("source_difficulty", "")).strip()] += 1
                spid = str(m.get("source_problem_id", "")).strip()
                if spid:
                    unique_problem_ids.add(spid)
                try:
                    row_idx = int(m.get("source_row_index", -1))
                    if row_idx >= 0:
                        row_index_set.add(row_idx)
                except (TypeError, ValueError):
                    pass

            etype = medoid["_type"]
            per_type_counts[etype] += 1

            entry = LibraryEntry(
                entry_id="",  # assigned after global sort
                type=etype,
                name=str(medoid.get("name", "")).strip(),
                principle=str(medoid.get("principle", "")).strip(),
                when_to_apply=str(medoid.get("when_to_apply", "")).strip(),
                domain=_top_item(domain_counter),
                hit_count=cluster_size,
                num_unique_problems=len(unique_problem_ids),
                source_problem_ids=sorted(unique_problem_ids),
                source_row_indices=sorted(row_index_set),
                source_difficulties=dict(difficulty_counter.most_common()),
                domains_seen=dict(domain_counter.most_common()),
                domains_seen_raw=dict(domain_raw_counter.most_common()),
                cluster_members=(
                    [
                        {
                            "name": str(m.get("name", "")).strip(),
                            "principle": str(m.get("principle", "")).strip(),
                            "domain": str(m.get("domain", "")).strip(),
                            "domain_canonical": str(m.get("_domain_canon", "")).strip(),
                            "source_problem_id": str(m.get("source_problem_id", "")).strip(),
                        }
                        for m in members
                    ]
                    if keep_cluster_members
                    else []
                ),
            )
            pending.append((entry, medoid_embedding))

    pending.sort(key=lambda pair: (-pair[0].hit_count, pair[0].type, pair[0].name.lower()))
    entries: list[LibraryEntry] = [pair[0] for pair in pending]
    type_rank: Counter[str] = Counter()
    for e in entries:
        type_rank[e.type] += 1
        e.entry_id = f"{e.type}-{type_rank[e.type]:06d}"

    entry_embeddings: np.ndarray | None
    if method == "text" or embed_dim is None:
        entry_embeddings = None
    else:
        rows = [
            (pair[1] if pair[1] is not None else np.zeros((embed_dim,), dtype=np.float32))
            for pair in pending
        ]
        entry_embeddings = (
            np.stack(rows).astype(np.float32, copy=False) if rows else np.zeros((0, embed_dim), dtype=np.float32)
        )

    stats = {
        "n_raw": n_raw,
        "n_after_filter": n_filtered,
        "n_final": len(entries),
        "per_type_counts": dict(per_type_counts),
        "dropped": dict(drop_counts),
        "avg_cluster_size": float(mean(cluster_sizes)) if cluster_sizes else 0.0,
        "median_cluster_size": float(median(cluster_sizes)) if cluster_sizes else 0.0,
        "normalize_domains": bool(normalize_domains),
        "n_domains_raw_distinct": len(raw_domain_counter),
        "n_domains_canonical_distinct": len(canon_domain_counter),
        "top_canonical_domains": dict(canon_domain_counter.most_common(20)),
        "method": method,
        "semantic_ratio": float(semantic_ratio),
        "text_ratio": float(text_ratio),
        "embedder": embedder if method != "text" else None,
        "device": resolved_device if method != "text" else None,
        "embed_include_domain": bool(embed_include_domain) if method != "text" else None,
        "embed_dim": embed_dim,
        "embed_batch_size": int(embed_batch_size) if method != "text" else None,
    }
    return entries, stats, dropped, entry_embeddings


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_library_jsonl(entries: list[LibraryEntry], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")


def write_dropped_jsonl(dropped: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in dropped:
            abstraction = {
                k: v for k, v in item["abstraction"].items() if not str(k).startswith("_")
            }
            record = {"reason": item["reason"], "abstraction": abstraction}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_meta_json(meta: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def write_embeddings_npy(
    out_dir: Path,
    embeddings: np.ndarray,
    *,
    model: str,
    embed_include_domain: bool,
    method: str,
    device: str | None,
    n_entries: int,
) -> tuple[Path, Path]:
    """Persist L2-normalized entry embeddings + a sidecar metadata JSON.

    ``embeddings.npy`` is shape ``(n_entries, D)`` float32, row order matches
    ``library.jsonl``. ``embeddings_meta.json`` records what produced it so a
    later retrieval step can verify model + recipe match.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / "embeddings.npy"
    meta_path = out_dir / "embeddings_meta.json"

    arr = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
    if arr.ndim != 2 or arr.shape[0] != n_entries:
        raise ValueError(
            f"embeddings shape {arr.shape} does not match n_entries={n_entries}"
        )
    np.save(npy_path, arr, allow_pickle=False)

    meta = {
        "model": model,
        "revision": "unknown",
        "dim": int(arr.shape[1]) if arr.size else 0,
        "normalize": True,
        "composed_text_recipe": COMPOSED_TEXT_RECIPE,
        "query_recipe": QUERY_RECIPE_V1,
        "embed_include_domain": bool(embed_include_domain),
        "method": method,
        "device": device,
        "n_entries": int(arr.shape[0]),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return npy_path, meta_path


def format_top_n_text(
    entries: list[LibraryEntry],
    *,
    top_n: int = 5,
    rule_char: str = "=",
    rule_width: int = 80,
) -> str:
    """Plain-text top-N-per-type preview suitable for terminal output.

    Avoids markdown tables (which wrap badly in narrow terminals); each entry is
    rendered as a small block of labeled lines.
    """
    rule = rule_char * rule_width
    strategies = [e for e in entries if e.type == "strategy"]
    cautions = [e for e in entries if e.type == "caution"]

    def render_section(label: str, items: list[LibraryEntry]) -> list[str]:
        lines: list[str] = ["", rule, f"Top {min(top_n, len(items))} {label} (of {len(items)})", rule]
        if not items:
            lines.append("(no entries)")
            return lines
        for i, e in enumerate(items[:top_n], start=1):
            example = e.source_problem_ids[0] if e.source_problem_ids else "-"
            lines.append(
                f"[{i}] hit={e.hit_count} unique_problems={e.num_unique_problems} "
                f"id={e.entry_id} domain={e.domain or '-'}"
            )
            lines.append(f"    name      : {e.name}")
            lines.append(f"    principle : {e.principle}")
            if e.when_to_apply:
                lines.append(f"    when      : {e.when_to_apply}")
            lines.append(f"    example   : {example}")
        return lines

    out: list[str] = []
    out.extend(render_section("strategies (by hit_count)", strategies))
    out.extend(render_section("cautions (by hit_count)", cautions))
    return "\n".join(out)


def _md_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )


_MD_PREAMBLE_KEYS = (
    "built_at",
    "source_file",
    "n_raw",
    "n_after_filter",
    "n_final",
    "per_type_counts",
    "method",
    "semantic_ratio",
    "text_ratio",
    "embedder",
    "device",
    "embed_dim",
    "embed_include_domain",
    "min_chars",
    "max_chars",
    "per_type",
    "keep_cluster_members",
    "dropped",
    "avg_cluster_size",
    "median_cluster_size",
)


def write_library_md(
    entries: list[LibraryEntry],
    meta: dict[str, Any],
    out_path: Path,
    *,
    top_n_per_type: int = 50,
) -> None:
    strategies = [e for e in entries if e.type == "strategy"]
    cautions = [e for e in entries if e.type == "caution"]
    singleton_strategy = sum(1 for e in strategies if e.hit_count == 1)
    singleton_caution = sum(1 for e in cautions if e.hit_count == 1)

    lines: list[str] = []
    method_label = str(meta.get("method", "semantic"))
    lines.append(f"# HRLib v1 (flat, method={method_label})")
    lines.append("")
    lines.append("Each row below = one deduplicated abstraction. Column definitions:")
    lines.append("")
    lines.append("- **#** — 1-indexed rank within this type, after sorting by `hit_count` desc.")
    lines.append("- **hit** — number of raw abstractions merged into this cluster (`hit_count`).")
    lines.append(
        "- **name** — the cluster medoid's short label (may vary across LLM calls; principle is the stable field)."
    )
    lines.append("- **principle** — the medoid's one-sentence reasoning advice.")
    lines.append(
        "- **domain** — most common `domain` across cluster members (alphabetical tiebreak)."
    )
    lines.append(
        "- **example source** — first `source_problem_id` in the cluster, formatted "
        "`data_source|split|row_index`."
    )
    lines.append("")
    lines.append(
        "Full records (including `when_to_apply`, every `source_problem_id`, domain/difficulty "
        "distributions, and the raw `cluster_members`) live in `library.jsonl`. Unmerged singletons "
        "are included in `library.jsonl` too; this document only surfaces the top-N per type."
    )
    lines.append("")

    lines.append("## Run metadata")
    lines.append("")
    for key in _MD_PREAMBLE_KEYS:
        if key in meta:
            lines.append(f"- **{key}**: {json.dumps(meta[key], ensure_ascii=False)}")
    lines.append("")

    def render_table(header: str, items: list[LibraryEntry]) -> None:
        lines.append(f"## {header} (showing top {min(top_n_per_type, len(items))} of {len(items)})")
        lines.append("")
        if not items:
            lines.append("_(no entries)_")
            lines.append("")
            return
        lines.append("| # | hit | name | principle | domain | example source |")
        lines.append("|---|-----|------|-----------|--------|----------------|")
        for i, e in enumerate(items[:top_n_per_type], start=1):
            example = e.source_problem_ids[0] if e.source_problem_ids else ""
            lines.append(
                f"| {i} | {e.hit_count} | {_md_escape(e.name)} | "
                f"{_md_escape(e.principle)} | {_md_escape(e.domain)} | "
                f"{_md_escape(example)} |"
            )
        lines.append("")

    render_table("Top strategies (by hit_count)", strategies)
    render_table("Top cautions (by hit_count)", cautions)

    lines.append("## Long tail (hit_count == 1) summary")
    lines.append("")
    lines.append(f"- {singleton_strategy} singleton strategies")
    lines.append(f"- {singleton_caution} singleton cautions")
    lines.append("- All singletons remain in `library.jsonl`; omitted here to keep the preview readable.")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
