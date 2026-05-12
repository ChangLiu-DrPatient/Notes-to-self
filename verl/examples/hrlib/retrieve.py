#!/usr/bin/env python3
"""FlatRetriever: cosine top-k over a prebuilt HRLib library.

This module is deliberately minimal for Stage 0 v1 (see
implementation_plan_stage0_3.md §5.2.4):

- Loads ``library.jsonl`` (entries) + ``embeddings.npy`` (L2-normalized,
  float32, shape ``(n_entries, D)``) + ``embeddings_meta.json`` (model name,
  dim, composed-text recipe, query recipe).
- Lazy-loads the sentence-transformers model named in the sidecar; asserts the
  model loads to the same dim the library was built with.
- ``retrieve(text, k)`` composes the query via the library's ``query_recipe``,
  encodes with the same normalization, does ``embeddings @ q``, picks top-k via
  ``argpartition``, returns a list of dicts sorted by descending cosine.

Not implemented here (deferred to Stage 1+):

- ``[domain]`` prefix on queries (requires passing ``subject`` through).
- Strategy/caution quotas and intra-result dedup.
- FAISS / sharded retrieval.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_V1_FALLBACK_QUERY_RECIPE = "{user_text}"


@dataclass
class RetrievalHit:
    """One row from ``library.jsonl`` plus the retriever-added score."""

    entry_id: str
    type: str
    name: str
    principle: str
    when_to_apply: str
    domain: str
    hit_count: int
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "type": self.type,
            "name": self.name,
            "principle": self.principle,
            "when_to_apply": self.when_to_apply,
            "domain": self.domain,
            "hit_count": self.hit_count,
            "score": float(self.score),
        }


def _load_library_entries(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _resolve_device(request: str) -> str:
    req = (request or "auto").strip().lower()
    if req == "cpu":
        return "cpu"
    try:
        import torch  # noqa: WPS433
    except ImportError:
        return "cpu"
    if req in {"auto", ""}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda"):
        return req if torch.cuda.is_available() else "cpu"
    return req


@functools.lru_cache(maxsize=4)
def _load_encoder(name: str, device: str):
    """Cached SentenceTransformer loader (shared across retriever instances)."""
    from sentence_transformers import SentenceTransformer  # noqa: WPS433

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return SentenceTransformer(name, device=device)


class FlatRetriever:
    """Cosine top-k retriever over a flat HRLib library.

    Parameters
    ----------
    library_dir : str | Path
        Directory produced by ``20_aggregate.py`` (must contain ``library.jsonl``,
        ``embeddings.npy``, and ``embeddings_meta.json``).
    device : str, default ``"auto"``
        ``auto`` picks CUDA when available, else CPU. ``cpu`` / ``cuda`` /
        ``cuda:N`` work as expected.
    lazy : bool, default True
        When True, the encoder is loaded on first ``encode_query`` call.
        Set False to pre-load (useful for long-running services to surface load
        errors at startup).
    """

    def __init__(
        self,
        library_dir: str | Path,
        *,
        device: str = "auto",
        lazy: bool = True,
    ) -> None:
        self.library_dir = Path(library_dir)
        if not self.library_dir.is_dir():
            raise FileNotFoundError(f"library_dir does not exist: {self.library_dir}")

        lib_path = self.library_dir / "library.jsonl"
        emb_path = self.library_dir / "embeddings.npy"
        meta_path = self.library_dir / "embeddings_meta.json"
        for p in (lib_path, emb_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(f"required library artifact missing: {p}")

        self.entries = _load_library_entries(lib_path)
        self.embeddings: np.ndarray = np.load(emb_path, mmap_mode="r")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if self.embeddings.ndim != 2:
            raise ValueError(
                f"embeddings.npy must be 2D, got shape {self.embeddings.shape}"
            )
        if self.embeddings.shape[0] != len(self.entries):
            raise ValueError(
                "library/embedding row mismatch: "
                f"library.jsonl={len(self.entries)}, "
                f"embeddings.npy={self.embeddings.shape[0]}"
            )
        if self.embeddings.dtype != np.float32:
            # Embedder writer guarantees float32; be defensive for future formats.
            self.embeddings = self.embeddings.astype(np.float32, copy=False)

        self.model_name: str = str(self.meta.get("model") or "")
        if not self.model_name:
            raise ValueError(f"embeddings_meta.json missing 'model': {meta_path}")
        self.expected_dim = int(self.meta.get("dim") or self.embeddings.shape[1])
        if self.expected_dim != self.embeddings.shape[1]:
            raise ValueError(
                f"meta.dim={self.expected_dim} vs embeddings dim={self.embeddings.shape[1]}"
            )

        self.query_recipe: str = str(
            self.meta.get("query_recipe") or _V1_FALLBACK_QUERY_RECIPE
        )

        self.device = _resolve_device(device)
        self._encoder = None
        if not lazy:
            self._get_encoder()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _get_encoder(self):
        if self._encoder is None:
            enc = _load_encoder(self.model_name, self.device)
            # Verify the loaded model matches the library's dim.
            # sentence-transformers>=3 renamed to get_embedding_dimension;
            # fall back through both names, and finally to a probe encode.
            out_dim: int | None = None
            for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
                fn = getattr(enc, attr, None)
                if callable(fn):
                    try:
                        out_dim = int(fn())
                    except Exception:
                        out_dim = None
                    if out_dim:
                        break
            if not out_dim:
                out_dim = int(
                    enc.encode(
                        "probe",
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    ).shape[-1]
                )
            if out_dim != self.expected_dim:
                raise ValueError(
                    "loaded encoder dim does not match library: "
                    f"encoder={self.model_name} dim={out_dim}, "
                    f"library dim={self.expected_dim}"
                )
            self._encoder = enc
        return self._encoder

    def _compose_query_text(self, user_text: str, **extra: Any) -> str:
        """Apply the library's ``query_recipe`` template.

        v1 recipe is literally ``{user_text}``. Stage 1 may switch to
        ``[{subject}] {user_text}``; any keys expected by the recipe must be
        passed as kwargs (e.g. ``subject=...``).
        """
        text = str(user_text).strip()
        fields = {"user_text": text, **{k: str(v).strip() for k, v in extra.items()}}
        try:
            return self.query_recipe.format(**fields)
        except KeyError as exc:
            missing = exc.args[0]
            raise ValueError(
                f"query_recipe {self.query_recipe!r} requires field {missing!r}; "
                f"pass it as a keyword argument to retrieve()."
            ) from exc

    def encode_query(self, text: str, **extra: Any) -> np.ndarray:
        """Encode a single query into an L2-normalized float32 vector of shape (D,)."""
        composed = self._compose_query_text(text, **extra)
        enc = self._get_encoder()
        vec = enc.encode(
            [composed],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != 1:
            raise RuntimeError(f"unexpected encoder output shape: {arr.shape}")
        return np.ascontiguousarray(arr[0])

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _topk_hits_from_scores(self, scores: np.ndarray, k: int) -> list[RetrievalHit]:
        """Build sorted top-k hits from a full cosine score vector."""
        if k <= 0:
            return []
        n = self.embeddings.shape[0]
        if n == 0:
            return []
        k = min(k, n)

        # argpartition for O(N) top-k, then sort the k-slice descending
        if k < n:
            top_idx = np.argpartition(-scores, k - 1)[:k]
        else:
            top_idx = np.arange(n)
        order = top_idx[np.argsort(-scores[top_idx])]

        hits: list[RetrievalHit] = []
        for idx in order.tolist():
            entry = self.entries[idx]
            hits.append(
                RetrievalHit(
                    entry_id=str(entry.get("entry_id", "")),
                    type=str(entry.get("type", "")),
                    name=str(entry.get("name", "")),
                    principle=str(entry.get("principle", "")),
                    when_to_apply=str(entry.get("when_to_apply", "")),
                    domain=str(entry.get("domain", "")),
                    hit_count=int(entry.get("hit_count", 0)),
                    score=float(scores[idx]),
                )
            )
        return hits

    def retrieve_with_all_scores(
        self, text: str, k: int = 6, **extra: Any
    ) -> tuple[list[RetrievalHit], np.ndarray]:
        """Return top-k hits and the full (N,) cosine score vector.

        ``extra`` keyword arguments are forwarded to ``encode_query`` to fill any
        placeholders in the library's ``query_recipe`` (v1 has none).
        """
        n = self.embeddings.shape[0]
        if n == 0:
            return [], np.zeros((0,), dtype=np.float32)
        q = self.encode_query(text, **extra)
        scores = np.asarray(self.embeddings @ q, dtype=np.float32)
        hits = self._topk_hits_from_scores(scores, k)
        return hits, scores

    def retrieve(self, text: str, k: int = 6, **extra: Any) -> list[RetrievalHit]:
        """Return the top-``k`` library hits by cosine similarity.

        ``extra`` keyword arguments are forwarded to ``encode_query`` to fill any
        placeholders in the library's ``query_recipe`` (v1 has none).
        """
        hits, _ = self.retrieve_with_all_scores(text, k=k, **extra)
        return hits

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Lightweight summary for logging / injection_meta."""
        return {
            "library_dir": str(self.library_dir),
            "n_entries": len(self.entries),
            "model": self.model_name,
            "dim": self.expected_dim,
            "device": self.device,
            "query_recipe": self.query_recipe,
            "composed_text_recipe": self.meta.get("composed_text_recipe"),
        }
