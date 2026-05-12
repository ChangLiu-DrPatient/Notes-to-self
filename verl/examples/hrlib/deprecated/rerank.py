#!/usr/bin/env python3
"""Cross-encoder reranking utilities for HRLib retrieval candidates."""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


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
def _load_cross_encoder(model_name: str, device: str, max_length: int):
    from sentence_transformers import CrossEncoder  # noqa: WPS433

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    kwargs: dict[str, Any] = {"device": device}
    if max_length > 0:
        kwargs["max_length"] = int(max_length)
    return CrossEncoder(model_name, **kwargs)


def _to_1d_scores(raw: Any) -> np.ndarray:
    arr = np.asarray(raw)
    if arr.ndim == 1:
        return np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return np.asarray(arr[:, 0], dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        # Many rerankers return 2 logits; positive class is conventionally last.
        return np.asarray(arr[:, -1], dtype=np.float32)
    raise RuntimeError(f"unexpected cross-encoder score shape: {arr.shape}")


@dataclass
class RerankResult:
    """Reranking outputs aligned to bi-encoder candidate order."""

    scores_by_bi_rank: np.ndarray
    rerank_order: list[int]


class CrossEncoderReranker:
    """Thin wrapper around sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
        lazy: bool = True,
    ) -> None:
        self.model_name = str(model_name).strip()
        if not self.model_name:
            raise ValueError("rerank model name must be non-empty")
        self.device = _resolve_device(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self._model = None
        if not lazy:
            self._get_model()

    def _get_model(self):
        if self._model is None:
            self._model = _load_cross_encoder(self.model_name, self.device, self.max_length)
        return self._model

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        if not pairs:
            return np.zeros((0,), dtype=np.float32)
        model = self._get_model()
        raw = model.predict(
            list(pairs),
            batch_size=int(batch_size or self.batch_size),
            show_progress_bar=False,
        )
        return _to_1d_scores(raw)

    def rerank_texts(
        self,
        query_text: str,
        candidate_texts: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> RerankResult:
        if not candidate_texts:
            return RerankResult(scores_by_bi_rank=np.zeros((0,), dtype=np.float32), rerank_order=[])
        pairs = [(str(query_text), str(text)) for text in candidate_texts]
        scores = self.score_pairs(pairs, batch_size=batch_size)
        order = np.argsort(-scores, kind="stable").tolist()
        return RerankResult(scores_by_bi_rank=scores, rerank_order=order)


def hit_to_rerank_text(hit: Any) -> str:
    """Render one RetrievalHit-like object into reranker passage text."""
    kind = str(getattr(hit, "type", "") or "").strip() or "note"
    principle = str(getattr(hit, "principle", "") or "").strip()
    when = str(getattr(hit, "when_to_apply", "") or "").strip()
    domain = str(getattr(hit, "domain", "") or "").strip()

    lines = [f"[{kind}] {principle}".strip()]
    if when:
        lines.append(f"when: {when}")
    if domain:
        lines.append(f"domain: {domain}")
    return "\n".join(line for line in lines if line).strip()
