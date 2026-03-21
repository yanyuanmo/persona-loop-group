"""EmbeddingMemory — dense-vector memory backend using sentence-transformers.

Drop-in replacement for ChromaMemory.  Embeddings are computed lazily on first
use so that importing this module has no heavy dependencies at parse time.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from persona_loop.memory.base_memory import BaseMemory

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingMemory(BaseMemory):
    """Cosine-similarity memory with a recency bonus (same formula as ChromaMemory)."""

    def __init__(self) -> None:
        self._store: List[str] = []
        self._embeddings: List[np.ndarray] = []
        self._model = None  # lazy-loaded

    def reset(self) -> None:
        self._store = []
        self._embeddings = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(_MODEL_NAME)
        return self._model

    def _embed(self, text: str) -> np.ndarray:
        model = self._load_model()
        vec: np.ndarray = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        # Both vectors are already L2-normalised by normalize_embeddings=True.
        return float(np.dot(a, b))

    def _rank(self, query: str) -> List[Tuple[float, int, str]]:
        if not self._store:
            return []
        q_vec = self._embed(query)
        n = len(self._store)
        ranked: List[Tuple[float, int, str]] = []
        for idx, (text, doc_vec) in enumerate(zip(self._store, self._embeddings)):
            sim = self._cosine(q_vec, doc_vec)
            recency_bonus = 0.03 * ((idx + 1) / max(1, n))
            score = sim + recency_bonus
            ranked.append((score, idx, text))
        ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return ranked

    # ------------------------------------------------------------------
    # BaseMemory interface
    # ------------------------------------------------------------------

    def add(self, text: str) -> None:
        normalized = str(text).strip()
        if not normalized:
            return
        # Deduplicate: avoid storing the same text twice (common in eval mode
        # where the same history turns are seen across many QA iterations).
        if normalized in self._store:
            return
        self._store.append(normalized)
        self._embeddings.append(self._embed(normalized))

    def search(self, query: str, top_k: int = 3) -> List[str]:
        if not self._store or top_k <= 0:
            return []
        unique: List[str] = []
        seen: set = set()
        for _score, _idx, item in self._rank(query):
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
            if len(unique) >= top_k:
                break
        return unique
