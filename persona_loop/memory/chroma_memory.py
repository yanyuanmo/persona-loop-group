from __future__ import annotations

from typing import List

from persona_loop.memory.base_memory import BaseMemory


class ChromaMemory(BaseMemory):
    def __init__(self) -> None:
        self._store: List[str] = []

    def add(self, text: str) -> None:
        self._store.append(text)

    def search(self, query: str, top_k: int = 3) -> List[str]:
        query_tokens = set(query.lower().split())
        scored = []
        for item in self._store:
            overlap = len(query_tokens.intersection(item.lower().split()))
            scored.append((overlap, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]
