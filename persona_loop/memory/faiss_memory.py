from __future__ import annotations

from typing import List

from persona_loop.memory.base_memory import BaseMemory


class FaissMemory(BaseMemory):
    def __init__(self) -> None:
        self._store: List[str] = []

    def add(self, text: str) -> None:
        self._store.append(text)

    def search(self, query: str, top_k: int = 3) -> List[str]:
        # Skeleton behavior: returns recent items to emulate fast approximate retrieval.
        return list(reversed(self._store[-top_k:]))
