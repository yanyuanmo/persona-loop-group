from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

from persona_loop.memory.base_memory import BaseMemory


class ChromaMemory(BaseMemory):
    _STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had", "has",
        "have", "he", "her", "hers", "him", "his", "i", "if", "in", "is", "it", "its", "me",
        "my", "of", "on", "or", "our", "ours", "she", "that", "the", "their", "them", "they",
        "this", "to", "was", "we", "were", "what", "when", "where", "who", "why", "with", "you",
        "your", "yours",
    }

    def __init__(self) -> None:
        self._store: List[str] = []
        self._doc_freq: Counter[str] = Counter()
        self._doc_tokens: List[Counter[str]] = []
        self._doc_lengths: List[int] = []

    def reset(self) -> None:
        self._store = []
        self._doc_freq = Counter()
        self._doc_tokens = []
        self._doc_lengths = []

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9']+", text.lower())
        return [token for token in tokens if token not in cls._STOPWORDS]

    def _bm25_score(self, query_tokens: List[str], doc_tf: Counter[str], doc_len: int) -> float:
        if not query_tokens or not doc_tf:
            return 0.0

        num_docs = max(1, len(self._store))
        avg_doc_len = max(1.0, sum(self._doc_lengths) / max(1, len(self._doc_lengths)))
        k1 = 1.5
        b = 0.75
        score = 0.0

        for token in query_tokens:
            term_freq = doc_tf.get(token, 0)
            if term_freq <= 0:
                continue
            doc_freq = self._doc_freq.get(token, 0)
            idf = math.log(1.0 + (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            norm = term_freq + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
            score += idf * ((term_freq * (k1 + 1.0)) / max(norm, 1e-8))
        return score

    def _rank(self, query: str) -> List[Tuple[float, int, str]]:
        query_tokens = self._tokenize(query)
        ranked: List[Tuple[float, int, str]] = []
        for idx, item in enumerate(self._store):
            bm25 = self._bm25_score(query_tokens, self._doc_tokens[idx], self._doc_lengths[idx])
            recency_bonus = 0.03 * ((idx + 1) / max(1, len(self._store)))
            score = bm25 + recency_bonus
            ranked.append((score, idx, item))
        ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return ranked

    def add(self, text: str) -> None:
        normalized = str(text).strip()
        if not normalized:
            return
        self._store.append(normalized)
        tf = Counter(self._tokenize(normalized))
        self._doc_tokens.append(tf)
        self._doc_lengths.append(sum(tf.values()))
        for token in tf.keys():
            self._doc_freq[token] += 1

    def search(self, query: str, top_k: int = 3) -> List[str]:
        if not self._store:
            return []

        unique: List[str] = []
        seen = set()
        for score, _, item in self._rank(query):
            if item in seen:
                continue
            if score <= 0 and unique:
                continue
            unique.append(item)
            seen.add(item)
            if len(unique) >= top_k:
                return unique

        if unique:
            return unique[:top_k]

        return list(dict.fromkeys(reversed(self._store)))[:top_k]
