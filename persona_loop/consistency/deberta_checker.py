from __future__ import annotations

from persona_loop.consistency.base_checker import BaseChecker


class DebertaChecker(BaseChecker):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def score(self, premise: str, hypothesis: str) -> float:
        # Placeholder lexical overlap score. Replace with a real NLI model.
        p = set(premise.lower().split())
        h = set(hypothesis.lower().split())
        if not h:
            return 0.0
        return round(len(p.intersection(h)) / max(1, len(h)), 4)
