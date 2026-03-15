from __future__ import annotations

import re
from typing import ClassVar, Dict, Optional

from persona_loop.consistency.base_checker import BaseChecker
from persona_loop.eval.nli_scorer import NLIScorer


class DebertaChecker(BaseChecker):
    _SCORER_CACHE: ClassVar[Dict[str, NLIScorer]] = {}

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._scorer: Optional[NLIScorer] = None
        self._load_error: Optional[str] = None

        cached = self._SCORER_CACHE.get(model_name)
        if cached is not None:
            self._scorer = cached
            return

        try:
            scorer = NLIScorer(model_name=model_name)
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
        else:
            self._scorer = scorer
            self._SCORER_CACHE[model_name] = scorer

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9']+", text.lower()))

    def _fallback_score(self, premise: str, hypothesis: str) -> float:
        premise_tokens = self._tokenize(premise)
        hypothesis_tokens = self._tokenize(hypothesis)
        if not premise_tokens or not hypothesis_tokens:
            return 0.0

        overlap = len(premise_tokens.intersection(hypothesis_tokens)) / max(1, len(hypothesis_tokens))
        negation_penalty = 0.0
        if ({"not", "never", "no"} & premise_tokens) ^ ({"not", "never", "no"} & hypothesis_tokens):
            negation_penalty = 0.2
        return max(-1.0, min(1.0, round(overlap - negation_penalty, 4)))

    def score(self, premise: str, hypothesis: str) -> float:
        premise = premise.strip()
        hypothesis = hypothesis.strip()
        if not premise or not hypothesis:
            return 0.0

        if self._scorer is None:
            return self._fallback_score(premise=premise, hypothesis=hypothesis)

        label_scores = self._scorer.score(premise=premise, hypothesis=hypothesis)
        entailment = float(label_scores.get("entailment", 0.0))
        contradiction = float(label_scores.get("contradiction", 0.0))
        return round(entailment - contradiction, 4)
