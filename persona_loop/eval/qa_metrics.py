from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(_normalize(prediction) == _normalize(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_scores(prediction: str, gold: str) -> Dict[str, float]:
    return {
        "em": exact_match(prediction, gold),
        "f1": token_f1(prediction, gold),
    }
