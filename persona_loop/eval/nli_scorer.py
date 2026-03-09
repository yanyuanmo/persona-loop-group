from __future__ import annotations

from typing import Dict, List


class NLIScorer:
    def __init__(self, model_name: str):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Real NLI scoring requires 'transformers' and 'torch'. "
                "Install with: pip install transformers torch"
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()

        id2label = {int(k): v.lower() for k, v in self._model.config.id2label.items()}
        self._label_order: List[str] = [id2label[i] for i in sorted(id2label.keys())]

    def score(self, premise: str, hypothesis: str) -> Dict[str, float]:
        encoded = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with self._torch.no_grad():
            logits = self._model(**encoded).logits[0]
            probs = self._torch.softmax(logits, dim=-1).tolist()

        label_probs = {self._label_order[i]: float(probs[i]) for i in range(len(probs))}
        return {
            "entailment": label_probs.get("entailment", 0.0),
            "neutral": label_probs.get("neutral", 0.0),
            "contradiction": label_probs.get("contradiction", 0.0),
        }
