from __future__ import annotations

from typing import Dict, List


def compute_consistency_metrics(outputs: List[Dict[str, object]]) -> Dict[str, float]:
    scores = [o.get("consistency") for o in outputs if o.get("consistency") is not None]
    if not scores:
        return {
            "consistency_mean": 0.0,
            "consistency_count": 0.0,
        }

    mean_score = sum(float(s) for s in scores) / len(scores)
    return {
        "consistency_mean": round(mean_score, 4),
        "consistency_count": float(len(scores)),
    }
