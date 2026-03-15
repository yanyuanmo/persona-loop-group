from __future__ import annotations

import re
from statistics import mean
from typing import Dict, List, Optional


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _select_relevant_facts(prediction: str, facts: List[Dict[str, object]], top_k: int) -> List[Dict[str, object]]:
    if not facts:
        return []
    if top_k <= 0 or top_k >= len(facts):
        return facts

    pred_tokens = _tokens(prediction)
    scored: List[tuple[int, Dict[str, object]]] = []
    for fact in facts:
        ft = str(fact.get("fact_text", ""))
        overlap = len(pred_tokens.intersection(_tokens(ft)))
        scored.append((overlap, fact))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [fact for _, fact in scored[:top_k]]


def compute_persona_metrics(
    prediction: str,
    fact_bank: List[Dict[str, object]],
    nli: Optional[object],
    top_k: int = 5,
    support_threshold: float = 0.5,
    conflict_threshold: float = 0.5,
    question_subjects: Optional[List[str]] = None,
    allowed_slots: Optional[set[str]] = None,
    min_filtered_facts: int = 2,
) -> Dict[str, object]:
    pool = list(fact_bank)

    # Optional relevance constraints for evaluation: prefer facts tied to
    # question subjects/slots, but back off if filtering becomes too sparse.
    if question_subjects:
        subj_lc = {s.strip().lower() for s in question_subjects if str(s).strip()}
        by_owner = [
            f
            for f in pool
            if str(f.get("owner", "")).strip().lower() in subj_lc
        ]
        if len(by_owner) >= int(min_filtered_facts):
            pool = by_owner

    if allowed_slots:
        by_slot = [
            f
            for f in pool
            if str(f.get("slot", "")).strip() in allowed_slots
        ]
        if len(by_slot) >= int(min_filtered_facts):
            pool = by_slot

    selected = _select_relevant_facts(prediction=prediction, facts=pool, top_k=top_k)

    if not selected or nli is None:
        return {
            "persona_facts_total": len(fact_bank),
            "persona_facts_used": len(selected),
            "persona_entailment": 0.0,
            "persona_contradiction": 0.0,
            "persona_supported_ratio": 0.0,
            "persona_conflict_ratio": 0.0,
            "persona_pcs": 0.0,
            "persona_fact_texts": [str(f.get("fact_text", "")) for f in selected],
        }

    entailments: List[float] = []
    contradictions: List[float] = []
    supported = 0
    conflicted = 0

    for fact in selected:
        hyp = str(fact.get("fact_text", ""))
        if not hyp:
            continue
        score = nli.score(premise=prediction, hypothesis=hyp)
        ent = float(score.get("entailment", 0.0))
        con = float(score.get("contradiction", 0.0))
        entailments.append(ent)
        contradictions.append(con)
        if ent >= support_threshold:
            supported += 1
        if con >= conflict_threshold:
            conflicted += 1

    if not entailments:
        return {
            "persona_facts_total": len(fact_bank),
            "persona_facts_used": len(selected),
            "persona_entailment": 0.0,
            "persona_contradiction": 0.0,
            "persona_supported_ratio": 0.0,
            "persona_conflict_ratio": 0.0,
            "persona_pcs": 0.0,
            "persona_fact_texts": [str(f.get("fact_text", "")) for f in selected],
        }

    avg_ent = float(mean(entailments))
    avg_con = float(mean(contradictions))
    used = len(entailments)

    return {
        "persona_facts_total": len(fact_bank),
        "persona_facts_used": used,
        "persona_entailment": avg_ent,
        "persona_contradiction": avg_con,
        "persona_supported_ratio": supported / used,
        "persona_conflict_ratio": conflicted / used,
        "persona_pcs": avg_ent - avg_con,
        "persona_fact_texts": [str(f.get("fact_text", "")) for f in selected],
    }
