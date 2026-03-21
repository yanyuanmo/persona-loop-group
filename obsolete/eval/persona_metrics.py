from __future__ import annotations

import re
from statistics import mean
from typing import Dict, List, Optional


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _select_relevant_facts(query: str, facts: List[Dict[str, object]], top_k: int) -> List[Dict[str, object]]:
    """Select the top_k most question-relevant facts for NLI evaluation.

    Uses the *question* (not the prediction) to score relevance so that fact
    selection is independent of the model's output, avoiding circular bias.
    """
    if not facts:
        return []
    if top_k <= 0 or top_k >= len(facts):
        return facts

    query_tokens = _tokens(query)
    scored: List[tuple[int, Dict[str, object]]] = []
    for fact in facts:
        ft = str(fact.get("fact_text", ""))
        overlap = len(query_tokens.intersection(_tokens(ft)))
        scored.append((overlap, fact))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Drop zero-overlap facts when there are positively-overlapping alternatives.
    # Comparing completely unrelated text pairs inflates contradiction scores.
    if scored and scored[0][0] > 0:
        scored = [item for item in scored if item[0] > 0]

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
    question: str = "",
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

    # Use the question (not the prediction) to select relevant facts so that
    # selection is not biased by what the model chose to say or omit.
    selection_query = question if question.strip() else prediction
    selected = _select_relevant_facts(query=selection_query, facts=pool, top_k=top_k)

    if not selected or nli is None:
        return {
            "persona_facts_total": len(fact_bank),
            "persona_facts_used": len(selected),
            "persona_entailment": 0.0,
            "persona_contradiction": 0.0,
            "persona_contradiction_max": 0.0,
            "persona_any_contradiction": False,
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
        fact_text = str(fact.get("fact_text", ""))
        if not fact_text:
            continue
        # Dialogue NLI standard direction (Welleck et al. 2019):
        #   premise = persona fact, hypothesis = response
        # This checks whether the response is consistent with / contradicts the
        # persona fact, NOT whether the response re-asserts the fact.
        score = nli.score(premise=fact_text, hypothesis=prediction)
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
            "persona_contradiction_max": 0.0,
            "persona_any_contradiction": False,
            "persona_supported_ratio": 0.0,
            "persona_conflict_ratio": 0.0,
            "persona_pcs": 0.0,
            "persona_fact_texts": [str(f.get("fact_text", "")) for f in selected],
        }

    avg_ent = float(mean(entailments))
    avg_con = float(mean(contradictions))
    max_con = float(max(contradictions))
    used = len(entailments)

    return {
        "persona_facts_total": len(fact_bank),
        "persona_facts_used": used,
        "persona_entailment": avg_ent,
        "persona_contradiction": avg_con,
        "persona_contradiction_max": max_con,
        "persona_any_contradiction": max_con >= conflict_threshold,
        "persona_supported_ratio": supported / used,
        "persona_conflict_ratio": conflicted / used,
        "persona_pcs": avg_ent - avg_con,
        "persona_fact_texts": [str(f.get("fact_text", "")) for f in selected],
    }
