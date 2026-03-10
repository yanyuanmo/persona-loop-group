from __future__ import annotations

import re
from typing import Dict, List


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" .,!?:;\t\n\r")
    return text


def _lower_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _make_fact(slot: str, value: str, dia_id: str, confidence: float) -> Dict[str, object]:
    value = _clean(value)
    return {
        "slot": slot,
        "value": value,
        "fact_text": f"{slot.replace('_', ' ')}: {value}",
        "dia_id": dia_id,
        "confidence": confidence,
        "source": "rule",
    }


_SINGLETON_SLOTS = {
    "relationship_status",
    "occupation",
    "age",
    "location",
    "pet_name",
}


def _push(
    out: List[Dict[str, object]],
    slot: str,
    value: str,
    dia_id: str,
    confidence: float,
    turn_index: int,
) -> None:
    value = _clean(value)
    if not value:
        return
    fact = _make_fact(slot=slot, value=value, dia_id=dia_id, confidence=confidence)
    fact["turn_index"] = turn_index
    out.append(fact)


def _extract_turn_facts(turn_text: str, dia_id: str, turn_index: int) -> List[Dict[str, object]]:
    facts: List[Dict[str, object]] = []

    # Relationship status.
    for m in re.finditer(
        r"\b(?:i am|i'm)\s+(single|married|divorced|engaged|widowed)\b",
        turn_text,
        flags=re.IGNORECASE,
    ):
        _push(facts, "relationship_status", m.group(1), dia_id, 0.92, turn_index)

    # Occupation / role.
    for m in re.finditer(
        r"\b(?:i work as|i'm\s+an?|i am\s+an?)\s+([a-z][a-z\- ]{2,30})\b",
        turn_text,
        flags=re.IGNORECASE,
    ):
        value = _clean(m.group(1)).lower()
        if value in {"single", "married", "divorced", "engaged", "widowed"}:
            continue
        if any(k in value for k in ["swamped", "tired", "happy", "sad", "keen", "working"]):
            continue
        _push(facts, "occupation", value, dia_id, 0.72, turn_index)

    # Age.
    for m in re.finditer(r"\b(?:i am|i'm)\s+(\d{1,2})\s*(?:years old|yo)?\b", turn_text, flags=re.IGNORECASE):
        _push(facts, "age", m.group(1), dia_id, 0.9, turn_index)

    # Location.
    for m in re.finditer(
        r"\b(?:i live in|i'm from|i am from)\s+([a-z][a-z\- ]{1,40})\b",
        turn_text,
        flags=re.IGNORECASE,
    ):
        _push(facts, "location", m.group(1), dia_id, 0.86, turn_index)

    # Pet / relation indicators.
    for m in re.finditer(r"\bmy\s+pet\s+([A-Z][a-z]+)\b", turn_text):
        _push(facts, "pet_name", m.group(1), dia_id, 0.94, turn_index)

    for m in re.finditer(r"\b([A-Z][a-z]+)\s+is\s+my\s+pet\b", turn_text):
        _push(facts, "pet_name", m.group(1), dia_id, 0.94, turn_index)

    # Preferences.
    for m in re.finditer(r"\bi\s+(?:like|love|enjoy|prefer)\s+([^.!?]{2,80})", turn_text, flags=re.IGNORECASE):
        _push(facts, "likes", m.group(1), dia_id, 0.78, turn_index)

    for m in re.finditer(r"\bi\s+(?:hate|dislike)\s+([^.!?]{2,80})", turn_text, flags=re.IGNORECASE):
        _push(facts, "dislikes", m.group(1), dia_id, 0.78, turn_index)

    # Possessions / relations.
    for m in re.finditer(r"\bmy\s+([a-z ]{2,24})\s+is\s+([^.!?]{2,80})", turn_text, flags=re.IGNORECASE):
        key = _clean(m.group(1)).lower().replace(" ", "_")
        if key:
            _push(facts, f"my_{key}", m.group(2), dia_id, 0.84, turn_index)

    for m in re.finditer(r"\bi\s+(?:have|own|got)\s+([^.!?]{2,80})", turn_text, flags=re.IGNORECASE):
        _push(facts, "has", m.group(1), dia_id, 0.76, turn_index)

    # Personal experiences that often encode stable identity context.
    for m in re.finditer(r"\bi\s+went\s+to\s+([^.!?]{2,80})", turn_text, flags=re.IGNORECASE):
        _push(facts, "recent_experience", m.group(1), dia_id, 0.74, turn_index)

    for m in re.finditer(r"\bi\s+celebrated\s+([^.!?]{2,80})", turn_text, flags=re.IGNORECASE):
        _push(facts, "life_event", m.group(1), dia_id, 0.74, turn_index)

    for m in re.finditer(r"\b([A-Z][a-z]+)\s+is\s+my\s+([a-z ]{2,24})\b", turn_text):
        person = _clean(m.group(1))
        rel = _clean(m.group(2)).lower().replace(" ", "_")
        if person and rel:
            _push(facts, f"my_{rel}", person, dia_id, 0.86, turn_index)

    # Stable role identity patterns.
    for m in re.finditer(
        r"\b(?:i am|i'm)\s+(?:a|an)\s+(mother|father|parent|student|teacher|nurse|engineer|artist|designer|developer)\b",
        turn_text,
        flags=re.IGNORECASE,
    ):
        _push(facts, "identity_role", m.group(1), dia_id, 0.85, turn_index)

    return facts


def extract_persona_facts(visible_turns: List[Dict[str, str]], max_facts: int = 24) -> List[Dict[str, object]]:
    """Extract lightweight persona facts from visible dialogue turns.

    This is intentionally rule-based for reproducibility and low cost.
    """
    facts: List[Dict[str, object]] = []

    for idx, turn in enumerate(visible_turns):
        dia_id = str(turn.get("dia_id", ""))
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        facts.extend(_extract_turn_facts(turn_text=text, dia_id=dia_id, turn_index=idx))

    # De-duplicate by (slot, value), keep the most recent turn occurrence.
    dedup: Dict[str, Dict[str, object]] = {}
    for fact in facts:
        key = f"{fact['slot']}::{str(fact['value']).lower()}"
        old = dedup.get(key)
        if old is None or int(fact.get("turn_index", -1)) >= int(old.get("turn_index", -1)):
            dedup[key] = fact

    dedup_facts = list(dedup.values())

    # Resolve singleton slots with latest-wins policy.
    singleton_latest: Dict[str, Dict[str, object]] = {}
    multi_facts: List[Dict[str, object]] = []
    for fact in dedup_facts:
        slot = str(fact.get("slot", ""))
        if slot in _SINGLETON_SLOTS:
            prev = singleton_latest.get(slot)
            if prev is None or int(fact.get("turn_index", -1)) >= int(prev.get("turn_index", -1)):
                singleton_latest[slot] = fact
        else:
            multi_facts.append(fact)

    out = multi_facts + list(singleton_latest.values())
    out.sort(key=lambda x: int(x.get("turn_index", -1)))

    # Remove helper field from outputs.
    for fact in out:
        fact.pop("turn_index", None)

    if max_facts > 0:
        out = out[-max_facts:]
    return out
