from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from persona_loop.llm.base_llm import BaseLLM


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" .,!?:;\t\n\r")
    return text


def _clamp_confidence(value: object, default: float = 0.6) -> float:
    try:
        v = float(value)
    except Exception:  # noqa: BLE001
        v = default
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _infer_polarity(text: str) -> int:
    t = str(text or "").lower()
    neg_markers = [" didn't ", " did not ", " never ", " no longer ", " don't ", " do not ", " not "]
    pad = f" {t} "
    return -1 if any(m in pad for m in neg_markers) else 1


def _extract_time_text(text: str) -> str:
    t = str(text or "").strip()
    if not t:
        return ""
    patterns = [
        r"\b(yesterday|today|tonight|last\s+night|this\s+morning)\b",
        r"\b(last|next|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month|year|weekend)\b",
        r"\b\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b",
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
    ]
    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            return _clean(m.group(0)).lower()
    return ""


def _normalize_time_text(time_text: str) -> str:
    t = _clean(str(time_text or "")).lower()
    if not t:
        return ""
    return re.sub(r"\s+", " ", t)


def _lower_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _make_fact(
    slot: str,
    value: str,
    dia_id: str,
    confidence: float,
    owner: str = "",
    polarity: int = 1,
    time_text: str = "",
    time_norm: str = "",
) -> Dict[str, object]:
    value = _clean(value)
    norm_conf = _clamp_confidence(confidence)
    owner_norm = _clean(owner) or "unknown"
    polarity_norm = -1 if int(polarity) < 0 else 1
    time_text_norm = _clean(time_text).lower()
    time_norm_norm = _normalize_time_text(time_norm or time_text_norm)
    out = {
        "slot": slot,
        "value": value,
        "fact_text": f"no longer {slot.replace('_', ' ')}: {value}" if polarity_norm < 0 else f"{slot.replace('_', ' ')}: {value}",
        "dia_id": dia_id,
        "confidence": norm_conf,
        "extract_confidence": norm_conf,
        "owner": owner_norm,
        "polarity": polarity_norm,
        "time_text": time_text_norm,
        "time_norm": time_norm_norm,
        "source": "rule",
    }
    return out


_SINGLETON_SLOTS = {
    "relationship_status",
    "occupation",
    "age",
    "location",
    "pet_name",
    "my_goal",
    "identity_role",
}

_REL_STATUS_MAP = {
    "single": "single",
    "married": "married",
    "divorced": "divorced",
    "engaged": "engaged",
    "widowed": "widowed",
}

_OCCUPATION_MAP = {
    "software engineer": "software engineer",
    "developer": "developer",
    "nurse": "nurse",
    "teacher": "teacher",
    "student": "student",
    "artist": "artist",
    "designer": "designer",
    "counselor": "counselor",
    "counselling": "counselor",
    "mental health": "mental health",
}

_KNOWN_SLOTS = {
    "relationship_status",
    "occupation",
    "location",
    "likes",
    "dislikes",
    "identity_role",
    "life_event",
    "recent_experience",
    "age",
    "has",
    "my_goal",
}


def _normalize_value(slot: str, value: str) -> str:
    v = _clean(value)
    lv = v.lower()

    if slot == "relationship_status":
        return _REL_STATUS_MAP.get(lv, lv)

    if slot in {"occupation", "identity_role"}:
        for k, mapped in _OCCUPATION_MAP.items():
            if k in lv:
                return mapped
        return lv

    if slot in {"location", "pet_name"}:
        # Keep proper casing for readability while preserving canonical space.
        return " ".join(x.capitalize() for x in lv.split())

    return v


def _push(
    out: List[Dict[str, object]],
    slot: str,
    value: str,
    dia_id: str,
    confidence: float,
    turn_index: int,
    owner: str = "",
    source_text: str = "",
) -> None:
    value = _normalize_value(slot=slot, value=value)
    if not value:
        return
    source_blob = source_text or value
    polarity = _infer_polarity(source_blob)
    time_text = _extract_time_text(source_blob)
    fact = _make_fact(
        slot=slot,
        value=value,
        dia_id=dia_id,
        confidence=confidence,
        owner=owner,
        polarity=polarity,
        time_text=time_text,
        time_norm=_normalize_time_text(time_text),
    )
    fact["turn_index"] = turn_index
    out.append(fact)


def _extract_turn_facts(turn_text: str, dia_id: str, turn_index: int, speaker: str = "") -> List[Dict[str, object]]:
    facts: List[Dict[str, object]] = []

    # Relationship status.
    for m in re.finditer(
        r"\b(?:i am|i'm)\s+(single|married|divorced|engaged|widowed)\b",
        turn_text,
        flags=re.IGNORECASE,
    ):
        _push(facts, "relationship_status", m.group(1), dia_id, 0.92, turn_index, owner=speaker, source_text=turn_text)

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
        _push(facts, "occupation", value, dia_id, 0.72, turn_index, owner=speaker, source_text=turn_text)

    # Age.
    for m in re.finditer(r"\b(?:i am|i'm)\s+(\d{1,2})\s*(?:years old|yo)?\b", turn_text, flags=re.IGNORECASE):
        _push(facts, "age", m.group(1), dia_id, 0.9, turn_index, owner=speaker, source_text=turn_text)

    # Location.
    for m in re.finditer(
        r"\b(?:i live in|i'm from|i am from)\s+([a-z][a-z\- ]{1,40})\b",
        turn_text,
        flags=re.IGNORECASE,
    ):
        _push(facts, "location", m.group(1), dia_id, 0.86, turn_index, owner=speaker, source_text=turn_text)

    # Pet / relation indicators.
    for m in re.finditer(r"\bmy\s+pet\s+([A-Z][a-z]+)\b", turn_text):
        _push(facts, "pet_name", m.group(1), dia_id, 0.94, turn_index, owner=speaker, source_text=turn_text)

    for m in re.finditer(r"\b([A-Z][a-z]+)\s+is\s+my\s+pet\b", turn_text):
        _push(facts, "pet_name", m.group(1), dia_id, 0.94, turn_index, owner=speaker, source_text=turn_text)

    # Preferences.
    for m in re.finditer(r"\bi\s+(?:like|love|enjoy|prefer)\s+([^.!?]{2,80})", turn_text, flags=re.IGNORECASE):
        _push(facts, "likes", m.group(1), dia_id, 0.78, turn_index, owner=speaker, source_text=turn_text)

    for m in re.finditer(r"\bi\s+(?:hate|dislike)\s+([^.!?]{2,80})", turn_text, flags=re.IGNORECASE):
        _push(facts, "dislikes", m.group(1), dia_id, 0.78, turn_index, owner=speaker, source_text=turn_text)

    # Possessions / relations.
    for m in re.finditer(r"\bmy\s+([a-z ]{2,24})\s+is\s+([^.!?]{2,80})", turn_text, flags=re.IGNORECASE):
        key = _clean(m.group(1)).lower().replace(" ", "_")
        if key:
            _push(facts, f"my_{key}", m.group(2), dia_id, 0.84, turn_index, owner=speaker, source_text=turn_text)

    for m in re.finditer(r"\bi\s+(?:have|own|got)\s+([^.!?]{2,80})", turn_text, flags=re.IGNORECASE):
        _push(facts, "has", m.group(1), dia_id, 0.76, turn_index, owner=speaker, source_text=turn_text)

    # Personal experiences that often encode stable identity context.
    for m in re.finditer(r"\bi\s+went\s+to\s+([^.!?]{2,80})", turn_text, flags=re.IGNORECASE):
        _push(facts, "recent_experience", m.group(1), dia_id, 0.74, turn_index, owner=speaker, source_text=turn_text)

    for m in re.finditer(r"\bi\s+celebrated\s+([^.!?]{2,80})", turn_text, flags=re.IGNORECASE):
        _push(facts, "life_event", m.group(1), dia_id, 0.74, turn_index, owner=speaker, source_text=turn_text)

    for m in re.finditer(r"\b([A-Z][a-z]+)\s+is\s+my\s+([a-z ]{2,24})\b", turn_text):
        person = _clean(m.group(1))
        rel = _clean(m.group(2)).lower().replace(" ", "_")
        if person and rel:
            _push(facts, f"my_{rel}", person, dia_id, 0.86, turn_index, owner=speaker, source_text=turn_text)

    # Stable role identity patterns.
    for m in re.finditer(
        r"\b(?:i am|i'm)\s+(?:a|an)\s+(mother|father|parent|student|teacher|nurse|engineer|artist|designer|developer)\b",
        turn_text,
        flags=re.IGNORECASE,
    ):
        _push(facts, "identity_role", m.group(1), dia_id, 0.85, turn_index, owner=speaker, source_text=turn_text)

    return facts


def extract_persona_facts_with_stats(
    visible_turns: List[Dict[str, str]], max_facts: int = 24
) -> Dict[str, object]:
    """Extract persona facts with audit stats for reproducibility."""
    facts: List[Dict[str, object]] = []

    for idx, turn in enumerate(visible_turns):
        dia_id = str(turn.get("dia_id", ""))
        text = str(turn.get("text", "")).strip()
        speaker = str(turn.get("speaker", "")).strip()
        if not text:
            continue
        facts.extend(_extract_turn_facts(turn_text=text, dia_id=dia_id, turn_index=idx, speaker=speaker))

    raw_candidates = len(facts)

    # De-duplicate by semantic key (owner, slot, value, time_norm), keep latest.
    # Polarity is intentionally excluded from the key so that a negation of the same
    # value (e.g. "no longer likes hiking") supersedes the earlier affirmation via
    # latest-wins, instead of both surviving as conflicting facts.
    dedup: Dict[str, Dict[str, object]] = {}
    for fact in facts:
        key = (
            f"{str(fact.get('owner', 'unknown')).lower()}::"
            f"{fact['slot']}::{str(fact['value']).lower()}::"
            f"{str(fact.get('time_norm', '')).lower()}"
        )
        old = dedup.get(key)
        if old is None:
            fact["first_seen_turn"] = int(fact.get("turn_index", -1))
            fact["last_seen_turn"] = int(fact.get("turn_index", -1))
            fact["mentions"] = 1
            fact["evidence_dia_ids"] = [str(fact.get("dia_id", ""))]
            dedup[key] = fact
            continue

        cur_turn = int(fact.get("turn_index", -1))
        old["first_seen_turn"] = min(int(old.get("first_seen_turn", cur_turn)), cur_turn)
        old["last_seen_turn"] = max(int(old.get("last_seen_turn", cur_turn)), cur_turn)
        old["mentions"] = int(old.get("mentions", 1)) + 1
        dia_ids = list(old.get("evidence_dia_ids", []))
        dia = str(fact.get("dia_id", ""))
        if dia and dia not in dia_ids:
            dia_ids.append(dia)
        old["evidence_dia_ids"] = dia_ids
        if cur_turn >= int(old.get("turn_index", -1)):
            old["turn_index"] = cur_turn
            old["dia_id"] = dia
            old["confidence"] = fact.get("confidence", old.get("confidence", 0.0))
            old["extract_confidence"] = fact.get("extract_confidence", old.get("extract_confidence", 0.0))
            old["polarity"] = fact.get("polarity", old.get("polarity", 1))
            old["time_text"] = fact.get("time_text", old.get("time_text", ""))
            old["time_norm"] = fact.get("time_norm", old.get("time_norm", ""))

    dedup_facts = list(dedup.values())
    dedup_candidates = len(dedup_facts)

    # Resolve singleton slots with latest-wins policy.
    singleton_latest: Dict[str, Dict[str, object]] = {}
    singleton_values: Dict[str, set[str]] = {}
    multi_facts: List[Dict[str, object]] = []
    for fact in dedup_facts:
        slot = str(fact.get("slot", ""))
        owner = str(fact.get("owner", "unknown")).strip().lower() or "unknown"
        singleton_key = f"{owner}::{slot}"
        if slot in _SINGLETON_SLOTS:
            singleton_values.setdefault(singleton_key, set()).add(str(fact.get("value", "")).lower())
            prev = singleton_latest.get(singleton_key)
            if prev is None or int(fact.get("turn_index", -1)) >= int(prev.get("turn_index", -1)):
                singleton_latest[singleton_key] = fact
        else:
            multi_facts.append(fact)

    out = multi_facts + list(singleton_latest.values())
    out.sort(key=lambda x: int(x.get("last_seen_turn", x.get("turn_index", -1))))

    singleton_conflict_slots = sum(1 for vals in singleton_values.values() if len(vals) > 1)
    singleton_conflict_values = sum(max(0, len(vals) - 1) for vals in singleton_values.values())
    unique_slots = len({f"{str(f.get('owner', 'unknown')).lower()}::{str(f.get('slot', ''))}" for f in out if str(f.get("slot", ""))})

    # Remove helper field from outputs.
    for fact in out:
        fact.pop("turn_index", None)

    if max_facts > 0:
        out = out[-max_facts:]

    return {
        "facts": out,
        "stats": {
            "raw_candidates": raw_candidates,
            "dedup_candidates": dedup_candidates,
            "unique_slots": unique_slots,
            "singleton_conflict_slots": singleton_conflict_slots,
            "singleton_conflict_values": singleton_conflict_values,
            "unknown_owner_count": int(
                sum(1 for f in out if str(f.get("owner", "unknown")).strip().lower() in {"", "unknown"})
            ),
        },
    }


def extract_persona_facts(visible_turns: List[Dict[str, str]], max_facts: int = 24) -> List[Dict[str, object]]:
    """Backward-compatible wrapper returning only facts."""
    return list(extract_persona_facts_with_stats(visible_turns=visible_turns, max_facts=max_facts)["facts"])


def extract_persona_facts_llm_with_stats(
    visible_turns: List[Dict[str, str]],
    llm: BaseLLM,
    max_facts: int = 24,
) -> Dict[str, object]:
    """Pure LLM extraction — no regex rules, just prompt + grounding check."""
    result = _llm_extract_persona_facts(visible_turns=visible_turns, llm=llm, max_facts=max_facts)
    facts = list(result.get("facts", []))
    debug = dict(result.get("debug", {}))
    stats = {
        "raw_candidates": int(debug.get("llm_candidate_count", 0)),
        "dedup_candidates": int(debug.get("llm_valid_fact_count", 0)),
        "unique_slots": len({str(f.get("slot", "")) for f in facts}),
        "singleton_conflict_slots": 0,
        "singleton_conflict_values": 0,
        "unknown_owner_count": int(
            sum(1 for f in facts if str(f.get("owner", "unknown")).strip().lower() in {"", "unknown"})
        ),
        "hybrid_used_llm": True,
        "llm_raw_len": int(debug.get("llm_raw_len", 0)),
        "llm_json_parsed": bool(debug.get("llm_json_parsed", False)),
        "llm_structured_success": bool(debug.get("llm_structured_success", False)),
        "llm_candidate_count": int(debug.get("llm_candidate_count", 0)),
        "llm_valid_fact_count": int(debug.get("llm_valid_fact_count", 0)),
        "llm_fallback_used": bool(debug.get("llm_fallback_used", False)),
        "llm_repair_used": bool(debug.get("llm_repair_used", False)),
        "llm_repair_success": bool(debug.get("llm_repair_success", False)),
    }
    return {"facts": facts, "stats": stats}


def _build_llm_extract_prompt(max_facts: int) -> str:
    return (
        "Task: extract stable persona facts from dialogue. "
        "Output must be valid JSON and nothing else (no markdown, no explanation). "
        "Schema: {\"facts\":[{\"slot\":\"...\",\"value\":\"...\",\"dia_id\":\"...\",\"owner\":\"...\",\"confidence\":0.0,\"polarity\":1,\"time_text\":\"...\",\"time_norm\":\"...\"}]}. "
        "Keep slot names concise: relationship_status, occupation, location, likes, dislikes, "
        "identity_role, life_event, recent_experience, age, has, my_goal. "
        "Always provide owner if inferable from speaker or dia_id. Use owner='unknown' when uncertain. "
        "Use polarity=-1 for negated facts, polarity=1 otherwise. "
        "Only include facts explicitly supported by context. "
        "Prefer adding new slot types that are not obvious duplicates. "
        f"Return at most {max_facts} facts."
    )


def _build_llm_repair_prompt(max_facts: int) -> str:
    return (
        "Rewrite the provided text into STRICT JSON only with schema "
        '{"facts":[{"slot":"...","value":"...","dia_id":"...","owner":"...","confidence":0.0,"polarity":1,"time_text":"...","time_norm":"..."}]}. '
        "Do not add explanation. Keep at most "
        f"{max_facts} facts."
    )


def _extract_json_object(text: str) -> Optional[Dict[str, object]]:
    s = (text or "").strip()
    if not s:
        return None

    # Strip common markdown wrappers.
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s).strip()

    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else None
    except Exception:  # noqa: BLE001
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(s[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:  # noqa: BLE001
        pass

    # Fallback: tolerate single-quoted pseudo-JSON when structure is simple.
    candidate = s[start : end + 1] if start >= 0 and end > start else s
    candidate = re.sub(r"(?<!\\)'", '"', candidate)
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _normalize_slot_name(slot_raw: str) -> str:
    slot = _clean(slot_raw).lower().replace(" ", "_")
    aliases = {
        "role": "identity_role",
        "identity": "identity_role",
        "job": "occupation",
        "profession": "occupation",
        "experience": "recent_experience",
        "event": "life_event",
        "goal": "my_goal",
    }
    slot = aliases.get(slot, slot)
    return slot


def _extract_facts_from_text_fallback(raw: str, max_facts: int) -> List[Dict[str, object]]:
    """Fallback parser for non-JSON responses like 'slot: value' lines."""
    out: List[Dict[str, object]] = []
    text = (raw or "").strip()
    if not text:
        return out

    # Capture patterns like "occupation: teacher" possibly with dia_id hints.
    pattern = re.compile(r"([a-zA-Z_ ]{2,30})\s*[:=-]\s*([^\n;,]{1,140})")
    for m in pattern.finditer(text):
        slot = _normalize_slot_name(m.group(1))
        if slot not in _KNOWN_SLOTS and not slot.startswith("my_"):
            continue
        value = _normalize_value(slot=slot, value=m.group(2))
        if not value:
            continue
        tail = text[m.end() : m.end() + 60]
        dia_match = re.search(r"(D\d+:\d+)", tail)
        dia_id = dia_match.group(1) if dia_match else ""
        fact = _make_fact(slot=slot, value=value, dia_id=dia_id, confidence=0.55, owner="unknown")
        fact["source"] = "llm"
        out.append(fact)
        if max_facts > 0 and len(out) >= max_facts:
            break
    return out


def _canonical_owner(owner_raw: str, participant_names: List[str]) -> str:
    owner = _clean(str(owner_raw or ""))
    if not owner:
        return "unknown"
    by_lc = {name.lower(): name for name in participant_names if name}
    return by_lc.get(owner.lower(), "unknown")


def _token_overlap_ratio(a: str, b: str) -> float:
    ta = set(_lower_tokens(a))
    tb = set(_lower_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta.intersection(tb)) / max(1, len(ta))


def _is_llm_fact_supported(slot: str, value: str, source_text: str) -> bool:
    # High-precision cheap guard to remove hallucinated facts.
    v = _clean(value)
    s = str(source_text or "")
    if not v or not s:
        return False
    if v.lower() in s.lower():
        return True
    # Short values are prone to accidental overlap; require stricter match.
    min_overlap = 0.60 if len(_lower_tokens(v)) <= 2 else 0.28
    if slot in {"age", "relationship_status", "location", "pet_name"}:
        min_overlap = max(min_overlap, 0.5)
    return _token_overlap_ratio(v, s) >= min_overlap


def _llm_extract_persona_facts(
    visible_turns: List[Dict[str, str]], llm: BaseLLM, max_facts: int
) -> Dict[str, object]:
    context_lines: List[str] = []
    dia_owner: Dict[str, str] = {}
    dia_text: Dict[str, str] = {}
    participant_names: List[str] = []
    for turn in visible_turns:
        dia_id = str(turn.get("dia_id", "")).strip()
        speaker = str(turn.get("speaker", "")).strip()
        text = str(turn.get("text", "")).strip()
        if dia_id and text:
            speaker_tag = f"[speaker={speaker}]" if speaker else "[speaker=unknown]"
            context_lines.append(f"{dia_id} {speaker_tag} {text}")
            if speaker:
                dia_owner[dia_id] = speaker
                participant_names.append(speaker)
            dia_text[dia_id] = text
    participant_names = sorted({n for n in participant_names if n})
    context = "\n".join(context_lines)
    if not context:
        return {
            "facts": [],
            "debug": {
                "llm_raw_len": 0,
                "llm_json_parsed": False,
                "llm_candidate_count": 0,
                "llm_valid_fact_count": 0,
                "llm_fallback_used": False,
                "llm_repair_used": False,
                "llm_repair_success": False,
            },
        }

    parsed = llm.generate_json(prompt=_build_llm_extract_prompt(max_facts=max_facts), context=context)
    raw = ""
    raw_len = 0
    structured_success = parsed is not None
    if not parsed:
        raw = llm.generate(prompt=_build_llm_extract_prompt(max_facts=max_facts), context=context)
        raw_len = len(raw or "")
        parsed = _extract_json_object(raw)
    repair_used = False
    repair_success = False
    if not parsed:
        repair_used = True
        repaired = llm.generate(prompt=_build_llm_repair_prompt(max_facts=max_facts), context=raw)
        parsed = _extract_json_object(repaired)
        repair_success = parsed is not None

    if not parsed:
        fb = _extract_facts_from_text_fallback(raw=raw, max_facts=max_facts)
        return {
            "facts": fb,
            "debug": {
                "llm_raw_len": raw_len,
                "llm_json_parsed": False,
                "llm_structured_success": structured_success,
                "llm_candidate_count": len(fb),
                "llm_valid_fact_count": len(fb),
                "llm_fallback_used": len(fb) > 0,
                "llm_repair_used": repair_used,
                "llm_repair_success": repair_success,
            },
        }
    arr = parsed.get("facts", [])
    if not isinstance(arr, list):
        return {
            "facts": [],
            "debug": {
                "llm_raw_len": raw_len,
                "llm_json_parsed": True,
                "llm_structured_success": structured_success,
                "llm_candidate_count": 0,
                "llm_valid_fact_count": 0,
                "llm_fallback_used": False,
                "llm_repair_used": repair_used,
                "llm_repair_success": repair_success,
            },
        }

    out: List[Dict[str, object]] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        slot = _clean(str(item.get("slot", ""))).lower().replace(" ", "_")
        value = _normalize_value(slot=slot, value=str(item.get("value", "")))
        if not slot or not value:
            continue
        dia_id = _clean(str(item.get("dia_id", "")))

        # Fact must be explicitly grounded to a visible dialogue turn.
        if not dia_id or dia_id not in dia_text:
            continue

        owner = _clean(str(item.get("owner", "") or item.get("speaker", "")))
        if not owner:
            owner = dia_owner.get(dia_id, "")
        owner = _canonical_owner(owner_raw=owner, participant_names=participant_names)

        confidence = _clamp_confidence(item.get("confidence", 0.6), default=0.6)

        # If turn speaker is known, softly enforce owner consistency instead of dropping.
        if dia_id in dia_owner:
            speaker_owner = dia_owner.get(dia_id, "")
            if owner == "unknown":
                owner = speaker_owner
                confidence = max(0.0, confidence * 0.92)
            elif owner != speaker_owner:
                owner = speaker_owner
                confidence = max(0.0, confidence * 0.85)

        # Lightweight semantic verification against the source turn text.
        if not _is_llm_fact_supported(slot=slot, value=value, source_text=dia_text.get(dia_id, "")):
            confidence = max(0.0, confidence * 0.78)

        # Drop only very weak candidates; moderate/strong ones are deferred to downstream thresholding.
        if confidence < 0.35:
            continue

        polarity = -1 if int(item.get("polarity", 1) or 1) < 0 else 1
        time_text = _clean(str(item.get("time_text", ""))).lower()
        if not time_text:
            time_text = _extract_time_text(value)
        time_norm = _normalize_time_text(str(item.get("time_norm", "")) or time_text)
        fact = _make_fact(
            slot=slot,
            value=value,
            dia_id=dia_id,
            confidence=confidence,
            owner=owner,
            polarity=polarity,
            time_text=time_text,
            time_norm=time_norm,
        )
        fact["source"] = "llm"
        out.append(fact)

    out = out[: max(0, max_facts)]
    return {
        "facts": out,
        "debug": {
            "llm_raw_len": raw_len,
            "llm_json_parsed": True,
            "llm_structured_success": structured_success,
            "llm_candidate_count": len(arr),
            "llm_valid_fact_count": len(out),
            "llm_fallback_used": False,
            "llm_repair_used": repair_used,
            "llm_repair_success": repair_success,
        },
    }


def extract_persona_facts_hybrid_with_stats(
    visible_turns: List[Dict[str, str]],
    llm: Optional[BaseLLM],
    max_facts: int = 24,
    min_rule_facts: int = 3,
    llm_max_facts: int = 8,
) -> Dict[str, object]:
    """Rule-first extraction with optional LLM recall boost."""
    base = extract_persona_facts_with_stats(visible_turns=visible_turns, max_facts=max_facts)
    rule_facts = list(base.get("facts", []))
    stats = dict(base.get("stats", {}))

    if llm is None or len(rule_facts) >= min_rule_facts or llm_max_facts <= 0:
        stats["hybrid_used_llm"] = False
        stats["llm_added_facts"] = 0
        stats["llm_raw_len"] = 0
        stats["llm_json_parsed"] = False
        stats["llm_structured_success"] = False
        stats["llm_candidate_count"] = 0
        stats["llm_valid_fact_count"] = 0
        stats["llm_fallback_used"] = False
        stats["llm_repair_used"] = False
        stats["llm_repair_success"] = False
        return {"facts": rule_facts, "stats": stats}

    llm_result = _llm_extract_persona_facts(
        visible_turns=visible_turns,
        llm=llm,
        max_facts=llm_max_facts,
    )
    llm_facts = list(llm_result.get("facts", []))
    llm_debug = dict(llm_result.get("debug", {}))

    seen = {
        f"{str(f.get('owner', 'unknown')).lower()}::{str(f.get('slot', ''))}::{str(f.get('value', '')).lower()}::{int(f.get('polarity', 1))}::{str(f.get('time_norm', '')).lower()}"
        for f in rule_facts
    }
    merged = list(rule_facts)
    added = 0
    for fact in llm_facts:
        key = (
            f"{str(fact.get('owner', 'unknown')).lower()}::"
            f"{str(fact.get('slot', ''))}::{str(fact.get('value', '')).lower()}::"
            f"{int(fact.get('polarity', 1))}::{str(fact.get('time_norm', '')).lower()}"
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(fact)
        added += 1

    if max_facts > 0:
        merged = merged[-max_facts:]

    stats["hybrid_used_llm"] = True
    stats["llm_added_facts"] = added
    stats["llm_raw_len"] = int(llm_debug.get("llm_raw_len", 0))
    stats["llm_json_parsed"] = bool(llm_debug.get("llm_json_parsed", False))
    stats["llm_structured_success"] = bool(llm_debug.get("llm_structured_success", False))
    stats["llm_candidate_count"] = int(llm_debug.get("llm_candidate_count", 0))
    stats["llm_valid_fact_count"] = int(llm_debug.get("llm_valid_fact_count", 0))
    stats["llm_fallback_used"] = bool(llm_debug.get("llm_fallback_used", False))
    stats["llm_repair_used"] = bool(llm_debug.get("llm_repair_used", False))
    stats["llm_repair_success"] = bool(llm_debug.get("llm_repair_success", False))
    return {"facts": merged, "stats": stats}
