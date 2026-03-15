from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
import time
from typing import Any, Dict, List, Optional

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from persona_loop.core.factories import create_agent
from persona_loop.core.factories import create_checker
from persona_loop.core.factories import create_llm
from persona_loop.core.factories import create_memory
from persona_loop.eval.nli_scorer import NLIScorer
from persona_loop.eval.persona_extractor import extract_persona_facts_hybrid_with_stats
from persona_loop.eval.persona_extractor import extract_persona_facts_with_stats
from persona_loop.eval.persona_metrics import compute_persona_metrics
from persona_loop.eval.qa_metrics import qa_scores


def load_locomo(path: Path) -> List[Dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit(root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_slice_entries(path: Path) -> List[Dict[str, object]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        entries = raw.get("entries", [])
        if not isinstance(entries, list):
            raise ValueError("Slice file entries must be a list")
        return entries
    if isinstance(raw, list):
        return raw
    raise ValueError("Slice file must be a list or an object with an 'entries' list")


def flatten_conversation(conv: Dict[str, object]) -> List[Dict[str, str]]:
    flat: List[Dict[str, str]] = []
    for idx in range(1, 100):
        key = f"session_{idx}"
        if key not in conv:
            break
        turns = conv[key]
        for turn in turns:
            dia_id = str(turn.get("dia_id", ""))
            text = str(turn.get("text", "")).strip()
            speaker = str(turn.get("speaker", "")).strip()
            if dia_id and text:
                row = {"dia_id": dia_id, "text": text}
                if speaker:
                    row["speaker"] = speaker
                flat.append(row)
    return flat


def build_history(turns: List[Dict[str, str]], max_turns: int) -> List[str]:
    items = [f"{t['dia_id']} {t['text']}" for t in turns]
    if max_turns <= 0:
        return items
    return items[-max_turns:]


def answer_question(
    agent,
    question: str,
    history: List[str],
    persona_lines: Optional[List[str]] = None,
) -> Dict[str, Any]:
    blocks: List[str] = []
    if persona_lines:
        blocks.extend([f"[PERSONA] {x}" for x in persona_lines])
    blocks.extend([f"[HISTORY] {x}" for x in history])
    context = "\n".join(blocks)
    return dict(agent.run_turn(prompt=question, context=context))


# ---------------------------------------------------------------------------
# Question-type classification and slot-relevance mapping
# ---------------------------------------------------------------------------

_QTYPE_TO_SLOTS: Dict[str, set] = {
    "time":       {"recent_experience", "life_event", "has"},
    "location":   {"location", "recent_experience", "life_event"},
    "identity":   {"identity_role", "occupation", "relationship_status", "age"},
    "preference": {"likes", "dislikes", "my_goal"},
    "status":     {"relationship_status", "identity_role"},
    "occupation": {"occupation", "identity_role"},
    "event":      {"life_event", "recent_experience", "has"},
    "goal":       {"my_goal", "life_event"},
}

_QTYPE_TO_INJECT_TOPK_DEFAULTS: Dict[str, int] = {
    "time": 3,
    "location": 3,
    "identity": 4,
    "preference": 5,
    "status": 4,
    "occupation": 4,
    "event": 4,
    "goal": 4,
}


def _classify_question_types(question: str) -> List[str]:
    """Return broad semantic question type labels for slot-relevance filtering."""
    q = question.lower()
    types: List[str] = []
    if re.search(
        r"\b(when|what time|what year|what month|what date|what day"
        r"|how long|how many (?:years|months|days|weeks))\b", q
    ):
        types.append("time")
    if re.search(
        r"\b(where|which (?:city|place|country|location)|location|lived|moved)\b", q
    ):
        types.append("location")
    if re.search(
        r"\b(who(?:'s|s)?|(?:what|what's) \w+(?:'s)?\s+(?:role|name|identity))"
        r"|\bidentity\b", q
    ):
        types.append("identity")
    if re.search(
        r"\b(like[sd]?|enjoy[sd]?|hate[sd]?|dislike[sd]?|favorite|feel about"
        r"|opinion|think about|prefer(?:red|s)?)\b", q
    ):
        types.append("preference")
    if re.search(
        r"\b(relationship|married|dating|divorced|single|partner|spouse"
        r"|boyfriend|girlfriend|husband|wife|engaged)\b", q
    ):
        types.append("status")
    if re.search(
        r"\b(job|career|work(?:ing)?|profession|occupation|study|studied"
        r"|major|degree|employ(?:ed|ment)?)\b", q
    ):
        types.append("occupation")
    if re.search(
        r"\b(did|happened|went|join(?:ed)?|attend(?:ed)?|visit(?:ed)?"
        r"|start(?:ed)?|finish(?:ed)?|complet(?:ed)?|graduate[d]?"
        r"|born|died|mov(?:ed)?|got|have|had|been|became|go to)\b", q
    ):
        types.append("event")
    if re.search(
        r"\b(want(?:s|ed)?|plan(?:n(?:ing|ed))?|trying|hoping"
        r"|goal|aim|intend(?:s|ed)?|looking forward)\b", q
    ):
        types.append("goal")
    return types


def _extract_question_subjects(question: str, candidate_names: List[str]) -> List[str]:
    q = str(question).lower()
    out: List[str] = []
    unique = sorted({str(n).strip() for n in candidate_names if str(n).strip()}, key=len, reverse=True)
    for name in unique:
        if re.search(rf"\b{re.escape(name.lower())}\b", q):
            out.append(name)
    return out


def _parse_qtype_topk_overrides(spec: str) -> Dict[str, int]:
    """Parse top-k overrides like 'time=2,event=4' for adaptive persona injection."""
    out: Dict[str, int] = {}
    for token in str(spec or "").split(","):
        token = token.strip()
        if not token or "=" not in token:
            continue
        key, val = token.split("=", 1)
        key = key.strip().lower()
        if key not in _QTYPE_TO_INJECT_TOPK_DEFAULTS:
            continue
        try:
            parsed = int(val.strip())
        except Exception:  # noqa: BLE001
            continue
        if parsed > 0:
            out[key] = parsed
    return out


def _resolve_persona_injection_topk(
    base_topk: int,
    question_types: List[str],
    adaptive_enabled: bool,
    qtype_topk_overrides: Dict[str, int],
) -> int:
    if not adaptive_enabled or not question_types:
        return max(1, int(base_topk))
    resolved = max(1, int(base_topk))
    for qt in question_types:
        key = str(qt).strip().lower()
        if not key:
            continue
        qt_topk = qtype_topk_overrides.get(key, _QTYPE_TO_INJECT_TOPK_DEFAULTS.get(key))
        if qt_topk is not None:
            resolved = min(resolved, max(1, int(qt_topk)))
    return resolved


def _fact_confidence(fact: Dict[str, object]) -> float:
    raw = fact.get("extract_confidence", fact.get("confidence", 0.0))
    try:
        return float(raw)
    except Exception:  # noqa: BLE001
        return 0.0


def _filter_persona_facts_for_question(
    fact_bank: List[Dict[str, object]],
    subjects: List[str],
    min_confidence: float,
    allow_unknown_owner: bool,
    single_owner_when_ambiguous: bool,
    allowed_slots: Optional[set] = None,
    slot_filter_min_facts: int = 2,
) -> List[Dict[str, object]]:
    eligible = [f for f in fact_bank if _fact_confidence(f) >= float(min_confidence)]
    if not allow_unknown_owner:
        eligible = [
            f
            for f in eligible
            if str(f.get("owner", "unknown")).strip().lower() not in {"", "unknown"}
        ]

    if not subjects:
        owner_filtered = eligible if eligible else fact_bank
        if single_owner_when_ambiguous and owner_filtered:
            # Ambiguous question subject: avoid mixing multi-owner persona lines.
            buckets: Dict[str, List[Dict[str, object]]] = {}
            for f in owner_filtered:
                owner = str(f.get("owner", "")).strip().lower()
                if not owner or owner == "unknown":
                    continue
                buckets.setdefault(owner, []).append(f)
            if buckets:
                best_owner = max(
                    buckets.keys(),
                    key=lambda o: (
                        len(buckets[o]),
                        sum(_fact_confidence(x) for x in buckets[o]) / max(1, len(buckets[o])),
                    ),
                )
                owner_filtered = buckets[best_owner]
    else:
        subject_lc = {s.lower() for s in subjects if s}
        owner_filtered = [
            f
            for f in eligible
            if str(f.get("owner", "unknown")).strip().lower() in subject_lc
        ]
    # If the question explicitly targets subject(s), do not back off to non-matching
    # owners. Cross-owner injection is a major source of subject swaps.

    if not allowed_slots:
        return owner_filtered

    # Slot-relevance filter: keep only facts whose slot is relevant to the question type.
    # Fall back to full owner-filtered list when too few slot-matched facts exist.
    slot_filtered = [
        f for f in owner_filtered
        if str(f.get("slot", "")).strip() in allowed_slots
    ]
    return slot_filtered if len(slot_filtered) >= slot_filter_min_facts else owner_filtered


def _question_has_negative_cue(question: str) -> bool:
    q = question.lower()
    return bool(re.search(r"\b(not|never|no longer|still|anymore|would|if .* hadn't)\b", q))


def _is_abstract_fact(slot: str, value: str) -> bool:
    lv = value.lower()
    abstract_slots = {"life_event", "has"}
    abstract_terms = {
        "feeling",
        "overwhelmed",
        "responsibilities",
        "accepted",
        "courage",
        "understanding",
        "empathy",
        "support",
        "confidence",
    }
    if slot not in abstract_slots:
        return False
    return any(t in lv for t in abstract_terms)


def _dedup_facts_keep_high_confidence(facts: List[Dict[str, object]]) -> List[Dict[str, object]]:
    best: Dict[tuple, Dict[str, object]] = {}
    for f in facts:
        key = (
            str(f.get("owner", "")).strip().lower(),
            str(f.get("slot", "")).strip().lower(),
            str(f.get("value", "")).strip().lower(),
        )
        if key not in best or _fact_confidence(f) > _fact_confidence(best[key]):
            best[key] = f
    return list(best.values())


def _fact_has_time_anchor(fact: Dict[str, object]) -> bool:
    return bool(str(fact.get("time_text", "")).strip() or str(fact.get("time_norm", "")).strip())


def _is_temporal_slot(slot: str) -> bool:
    s = str(slot).strip().lower()
    if not s:
        return False
    temporal_slots = {
        "recent_experience",
        "life_event",
        "my_goal",
        "my_plan",
        "my_vision",
    }
    if s in temporal_slots:
        return True
    return any(t in s for t in ("time", "date", "year", "month", "day"))


def _preinject_conflict_gate(
    facts: List[Dict[str, object]],
    scope: str,
) -> tuple[List[Dict[str, object]], Dict[str, int]]:
    """Resolve owner-slot multi-value conflicts before ranking/injection.

    For each conflicting owner+slot group, keep one best candidate by confidence
    (and by time anchor / rule source as tie breakers) to reduce contradiction pressure.
    """
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for f in facts:
        key = (
            str(f.get("owner", "")).strip().lower(),
            str(f.get("slot", "")).strip().lower(),
        )
        grouped.setdefault(key, []).append(f)

    kept: List[Dict[str, object]] = []
    groups_resolved = 0
    facts_removed = 0

    for (_, slot), bucket in grouped.items():
        values = {
            str(x.get("value", "")).strip().lower()
            for x in bucket
            if str(x.get("value", "")).strip()
        }
        has_conflict = len(values) > 1
        is_temporal_group = _is_temporal_slot(slot) or any(_fact_has_time_anchor(x) for x in bucket)
        should_resolve = has_conflict and (scope == "all" or (scope == "time" and is_temporal_group))

        if not should_resolve:
            kept.extend(bucket)
            continue

        groups_resolved += 1

        def _score(f: Dict[str, object]) -> tuple:
            source = str(f.get("source", "")).strip().lower()
            source_rule = 1 if source == "rule" else 0
            return (
                _fact_confidence(f),
                1 if _fact_has_time_anchor(f) else 0,
                source_rule,
            )

        best = max(bucket, key=_score)
        kept.append(best)
        facts_removed += max(0, len(bucket) - 1)

    return _dedup_facts_keep_high_confidence(kept), {
        "groups_resolved": int(groups_resolved),
        "facts_removed": int(facts_removed),
    }


def _risk_filter_persona_facts(
    facts: List[Dict[str, object]],
    question: str,
    question_types: List[str],
    min_llm_confidence: float,
    drop_negative: bool,
    drop_abstract: bool,
    drop_relative_time: bool,
) -> List[Dict[str, object]]:
    qtypes = set(question_types)
    allow_abstract_for = {"event", "goal", "preference"}
    keep: List[Dict[str, object]] = []
    neg_q = _question_has_negative_cue(question)
    relative_time_re = re.compile(
        r"\b(yesterday|today|tonight|last\s+(night|week|month|year|weekend|friday|monday)|"
        r"this\s+(week|month|year|weekend)|"
        r"\d+\s+(day|days|week|weeks|month|months|year|years)\s+ago|"
        r"a\s+few\s+(days|weeks|months)\s+ago|recently)\b",
        re.IGNORECASE,
    )

    for f in facts:
        slot = str(f.get("slot", "")).strip()
        value = str(f.get("value", "")).strip()
        source = str(f.get("source", "")).strip().lower()
        conf = _fact_confidence(f)
        polarity = int(f.get("polarity", 1))

        if source == "llm" and conf < float(min_llm_confidence):
            continue
        if drop_negative and polarity < 0 and not neg_q:
            continue
        if drop_abstract and _is_abstract_fact(slot=slot, value=value) and not (qtypes & allow_abstract_for):
            continue
        if drop_relative_time:
            tt = str(f.get("time_text", "")).strip()
            tn = str(f.get("time_norm", "")).strip()
            hay = " ".join([slot, value, tt, tn]).strip()
            if hay and relative_time_re.search(hay):
                continue
        keep.append(f)

    return _dedup_facts_keep_high_confidence(keep)


def _rank_persona_facts_for_question(
    facts: List[Dict[str, object]],
    question_types: List[str],
    rank_target_types: Optional[set] = None,
    min_candidates: int = 2,
    min_score_gap: float = 0.0,
) -> tuple[List[Dict[str, object]], bool, float]:
    if not facts:
        return facts, False, 0.0

    qtypes = set(question_types)
    if rank_target_types is not None and qtypes and not (qtypes & rank_target_types):
        return facts, False, 0.0
    slot_priority = {
        "time": {"recent_experience", "life_event", "goal"},
        "location": {"location", "life_event", "recent_experience"},
        "identity": {"identity_role", "relationship_status", "occupation"},
        "preference": {"likes", "dislikes", "my_favorite"},
        "status": {"relationship_status", "has", "identity_role"},
        "occupation": {"occupation", "identity_role"},
        "event": {"recent_experience", "life_event", "goal"},
        "goal": {"goal", "my_goal", "my_plan", "my_vision"},
    }

    preferred_slots: set = set()
    for qt in qtypes:
        preferred_slots.update(slot_priority.get(qt, set()))

    def _score_tuple_and_value(f: Dict[str, object]) -> tuple[tuple, float]:
        slot = str(f.get("slot", "")).strip()
        has_time_anchor = 1 if (str(f.get("time_text", "")).strip() or str(f.get("time_norm", "")).strip()) else 0
        slot_hit = 1 if (preferred_slots and slot in preferred_slots) else 0
        source = str(f.get("source", "")).strip().lower()
        source_rule = 1 if source == "rule" else 0
        conf = _fact_confidence(f)
        # Keep the original ranking precedence to preserve prior behavior.
        if "time" in qtypes:
            score_tuple = (slot_hit, has_time_anchor, conf, source_rule)
            score_value = slot_hit + 0.1 * has_time_anchor + 0.01 * conf + 0.001 * source_rule
            return score_tuple, score_value
        score_tuple = (slot_hit, conf, has_time_anchor, source_rule)
        score_value = slot_hit + 0.1 * conf + 0.01 * has_time_anchor + 0.001 * source_rule
        return score_tuple, score_value

    def _score_tuple(f: Dict[str, object]) -> tuple:
        return _score_tuple_and_value(f)[0]

    def _score_value(f: Dict[str, object]) -> float:
        return _score_tuple_and_value(f)[1]

    ranked = sorted(facts, key=_score_tuple, reverse=True)
    if len(ranked) < max(1, int(min_candidates)):
        return facts, False, 0.0

    top_score = _score_value(ranked[0])
    second_score = _score_value(ranked[1]) if len(ranked) > 1 else top_score
    score_gap = max(0.0, top_score - second_score)
    if score_gap < float(min_score_gap):
        return facts, False, score_gap

    return ranked, True, score_gap


def _format_persona_lines(
    fact_bank: List[Dict[str, object]],
    top_k: int = 6,
    line_style: str = "tagged",
) -> List[str]:
    out: List[str] = []
    for fact in fact_bank[: max(0, top_k)]:
        slot = str(fact.get("slot", "")).strip()
        value = str(fact.get("value", "")).strip()
        owner = str(fact.get("owner", "")).strip()
        if line_style == "plain":
            # Plain style minimizes scaffolding tokens to reduce style over-conditioning.
            if value:
                out.append(value)
            continue
        if slot and value:
            line = f"{slot}: {value}"
            out.append(f"[{owner}] {line}" if owner else line)
        elif value:
            out.append(f"[{owner}] {value}" if owner else value)
    return out


def _dia_id_from_history_item(item: str) -> str:
    parts = item.split(" ", 1)
    return parts[0] if parts else ""


def _is_evidence_visible(history: List[str], evidence_ids: List[str]) -> bool:
    if not evidence_ids:
        return False
    visible_ids = {_dia_id_from_history_item(item) for item in history}
    return any(e in visible_ids for e in evidence_ids)


def _build_eval_history(
    eval_mode: str,
    question: str,
    visible_turns: List[Dict[str, str]],
    evidence_ids: List[str],
    max_turns: int,
    retrieval_topk: int,
) -> List[str]:
    if eval_mode == "open_book":
        return build_history(visible_turns, max_turns=max_turns)

    if eval_mode == "hide_evidence":
        filtered_turns = [t for t in visible_turns if str(t.get("dia_id", "")) not in set(evidence_ids)]
        return build_history(filtered_turns, max_turns=max_turns)

    # eval_mode == memory_only
    memory = create_memory(memory_type="chroma")
    if memory is None:
        return []

    for turn in visible_turns:
        dia_id = str(turn.get("dia_id", ""))
        text = str(turn.get("text", "")).strip()
        if dia_id and text:
            memory.add(text=f"{dia_id} {text}")

    hits = memory.search(query=question, top_k=max(1, retrieval_topk))
    return [str(h).strip() for h in hits if str(h).strip()]


def _enrich_fact_owners(fact_bank: List[Dict[str, object]], visible_turns: List[Dict[str, str]]) -> List[Dict[str, object]]:
    dia_to_speaker = {
        str(t.get("dia_id", "")).strip(): str(t.get("speaker", "")).strip()
        for t in visible_turns
        if str(t.get("dia_id", "")).strip()
    }
    out: List[Dict[str, object]] = []
    for fact in fact_bank:
        row = dict(fact)
        owner = str(row.get("owner", "")).strip()
        if not owner:
            dia = str(row.get("dia_id", "")).strip()
            owner = dia_to_speaker.get(dia, "")
            if owner:
                row["owner"] = owner
        out.append(row)
    return out


def print_progress_bar(current: int, total: int, start_time: float, width: int = 30) -> None:
    if total <= 0:
        return
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - start_time
    rate = current / elapsed if elapsed > 0 else 0.0
    remaining = (total - current) / rate if rate > 0 else 0.0
    msg = f"\r[{bar}] {current}/{total} ({ratio * 100:5.1f}%) | ETA {remaining:6.1f}s"
    print(msg, end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoCoMo QA + NLI consistency evaluation.")
    parser.add_argument("--data", default="data/locomo10.json")
    parser.add_argument("--agent", default="continuous")
    parser.add_argument("--llm-provider", default="hf")
    parser.add_argument("--llm-model", default="mistral-7b-instruct")
    parser.add_argument("--llm-base-url", nargs="?", const="", default=None)
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-qa", type=int, default=0)
    parser.add_argument("--max-qa-per-sample", type=int, default=0)
    parser.add_argument("--qa-offset", type=int, default=0)
    parser.add_argument("--slice-file", default="")
    parser.add_argument("--skip-nli", action="store_true")
    parser.add_argument("--eval-mode", choices=["open_book", "hide_evidence", "memory_only"], default="open_book")
    parser.add_argument("--retrieval-topk", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--persona-mode", choices=["off", "derived", "hybrid", "file"], default="derived")
    parser.add_argument("--persona-file", default="")
    parser.add_argument("--persona-topk", type=int, default=5)
    parser.add_argument(
        "--persona-metrics-filtered-facts",
        action="store_true",
        help="Compute persona metrics on question-filtered persona facts instead of full fact bank.",
    )
    parser.add_argument(
        "--persona-metrics-filter-min-facts",
        type=int,
        default=2,
        help="Minimum filtered facts required before owner/slot filtering is applied in persona metric computation.",
    )
    parser.add_argument(
        "--persona-min-confidence",
        type=float,
        default=0.55,
        help="Minimum extraction confidence for persona facts considered for injection.",
    )
    parser.add_argument(
        "--persona-allow-unknown-owner",
        action="store_true",
        help="Allow owner=unknown persona facts to be injected (disabled by default).",
    )
    parser.add_argument(
        "--persona-single-owner-when-ambiguous",
        action="store_true",
        help="When question subject is ambiguous, keep persona facts from one dominant owner to avoid cross-owner injections.",
    )
    parser.add_argument(
        "--persona-slot-filter",
        action="store_true",
        help="Filter injected persona facts to slots relevant to the question type before injection.",
    )
    parser.add_argument(
        "--persona-slot-filter-min-facts",
        type=int,
        default=2,
        help="Minimum slot-matched facts required to apply slot filter; falls back to owner-only filter otherwise.",
    )
    parser.add_argument(
        "--persona-risk-filter",
        action="store_true",
        help="Apply heuristic conflict-risk filtering to persona facts before injection.",
    )
    parser.add_argument(
        "--persona-risk-min-llm-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence required for LLM-sourced persona facts when risk filter is enabled.",
    )
    parser.add_argument(
        "--persona-risk-drop-negative",
        action="store_true",
        help="Drop negative-polarity persona facts unless the question has explicit negative cues.",
    )
    parser.add_argument(
        "--persona-risk-drop-abstract",
        action="store_true",
        help="Drop abstract life_event/has facts for non-event question types when risk filter is enabled.",
    )
    parser.add_argument(
        "--persona-risk-drop-relative-time",
        action="store_true",
        help="Drop relative-time persona facts (e.g., yesterday/last week/2 days ago) when risk filter is enabled.",
    )
    parser.add_argument(
        "--persona-preinject-conflict-gate",
        action="store_true",
        help="Resolve owner-slot conflicting persona values before ranking/injection.",
    )
    parser.add_argument(
        "--persona-preinject-conflict-scope",
        choices=["time", "all"],
        default="time",
        help="Scope for pre-injection conflict resolution: only temporal groups or all owner-slot groups.",
    )
    parser.add_argument(
        "--persona-rank-by-question-type",
        action="store_true",
        help="Re-rank persona facts by question-type slot relevance and confidence before top-k injection.",
    )
    parser.add_argument(
        "--persona-rank-target-types",
        default="",
        help="Comma-separated question types where ranking is applied (e.g., time,event,location). Empty means all types.",
    )
    parser.add_argument(
        "--persona-rank-min-candidates",
        type=int,
        default=2,
        help="Minimum number of eligible persona facts required before rank-by-question-type is considered.",
    )
    parser.add_argument(
        "--persona-rank-min-score-gap",
        type=float,
        default=0.0,
        help="Minimum top1-top2 ranking score gap required to apply ranking (acts as a conservative gate).",
    )
    parser.add_argument(
        "--persona-adaptive-topk",
        action="store_true",
        help="Adapt persona injection top-k by question type to reduce prompt intrusiveness.",
    )
    parser.add_argument(
        "--persona-adaptive-topk-map",
        default="",
        help="Optional overrides for adaptive top-k, e.g. 'time=2,event=3,identity=4'.",
    )
    parser.add_argument(
        "--persona-min-history",
        type=int,
        default=0,
        help="Minimum number of visible history turns required before persona lines are injected.",
    )
    parser.add_argument(
        "--persona-line-style",
        choices=["tagged", "plain"],
        default="tagged",
        help="Format style for injected persona lines.",
    )
    parser.add_argument(
        "--inject-persona-for-all-agents",
        action="store_true",
        help="Inject persona lines for all agents (not only persona_loop) for fair apples-to-apples comparison.",
    )
    parser.add_argument(
        "--save-turn-debug",
        action="store_true",
        help="Save per-turn debug payloads (context used, persona lines, loop counters) to turn_debug.json.",
    )
    parser.add_argument("--persona-max-facts", type=int, default=24)
    parser.add_argument("--persona-hybrid-min-rule-facts", type=int, default=3)
    parser.add_argument("--persona-hybrid-llm-max-facts", type=int, default=8)
    parser.add_argument("--persona-llm-provider", default="",
                        help="LLM provider for persona extraction head (overrides --llm-provider).")
    parser.add_argument("--persona-llm-model", default="",
                        help="LLM model for persona extraction head (overrides --llm-model).")
    parser.add_argument("--persona-llm-base-url", nargs="?", const="", default=None,
                        help="base_url for persona extraction LLM. Bare flag (no value) = real OpenAI. Omit = inherit OPENAI_BASE_URL env var.")
    parser.add_argument("--persona-cache", default="",
                        help="Path to persona fact cache JSON. Facts are loaded from cache on hit and written back on completion.")
    parser.add_argument("--loop-interval", type=int, default=8)
    parser.add_argument("--loop-retrieval-topk", type=int, default=3)
    parser.add_argument("--loop-recent-turns", type=int, default=3)
    parser.add_argument("--loop-nli-threshold", type=float, default=0.2)
    parser.add_argument("--loop-max-corrections", type=int, default=2)
    parser.add_argument("--loop-persona-facts", type=int, default=6)
    parser.add_argument(
        "--loop-reset-require-low-consistency",
        action="store_true",
        help="Only perform Persona Loop reset when recent context has low consistency against persona facts.",
    )
    parser.add_argument(
        "--loop-rerank-relevance-weight",
        type=float,
        default=0.45,
        help="Relevance weight when reranking retrieved snippets in Persona Loop.",
    )
    parser.add_argument(
        "--loop-rerank-support-weight",
        type=float,
        default=0.55,
        help="Persona-support weight when reranking retrieved snippets in Persona Loop.",
    )
    parser.add_argument(
        "--loop-summary-max-items",
        type=int,
        default=0,
        help="Number of middle-history summary lines to include after a reset. 0 disables summary.",
    )
    parser.add_argument(
        "--loop-min-history",
        type=int,
        default=0,
        help="Minimum number of visible [HISTORY] turns required before Persona Loop reset is allowed.",
    )
    parser.add_argument(
        "--loop-ablation",
        default="",
        help=(
            "Comma-separated Persona Loop ablations: "
            "disable_persona_persist,disable_nli_rerank,disable_corrections"
        ),
    )
    parser.add_argument("--output", default="artifacts/locomo_eval")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_kwargs: Dict[str, object] = {}
    if args.llm_base_url is not None:
        llm_kwargs["base_url"] = args.llm_base_url
    llm = create_llm(provider=args.llm_provider, model_name=args.llm_model, **llm_kwargs)
    # Separate LLM for persona extraction (e.g. OpenAI gpt-4o-mini while QA uses a local model).
    if args.persona_llm_provider and args.persona_llm_model:
        # --persona-llm-base-url (bare flag) sets '' meaning real OpenAI; omitting the flag
        # leaves args.persona_llm_base_url=None, and OpenAILLM's _UNSET sentinel then inherits
        # OPENAI_BASE_URL env var. Pass the raw string so the sentinel logic works correctly.
        persona_base_url_kwarg: dict = {}
        if args.persona_llm_base_url is not None:
            persona_base_url_kwarg["base_url"] = args.persona_llm_base_url
        persona_llm = create_llm(
            provider=args.persona_llm_provider,
            model_name=args.persona_llm_model,
            **persona_base_url_kwarg,
        )
    else:
        persona_llm = llm

    ablation_tokens = {
        x.strip().lower()
        for x in str(args.loop_ablation).split(",")
        if x.strip()
    }
    allowed_ablation_tokens = {
        "disable_persona_persist",
        "disable_nli_rerank",
        "disable_corrections",
    }
    invalid_ablation_tokens = sorted(ablation_tokens - allowed_ablation_tokens)
    if invalid_ablation_tokens:
        raise ValueError(
            "Invalid --loop-ablation values: "
            + ", ".join(invalid_ablation_tokens)
            + ". Allowed: "
            + ", ".join(sorted(allowed_ablation_tokens))
        )

    def _build_eval_agent():
        memory = create_memory(memory_type="chroma" if args.agent in {"rag", "persona_loop"} else None)
        loop_checker = None
        if args.agent == "persona_loop":
            # Keep loop contradiction detection on by default to match the proposal behavior.
            loop_checker = create_checker(enabled=True, checker_type="deberta", model_name=args.nli_model)
        agent_kwargs: Dict[str, object] = {}
        if args.agent == "persona_loop":
            agent_kwargs = {
                "loop_interval": args.loop_interval,
                "retrieval_top_k": args.loop_retrieval_topk,
                "recent_turns": args.loop_recent_turns,
                "nli_threshold": args.loop_nli_threshold,
                "max_corrections": args.loop_max_corrections,
                "min_history_for_reset": args.loop_min_history,
                "reset_require_low_consistency": bool(args.loop_reset_require_low_consistency),
                "rerank_relevance_weight": args.loop_rerank_relevance_weight,
                "rerank_support_weight": args.loop_rerank_support_weight,
                "summary_max_items": args.loop_summary_max_items,
                "disable_persona_persist": "disable_persona_persist" in ablation_tokens,
                "disable_nli_rerank": "disable_nli_rerank" in ablation_tokens,
                "disable_corrections": "disable_corrections" in ablation_tokens,
            }
        return create_agent(name=args.agent, llm=llm, memory=memory, checker=loop_checker, **agent_kwargs)
    nli = None if args.skip_nli else NLIScorer(model_name=args.nli_model)

    rows: List[Dict[str, object]] = []
    persona_debug_rows: List[Dict[str, object]] = []
    full_persona_fact_bank_rows: List[Dict[str, object]] = []
    turn_debug_rows: List[Dict[str, object]] = []

    # Load persona fact cache (keyed by "sample_id:qa_index") to skip redundant LLM calls.
    persona_cache: Dict[str, Dict[str, object]] = {}
    persona_cache_path: Optional[Path] = Path(args.persona_cache) if args.persona_cache else None
    persona_cache_hits = 0
    persona_cache_misses = 0
    if persona_cache_path and persona_cache_path.exists():
        for row in json.loads(persona_cache_path.read_text(encoding="utf-8")):
            key = f"{row['sample_id']}:{row['qa_index']}"
            persona_cache[key] = row
        print(f"[PERSONA_CACHE] Loaded {len(persona_cache)} cached entries from {persona_cache_path}")

    samples = load_locomo(data_path)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    slice_map: Dict[str, List[int]] = {}
    if args.slice_file:
        slice_path = Path(args.slice_file)
        entries = load_slice_entries(slice_path)
        grouped: Dict[str, List[int]] = {}
        for entry in entries:
            sample_id = str(entry.get("sample_id", "")).strip()
            if not sample_id:
                continue
            qa_index = int(entry.get("qa_index", -1))
            if qa_index < 0:
                continue
            grouped.setdefault(sample_id, []).append(qa_index)
        for sample_id, idxs in grouped.items():
            slice_map[sample_id] = sorted(set(idxs))

    rank_target_types: Optional[set] = None
    if args.persona_rank_target_types:
        rank_target_types = {
            t.strip().lower()
            for t in str(args.persona_rank_target_types).split(",")
            if t.strip()
        }
    qtype_topk_overrides = _parse_qtype_topk_overrides(str(args.persona_adaptive_topk_map))

    total_qa = 0
    for sample in samples:
        sample_id = str(sample.get("sample_id", "")).strip()
        qas = list(sample.get("qa", []))
        if slice_map:
            sel = [i for i in slice_map.get(sample_id, []) if 0 <= i < len(qas)]
            total_qa += len(sel)
        else:
            if args.qa_offset > 0:
                qas = qas[args.qa_offset :]
            if args.max_qa_per_sample > 0:
                qas = qas[: args.max_qa_per_sample]
            total_qa += len(qas)
    if args.max_qa > 0:
        total_qa = min(total_qa, args.max_qa)

    _persona_llm_tag = (
        f"{args.persona_llm_provider}:{args.persona_llm_model}"
        if args.persona_llm_provider and args.persona_llm_model
        else "same_as_llm"
    )
    print(
        f"Starting LoCoMo eval: samples={len(samples)}, target_qa={total_qa}, "
        f"agent={args.agent}, llm={args.llm_provider}:{args.llm_model}, "
        f"persona_llm={_persona_llm_tag}, "
        f"eval_mode={args.eval_mode}, skip_nli={args.skip_nli}"
    )
    start_time = time.time()
    processed = 0
    stopped_early = False
    persona_file_map: Dict[str, List[Dict[str, object]]] = {}
    if args.persona_mode == "file":
        if not args.persona_file:
            raise ValueError("--persona-file is required when --persona-mode=file")
        persona_file_path = Path(args.persona_file)
        persona_file_map = json.loads(persona_file_path.read_text(encoding="utf-8"))

    for sample in samples:
        # Each sample is an independent conversation; reset agent state/memory to avoid leakage.
        agent = _build_eval_agent()
        sample_id = str(sample.get("sample_id", ""))
        conv = sample.get("conversation", {})
        turns = flatten_conversation(conv)
        dia2idx = {t["dia_id"]: i for i, t in enumerate(turns)}
        qas = list(sample.get("qa", []))
        qa_pairs: List[tuple[int, Dict[str, object]]] = []
        if slice_map:
            for idx in slice_map.get(sample_id, []):
                if 0 <= idx < len(qas):
                    qa_pairs.append((idx, qas[idx]))
        else:
            start = max(0, int(args.qa_offset))
            trimmed = qas[start:]
            if args.max_qa_per_sample > 0:
                trimmed = trimmed[: args.max_qa_per_sample]
            qa_pairs = [(start + i, qa) for i, qa in enumerate(trimmed)]

        for qa_index, qa in qa_pairs:
            if args.max_qa > 0 and processed >= args.max_qa:
                stopped_early = True
                break

            question = str(qa.get("question", "")).strip()
            if not question:
                continue

            evidence = [str(x) for x in qa.get("evidence", [])]
            if evidence:
                max_idx = max(dia2idx.get(e, -1) for e in evidence)
                visible_turns = turns[: max_idx + 1] if max_idx >= 0 else turns
            else:
                visible_turns = turns

            history = _build_eval_history(
                eval_mode=args.eval_mode,
                question=question,
                visible_turns=visible_turns,
                evidence_ids=evidence,
                max_turns=args.max_turns,
                retrieval_topk=args.retrieval_topk,
            )
            evidence_visible = _is_evidence_visible(history=history, evidence_ids=evidence)

            fact_bank: List[Dict[str, object]] = []
            persona_extract_stats: Dict[str, object] = {
                "raw_candidates": 0,
                "dedup_candidates": 0,
                "unique_slots": 0,
                "singleton_conflict_slots": 0,
                "singleton_conflict_values": 0,
            }
            hybrid_used_llm_runtime = False
            _cache_key = f"{sample_id}:{qa_index}"
            if persona_cache_path and _cache_key in persona_cache:
                _cached = persona_cache[_cache_key]
                fact_bank = list(_cached.get("fact_bank", []))
                persona_extract_stats = dict(_cached.get("persona_extract_stats", {}))
                persona_extract_stats["from_cache"] = True
                persona_cache_hits += 1
            elif args.persona_mode == "derived":
                if persona_cache_path:
                    persona_cache_misses += 1
                extracted = extract_persona_facts_with_stats(
                    visible_turns=visible_turns,
                    max_facts=args.persona_max_facts,
                )
                fact_bank = list(extracted.get("facts", []))
                persona_extract_stats = dict(extracted.get("stats", {}))
            elif args.persona_mode == "hybrid":
                if persona_cache_path:
                    persona_cache_misses += 1
                extracted = extract_persona_facts_hybrid_with_stats(
                    visible_turns=visible_turns,
                    llm=persona_llm,
                    max_facts=args.persona_max_facts,
                    min_rule_facts=args.persona_hybrid_min_rule_facts,
                    llm_max_facts=args.persona_hybrid_llm_max_facts,
                )
                fact_bank = list(extracted.get("facts", []))
                persona_extract_stats = dict(extracted.get("stats", {}))
                hybrid_used_llm_runtime = bool(persona_extract_stats.get("hybrid_used_llm", False))
            elif args.persona_mode == "file":
                fact_bank = list(persona_file_map.get(sample_id, []))

            fact_bank = _enrich_fact_owners(fact_bank=fact_bank, visible_turns=visible_turns)

            persona_extract_stats.setdefault("from_cache", False)

            # Update in-memory cache for new extractions (written to disk at end).
            if persona_cache_path and _cache_key not in persona_cache:
                persona_cache[_cache_key] = {
                    "sample_id": sample_id,
                    "qa_index": qa_index,
                    "question": question,
                    "persona_mode": args.persona_mode,
                    "persona_extract_stats": copy.deepcopy(persona_extract_stats),
                    "fact_bank": copy.deepcopy(fact_bank),
                }
            elif persona_cache_path and _cache_key in persona_cache:
                persona_cache[_cache_key]["fact_bank"] = copy.deepcopy(fact_bank)

            participant_names = [str(t.get("speaker", "")).strip() for t in visible_turns if str(t.get("speaker", "")).strip()]
            question_subjects = _extract_question_subjects(question=question, candidate_names=participant_names)
            question_types: List[str] = []
            allowed_slots: Optional[set] = None
            if args.persona_slot_filter or args.persona_rank_by_question_type or args.persona_adaptive_topk:
                question_types = _classify_question_types(question)
            if args.persona_slot_filter:
                if question_types:
                    allowed_slots = set()
                    for _qt in question_types:
                        allowed_slots.update(_QTYPE_TO_SLOTS.get(_qt, set()))
            subject_facts = _filter_persona_facts_for_question(
                fact_bank=fact_bank,
                subjects=question_subjects,
                min_confidence=args.persona_min_confidence,
                allow_unknown_owner=bool(args.persona_allow_unknown_owner),
                single_owner_when_ambiguous=bool(args.persona_single_owner_when_ambiguous),
                allowed_slots=allowed_slots,
                slot_filter_min_facts=int(args.persona_slot_filter_min_facts),
            )
            pre_risk_count = len(subject_facts)
            if args.persona_risk_filter:
                subject_facts = _risk_filter_persona_facts(
                    facts=subject_facts,
                    question=question,
                    question_types=question_types,
                    min_llm_confidence=float(args.persona_risk_min_llm_confidence),
                    drop_negative=bool(args.persona_risk_drop_negative),
                    drop_abstract=bool(args.persona_risk_drop_abstract),
                    drop_relative_time=bool(args.persona_risk_drop_relative_time),
                )
            conflict_gate_stats = {"groups_resolved": 0, "facts_removed": 0}
            if args.persona_preinject_conflict_gate:
                subject_facts, conflict_gate_stats = _preinject_conflict_gate(
                    facts=subject_facts,
                    scope=str(args.persona_preinject_conflict_scope),
                )
            rank_applied = False
            rank_score_gap = 0.0
            if args.persona_rank_by_question_type:
                subject_facts, rank_applied, rank_score_gap = _rank_persona_facts_for_question(
                    facts=subject_facts,
                    question_types=question_types,
                    rank_target_types=rank_target_types,
                    min_candidates=int(args.persona_rank_min_candidates),
                    min_score_gap=float(args.persona_rank_min_score_gap),
                )
            persona_injection_topk = _resolve_persona_injection_topk(
                base_topk=int(args.loop_persona_facts),
                question_types=question_types,
                adaptive_enabled=bool(args.persona_adaptive_topk),
                qtype_topk_overrides=qtype_topk_overrides,
            )
            persona_lines = _format_persona_lines(
                fact_bank=subject_facts,
                top_k=persona_injection_topk,
                line_style=str(args.persona_line_style),
            )
            persona_lines_for_turn: Optional[List[str]] = None
            should_inject_persona = args.agent == "persona_loop" or bool(args.inject_persona_for_all_agents)
            if should_inject_persona and len(history) >= max(0, int(args.persona_min_history)):
                persona_lines_for_turn = persona_lines
            turn_result = answer_question(
                agent=agent,
                question=question,
                history=history,
                persona_lines=persona_lines_for_turn,
            )
            prediction = str(turn_result.get("response", ""))
            loop_reset = bool(turn_result.get("loop_reset", False))
            loop_recent_persisted = int(turn_result.get("loop_recent_persisted", 0))
            loop_retrieved_count = int(turn_result.get("loop_retrieved_count", 0))
            loop_corrections_count = int(turn_result.get("loop_corrections_count", 0))
            loop_summary_count = int(turn_result.get("loop_summary_count", 0))

            metric_subjects = question_subjects if args.persona_metrics_filtered_facts else None
            metric_slots = allowed_slots if args.persona_metrics_filtered_facts else None
            persona_metrics = compute_persona_metrics(
                prediction=prediction,
                fact_bank=fact_bank,
                nli=nli,
                top_k=args.persona_topk,
                question_subjects=metric_subjects,
                allowed_slots=metric_slots,
                min_filtered_facts=int(args.persona_metrics_filter_min_facts),
            )

            # Persist full extracted fact bank per QA for deeper audit and error analysis.
            full_persona_fact_bank_rows.append(
                {
                    "sample_id": sample_id,
                    "qa_index": qa_index,
                    "question": question,
                    "persona_mode": args.persona_mode,
                    "eval_mode": args.eval_mode,
                    "persona_extract_stats": persona_extract_stats,
                    "fact_bank": copy.deepcopy(fact_bank),
                }
            )

            gold = qa.get("answer")
            adv = qa.get("adversarial_answer")
            row: Dict[str, object] = {
                "sample_id": sample_id,
                "category": int(qa.get("category", -1)),
                "qa_index": qa_index,
                "question": question,
                "prediction": prediction,
                "gold_answer": gold,
                "adversarial_answer": adv,
                "persona_mode": args.persona_mode,
                "eval_mode": args.eval_mode,
                "history_items": len(history),
                "evidence_visible": evidence_visible,
                "persona_cache_hit": bool(persona_extract_stats.get("from_cache", False)),
                "persona_raw_candidates": int(persona_extract_stats.get("raw_candidates", 0)),
                "persona_dedup_candidates": int(persona_extract_stats.get("dedup_candidates", 0)),
                "persona_unique_slots": int(persona_extract_stats.get("unique_slots", 0)),
                "persona_conflict_slots": int(persona_extract_stats.get("singleton_conflict_slots", 0)),
                "persona_conflict_values": int(persona_extract_stats.get("singleton_conflict_values", 0)),
                "persona_hybrid_used_llm": bool(persona_extract_stats.get("hybrid_used_llm", False)),
                "persona_hybrid_llm_added_facts": int(persona_extract_stats.get("llm_added_facts", 0)),
                "persona_llm_raw_len": int(persona_extract_stats.get("llm_raw_len", 0)),
                "persona_llm_json_parsed": bool(persona_extract_stats.get("llm_json_parsed", False)),
                "persona_llm_structured_success": bool(persona_extract_stats.get("llm_structured_success", False)),
                "persona_llm_candidate_count": int(persona_extract_stats.get("llm_candidate_count", 0)),
                "persona_llm_valid_fact_count": int(persona_extract_stats.get("llm_valid_fact_count", 0)),
                "persona_llm_fallback_used": bool(persona_extract_stats.get("llm_fallback_used", False)),
                "persona_llm_repair_used": bool(persona_extract_stats.get("llm_repair_used", False)),
                "persona_llm_repair_success": bool(persona_extract_stats.get("llm_repair_success", False)),
                "persona_hybrid_used_llm_runtime": hybrid_used_llm_runtime,
                "loop_reset": loop_reset,
                "loop_recent_persisted": loop_recent_persisted,
                "loop_retrieved_count": loop_retrieved_count,
                "loop_corrections_count": loop_corrections_count,
                "loop_summary_count": loop_summary_count,
                "loop_ablation": ",".join(sorted(ablation_tokens)),
                "inject_persona_for_all_agents": bool(args.inject_persona_for_all_agents),
                "question_subject": question_subjects[0] if question_subjects else "",
                "question_subjects": list(question_subjects),
                "persona_owner_match_ratio": (
                    round(
                        (
                            sum(
                                1
                                for f in fact_bank
                                if question_subjects
                                and str(f.get("owner", "")).strip().lower() in {s.lower() for s in question_subjects}
                            )
                            / max(1, len(fact_bank))
                        ),
                        4,
                    )
                    if question_subjects
                    else 0.0
                ),
                "persona_unknown_owner_ratio": round(
                    (
                        sum(
                            1
                            for f in fact_bank
                            if str(f.get("owner", "unknown")).strip().lower() in {"", "unknown"}
                        )
                        / max(1, len(fact_bank))
                    ),
                    4,
                ) if fact_bank else 0.0,
                "persona_facts_eligible": len(subject_facts),
                "question_types": list(question_types),
                "persona_slot_filter_active": bool(allowed_slots),
                "persona_slot_filter_allowed": sorted(allowed_slots) if allowed_slots else [],
                "persona_risk_filter": bool(args.persona_risk_filter),
                "persona_preinject_conflict_gate": bool(args.persona_preinject_conflict_gate),
                "persona_preinject_conflict_scope": str(args.persona_preinject_conflict_scope),
                "persona_single_owner_when_ambiguous": bool(args.persona_single_owner_when_ambiguous),
                "persona_rank_by_question_type": bool(args.persona_rank_by_question_type),
                "persona_rank_target_types": sorted(rank_target_types) if rank_target_types else [],
                "persona_rank_min_candidates": int(args.persona_rank_min_candidates),
                "persona_rank_min_score_gap": float(args.persona_rank_min_score_gap),
                "persona_rank_applied": bool(rank_applied),
                "persona_rank_score_gap": float(rank_score_gap),
                "persona_adaptive_topk": bool(args.persona_adaptive_topk),
                "persona_adaptive_topk_map": dict(qtype_topk_overrides),
                "persona_injection_topk": int(persona_injection_topk),
                "persona_line_style": str(args.persona_line_style),
                "persona_risk_min_llm_confidence": float(args.persona_risk_min_llm_confidence),
                "persona_risk_drop_negative": bool(args.persona_risk_drop_negative),
                "persona_risk_drop_abstract": bool(args.persona_risk_drop_abstract),
                "persona_risk_drop_relative_time": bool(args.persona_risk_drop_relative_time),
                "persona_risk_removed": max(0, int(pre_risk_count - len(subject_facts))),
                "persona_preinject_conflict_groups_resolved": int(conflict_gate_stats.get("groups_resolved", 0)),
                "persona_preinject_conflict_facts_removed": int(conflict_gate_stats.get("facts_removed", 0)),
                "persona_metrics_filtered_facts": bool(args.persona_metrics_filtered_facts),
                "persona_metrics_filter_min_facts": int(args.persona_metrics_filter_min_facts),
            }
            row.update(persona_metrics)

            if isinstance(gold, str) and gold.strip():
                row.update(qa_scores(prediction, gold))
                if nli is not None:
                    nli_gold = nli.score(premise=prediction, hypothesis=gold)
                    row["nli_entailment_gold"] = nli_gold["entailment"]
                    row["nli_contradiction_gold"] = nli_gold["contradiction"]

            if isinstance(adv, str) and adv.strip():
                if nli is not None:
                    nli_adv = nli.score(premise=prediction, hypothesis=adv)
                    row["nli_entailment_adv"] = nli_adv["entailment"]
                    row["nli_contradiction_adv"] = nli_adv["contradiction"]

            rows.append(row)
            if args.save_turn_debug:
                turn_debug_rows.append(
                    {
                        "sample_id": sample_id,
                        "qa_index": qa_index,
                        "question": question,
                        "agent": args.agent,
                        "history_items": len(history),
                        "persona_injected": bool(persona_lines_for_turn),
                        "question_subjects": list(question_subjects),
                        "persona_facts_eligible": len(subject_facts),
                        "persona_lines": list(persona_lines_for_turn or []),
                        "context_used": str(turn_result.get("context", "")),
                        "consistency": turn_result.get("consistency", None),
                        "loop_reset": loop_reset,
                        "loop_recent_persisted": loop_recent_persisted,
                        "loop_retrieved_count": loop_retrieved_count,
                        "loop_corrections_count": loop_corrections_count,
                        "loop_summary_count": loop_summary_count,
                        "prediction": prediction,
                        "gold_answer": gold,
                        "f1": row.get("f1", 0.0),
                        "em": row.get("em", 0.0),
                    }
                )
            persona_debug_rows.append(
                {
                    "sample_id": sample_id,
                    "question": question,
                    "persona_mode": args.persona_mode,
                    "eval_mode": args.eval_mode,
                    "evidence_visible": evidence_visible,
                    "persona_cache_hit": row.get("persona_cache_hit", False),
                    "persona_facts_total": row.get("persona_facts_total", 0),
                    "persona_facts_used": row.get("persona_facts_used", 0),
                    "persona_fact_texts": row.get("persona_fact_texts", []),
                    "persona_raw_candidates": row.get("persona_raw_candidates", 0),
                    "persona_dedup_candidates": row.get("persona_dedup_candidates", 0),
                    "persona_unique_slots": row.get("persona_unique_slots", 0),
                    "persona_conflict_slots": row.get("persona_conflict_slots", 0),
                    "persona_conflict_values": row.get("persona_conflict_values", 0),
                    "persona_hybrid_used_llm": row.get("persona_hybrid_used_llm", False),
                    "persona_hybrid_llm_added_facts": row.get("persona_hybrid_llm_added_facts", 0),
                    "persona_llm_raw_len": row.get("persona_llm_raw_len", 0),
                    "persona_llm_json_parsed": row.get("persona_llm_json_parsed", False),
                    "persona_llm_structured_success": row.get("persona_llm_structured_success", False),
                    "persona_llm_candidate_count": row.get("persona_llm_candidate_count", 0),
                    "persona_llm_valid_fact_count": row.get("persona_llm_valid_fact_count", 0),
                    "persona_llm_fallback_used": row.get("persona_llm_fallback_used", False),
                    "persona_llm_repair_used": row.get("persona_llm_repair_used", False),
                    "persona_llm_repair_success": row.get("persona_llm_repair_success", False),
                    "persona_hybrid_used_llm_runtime": row.get("persona_hybrid_used_llm_runtime", False),
                    "loop_reset": row.get("loop_reset", False),
                    "loop_recent_persisted": row.get("loop_recent_persisted", 0),
                    "loop_retrieved_count": row.get("loop_retrieved_count", 0),
                    "loop_corrections_count": row.get("loop_corrections_count", 0),
                    "loop_summary_count": row.get("loop_summary_count", 0),
                }
            )
            processed += 1

            if not args.no_progress_bar:
                print_progress_bar(processed, total_qa, start_time)

            if args.progress_every > 0 and (
                processed % args.progress_every == 0 or processed == total_qa
            ):
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0.0
                print(
                    f"Progress: {processed}/{total_qa} QA, elapsed={elapsed:.1f}s, rate={rate:.2f} qa/s"
                )
                if persona_cache_path:
                    cache_total = persona_cache_hits + persona_cache_misses
                    cache_hit_rate = (persona_cache_hits / cache_total) if cache_total > 0 else 0.0
                    print(
                        f"[PERSONA_CACHE] hits={persona_cache_hits} misses={persona_cache_misses} "
                        f"hit_rate={cache_hit_rate:.2%}"
                    )

        if stopped_early:
            break

    def avg(key: str) -> float:
        vals = [float(r[key]) for r in rows if key in r]
        return round(mean(vals), 4) if vals else 0.0

    metrics = {
        "count": len(rows),
        "em": avg("em"),
        "f1": avg("f1"),
        "nli_entailment_gold": avg("nli_entailment_gold"),
        "nli_contradiction_gold": avg("nli_contradiction_gold"),
        "nli_entailment_adv": avg("nli_entailment_adv"),
        "nli_contradiction_adv": avg("nli_contradiction_adv"),
        "persona_entailment": avg("persona_entailment"),
        "persona_contradiction": avg("persona_contradiction"),
        "persona_supported_ratio": avg("persona_supported_ratio"),
        "persona_conflict_ratio": avg("persona_conflict_ratio"),
        "persona_pcs": avg("persona_pcs"),
        "persona_facts_total_avg": avg("persona_facts_total"),
        "persona_facts_used_avg": avg("persona_facts_used"),
        "persona_raw_candidates_avg": avg("persona_raw_candidates"),
        "persona_dedup_candidates_avg": avg("persona_dedup_candidates"),
        "persona_unique_slots_avg": avg("persona_unique_slots"),
        "persona_conflict_slots_avg": avg("persona_conflict_slots"),
        "persona_conflict_values_avg": avg("persona_conflict_values"),
        "persona_cache_hit_ratio": avg("persona_cache_hit"),
        "persona_cache_hits": int(persona_cache_hits),
        "persona_cache_misses": int(persona_cache_misses),
        "persona_hybrid_used_llm_ratio": avg("persona_hybrid_used_llm"),
        "persona_hybrid_used_llm_runtime_ratio": avg("persona_hybrid_used_llm_runtime"),
        "loop_reset_ratio": avg("loop_reset"),
        "loop_recent_persisted_avg": avg("loop_recent_persisted"),
        "loop_retrieved_count_avg": avg("loop_retrieved_count"),
        "loop_corrections_count_avg": avg("loop_corrections_count"),
        "loop_summary_count_avg": avg("loop_summary_count"),
        "loop_ablation": ",".join(sorted(ablation_tokens)),
        "inject_persona_for_all_agents": bool(args.inject_persona_for_all_agents),
        "persona_owner_match_ratio_avg": avg("persona_owner_match_ratio"),
        "persona_unknown_owner_ratio_avg": avg("persona_unknown_owner_ratio"),
        "persona_facts_eligible_avg": avg("persona_facts_eligible"),
        "persona_preinject_conflict_gate": bool(args.persona_preinject_conflict_gate),
        "persona_preinject_conflict_scope": str(args.persona_preinject_conflict_scope),
        "persona_preinject_conflict_groups_resolved_avg": avg("persona_preinject_conflict_groups_resolved"),
        "persona_preinject_conflict_facts_removed_avg": avg("persona_preinject_conflict_facts_removed"),
        "persona_single_owner_when_ambiguous": bool(args.persona_single_owner_when_ambiguous),
        "persona_rank_by_question_type": bool(args.persona_rank_by_question_type),
        "persona_rank_target_types": sorted(rank_target_types) if rank_target_types else [],
        "persona_rank_min_candidates": int(args.persona_rank_min_candidates),
        "persona_rank_min_score_gap": float(args.persona_rank_min_score_gap),
        "persona_rank_applied_ratio": avg("persona_rank_applied"),
        "persona_rank_score_gap_avg": avg("persona_rank_score_gap"),
        "persona_adaptive_topk": bool(args.persona_adaptive_topk),
        "persona_adaptive_topk_map": dict(qtype_topk_overrides),
        "persona_injection_topk_avg": avg("persona_injection_topk"),
        "persona_line_style": str(args.persona_line_style),
        "save_turn_debug": bool(args.save_turn_debug),
        "persona_hybrid_llm_added_facts_avg": avg("persona_hybrid_llm_added_facts"),
        "persona_llm_raw_len_avg": avg("persona_llm_raw_len"),
        "persona_llm_json_parsed_ratio": avg("persona_llm_json_parsed"),
        "persona_llm_structured_success_ratio": avg("persona_llm_structured_success"),
        "persona_llm_candidate_count_avg": avg("persona_llm_candidate_count"),
        "persona_llm_valid_fact_count_avg": avg("persona_llm_valid_fact_count"),
        "persona_llm_fallback_used_ratio": avg("persona_llm_fallback_used"),
        "persona_llm_repair_used_ratio": avg("persona_llm_repair_used"),
        "persona_llm_repair_success_ratio": avg("persona_llm_repair_success"),
        "persona_risk_drop_relative_time": bool(args.persona_risk_drop_relative_time),
        "evidence_visible_ratio": avg("evidence_visible"),
        "history_items_avg": avg("history_items"),
        "adversarial_count": int(sum(1 for r in rows if isinstance(r.get("adversarial_answer"), str) and str(r.get("adversarial_answer", "")).strip())),
    }

    (output_dir / "qa_predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "persona_facts_debug.json").write_text(
        json.dumps(persona_debug_rows, indent=2), encoding="utf-8"
    )
    (output_dir / "full_persona_fact_bank.json").write_text(
        json.dumps(full_persona_fact_bank_rows, indent=2), encoding="utf-8"
    )
    if args.save_turn_debug:
        (output_dir / "turn_debug.json").write_text(
            json.dumps(turn_debug_rows, indent=2), encoding="utf-8"
        )
    (output_dir / "qa_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Write updated persona cache (new extractions merged with previously cached entries).
    if persona_cache_path:
        persona_cache_path.parent.mkdir(parents=True, exist_ok=True)
        persona_cache_path.write_text(
            json.dumps(list(persona_cache.values()), indent=2), encoding="utf-8"
        )
        print(f"[PERSONA_CACHE] Saved {len(persona_cache)} entries to {persona_cache_path}")

    run_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(ROOT),
        "script": "scripts/run_locomo_eval.py",
        "args": vars(args),
        "data_path": str(data_path),
        "data_sha256": file_sha256(data_path) if data_path.exists() else "missing",
        "slice_file": args.slice_file,
        "slice_count": int(sum(len(v) for v in slice_map.values())),
        "result_count": len(rows),
        "metrics_file": str(output_dir / "qa_metrics.json"),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    if not args.no_progress_bar:
        print()

    elapsed = time.time() - start_time
    print(f"Finished LoCoMo eval in {elapsed:.1f}s. Output: {output_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
