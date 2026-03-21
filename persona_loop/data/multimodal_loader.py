"""Loader for the multimodal_dialog dataset.

Data format (per agent JSON file):
    {
        "name": "Yangyang",
        "persona_summary": "Yangyang is a lively and warm-hearted...",
        "session_1_date_time": "8:02 pm on 6 March, 2023",
        "session_1": [
            {
                "speaker": "Yangyang",
                "clean_text": "Hey Chenmo! ...",
                "text": "Hello Chenmo! ...",
                "dia_id": "D1:1"
            }, ...
        ],
        "session_1_facts": {
            "Yangyang": [["Yangyang saw a squirrel...", "D1:1"], ...],
            "Chenmo":   [["Chenmo has been writing short stories...", "D1:2"], ...]
        },
        "session_1_summary": "On 8:02 pm on 6 March, ...",
        "session_2_date_time": ...,
        "session_2": [...],
        ...
    }

Each pair/ directory has agent_a.json and agent_b.json.
The session turns are SHARED — both files contain the same conversation;
only persona_summary (and name) differs per agent.

Usage:
    from persona_loop.data.multimodal_loader import load_pair, DialogSample

    sample = load_pair("data/multimodal_dialog/pair1")
    print(sample.agent_a.persona_summary)
    for turn in sample.turns:          # all sessions flattened
        print(turn.dia_id, turn.speaker, turn.text)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Turn:
    dia_id: str
    speaker: str
    text: str          # clean_text preferred; falls back to text
    session: int       # 1-indexed session number


@dataclass
class AgentData:
    name: str
    persona_summary: str
    # per-session metadata
    session_date_times: Dict[int, str]   # session_num -> datetime string
    session_summaries: Dict[int, str]    # session_num -> summary text
    # per-session observed facts: {session_num -> {speaker -> [(fact_text, dia_id), ...]}}
    session_facts: Dict[int, Dict[str, List[tuple]]]


@dataclass
class DialogSample:
    pair_id: str                  # e.g. "pair1"
    agent_a: AgentData
    agent_b: AgentData
    turns: List[Turn]             # all sessions flattened in order
    sessions: Dict[int, List[Turn]]  # sessions keyed by session number (1-indexed)

    def turns_up_to_session(self, max_session: int) -> List[Turn]:
        """Return all turns from sessions 1..max_session."""
        return [t for t in self.turns if t.session <= max_session]

    def turns_in_session(self, session: int) -> List[Turn]:
        return self.sessions.get(session, [])

    @property
    def session_count(self) -> int:
        return len(self.sessions)


def _parse_agent(data: dict) -> AgentData:
    name = str(data.get("name", "")).strip()
    persona_summary = str(data.get("persona_summary", "")).strip()

    session_date_times: Dict[int, str] = {}
    session_summaries: Dict[int, str] = {}
    session_facts: Dict[int, Dict[str, List[tuple]]] = {}

    for n in range(1, 20):
        dt_key = f"session_{n}_date_time"
        sm_key = f"session_{n}_summary"
        fa_key = f"session_{n}_facts"
        if dt_key not in data and sm_key not in data:
            break
        session_date_times[n] = str(data.get(dt_key, ""))
        session_summaries[n] = str(data.get(sm_key, ""))
        raw_facts = data.get(fa_key, {})
        if isinstance(raw_facts, dict):
            parsed_facts: Dict[str, List[tuple]] = {}
            for speaker, items in raw_facts.items():
                parsed_facts[str(speaker)] = [
                    (str(item[0]), str(item[1])) for item in items if len(item) >= 2
                ]
            session_facts[n] = parsed_facts
        else:
            session_facts[n] = {}

    return AgentData(
        name=name,
        persona_summary=persona_summary,
        session_date_times=session_date_times,
        session_summaries=session_summaries,
        session_facts=session_facts,
    )


def _parse_turns(data: dict) -> Dict[int, List[Turn]]:
    sessions: Dict[int, List[Turn]] = {}
    for n in range(1, 20):
        key = f"session_{n}"
        if key not in data:
            break
        raw_turns = data[key]
        if not isinstance(raw_turns, list):
            continue
        turns: List[Turn] = []
        for t in raw_turns:
            dia_id = str(t.get("dia_id", "")).strip()
            speaker = str(t.get("speaker", "")).strip()
            # Prefer clean_text (naturalised), fall back to text
            text = str(t.get("clean_text") or t.get("text", "")).strip()
            if dia_id and text:
                turns.append(Turn(dia_id=dia_id, speaker=speaker, text=text, session=n))
        sessions[n] = turns
    return sessions


def load_pair(pair_dir: str | Path) -> DialogSample:
    """Load a pair directory containing agent_a.json and agent_b.json."""
    pair_path = Path(pair_dir)
    pair_id = pair_path.name

    data_a = json.loads((pair_path / "agent_a.json").read_text(encoding="utf-8"))
    data_b = json.loads((pair_path / "agent_b.json").read_text(encoding="utf-8"))

    agent_a = _parse_agent(data_a)
    agent_b = _parse_agent(data_b)

    # Turns are shared; parse from agent_a (both are identical)
    sessions = _parse_turns(data_a)
    flat_turns: List[Turn] = []
    for n in sorted(sessions.keys()):
        flat_turns.extend(sessions[n])

    return DialogSample(
        pair_id=pair_id,
        agent_a=agent_a,
        agent_b=agent_b,
        turns=flat_turns,
        sessions=sessions,
    )


def load_all_pairs(data_dir: str | Path, exclude: Optional[List[str]] = None) -> List[DialogSample]:
    """Load all valid pair directories from data_dir."""
    root = Path(data_dir)
    exclude_set = set(exclude or [])
    samples: List[DialogSample] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if d.name in exclude_set:
            continue
        if not (d / "agent_a.json").exists() or not (d / "agent_b.json").exists():
            continue
        samples.append(load_pair(d))
    return samples
