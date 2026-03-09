from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from persona_loop.core.factories import create_agent
from persona_loop.core.factories import create_checker
from persona_loop.core.factories import create_llm
from persona_loop.core.factories import create_memory
from persona_loop.context_builder.priority_builder import build_priority_context
from persona_loop.eval.nli_consistency import compute_consistency_metrics


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _join_prefixed(prefix: str, items: List[str]) -> List[str]:
    return [f"[{prefix}] {x}" for x in items]


def build_agent_context(
    cfg: DictConfig,
    agent_name: str,
    prompt: str,
    turn_id: int,
    persona_facts: List[str],
    corrections: List[str],
    history: List[str],
) -> str:
    if agent_name == "persona_loop":
        return build_priority_context(
            persona_facts=persona_facts,
            corrections=corrections,
            history=history[-int(cfg.context_builder.max_history_turns) :],
            recent_turn=prompt,
            max_items=int(cfg.experiment.k),
        )

    if agent_name == "continuous":
        hist = history[-int(cfg.benchmark.continuous.history_turns) :]
        blocks: List[str] = []
        if bool(cfg.benchmark.continuous.include_persona):
            blocks.extend(_join_prefixed("PERSONA", persona_facts))
        blocks.extend(_join_prefixed("HISTORY", hist))
        blocks.append(f"[USER] {prompt}")
        return "\n".join(blocks)

    if agent_name == "sliding_window":
        hist = history[-int(cfg.benchmark.sliding_window.history_turns) :]
        blocks = []
        if bool(cfg.benchmark.sliding_window.include_persona):
            blocks.extend(_join_prefixed("PERSONA", persona_facts))
        blocks.extend(_join_prefixed("HISTORY", hist))
        blocks.append(f"[USER] {prompt}")
        return "\n".join(blocks)

    if agent_name == "periodic_remind":
        hist = history[-int(cfg.benchmark.periodic_remind.history_turns) :]
        interval = max(1, int(cfg.benchmark.periodic_remind.interval))
        include_persona = turn_id % interval == 0
        include_persona = include_persona or bool(
            cfg.benchmark.periodic_remind.include_persona_on_non_periodic
        )
        blocks = []
        if include_persona:
            blocks.extend(_join_prefixed("PERSONA", persona_facts))
        blocks.extend(_join_prefixed("HISTORY", hist))
        blocks.append(f"[USER] {prompt}")
        return "\n".join(blocks)

    if agent_name == "rag":
        hist = history[-int(cfg.benchmark.rag.history_turns) :]
        blocks = _join_prefixed("HISTORY", hist)
        blocks.append(f"[USER] {prompt}")
        return "\n".join(blocks)

    if agent_name == "ppa":
        hist = history[-int(cfg.benchmark.ppa.history_turns) :]
        blocks = _join_prefixed("PERSONA", persona_facts)
        blocks.extend(_join_prefixed("HISTORY", hist))
        blocks.append("[INSTRUCTION] Prefer persona-aligned and stable answers.")
        blocks.append(f"[USER] {prompt}")
        return "\n".join(blocks)

    return build_priority_context(
        persona_facts=persona_facts,
        corrections=corrections,
        history=history[-int(cfg.context_builder.max_history_turns) :],
        recent_turn=prompt,
        max_items=int(cfg.experiment.k),
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.experiment.seed))

    run_name = str(cfg.experiment.run_name)
    artifact_dir = Path("artifacts") / run_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    llm = create_llm(provider=cfg.llm.provider, model_name=cfg.llm.model_name)
    memory = create_memory(memory_type=cfg.memory.type)
    checker = create_checker(
        enabled=bool(cfg.consistency.enabled),
        checker_type=cfg.consistency.type,
        model_name=cfg.consistency.model_name,
    )
    agent = create_agent(name=cfg.agent.name, llm=llm, memory=memory, checker=checker)

    prompts: List[str] = list(cfg.dataset.samples)
    persona_facts: List[str] = list(cfg.dataset.persona_facts)
    corrections: List[str] = list(getattr(cfg.dataset, "corrections", []))
    outputs: List[Dict[str, object]] = []

    for turn_id, prompt in enumerate(prompts):
        history = [str(o["response"]) for o in outputs]
        context = build_agent_context(
            cfg=cfg,
            agent_name=str(cfg.agent.name),
            prompt=prompt,
            turn_id=turn_id,
            persona_facts=persona_facts,
            corrections=corrections,
            history=history,
        )
        result = agent.run_turn(prompt=prompt, context=context)
        outputs.append({"turn_id": turn_id, **result})

    metrics = compute_consistency_metrics(outputs)

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    (artifact_dir / "resolved_config.json").write_text(
        json.dumps(resolved_cfg, indent=2), encoding="utf-8"
    )
    (artifact_dir / "predictions.json").write_text(
        json.dumps(outputs, indent=2), encoding="utf-8"
    )
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"Run complete: {run_name}")
    print(f"Artifacts: {artifact_dir}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
