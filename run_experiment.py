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
    outputs: List[Dict[str, object]] = []

    for turn_id, prompt in enumerate(prompts):
        context = build_priority_context(
            persona_facts=list(cfg.dataset.persona_facts),
            corrections=[],
            history=[o["response"] for o in outputs[-int(cfg.context_builder.max_history_turns) :]],
            recent_turn=prompt,
            max_items=int(cfg.experiment.k),
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
