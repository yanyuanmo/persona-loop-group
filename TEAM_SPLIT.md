# Team Split Plan (Persona Loop)

## Branch Strategy

- Each member works on one module branch: `feature/<module>-<name>`
- Keep `run_experiment.py` and `persona_loop/core/factories.py` owned by integrator only
- Merge order: interfaces first, implementations second, experiments last

## Role A: Agent/Baseline Owner

- Folder: `persona_loop/agents/`
- Deliverables:
  - Implement baseline logic in separate files
  - Keep `run_turn(prompt, context)` signature unchanged
  - Add unit tests for each new baseline behavior

## Role B: Memory/Retrieval Owner

- Folder: `persona_loop/memory/`
- Deliverables:
  - Implement retrieval backend adapters
  - Keep `add()` and `search()` API contract stable
  - Add config entries under `configs/memory/`

## Role C: Consistency/NLI Owner

- Folder: `persona_loop/consistency/`
- Deliverables:
  - Replace placeholder checker with real NLI model
  - Add checker configs under `configs/consistency/`
  - Provide calibration notes for thresholds and score mapping

## Role D: Eval/Analysis Owner

- Folder: `persona_loop/eval/` and `scripts/`
- Deliverables:
  - Add core metrics and error analysis scripts
  - Export tables for baseline and ablation comparisons
  - Maintain reproducibility docs for metric computation

## Role E: Infra/Experiment Owner

- Folder: `configs/`, `run_experiment.py`, `README.md`
- Deliverables:
  - Manage Hydra presets for all official runs
  - Maintain experiment matrix and naming conventions
  - Ensure outputs include resolved configs and metrics

## Integration Checklist

- New module registered in `persona_loop/core/factories.py`
- New config file added in matching `configs/<group>/`
- Smoke test passes: `python run_experiment.py`
- K sweep passes: `scripts/run_k_sweep.ps1`
- Documentation updated in `README.md`
