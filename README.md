# Persona Loop Research Skeleton

A minimal, publication-oriented framework for persona consistency experiments.

## What Is Ready

- Plug-and-play modules for `agent`, `llm`, `memory`, `consistency`, and `context_builder`
- Hydra config groups for fast ablation and fair baseline comparison
- Unified experiment entrypoint with artifact export
- Team split document: `TEAM_SPLIT.md`

## Project Layout

```text
persona_loop/
├── agents/                  # method and baseline implementations
├── consistency/             # NLI checker interface and implementations
├── context_builder/         # context reconstruction strategies
├── core/                    # component factories and orchestration helpers
├── eval/                    # metrics and judge hooks
├── llm/                     # LLM backend adapters
├── memory/                  # vector memory backends
└── __init__.py
configs/
├── config.yaml              # default composition entrypoint
├── agent/
├── llm/
├── memory/
├── consistency/
├── context_builder/
├── dataset/
└── experiment/
scripts/
├── run_k_sweep.ps1
tests/
├── test_priority_builder.py
run_experiment.py
TEAM_SPLIT.md
requirements.txt
```

## Quick Start

1. Use the prepared conda env and install deps:

```powershell
conda activate persona-loop
pip install -r requirements.txt
```

2. Create local secret file from template:

```powershell
Copy-Item .env.example .env
```

Fill `.env` with your local keys (`OPENAI_API_KEY`, `WANDB_API_KEY`, `HF_TOKEN`).

3. Run default experiment:

```powershell
python run_experiment.py
```

4. Run K=5 preset:

```powershell
python run_experiment.py experiment=k5
```

## Safe Push Checklist

- Keep real keys only in `.env` (ignored by git)
- Never hardcode keys in `.py` or `.yaml`
- Commit only `.env.example` as template
- Before push, quickly check staged diff:

```powershell
git diff --staged
```

## Common Overrides

Run RAG baseline:

```powershell
python run_experiment.py agent=rag
```

Run direct full-context baseline:

```powershell
python run_experiment.py agent=continuous experiment.run_name=baseline_continuous
```

Run sliding-window baseline with custom window size:

```powershell
python run_experiment.py agent=sliding_window benchmark.sliding_window.history_turns=5
```

Run periodic-reminder baseline with K-turn reminder interval:

```powershell
python run_experiment.py agent=periodic_remind benchmark.periodic_remind.interval=4
```

## LoCoMo Consistency Evaluation

Run QA + real NLI consistency evaluation on local LoCoMo JSON:

```powershell
python scripts/run_locomo_eval.py --data data/locomo10.json --agent continuous --max-turns 100 --max-samples 2
```

Main outputs:

- `artifacts/locomo_eval/qa_predictions.json`
- `artifacts/locomo_eval/qa_metrics.json`

Metrics focus on consistency:

- `nli_entailment_gold` (higher is better)
- `nli_contradiction_gold` (lower is better)
- `nli_entailment_adv` (lower is better)
- `nli_contradiction_adv` (higher is better)

`em`/`f1` are auxiliary QA sanity metrics.

Disable NLI checker:

```powershell
python run_experiment.py consistency=off
```

Switch memory backend:

```powershell
python run_experiment.py memory=faiss
```

Use OpenAI backend config:

```powershell
python run_experiment.py llm=openai_gpt4o_mini
```

Use local Qwen backend (default):

```powershell
python run_experiment.py llm=qwen_local_7b
```

Use Kimi backend:

```powershell
python run_experiment.py llm=kimi_kimi_latest
```

Kimi requires `KIMI_API_KEY` in `.env` (never commit real keys).

Change run name and K directly:

```powershell
python run_experiment.py experiment.run_name=my_run experiment.k=6
```

## Reproducibility

- Global seed is controlled by `experiment.seed`
- All outputs are saved to `artifacts/<run_name>/`
- Resolved config is exported as `resolved_config.json`
- Predictions and metrics are exported as JSON for later analysis

## Team Workflow

- New method implementation: edit `persona_loop/agents/`
- New retrieval backend: edit `persona_loop/memory/`
- New checker: edit `persona_loop/consistency/`
- New experiment setup: add file under `configs/*/`
- Keep API contracts from `base_*` classes stable to avoid merge conflicts

Detailed owner-based split is in `TEAM_SPLIT.md`.
