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
