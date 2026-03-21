# obsolete/

This folder contains code from the original **LoComo QA** experiment phase.
It was moved here when the project pivoted to the `multimodal_dialog` dataset
and a clean persona-vs-memory architecture.

## Why it was retired

| Problem | Detail |
|---------|--------|
| Wrong `[PERSONA]` representation | Code extracted slot-value pairs from conversation turns (e.g. `recent_experience: LGBTQ support group yesterday`). Should be a fixed natural-language character description. |
| Stage B NLI on wrong target | NLI was checking history dialogue turns, not the agent's own generated responses. |
| `eval_mode` bypassed Stage B | Loop reset never exercised during evaluation runs. |
| PCS zero-overlap false positives | Selecting facts with zero-overlap produced ~40% spurious contradiction signal. |
| LoComo data dependency | All experiments tied to a single QA benchmark with no roleplay simulation. |

## Contents

### `run_experiment.py` + `configs/`

Hydra-based experiment entry point and config tree (agent/, llm/, memory/, consistency/, context_builder/, dataset/, experiment/ groups). Tied entirely to the LoComo QA benchmark and the old slot-value persona representation. Replaced by `scripts/run_multimodal_eval.py`.

### `context_builder/`

`priority_builder.py` — builds `[PERSONA] / [CORRECTION] / [HISTORY] / [RECENT]` context for QA mode. Superseded by the inline context building inside `PersonaLoopAgent.run_roleplay_turn()`.

### `prompts/`, `results/`, `persona_cache/`, `outputs/`

LoComo-era artefacts: system prompt templates, CSV/JSON result tables, per-conversation persona caches built by `build_persona_cache.py`, and Hydra run outputs.

### `scripts/`

| File | Purpose |
|------|---------|
| `run_locomo_eval.py` | 1700-line LoComo evaluation harness (QA F1/EM) |
| `analyze_pcs.py` | PCS post-analysis on LoComo artifacts |
| `build_locomo_slices.py` | Slice-building for LoComo conversation data |
| `build_persona_cache.py` | Extract per-conversation persona facts via LLM |
| `eval_persona_style.py` | Style evaluation prototype |
| `probe_local_context.py` | Debug: probe how context tokens are used |
| `profile_local_llm.py` | Throughput / latency profiling for local LLMs |
| `run_baselines.ps1` | PowerShell: run all baseline agents on LoComo |
| `run_hybrid_window_fairness_matrix.ps1` | Parameter sweep for hybrid agents |
| `run_k_sweep.ps1` | Sweep over loop-interval K |
| `run_local_llamacpp_server.ps1` | Helper to launch llama-server |
| `run_local_profile.ps1` | Profile local llama.cpp model |
| `run_locomo_matrix.ps1` | Full matrix run on LoComo |
| `run_locomo_preset.ps1` | Preset-based LoComo run |
| `run_memory_only_fairness_matrix.ps1` | Memory-only fairness sweep |
| `_tmp_analyze_debug.py` | Temporary debug analysis |
| `_tmp_test_json_schema.py` | Temporary JSON schema test |

### `agents/`

| File | Purpose |
|------|---------|
| `persona_loop_agent.py` | Old PersonaLoopAgent — incorrect Stage B, wrong persona representation |
| `continuous_agent.py` | Old ContinuousAgent (superseded by `continuous_agent_v2.py`) |
| `periodic_remind_agent.py` | Periodic persona-reminder injection agent |
| `ppa_agent.py` | Persona-Persistent Agent prototype |
| `rag_agent.py` | RAG-based agent (retrieval-augmented context) |
| `sliding_window_agent.py` | Fixed sliding-window context agent |

### `eval/`

| File | Purpose |
|------|---------|
| `persona_extractor.py` | LoComo slot-value persona extractor (per-QA extraction, wrong approach) |
| `llm_judge.py` | LLM-as-judge evaluation wrapper |
| `nli_consistency.py` | Old NLI consistency evaluator |
| `qa_metrics.py` | LoComo QA F1/EM metrics |
| `persona_metrics.py` | Per-fact NLI scoring tied to QA fact-banks |

### `tests/`

| File | Purpose |
|------|---------|
| `test_factories.py` | Tests for old factory signatures (periodic_remind, rag, sliding_window) |
| `test_priority_builder.py` | Tests for `priority_builder.py` (moved to context_builder above) |
