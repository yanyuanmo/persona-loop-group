# Persona Loop

A framework for evaluating **persona consistency** in long-form roleplay dialogue,
using a periodic context-reconstruction mechanism (Persona Loop) to keep a language
model on-character across a multi-session conversation.

The core research question: does periodically refreshing the LLM context with
NLI-detected corrections and retrieved memory help the model stay more consistent
with its fixed character description than a simple baseline?

---

## How It Works

### Experiment Setup

The dataset contains real multi-session dialogues between two named characters
(e.g. Yangyang and Chenmo). Each character has a `persona_summary` — a fixed
natural-language description of who they are (personality, background, habits, etc.).

The experiment **replays** the original conversation:
- On **partner turns**: use the original ground-truth text directly (not generated).
- On **agent turns**: have the LLM generate a response *in character*.

After the full replay, every generated response is evaluated against the character's
`persona_summary` via NLI and/or LLM-as-judge to get the Persona Consistency Score (PCS).

### What the LLM Sees

Each call uses a **system + user** chat structure (`temperature=0.7`):

```
[system]
You are Yangyang. Stay fully in character at all times.

Your character description:
Yangyang is a lively and warm-hearted young woman who loves music...

Priority rule: [CORRECTION] and [PERSONA] are authoritative ground truth.
If they contradict [HISTORY] or [MEMORY], trust [CORRECTION]/[PERSONA] first.

Context tags:
- [PERSONA]    Fixed character description.
- [MEMORY]     Retrieved snippets from past conversations.
- [HISTORY]    Recent verbatim conversation turns.
- [CORRECTION] A detected out-of-character statement — correct course.

[user]
[CORRECTION] You said 'I don't know my family well' — this contradicts your persona...
[MEMORY]     Chenmo: 你最近去哪玩了？
             Yangyang: 去了上海，特别喜欢黄浦江的夜景
[HISTORY]    Chenmo: ...
             Yangyang: ...

Chenmo says: "你之前说过你家在北京对么？"
Respond naturally as Yangyang in 1-3 sentences.
```

The `context_extra` block (CORRECTION + MEMORY + HISTORY) is what distinguishes
Persona Loop from the baseline. The baseline always passes an empty `context_extra`.

---

## Agents

### Continuous (Baseline)

`ContinuousAgent` — no memory, no loop, no correction.

Each agent turn calls:
```python
llm.generate_roleplay(speaker, partner, partner_text, persona_summary, context_extra="")
```

The system prompt contains only the `persona_summary`. The LLM has no access to
history or corrections. A configurable `--max-history-window` (default 20) controls
how many recent turns are kept in context.

### Persona Loop

`PersonaLoopAgent` — runs Stage A/B/C/D every K turns (default K=8).

**Non-loop turns (1 to K−1):**

Generate a response using whatever `context_extra` was built at the last loop
(or empty at the start). After each turn, append the raw dialogue to
`_recent_buffer` and the agent's response to `_agent_responses`.

**Loop turn (every K-th turn) — Stage A→B→C→D:**

| Stage | What happens |
|-------|--------------|
| **A** | **Persist**: write all K turns in `_recent_buffer` into the vector memory store (`EmbeddingMemory`). Each item is `"{partner}: {text}\n{speaker}: {response}"`. |
| **B** | **Correct**: for each response in `_agent_responses`, run NLI with `premise=persona_summary, hypothesis=response`. If score < −threshold, emit a `[CORRECTION]` hint. Max `max_corrections` hints per loop. |
| **C** | **Retrieve**: use the partner's current message as a query, fetch top-k semantically similar items from memory. |
| **D** | **Rebuild**: assemble `context_extra = [CORRECTION]s + [MEMORY]s + last recent_turns [HISTORY]`. Clear `_recent_buffer` and `_agent_responses` for the next K-window. |

Then generate the response normally with the fresh `context_extra`.

**Default parameters:**

| Parameter | Default | Meaning |
|-----------|---------|-------|
| `loop_interval` | `8` | K — turns between loop resets |
| `retrieval_top_k` | `3` | Stage C: memory snippets to retrieve |
| `recent_turns` | `3` | Stage D: recent history turns to keep in [HISTORY] |
| `nli_threshold` | `0.3` | Stage B: contradiction detection threshold |
| `max_corrections` | `2` | Stage B: max [CORRECTION] hints per loop |

### Ablation Modes

Use `--loop-ablation` to disable individual stages for component analysis:

| Token | Effect |
|-------|--------|
| `disable_corrections` | Disable Stage B (no NLI correction hints) |
| `disable_persona_persist` | Disable Stage A & C (no memory write/retrieve, context rebuild only) |

Example: `--loop-ablation disable_persona_persist`

---

## Evaluation Metrics

### Method A: NLI-Based PCS

**Persona Consistency Score** measures how well the generated responses agree
with the character's fixed identity using NLI.

For each generated response:
- NLI premise = `persona_summary`
- NLI hypothesis = response text
- Model: `cross-encoder/nli-deberta-v3-base` (default)

$$\text{PCS}_{\text{NLI}} = \frac{1}{N} \sum_{i=1}^{N} (p_{\text{entailment},i} - p_{\text{contradiction},i})$$

Higher PCS = more persona-consistent. Additional reported metrics:
- `persona_entailment` — mean entailment probability
- `persona_contradiction` — mean contradiction probability
- `persona_contradiction_max` — worst single turn
- `persona_any_contra_ratio` — fraction of turns where contradiction > 0.5

### Method B: LLM-as-Judge PCS

Uses GPT-4o (or another LLM) as a judge to rate persona consistency on a 1–5 scale,
then normalizes to [-1, 1] to mirror the NLI PCS sign convention.

$$\text{PCS}_{\text{judge}} = \frac{2 \cdot (\text{raw\_score} - 1)}{S - 1} - 1$$

where $S$ is the scale (default 5). Enable with `--judge`.

The judge evaluator captures **implicit** persona signals (tone, attitude, style) that
NLI models often miss, especially on natural-length dialogue responses (< 200 chars).

---

## Data

```
data/multimodal_dialog/
    pair1/         ← Yangyang & Chenmo      (3 sessions, ~60 turns)
    pair2/         ← Xiaoxuan & Minghao     (3 sessions, ~59 turns)
    pair3/         ← Lingling & Zihao       (3 sessions, ~60 turns)
    pair4/         ← Qingqing & Ruochen     (3 sessions, ~60 turns)
    pair5/         ← Mingzhi & Feifei       (5 sessions, ~1000 turns)
    pair6/         ← Mingzhi & Feifei       (8 sessions, ~480 turns)
    example/       ← sample data for reference
```

| Pair | Characters | Sessions | Approx Turns | Notes |
|------|-----------|----------|---|---|
| pair1 | Yangyang & Chenmo | 3 | 60 | Warm/empathetic vs quiet/witty |
| pair2 | Xiaoxuan & Minghao | 3 | 59 | Adventurous/loyal vs calm/thoughtful |
| pair3 | Lingling & Zihao | 3 | 60 | Dreamy/poetic vs warm/curious |
| pair4 | Qingqing & Ruochen | 3 | 60 | Independent/witty vs thoughtful/humble |
| pair5 | Mingzhi & Feifei | 5 | 1000 | Ultra-long sessions (200 turns each); shows degradation artifacts |
| pair6 | Mingzhi & Feifei | 8 | 480 | **Primary dataset** — balanced length (60 turns/session), 67-event graph |

**Recommendation:** Use **pair6** for most experiments. Pair1–4 are too short (60 turns)
to trigger meaningful K-sensitivity. Pair5's ultra-long sessions lead to repetition artifacts.

Each agent JSON contains:

| Field | Description |
|-------|-------------|
| `name` | Display name (e.g. `"Yangyang"`) |
| `persona_summary` | Fixed character description in natural language |
| `session_N` | List of `{speaker, clean_text, text, dia_id}` turns |
| `session_N_facts` | Per-speaker observed facts: `{speaker: [(fact_text, dia_id), ...]}` |
| `session_N_summary` | Narrative summary of the session |
| `session_N_date_time` | Date/time string for the session |

Both agents in a pair share the **same** conversation turns. Only `persona_summary`
(and `name`) differ between agent_a.json and agent_b.json.

The loader prefers `clean_text` over `text` per turn.

---

## Project Layout

```
persona_loop/
├── agents/
│   ├── base_agent.py
│   ├── continuous_agent_v2.py      ← baseline: no loop, no memory
│   └── persona_loop_agent_v2.py    ← Stage A/B/C/D loop agent
├── consistency/
│   ├── base_checker.py
│   └── deberta_checker.py          ← Stage B NLI contradiction checker
├── core/
│   └── factories.py                ← create_llm / create_memory / create_checker / create_agent
├── data/
│   └── multimodal_loader.py        ← load_pair / load_all_pairs / DialogSample
├── eval/
│   └── nli_scorer.py               ← NLIScorer for PCS computation
├── llm/
│   ├── base_llm.py                 ← build_roleplay_message (system+user tuple)
│   ├── openai_llm.py               ← generate_roleplay with temperature=0.7
│   ├── hf_llm.py                   ← HuggingFace fallback
│   └── kimi_llm.py                 ← Kimi API backend
└── memory/
    ├── base_memory.py              ← abstract interface with reset()
    ├── embedding_memory.py         ← sentence-transformers in-memory store (default)
    └── chroma_memory.py            ← ChromaDB persistent store (optional)
scripts/
├── run_multimodal_eval.py          ← main evaluation entry point
├── run_exp2_k_sensitivity.py       ← K-sensitivity sweep wrapper
├── run_k_sensitivity_pair1.sh      ← K∈{2,4,8,16,32} on pair1
├── run_k_sensitivity_pair4.sh      ← K∈{2,4,8,16,32} on pair4
├── run_k_sensitivity.sh            ← K-sensitivity on pair6
└── run_llamacpp_server.sh          ← start local llama.cpp server
tests/
└── test_smoke_v2.py                ← smoke tests for agents + data loader
data/
└── multimodal_dialog/pair1/ .. pair6/
artifacts/                          ← experiment output (summary.json, turn records, etc.)
```

---

## Quick Start

### 1. Environment

```bash
conda activate persona-loop
pip install -r requirements.txt
```

### 2. Start the LLM backend

```bash
# In a separate terminal:
bash scripts/run_llamacpp_server.sh
# Or manually:
./tools/llama.cpp/build/bin/llama-server \
    -m models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    --host 127.0.0.1 --port 8080 -c 4096
```

### 3. Set environment variables

```bash
export OPENAI_API_KEY="local"
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
```

### 4. Run evaluation

**Continuous baseline** (no loop, no memory):
```bash
python scripts/run_multimodal_eval.py \
    --data data/multimodal_dialog \
    --agent continuous \
    --llm-provider openai \
    --llm-model qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    --output artifacts/multimodal_continuous
```

**Persona Loop** (Stage A/B/C/D, loop every 8 turns):
```bash
python scripts/run_multimodal_eval.py \
    --data data/multimodal_dialog \
    --agent persona_loop \
    --llm-provider openai \
    --llm-model qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    --memory-backend embedding \
    --loop-interval 8 \
    --loop-retrieval-topk 3 \
    --loop-recent-turns 3 \
    --output artifacts/multimodal_pl
```

**With LLM-as-Judge evaluation** (GPT-4o scores each turn 1–5):
```bash
python scripts/run_multimodal_eval.py \
    --data data/multimodal_dialog \
    --agent persona_loop \
    --llm-provider openai \
    --llm-model qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    --judge --judge-provider openai --judge-base-url "" \
    --pairs pair6 --max-sessions 2 \
    --output artifacts/pair6_judge
```

**K-sensitivity sweep** (pair6, K=2/4/8/16/32 + continuous):
```bash
bash scripts/run_k_sensitivity.sh
```

**Component ablation** (disable memory):
```bash
python scripts/run_multimodal_eval.py \
    --agent persona_loop \
    --llm-provider openai \
    --llm-model qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    --loop-ablation disable_persona_persist \
    --pairs pair6 --max-sessions 2 \
    --output artifacts/pair6_no_memory
```

**Fast debug** (skip NLI, single pair):
```bash
python scripts/run_multimodal_eval.py \
    --data data/multimodal_dialog \
    --agent persona_loop \
    --llm-provider openai \
    --llm-model qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    --skip-nli --pairs pair1 \
    --output artifacts/debug_pl
```

### 5. Check results

```bash
python -c "import json; m=json.load(open('artifacts/multimodal_pl/summary.json')); print(json.dumps(m, indent=2))"
```

---

## CLI Reference

`scripts/run_multimodal_eval.py` arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| **Core** | | |
| `--data` | `data/multimodal_dialog` | Path to data directory |
| `--agent` | `persona_loop` | `continuous` or `persona_loop` |
| `--pairs` | *(all)* | Comma-separated pair names, e.g. `pair1,pair6` |
| `--max-sessions` | `0` (all) | Max sessions per pair (e.g. `2` = sessions 1–2 only) |
| `--output` | `artifacts/multimodal_eval` | Output directory |
| **LLM** | | |
| `--llm-provider` | `openai` | `openai`, `hf`, or `kimi` |
| `--llm-model` | *(required)* | Model name / filename |
| `--llm-base-url` | env var | Override `OPENAI_BASE_URL` |
| **NLI Evaluation** | | |
| `--nli-model` | `cross-encoder/nli-deberta-v3-base` | NLI model for PCS |
| `--skip-nli` | off | Skip NLI evaluation (no PCS computed) |
| **Judge Evaluation** | | |
| `--judge` | off | Enable LLM-as-judge scoring |
| `--judge-model` | *(llm-model)* | Judge LLM model (e.g. `gpt-4o`) |
| `--judge-provider` | *(llm-provider)* | Judge LLM provider |
| `--judge-base-url` | *(llm-base-url)* | Judge base URL; `""` for real OpenAI |
| `--judge-scale` | `5` | Rating scale (1–N) |
| **Persona Loop** | | |
| `--memory-backend` | `embedding` | `embedding` or `bm25` |
| `--loop-interval` | `8` | K: turns between loop resets |
| `--loop-retrieval-topk` | `3` | Stage C: memory snippets to retrieve |
| `--loop-recent-turns` | `3` | Stage D: recent history turns to keep |
| `--loop-nli-threshold` | `0.3` | Stage B: contradiction detection threshold |
| `--loop-ablation` | *(none)* | Comma-separated: `disable_corrections`, `disable_persona_persist` |
| **Continuous** | | |
| `--max-history-window` | `20` | Max history lines in context (0=unlimited) |

---

## Output Files

Each `--output` directory contains:

| File | Description |
|------|-------------|
| `summary.json` | Aggregate PCS (NLI + judge), per-role scores, loop stats, run metadata |
| `per_role_results.json` | Per-agent-role results (no turn records) |
| `turn_records.json` | Per-turn: `dia_id`, `gold_text`, `response`, `loop_reset`, `partner_text`, `judge_score` |
| `run_manifest.json` | Full argument snapshot for reproducibility |

---

## Smoke Tests

```bash
pytest tests/test_smoke_v2.py
```

Tests that: loop fires at the right intervals, data loads all pairs correctly,
`ContinuousAgent.run_roleplay_turn` returns without error.
