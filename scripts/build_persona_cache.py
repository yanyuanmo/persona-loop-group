"""Pre-extract persona facts for all (or a subset of) LoCoMo QA pairs and save to a cache file.

This lets you run the expensive GPT extraction ONCE, then reuse the results across many
eval runs (different agents, modes, etc.) without paying again.

Usage examples
--------------
# Hybrid extraction using local llama-server for the quick slice
python scripts/build_persona_cache.py \
    --data data/locomo10.json \
    --persona-mode hybrid \
    --llm-provider openai \
    --llm-model qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    --cache persona_cache/quick_hybrid.json \
    --slice-file configs/benchmark/slices/quick.json

# Hybrid extraction with GPT-4o-mini as extractor (no base_url = real OpenAI)
python scripts/build_persona_cache.py \
    --data data/locomo10.json \
    --persona-mode hybrid \
    --llm-provider openai \
    --llm-model gpt-4o-mini \
    --llm-base-url "" \
    --cache persona_cache/formal_hybrid_gpt4omini.json \
    --slice-file configs/benchmark/slices/formal.json

Then pass --persona-cache persona_cache/formal_hybrid_gpt4omini.json to run_locomo_eval.py
(or -PersonaCache in the ps1 scripts).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from persona_loop.core.factories import create_llm
from persona_loop.eval.persona_extractor import (
    extract_persona_facts_hybrid_with_stats,
    extract_persona_facts_llm_with_stats,
    extract_persona_facts_with_stats,
)


def load_locomo(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_local_env(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


def flatten_conversation(conv: Dict) -> List[Dict[str, str]]:
    flat: List[Dict[str, str]] = []
    for idx in range(1, 100):
        key = f"session_{idx}"
        if key not in conv:
            break
        for turn in conv[key]:
            dia_id = str(turn.get("dia_id", ""))
            text = str(turn.get("text", "")).strip()
            speaker = str(turn.get("speaker", "")).strip()
            if not speaker and ":" in dia_id:
                # LoCoMo dia_id is typically "<speaker>:<turn>", use it as a fallback.
                speaker = dia_id.split(":", 1)[0].strip()
            if dia_id and text:
                flat.append({"dia_id": dia_id, "text": text, "speaker": speaker})
    return flat


def load_slice_map(path: Path) -> Dict[str, List[int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("entries", raw) if isinstance(raw, dict) else raw
    grouped: Dict[str, List[int]] = {}
    for entry in entries:
        sid = str(entry.get("sample_id", "")).strip()
        qi = int(entry.get("qa_index", -1))
        if sid and qi >= 0:
            grouped.setdefault(sid, []).append(qi)
    return {k: sorted(set(v)) for k, v in grouped.items()}


def main() -> None:
    load_local_env(ROOT)
    parser = argparse.ArgumentParser(description="Pre-build persona fact cache.")
    parser.add_argument("--data", default="data/locomo10.json")
    parser.add_argument("--cache", required=True, help="Output cache JSON file path.")
    parser.add_argument("--persona-mode", choices=["derived", "hybrid", "llm"], default="hybrid")
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-base-url", nargs="?", const="", default=None,
                        help="base_url for the LLM. Bare flag = real OpenAI. Omit = inherit OPENAI_BASE_URL env var.")
    parser.add_argument("--persona-max-facts", type=int, default=24)
    parser.add_argument("--persona-hybrid-min-rule-facts", type=int, default=3)
    parser.add_argument("--persona-hybrid-llm-max-facts", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=0,
                        help="Limit visible history turns (0 = all).")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-qa", type=int, default=0)
    parser.add_argument("--slice-file", default="")
    parser.add_argument("--append", action="store_true",
                        help="If cache file exists, append new entries instead of overwriting.")
    args = parser.parse_args()

    # Build LLM
    llm_kwargs: dict = {}
    if args.llm_base_url is not None:
        # '' (bare flag) = real OpenAI; any URL = explicit base_url.
        # OpenAILLM uses _UNSET sentinel to distinguish "not passed" from "passed as empty".
        llm_kwargs["base_url"] = args.llm_base_url
    llm = create_llm(provider=args.llm_provider, model_name=args.llm_model, **llm_kwargs)

    # Load existing cache if appending
    cache_path = Path(args.cache)
    cache: Dict[str, dict] = {}
    if args.append and cache_path.exists():
        for row in json.loads(cache_path.read_text(encoding="utf-8")):
            key = f"{row['sample_id']}:{row['qa_index']}"
            cache[key] = row
        print(f"[CACHE] Loaded {len(cache)} existing entries from {cache_path}")

    # Load data
    samples = load_locomo(Path(args.data))
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    slice_map: Dict[str, List[int]] = {}
    if args.slice_file:
        slice_map = load_slice_map(Path(args.slice_file))

    # Count total
    total = 0
    for sample in samples:
        sid = str(sample.get("sample_id", ""))
        qas = list(sample.get("qa", []))
        if slice_map:
            total += len([i for i in slice_map.get(sid, []) if 0 <= i < len(qas)])
        else:
            total += len(qas)
    if args.max_qa > 0:
        total = min(total, args.max_qa)

    print(f"[BUILD] mode={args.persona_mode} llm={args.llm_provider}:{args.llm_model} "
          f"total_qa={total} cache={cache_path}")

    processed = 0
    skipped = 0
    start_time = time.time()

    for sample in samples:
        sid = str(sample.get("sample_id", ""))
        conv = sample.get("conversation", {})
        turns = flatten_conversation(conv)
        dia2idx = {t["dia_id"]: i for i, t in enumerate(turns)}
        qas = list(sample.get("qa", []))

        qa_pairs: List[tuple] = []
        if slice_map:
            for idx in slice_map.get(sid, []):
                if 0 <= idx < len(qas):
                    qa_pairs.append((idx, qas[idx]))
        else:
            qa_pairs = list(enumerate(qas))

        for qa_index, qa in qa_pairs:
            if args.max_qa > 0 and (processed + skipped) >= args.max_qa:
                break

            cache_key = f"{sid}:{qa_index}"
            if cache_key in cache:
                skipped += 1
                continue

            evidence = [str(x) for x in qa.get("evidence", [])]
            if evidence:
                max_idx = max(dia2idx.get(e, -1) for e in evidence)
                visible_turns = turns[: max_idx + 1] if max_idx >= 0 else turns
            else:
                visible_turns = turns

            if args.max_turns > 0:
                visible_turns = visible_turns[-args.max_turns:]

            if args.persona_mode == "derived":
                extracted = extract_persona_facts_with_stats(
                    visible_turns=visible_turns,
                    max_facts=args.persona_max_facts,
                )
            elif args.persona_mode == "llm":
                extracted = extract_persona_facts_llm_with_stats(
                    visible_turns=visible_turns,
                    llm=llm,
                    max_facts=args.persona_max_facts,
                )
            else:
                extracted = extract_persona_facts_hybrid_with_stats(
                    visible_turns=visible_turns,
                    llm=llm,
                    max_facts=args.persona_max_facts,
                    min_rule_facts=args.persona_hybrid_min_rule_facts,
                    llm_max_facts=args.persona_hybrid_llm_max_facts,
                )

            cache[cache_key] = {
                "sample_id": sid,
                "qa_index": qa_index,
                "question": str(qa.get("question", "")),
                "persona_mode": args.persona_mode,
                "persona_extract_stats": dict(extracted.get("stats", {})),
                "fact_bank": copy.deepcopy(list(extracted.get("facts", []))),
            }
            processed += 1

            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0.0
            remaining = (total - processed - skipped) / rate if rate > 0 else 0.0
            print(f"\r[{processed}/{total - skipped}] elapsed={elapsed:.0f}s rate={rate:.2f}qa/s ETA={remaining:.0f}s",
                  end="", flush=True)

    print()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(list(cache.values()), indent=2), encoding="utf-8")
    elapsed = time.time() - start_time
    print(f"[DONE] extracted={processed} skipped(cached)={skipped} total_saved={len(cache)} "
          f"elapsed={elapsed:.1f}s → {cache_path}")


if __name__ == "__main__":
    main()
