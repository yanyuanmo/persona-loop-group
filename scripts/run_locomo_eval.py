from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
import time
from typing import Dict, List, Optional

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from persona_loop.core.factories import create_agent
from persona_loop.core.factories import create_llm
from persona_loop.core.factories import create_memory
from persona_loop.eval.nli_scorer import NLIScorer
from persona_loop.eval.persona_extractor import extract_persona_facts
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
            if dia_id and text:
                flat.append({"dia_id": dia_id, "text": text})
    return flat


def build_history(turns: List[Dict[str, str]], max_turns: int) -> List[str]:
    items = [f"{t['dia_id']} {t['text']}" for t in turns]
    if max_turns <= 0:
        return items
    return items[-max_turns:]


def answer_question(agent, question: str, history: List[str]) -> str:
    context = "\n".join([f"[HISTORY] {x}" for x in history])
    result = agent.run_turn(prompt=question, context=context)
    return str(result["response"])


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
    parser.add_argument("--persona-mode", choices=["off", "derived", "file"], default="derived")
    parser.add_argument("--persona-file", default="")
    parser.add_argument("--persona-topk", type=int, default=5)
    parser.add_argument("--persona-max-facts", type=int, default=24)
    parser.add_argument("--output", default="artifacts/locomo_eval")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = create_llm(provider=args.llm_provider, model_name=args.llm_model)
    memory = create_memory(memory_type="chroma" if args.agent == "rag" else None)
    agent = create_agent(name=args.agent, llm=llm, memory=memory, checker=None)
    nli = None if args.skip_nli else NLIScorer(model_name=args.nli_model)

    rows: List[Dict[str, object]] = []
    persona_debug_rows: List[Dict[str, object]] = []
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

    print(
        f"Starting LoCoMo eval: samples={len(samples)}, target_qa={total_qa}, "
        f"agent={args.agent}, llm={args.llm_provider}:{args.llm_model}, "
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
            prediction = answer_question(agent=agent, question=question, history=history)

            fact_bank: List[Dict[str, object]] = []
            if args.persona_mode == "derived":
                fact_bank = extract_persona_facts(
                    visible_turns=visible_turns,
                    max_facts=args.persona_max_facts,
                )
            elif args.persona_mode == "file":
                fact_bank = list(persona_file_map.get(sample_id, []))

            persona_metrics = compute_persona_metrics(
                prediction=prediction,
                fact_bank=fact_bank,
                nli=nli,
                top_k=args.persona_topk,
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
            persona_debug_rows.append(
                {
                    "sample_id": sample_id,
                    "question": question,
                    "persona_mode": args.persona_mode,
                    "eval_mode": args.eval_mode,
                    "evidence_visible": evidence_visible,
                    "persona_facts_total": row.get("persona_facts_total", 0),
                    "persona_facts_used": row.get("persona_facts_used", 0),
                    "persona_fact_texts": row.get("persona_fact_texts", []),
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
        "evidence_visible_ratio": avg("evidence_visible"),
        "history_items_avg": avg("history_items"),
        "adversarial_count": int(sum(1 for r in rows if isinstance(r.get("adversarial_answer"), str) and str(r.get("adversarial_answer", "")).strip())),
    }

    (output_dir / "qa_predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "persona_facts_debug.json").write_text(
        json.dumps(persona_debug_rows, indent=2), encoding="utf-8"
    )
    (output_dir / "qa_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

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
