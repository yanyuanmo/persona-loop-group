from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from persona_loop.core.factories import create_llm


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return default


def _extract_first_json_object(text: str) -> Optional[dict]:
    s = text.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        pass

    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    blob = m.group(0)
    try:
        obj = json.loads(blob)
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_fact_index(rows: List[dict]) -> Dict[Tuple[str, int], List[dict]]:
    out: Dict[Tuple[str, int], List[dict]] = {}
    for row in rows:
        key = (str(row.get("sample_id", "")), int(row.get("qa_index", -1)))
        facts = list(row.get("fact_bank", []))
        out[key] = facts
    return out


def _pick_style_facts(facts: List[dict], subject: str, top_k: int = 8) -> List[str]:
    subject_lc = subject.strip().lower()

    def _fmt(f: dict) -> str:
        owner = str(f.get("owner", "")).strip() or "unknown"
        slot = str(f.get("slot", "")).strip()
        value = str(f.get("value", "")).strip()
        source = str(f.get("source", "")).strip() or "unknown"
        if slot and value:
            return f"[{owner}] {slot}: {value} (src={source})"
        return f"[{owner}] {str(f.get('fact_text', '')).strip()} (src={source})"

    if subject_lc:
        matched = [f for f in facts if str(f.get("owner", "")).strip().lower() == subject_lc]
    else:
        matched = []

    chosen = matched if matched else facts
    return [_fmt(f) for f in chosen[: max(0, top_k)]]


def _build_judge_prompt(question: str, prediction: str, style_facts: List[str]) -> str:
    facts_block = "\n".join(f"- {x}" for x in style_facts) if style_facts else "- (none)"
    return (
        "You are evaluating persona STYLE retention in a QA answer. "
        "Focus on style/voice/role consistency, not factual correctness.\n\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        "  \"voice_consistency\": number,\n"
        "  \"linguistic_signature\": number,\n"
        "  \"role_consistency\": number,\n"
        "  \"style_content_balance\": number,\n"
        "  \"overall\": number,\n"
        "  \"note\": string\n"
        "}\n\n"
        "Scoring rules:\n"
        "- Each score in [0,1].\n"
        "- overall should reflect the 4 sub-scores (roughly their average).\n"
        "- note should be concise (<= 25 words).\n"
        "- Do not include markdown or any extra text.\n\n"
        f"Question:\n{question}\n\n"
        f"Model answer:\n{prediction}\n\n"
        f"Persona style hints:\n{facts_block}\n"
    )


def _judge_one(llm, question: str, prediction: str, style_facts: List[str]) -> Optional[dict]:
    prompt = _build_judge_prompt(question=question, prediction=prediction, style_facts=style_facts)
    raw = llm.generate(prompt=prompt, context="")
    parsed = _extract_first_json_object(raw)
    if not parsed:
        return None

    keys = [
        "voice_consistency",
        "linguistic_signature",
        "role_consistency",
        "style_content_balance",
        "overall",
    ]
    out: dict = {k: min(1.0, max(0.0, _safe_float(parsed.get(k, 0.0)))) for k in keys}
    out["note"] = str(parsed.get("note", "")).strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate persona style retention for QA predictions.")
    parser.add_argument("--predictions", required=True, help="Path to qa_predictions.json")
    parser.add_argument(
        "--fact-bank",
        default="",
        help="Optional path to full_persona_fact_bank.json for building style hints.",
    )
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-base-url", nargs="?", const="", default=None)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    predictions = _load_json(pred_path)
    if not isinstance(predictions, list):
        raise ValueError("--predictions must point to a JSON array")

    rows = predictions[: args.max_rows] if args.max_rows > 0 else predictions

    fact_index: Dict[Tuple[str, int], List[dict]] = {}
    if args.fact_bank:
        fb_path = Path(args.fact_bank)
        if not fb_path.exists():
            raise FileNotFoundError(f"Fact bank file not found: {fb_path}")
        fb = _load_json(fb_path)
        if not isinstance(fb, list):
            raise ValueError("--fact-bank must point to a JSON array")
        fact_index = _build_fact_index(fb)

    llm_kwargs: Dict[str, object] = {}
    if args.llm_base_url is not None:
        llm_kwargs["base_url"] = args.llm_base_url
    llm = create_llm(provider=args.llm_provider, model_name=args.llm_model, **llm_kwargs)

    style_rows: List[dict] = []
    for idx, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id", "")).strip()
        qa_index = int(row.get("qa_index", -1))
        question = str(row.get("question", "")).strip()
        prediction = str(row.get("prediction", "")).strip()
        subject = str(row.get("question_subject", "")).strip()
        f1 = _safe_float(row.get("f1", 0.0))

        facts = fact_index.get((sample_id, qa_index), [])
        style_facts = _pick_style_facts(facts=facts, subject=subject, top_k=8)

        judge = _judge_one(llm=llm, question=question, prediction=prediction, style_facts=style_facts)
        if judge is None:
            style_rows.append(
                {
                    "sample_id": sample_id,
                    "qa_index": qa_index,
                    "question": question,
                    "f1": f1,
                    "judge_ok": False,
                    "style_facts_count": len(style_facts),
                    "style_facts": style_facts,
                }
            )
            print(f"[{idx}/{len(rows)}] judge_parse_failed sample={sample_id}:{qa_index}")
            continue

        style_rows.append(
            {
                "sample_id": sample_id,
                "qa_index": qa_index,
                "question": question,
                "f1": f1,
                "judge_ok": True,
                "style_facts_count": len(style_facts),
                "style_facts": style_facts,
                **judge,
            }
        )
        print(
            f"[{idx}/{len(rows)}] style_ok sample={sample_id}:{qa_index} "
            f"overall={judge['overall']:.3f}"
        )

    ok_rows = [r for r in style_rows if r.get("judge_ok")]
    def _avg(key: str) -> float:
        vals = [_safe_float(r.get(key, 0.0)) for r in ok_rows]
        return float(mean(vals)) if vals else 0.0

    summary = {
        "count": len(style_rows),
        "judge_ok_count": len(ok_rows),
        "judge_ok_ratio": round(len(ok_rows) / max(1, len(style_rows)), 4),
        "voice_consistency": round(_avg("voice_consistency"), 4),
        "linguistic_signature": round(_avg("linguistic_signature"), 4),
        "role_consistency": round(_avg("role_consistency"), 4),
        "style_content_balance": round(_avg("style_content_balance"), 4),
        "psrs": round(_avg("overall"), 4),
        "psrs_weighted_by_f1": round(
            float(mean([_safe_float(r.get("overall", 0.0)) * _safe_float(r.get("f1", 0.0)) for r in ok_rows]))
            if ok_rows
            else 0.0,
            4,
        ),
    }

    out_dir = Path(args.out_dir) if args.out_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "style_rows.json").write_text(
        json.dumps(style_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "style_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Finished style eval. Output:", out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
