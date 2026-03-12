from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


def parse_sizes(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def estimate_tps(latency_sec: float, response_text: str, usage: Any) -> float:
    if latency_sec <= 0:
        return 0.0
    completion_tokens = None
    if usage is not None:
        completion_tokens = getattr(usage, "completion_tokens", None)
    if completion_tokens is None:
        completion_tokens = len(response_text.split())
    return float(completion_tokens) / latency_sec


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile local OpenAI-compatible LLM latency by prompt size.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "dummy"))
    parser.add_argument(
        "--model",
        default=os.environ.get("LOCAL_MODEL_NAME", "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"),
    )
    parser.add_argument("--sizes", default="2000,8000,16000,32000,64000")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--output", default="artifacts/local_profile")
    args = parser.parse_args()

    sizes = parse_sizes(args.sizes)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for n in sizes:
        prompt = "x " * n
        t0 = time.time()
        row: Dict[str, Any] = {
            "prompt_words": n,
            "ok": False,
            "latency_sec": 0.0,
            "tokens_per_sec": 0.0,
            "error": "",
        }
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens,
                temperature=0,
            )
            latency = time.time() - t0
            text = (resp.choices[0].message.content or "").strip()
            row["ok"] = True
            row["latency_sec"] = round(latency, 3)
            row["tokens_per_sec"] = round(estimate_tps(latency, text, getattr(resp, "usage", None)), 3)
            row["preview"] = text[:80]
        except Exception as exc:  # noqa: BLE001
            latency = time.time() - t0
            row["latency_sec"] = round(latency, 3)
            row["error"] = str(exc)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True))

    summary = {
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "sizes": sizes,
        "results": rows,
    }
    (out_dir / "profile.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
