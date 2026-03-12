from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_data(path: Path) -> List[Dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_entries(sample_id: str, start: int, count: int) -> List[Dict[str, object]]:
    return [{"sample_id": sample_id, "qa_index": i} for i in range(start, start + count)]


def first_adv_index(sample: Dict[str, object]) -> int:
    qas = list(sample.get("qa", []))
    for idx, qa in enumerate(qas):
        adv = qa.get("adversarial_answer")
        if isinstance(adv, str) and adv.strip():
            return idx
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible LoCoMo slice files.")
    parser.add_argument("--data", default="data/locomo10.json")
    parser.add_argument("--out-dir", default="configs/benchmark/slices")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(data_path)
    if len(data) < 2:
        raise RuntimeError("Need at least 2 samples to build default slices")

    s0 = data[0]
    s1 = data[1]
    id0 = str(s0.get("sample_id", "sample0"))
    id1 = str(s1.get("sample_id", "sample1"))

    slices = {
        "quick": {
            "name": "quick",
            "description": "Single-sample quick sanity slice (first 10 QA).",
            "entries": make_entries(id0, 0, 10),
        },
        "formal": {
            "name": "formal",
            "description": "Single-sample formal slice (first 50 QA).",
            "entries": make_entries(id0, 0, 50),
        },
        "multisample": {
            "name": "multisample",
            "description": "Two-sample balanced slice (20 QA each from first two samples).",
            "entries": make_entries(id0, 0, 20) + make_entries(id1, 0, 20),
        },
        "advslice": {
            "name": "advslice",
            "description": "Adversarial-focused slice (20 QA from first adversarial index in first sample).",
            "entries": make_entries(id0, first_adv_index(s0), 20),
        },
    }

    for name, obj in slices.items():
        p = out_dir / f"{name}.json"
        p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
