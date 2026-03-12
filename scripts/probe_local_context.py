from __future__ import annotations

import os
import time

from openai import OpenAI


def main() -> None:
    base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    model = os.environ.get(
        "LOCAL_MODEL_NAME",
        "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    )

    size_env = os.environ.get("PROBE_SIZES", "").strip()
    if size_env:
        sizes = [int(x.strip()) for x in size_env.split(",") if x.strip()]
    else:
        sizes = [4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000, 36000]
    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"base_url={base_url}")
    print(f"model={model}")

    for n in sizes:
        prompt = "x " * n
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8,
                temperature=0,
            )
            dt = time.time() - t0
            text = (resp.choices[0].message.content or "").strip().replace("\n", " ")
            print(f"OK n={n} sec={dt:.2f} text={text[:60]!r}")
        except Exception as exc:
            dt = time.time() - t0
            print(f"FAIL n={n} sec={dt:.2f} err={exc}")


if __name__ == "__main__":
    main()
