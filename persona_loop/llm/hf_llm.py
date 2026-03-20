from __future__ import annotations

from persona_loop.llm.base_llm import BaseLLM


class HuggingFaceLLM(BaseLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFace local generation requires transformers. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self._pipeline = pipeline(
            task="text-generation",
            model=self.model_name,
            tokenizer=self.model_name,
            trust_remote_code=True,
        )

    def generate(self, prompt: str, context: str) -> str:
        self._load_pipeline()
        composed = self.build_message(prompt=prompt, context=context)
        outputs = self._pipeline(
            composed,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            return_full_text=False,
        )
        text = str(outputs[0]["generated_text"]).strip()
        return text if text else "Not sure based on the provided context."
