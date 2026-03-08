from __future__ import annotations

from typing import Any, Dict, Optional, Type

from persona_loop.agents.base_agent import BaseAgent
from persona_loop.agents.continuous_agent import ContinuousAgent
from persona_loop.agents.periodic_remind_agent import PeriodicRemindAgent
from persona_loop.agents.ppa_agent import PPAAgent
from persona_loop.agents.persona_loop_agent import PersonaLoopAgent
from persona_loop.agents.rag_agent import RAGAgent
from persona_loop.agents.sliding_window_agent import SlidingWindowAgent
from persona_loop.consistency.base_checker import BaseChecker
from persona_loop.consistency.deberta_checker import DebertaChecker
from persona_loop.llm.base_llm import BaseLLM
from persona_loop.llm.hf_llm import HuggingFaceLLM
from persona_loop.llm.openai_llm import OpenAILLM
from persona_loop.memory.base_memory import BaseMemory
from persona_loop.memory.chroma_memory import ChromaMemory
from persona_loop.memory.faiss_memory import FaissMemory

AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "persona_loop": PersonaLoopAgent,
    "continuous": ContinuousAgent,
    "periodic_remind": PeriodicRemindAgent,
    "rag": RAGAgent,
    "sliding_window": SlidingWindowAgent,
    "ppa": PPAAgent,
}

LLM_REGISTRY: Dict[str, Type[BaseLLM]] = {
    "openai": OpenAILLM,
    "hf": HuggingFaceLLM,
}

MEMORY_REGISTRY: Dict[str, Type[BaseMemory]] = {
    "chroma": ChromaMemory,
    "faiss": FaissMemory,
}

CHECKER_REGISTRY: Dict[str, Type[BaseChecker]] = {
    "deberta": DebertaChecker,
}


def _assert_known(name: str, registry: Dict[str, Any], kind: str) -> None:
    if name not in registry:
        allowed = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown {kind}: '{name}'. Allowed: {allowed}")


def create_llm(provider: str, model_name: str) -> BaseLLM:
    _assert_known(provider, LLM_REGISTRY, "llm provider")
    return LLM_REGISTRY[provider](model_name=model_name)


def create_memory(memory_type: Optional[str]) -> Optional[BaseMemory]:
    if memory_type is None:
        return None
    _assert_known(memory_type, MEMORY_REGISTRY, "memory backend")
    return MEMORY_REGISTRY[memory_type]()


def create_checker(enabled: bool, checker_type: Optional[str], model_name: Optional[str]) -> Optional[BaseChecker]:
    if not enabled:
        return None
    if checker_type is None:
        raise ValueError("consistency.type must be set when consistency.enabled=true")
    _assert_known(checker_type, CHECKER_REGISTRY, "consistency checker")
    if model_name is None:
        raise ValueError("consistency.model_name must be set when consistency.enabled=true")
    return CHECKER_REGISTRY[checker_type](model_name=model_name)


def create_agent(name: str, llm: BaseLLM, memory: Optional[BaseMemory], checker: Optional[BaseChecker]) -> BaseAgent:
    _assert_known(name, AGENT_REGISTRY, "agent")
    return AGENT_REGISTRY[name](llm=llm, memory=memory, checker=checker)
