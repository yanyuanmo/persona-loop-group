import pytest

from persona_loop.core.factories import create_checker
from persona_loop.core.factories import create_llm
from persona_loop.core.factories import create_memory


def test_create_llm_hf():
    llm = create_llm(provider="hf", model_name="test-model")
    assert llm.model_name == "test-model"


def test_create_memory_none():
    memory = create_memory(memory_type=None)
    assert memory is None


def test_create_checker_requires_model():
    with pytest.raises(ValueError):
        create_checker(enabled=True, checker_type="deberta", model_name=None)
