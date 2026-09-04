"""Agent-writing shape requirements for Honcho derivation prompts."""

import pytest

from src.deriver.prompts import minimal_deriver_prompt
from src.dialectic.prompts import agent_system_prompt
from src.writing_contract import (
    CONCLUSION_WRITING_CONTRACT,
    LLM_CONSUMED_WRITING_CONTRACT,
)


@pytest.fixture(autouse=True)
def clean_queue_tables():
    """Prompt tests are DB-free."""
    yield


@pytest.fixture(autouse=True)
def mock_tracked_db():
    """Prompt tests do not open tracked database sessions."""
    yield


def test_deriver_requires_atomic_conclusion_text():
    prompt = minimal_deriver_prompt(
        peer_id="peer",
        messages="USER: example",
    )

    assert CONCLUSION_WRITING_CONTRACT in prompt
    assert "Target 500 characters." in prompt
    assert "Never exceed 800 characters." in prompt
    assert "semantic compression, not concatenation" in prompt
    assert "keep every exact qualifier" not in prompt

    assert "25 words" not in prompt


def test_dialectic_requires_atomic_machine_context():
    prompt = agent_system_prompt(
        observer="peer",
        observed="peer",
        observer_peer_card=None,
        observed_peer_card=None,
    )
    assert LLM_CONSUMED_WRITING_CONTRACT in prompt
    block = prompt.split("## DIALECTIC OUTPUT", 1)[1]

    assert "Label missing evidence as `Unknown: <specific gap>`." in block
    assert "Preserve material uncertainty and evidence limits." in block
    assert "keep every exact qualifier" not in prompt

    assert "25 words" not in block
