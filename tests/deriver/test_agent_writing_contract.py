"""Agent-writing shape requirements for Honcho derivation prompts."""

import pytest

from src.deriver.prompts import minimal_deriver_prompt
from src.dialectic.prompts import agent_system_prompt


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

    for rule in (
        "One conclusion: one fact or preference.",
        "One conclusion: one short sentence.",
        "Length follows content: keep every exact qualifier, never pad with narrative.",
        "Split independent facts into separate conclusions.",
        "Exclude narrative and rationale.",
    ):
        assert rule in prompt

    assert "25 words" not in prompt


def test_dialectic_requires_atomic_machine_context():
    prompt = agent_system_prompt(
        observer="peer",
        observed="peer",
        observer_peer_card=None,
        observed_peer_card=None,
    )
    block = prompt.split("## OUTPUT CONTRACT", 1)[1]

    for rule in (
        "Prefer labeled bullets, checklists, or tables.",
        "One bullet: one fact, decision, contradiction, or gap.",
        "Length follows content: keep every exact qualifier, never pad with narrative.",
        "Label missing evidence as `Unknown: <specific gap>`.",
    ):
        assert rule in block

    assert "25 words" not in block
