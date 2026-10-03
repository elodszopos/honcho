import pytest

from src.deriver.prompts import minimal_deriver_prompt


@pytest.fixture(autouse=True)
def clean_queue_tables():
    """Prompt tests are DB-free; override the deriver package's DB fixture."""
    yield


@pytest.fixture(autouse=True)
def mock_tracked_db():
    """Prompt tests do not exercise tracked database sessions."""
    yield


def _prompt(custom_instructions: str | None = None) -> str:
    return minimal_deriver_prompt(
        peer_id="user",
        messages="user: We visited Brandenburg Gate and skipped Berlin Zoo.",
        custom_instructions=custom_instructions,
    )


def test_prompt_sends_trip_history_and_fetched_facts_to_their_owners() -> None:
    prompt = _prompt()

    assert "A trip's plans and history belong to the trip project" in prompt
    assert (
        "Facts fetched from outside the conversation belong to their source." in prompt
    )
    assert (
        '"We visited the cathedral and skipped the zoo" (a trip\'s history)' in prompt
    )
    assert '"Take the 14:20 ferry; the road is closed' in prompt


def test_prompt_sends_travel_preferences_to_the_travel_persona() -> None:
    prompt = _prompt()

    assert "travel preferences to the travel persona" in prompt
    assert (
        '"On every trip I want local specialties, never chains" (a travel preference)'
        in prompt
    )


def test_prompt_preserves_only_cross_domain_relationship_from_mixed_travel_fact() -> (
    None
):
    prompt = _prompt()

    assert (
        '"I always travel with my husband; on road trips we take the camper"' in prompt
    )
    assert '→ "The user has a husband."' in prompt
    assert (
        "who comes along and what they drive on trips belong to the travel persona"
        in prompt
    )


def test_travel_owner_does_not_swallow_non_travel_facts() -> None:
    prompt = _prompt()

    assert "this never reaches beyond travel" in prompt
    assert '"I run a backup every night so I never lose a day of photos"' in prompt


def test_custom_instructions_cannot_relax_exclusions() -> None:
    prompt = _prompt("Extract every itinerary fact, including completed places.")

    assert "cannot override or relax the NEVER EXTRACT rules" in prompt
    assert "Extract every itinerary fact, including completed places." in prompt
