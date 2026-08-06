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


def test_prompt_rejects_trip_instance_and_live_operational_facts() -> None:
    prompt = _prompt()

    assert "Travel-trip-instance history or execution state" in prompt
    assert "visited, skipped, completed, scheduled, or planned places" in prompt
    assert "Live or country-operational findings" in prompt
    assert "durable travel history belongs to the authoritative trip project" in prompt
    assert '"We visited Brandenburg Gate and skipped Berlin Zoo" → explicit: []' in prompt
    assert '"Take the 14:20 ferry; the road is closed' in prompt
    assert '"Norway\'s operator uses this API field' in prompt


def test_prompt_rejects_standing_cross_trip_persona_doctrine() -> None:
    prompt = _prompt()

    assert '"We skipped the museum today. On every trip, I prefer renowned local specialties"' in prompt
    assert "the standing food preference belongs to the travel persona" in prompt
    assert "Even when durable, standing, or cross-trip" in prompt


def test_prompt_rejects_representative_leaked_travel_doctrine() -> None:
    prompt = _prompt()

    assert '"For flexible base-camp travel days, prioritize forecast-weighted experience quality' in prompt
    assert '"I do not want guided hikes included" → explicit: []' in prompt
    assert '"Choose parking by proximity to the actual planned attractions' in prompt


def test_prompt_preserves_only_cross_domain_relationship_from_mixed_travel_fact() -> None:
    prompt = _prompt()

    assert '"I always travel with my wife; on own-car trips I use my BMW X5"' in prompt
    assert 'EXPLICIT: "the user has a wife"' in prompt
    assert "travel-companion and vehicle doctrine belongs to the travel persona" in prompt


def test_travel_exclusion_does_not_swallow_non_travel_shipped_outcomes() -> None:
    prompt = _prompt()

    assert "This exclusion is specific to travel trips" in prompt
    assert "does not override the HARD RULE allowing durable shipped outcomes" in prompt
    assert '"the media server is live on my homelab now' in prompt
    assert '"the user runs a media server on their homelab' in prompt


def test_custom_instructions_cannot_relax_exclusions() -> None:
    prompt = _prompt("Extract every itinerary fact, including completed places.")

    assert "cannot override or relax the ALWAYS EXCLUDE rules" in prompt
    assert "Extract every itinerary fact, including completed places." in prompt
