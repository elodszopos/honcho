"""Values this fork holds. Upstream rewrites its own tests on every pull, so a merge that
adopts upstream's value goes green unless the literal is asserted here."""

import json
import tomllib
from pathlib import Path
from typing import Any, get_type_hints

import pytest
from pydantic import ValidationError

from src import schemas
from src.config import DeriverSettings
from src.crud import document as crud_document
from src.crud import representation as crud_representation
from src.deriver import queue_manager
from src.deriver.prompts import minimal_deriver_prompt
from src.llm import registry as llm_registry
from src.llm.api import is_transient_llm_error
from src.utils import representation as representation_module

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_queue_tables():
    """Held-value assertions read settings, source and prompts; no database."""
    yield


@pytest.fixture(autouse=True)
def mock_tracked_db():
    yield


def _http_error(status: int, body: str) -> Exception:
    error = RuntimeError(body)
    error.status_code = status  # pyright: ignore[reportAttributeAccessIssue]
    return error


def _deriver_default(field: str) -> Any:
    """The declared default, not the resolved value -- .env overrides say nothing about the fork."""
    return DeriverSettings.model_fields[field].default


def test_deriver_work_unit_timeout_is_held_and_read():
    assert _deriver_default("WORK_UNIT_TIMEOUT_SECONDS") == 300

    source = Path(queue_manager.__file__).read_text()
    assert "settings.DERIVER.WORK_UNIT_TIMEOUT_SECONDS" in source


def test_dedup_distance_is_a_setting_and_not_a_literal():
    assert _deriver_default("DEDUPLICATE_MAX_DISTANCE") == 0.05

    source = Path(crud_document.__file__).read_text()
    assert "max_distance=settings.DERIVER.DEDUPLICATE_MAX_DISTANCE" in source
    assert "max_distance=0.05" not in source


def test_session_observation_cap_defaults_off_and_is_read():
    assert _deriver_default("MAX_OBSERVATIONS_PER_SESSION") == 0

    source = Path(crud_representation.__file__).read_text()
    assert "settings.DERIVER.MAX_OBSERVATIONS_PER_SESSION" in source


def test_quota_rejection_is_never_retried():
    assert is_transient_llm_error(_http_error(429, "usage_limit_reached")) is False
    assert is_transient_llm_error(_http_error(429, "rate limit exceeded")) is True
    assert is_transient_llm_error(_http_error(503, "upstream unavailable")) is True


def test_openai_clients_carry_no_retry_layer_of_their_own():
    source = Path(llm_registry.__file__).read_text()
    settings_found = source.count("max_retries=")

    assert settings_found > 0
    assert source.count("max_retries=0") == settings_found


def test_conclusion_create_carries_reinforcement_and_provenance():
    fields = schemas.ConclusionCreate.model_fields

    assert "times_derived" in fields
    assert "source_ids" in fields


def test_a_conclusion_cannot_be_written_without_an_admission():
    with pytest.raises(ValidationError, match="reason_for_entry"):
        schemas.ConclusionCreate(  # pyright: ignore[reportCallIssue]
            content="the user has a dog",
            observer_id="observer",
            observed_id="observed",
        )


def test_a_deriver_write_must_name_its_source_messages():
    with pytest.raises(ValidationError, match="deriver_agent conclusions require source_message_ids"):
        schemas.ConclusionCreate(
            content="the user has a dog",
            observer_id="observer",
            observed_id="observed",
            action="create",
            reason_for_entry="test",
            search_query="test",
            searched_conclusion_ids=[],
            source_tool_call_id="test",
            entry_origin="deriver_agent",
            agent_trace_id="test",
            agent_model="test",
        )


def _observation_input(**overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "content": "the user prefers tea",
        "action": "create",
        "reason_for_entry": "Distinct durable fact",
        "search_query": "tea",
        "searched_conclusion_ids": [],
    }
    fields.update(overrides)
    return schemas.ObservationInput(**fields)


def test_admission_input_refuses_blank_text_and_strips_nul_bytes():
    for field in ("content", "reason_for_entry", "search_query"):
        with pytest.raises(ValidationError, match="non-whitespace"):
            _observation_input(**{field: " \n\t "})

    assert _observation_input(content="a\x00b").content == "ab"


def test_a_representation_save_returns_a_count_so_an_empty_save_is_falsy():
    hints = get_type_hints(
        crud_representation.RepresentationManager.save_representation
    )

    assert hints["return"] is int


def test_the_two_representation_models_are_the_only_response_shapes():
    assert hasattr(representation_module, "ExtractedRepresentation")
    assert hasattr(representation_module, "AdmissionRepresentation")
    assert not hasattr(representation_module, "PromptRepresentation")


def test_the_deriver_writes_about_the_user_not_a_spelled_peer_id():
    prompt = minimal_deriver_prompt("elodszopos", "USER: I took my dog out")

    assert "the user" in prompt
    assert "as the target peer id" not in prompt
    assert "elodszopos is" not in prompt


def test_the_extraction_examples_are_fenced_and_disclaimed():
    prompt = minimal_deriver_prompt("elodszopos", "USER: I took my dog out")

    assert "Never emit a conclusion whose content comes from an example" in prompt
    assert (
        prompt.index("<examples>")
        < prompt.index("Positive -- clears all four criteria:")
        < prompt.index("</examples>")
    )


def test_both_sdks_declare_the_same_version():
    python_sdk = tomllib.loads(
        (REPO_ROOT / "sdks" / "python" / "pyproject.toml").read_text()
    )
    typescript_sdk = json.loads(
        (REPO_ROOT / "sdks" / "typescript" / "package.json").read_text()
    )

    assert python_sdk["project"]["version"] == typescript_sdk["version"]
