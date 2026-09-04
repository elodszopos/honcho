"""Values this fork holds. Upstream rewrites its own tests on every pull, so a merge that
adopts upstream's value goes green unless the literal is asserted here."""

import json
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from src import schemas
from src.config import settings
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


def test_deriver_work_unit_timeout_is_held_and_read():
    assert settings.DERIVER.WORK_UNIT_TIMEOUT_SECONDS == 300

    source = Path(queue_manager.__file__).read_text()
    assert "settings.DERIVER.WORK_UNIT_TIMEOUT_SECONDS" in source


def test_dedup_distance_stays_configurable():
    assert settings.DERIVER.DEDUPLICATE_MAX_DISTANCE is not None

    source = Path(crud_document.__file__).read_text()
    assert "max_distance=settings.DERIVER.DEDUPLICATE_MAX_DISTANCE" in source
    assert "max_distance=0.05" not in source


def test_session_observation_cap_defaults_off_and_is_read():
    assert settings.DERIVER.MAX_OBSERVATIONS_PER_SESSION == 0

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
    with pytest.raises(ValidationError):
        schemas.ConclusionCreate(  # pyright: ignore[reportCallIssue]
            content="the user has a dog",
            observer_id="observer",
            observed_id="observed",
        )


def test_a_deriver_write_must_name_its_source_messages():
    with pytest.raises(ValidationError):
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


def test_the_two_representation_models_are_the_only_response_shapes():
    assert hasattr(representation_module, "ExtractedRepresentation")
    assert hasattr(representation_module, "AdmissionRepresentation")
    assert not hasattr(representation_module, "PromptRepresentation")


def test_the_deriver_writes_about_the_user_not_a_spelled_peer_id():
    prompt = minimal_deriver_prompt("elodszopos", "USER: I took my dog out")

    assert "the user" in prompt
    assert "as the target peer id" not in prompt
    assert "elodszopos is" not in prompt


def test_both_sdks_declare_the_same_version():
    python_sdk = tomllib.loads(
        (REPO_ROOT / "sdks" / "python" / "pyproject.toml").read_text()
    )
    typescript_sdk = json.loads(
        (REPO_ROOT / "sdks" / "typescript" / "package.json").read_text()
    )

    assert python_sdk["project"]["version"] == typescript_sdk["version"]
