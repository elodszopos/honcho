"""Shared writing-contract coverage for every Honcho LLM text path."""

import pytest
from pydantic import ValidationError

from sdks.python.src.honcho.api_types import (
    ConclusionCreateParams as APIConclusionCreateParams,
)
from sdks.python.src.honcho.conclusions import (
    ConclusionCreateParams as SDKConclusionCreateParams,
)
from src import schemas
from src.deriver.prompts import minimal_deriver_prompt
from src.dialectic.prompts import agent_system_prompt
from src.dreamer.specialists import DeductionSpecialist, InductionSpecialist
from src.utils.agent_tools import TOOLS
from src.utils.representation import AdmissionDecision, ExtractedObservation
from src.utils.summarizer import long_summary_prompt, short_summary_prompt
from src.writing_contract import (
    CONCLUSION_TARGET_CHARS,
    LLM_CONSUMED_WRITING_CONTRACT,
    MAX_CONCLUSION_CHARS,
)


@pytest.fixture(autouse=True)
def clean_queue_tables():
    """Prompt and schema tests are DB-free."""
    yield


@pytest.fixture(autouse=True)
def mock_tracked_db():
    """Prompt and schema tests do not open database sessions."""
    yield


def test_shared_contract_has_approved_bounds_and_human_exception():
    assert CONCLUSION_TARGET_CHARS == 500
    assert MAX_CONCLUSION_CHARS == 800
    assert "another LLM consumes this text" in LLM_CONSUMED_WRITING_CONTRACT
    assert "explicitly identifies a human reader" in LLM_CONSUMED_WRITING_CONTRACT
    assert "explicitly requests another style" in LLM_CONSUMED_WRITING_CONTRACT


def test_every_honcho_llm_prompt_carries_the_shared_contract():
    prompts = {
        "deriver": minimal_deriver_prompt("peer", "USER: example"),
        "dialectic": agent_system_prompt("peer", "peer", None, None),
        "short_summary": short_summary_prompt("USER: example", 100, "None"),
        "long_summary": long_summary_prompt("USER: example", 100, "None"),
        "deduction": DeductionSpecialist().build_system_prompt("peer"),
        "induction": InductionSpecialist().build_system_prompt("peer"),
    }

    for name, prompt in prompts.items():
        assert LLM_CONSUMED_WRITING_CONTRACT in prompt, name


def test_summary_prompts_require_structured_machine_context():
    for prompt in (
        short_summary_prompt("USER: example", 100, "None"),
        long_summary_prompt("USER: example", 100, "None"),
    ):
        assert "chronological narrative" not in prompt
        assert "one fact, decision, request, or unresolved issue per bullet" in prompt
        assert "Preserve chronology only when sequence changes meaning" in prompt


def test_generated_conclusion_models_publish_the_hard_limit():
    models_and_fields = (
        (schemas.ConclusionCreate, "content"),
        (schemas.ObservationInput, "content"),
        (AdmissionDecision, "content"),
        (ExtractedObservation, "content"),
        (APIConclusionCreateParams, "content"),
        (SDKConclusionCreateParams, "content"),
    )

    for model, field in models_and_fields:
        assert model.model_json_schema()["properties"][field]["maxLength"] == 800


def test_conclusion_models_reject_text_over_the_hard_limit():
    too_long = "x" * (MAX_CONCLUSION_CHARS + 1)

    with pytest.raises(ValidationError):
        schemas.ConclusionCreate(
            content=too_long,
            observer_id="observer",
            observed_id="observed",
            action="create",
            reason_for_entry="test",
            search_query="test",
            searched_conclusion_ids=[],
            source_tool_call_id="test",
            entry_origin="operator_sdk",
            agent_trace_id="test",
            agent_model="test",
        )

    with pytest.raises(ValidationError):
        schemas.ObservationInput(
            content=too_long,
            action="create",
            reason_for_entry="test",
            search_query="test",
            searched_conclusion_ids=[],
        )

    with pytest.raises(ValidationError):
        SDKConclusionCreateParams(
            content=too_long,
            action="create",
            reason_for_entry="test",
            search_query="test",
            searched_conclusion_ids=[],
            source_tool_call_id="test",
            entry_origin="operator_sdk",
            agent_trace_id="test",
            agent_model="test",
        )


def test_agent_tool_schemas_publish_the_hard_limit():
    for tool_name in (
        "create_observations",
        "create_observations_deductive",
        "create_observations_inductive",
    ):
        item = TOOLS[tool_name]["input_schema"]["properties"]["observations"]["items"]
        assert item["properties"]["content"]["maxLength"] == 800
