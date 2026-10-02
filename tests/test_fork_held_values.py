"""Values this fork holds. Upstream rewrites its own tests on every pull, so a merge that
adopts upstream's value goes green unless the literal is asserted here."""

import ast
import datetime
import json
import re
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
from src.dreamer import dream_scheduler, surprisal
from src.llm import api as llm_api
from src.llm import registry as llm_registry
from src.llm.errors import is_transient_llm_error
from src.reconciler import sync_vectors
from src.utils import agent_tools
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


def test_deriver_work_units_actually_run_under_that_timeout():
    """Reading the setting is not the held value; wrapping the work in it is."""
    tree = ast.parse(Path(queue_manager.__file__).read_text())
    wrapped: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "wait_for"
            and node.args
        ):
            continue
        inner = node.args[0]
        if isinstance(inner, ast.Call):
            target = inner.func
            name = (
                target.attr
                if isinstance(target, ast.Attribute)
                else getattr(target, "id", None)
            )
            if name:
                wrapped.add(name)

    assert {"process_representation_batch", "process_item"} <= wrapped, (
        f"work-unit timeout no longer wraps the work; wait_for covers {sorted(wrapped)}"
    )


def test_unbounded_task_types_keep_the_multiplied_bound():
    representation = queue_manager._work_unit_timeout("representation")
    for task_type in ("dream", "deletion", "scope_backfill", "scope_removal"):
        assert queue_manager._work_unit_timeout(task_type) == representation * 6


def test_dedup_distance_is_a_setting_and_not_a_literal():
    assert _deriver_default("DEDUPLICATE_MAX_DISTANCE") == 0.05

    source = Path(crud_document.__file__).read_text()
    assert source.count("max_distance=settings.DERIVER.DEDUPLICATE_MAX_DISTANCE") == 2
    assert "_SEMANTIC_DUP_MAX_DISTANCE" not in source


def test_semantic_dedup_stays_unwired_from_every_write_path():
    for module in (crud_document, crud_representation, agent_tools):
        source = Path(module.__file__).read_text()
        assert "deduplicate=settings.DERIVER.DEDUPLICATE" not in source


def test_session_observation_cap_defaults_off_and_is_read():
    assert _deriver_default("MAX_OBSERVATIONS_PER_SESSION") == 0

    source = Path(crud_representation.__file__).read_text()
    assert "settings.DERIVER.MAX_OBSERVATIONS_PER_SESSION" in source


def test_quota_rejection_is_never_retried():
    assert is_transient_llm_error(_http_error(429, "usage_limit_reached")) is False
    assert is_transient_llm_error(_http_error(429, "rate limit exceeded")) is True
    assert is_transient_llm_error(_http_error(503, "upstream unavailable")) is True


def test_every_llm_retry_site_screens_quota_rejections():
    """The predicate only holds the line at sites that are told to consult it: the plain call,
    the tool loop's iterations, its final synthesis and its streamed setup."""
    sites = [
        (path, node)
        for path in sorted(Path(llm_api.__file__).parent.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "retry"
    ]
    assert {path.name for path, _ in sites} >= {"api.py", "tool_loop.py"}

    for path, site in sites:
        assert any(
            keyword.arg == "retry"
            and "is_transient_llm_error" in ast.unparse(keyword.value)
            for keyword in site.keywords
        ), (
            f"retry() at {path.name}:{site.lineno} retries every "
            "exception, so a quota 429 is re-spent instead of surfacing"
        )


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
    with pytest.raises(
        ValidationError, match="deriver_agent conclusions require source_message_ids"
    ):
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


def test_the_admission_block_names_the_id_of_every_explicit_conclusion():
    representation = representation_module.Representation(
        explicit=[
            representation_module.ExplicitObservation(
                id="conclusion-1",
                content="the user keeps bees",
                created_at=datetime.datetime.now(datetime.UTC),
                message_ids=[1],
                session_name="session",
            )
        ]
    )

    rendered = representation.format_as_markdown(include_ids=True)

    assert "[id:conclusion-1]" in rendered


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
        < prompt.index("\nEXTRACT:\n")
        < prompt.index("</examples>")
    )


_DELETED_AT_WRITE = re.compile(r"deleted_at\s*=\s*(?!=)")

_ALLOWED_DELETED_AT_WRITES = {
    ("src/crud/document.py", "doc.deleted_at = removed_at"),
    ("src/deriver/scope_backfill.py", "deleted_at=None,"),
}


def test_deleted_at_is_written_only_by_the_removal_chokepoint():
    """Every retirement records why. A module setting deleted_at itself bypasses that."""
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not _DELETED_AT_WRITE.search(stripped):
                continue
            if (relative, stripped) in _ALLOWED_DELETED_AT_WRITES:
                continue
            offenders.append(f"{relative}: {stripped}")

    assert offenders == []


def test_include_deleted_never_reaches_a_query_an_agent_runs():
    """A retired conclusion must not re-enter derivation through a widened search."""
    source = Path(crud_document.__file__).read_text()

    builder = source.split("def get_documents_with_filters(")[1].split("\nasync def ")[
        0
    ]
    assert source.count("include_deleted") == builder.count("include_deleted") > 0

    assert "include_deleted" not in Path(crud_representation.__file__).read_text()
    assert "include_deleted" not in Path(agent_tools.__file__).read_text()


def test_every_agent_removal_category_names_a_distinct_reason():
    assert {
        "duplicate_absorbed",
        "superseded",
        "contradicted",
        "misderived",
        "out_of_scope",
        "transient",
        "low_value",
    } == schemas.AGENT_REMOVAL_CATEGORIES
    assert {
        "superseded_by_enrichment",
        "semantic_dup_replaced",
        "scope_removed",
        "queued_delete",
    } == schemas.SYSTEM_REMOVAL_CATEGORIES
    assert not (schemas.AGENT_REMOVAL_CATEGORIES & schemas.SYSTEM_REMOVAL_CATEGORIES)


def test_a_removal_cannot_fabricate_an_actor_or_omit_a_reason():
    with pytest.raises(ValidationError, match="non-whitespace"):
        schemas.ConclusionRemoval(
            category="misderived",
            reason="  ",
            entry_origin="operator_sdk",
            agent_trace_id="t",
            agent_model="m",
        )

    with pytest.raises(ValidationError, match="carry no agent attribution"):
        schemas.ConclusionRemoval(
            category="scope_removed",
            reason="Session left the scope",
            entry_origin="system",
            agent_trace_id="pretend-agent",
            agent_model="pretend-model",
        )

    with pytest.raises(ValidationError, match="require agent_trace_id"):
        schemas.ConclusionRemoval(
            category="misderived",
            reason="The extraction misread the message",
            entry_origin="operator_sdk",
        )

    with pytest.raises(ValidationError, match="absorbed_into is required"):
        schemas.ConclusionRemoval(
            category="duplicate_absorbed",
            reason="Same memory as the survivor",
            entry_origin="operator_sdk",
            agent_trace_id="t",
            agent_model="m",
        )


def test_enrichment_locks_its_whole_batch_in_one_ordered_statement():
    """Locking targets one at a time in observation order deadlocks two batches
    enriching the same pair in opposite order, and no lock at all loses a
    concurrent absorb between read and write."""
    source = Path(crud_document.__file__).read_text()
    body = source.split("async def create_observations(")[1].split("\nasync def ")[0]

    batch_lock = body.split("locked_targets: dict[str, models.Document] = {}")[1].split(
        "for obs, embedding in zip("
    )[0]
    assert "sorted(" in batch_lock
    assert ".order_by(models.Document.id)" in batch_lock
    assert ".with_for_update()" in batch_lock
    assert "populate_existing=True" in batch_lock

    # The per-observation opt-in fetch is what deadlocked; it must not come back.
    assert "for_update: bool" not in source
    assert "for_update=True" not in source


# Every query over documents needs a verdict on retired rows. Retention made them
# permanent, so a missing `deleted_at` filter no longer self-heals in five minutes.
# Changing a count here means going to the site and deciding.
_DOCUMENT_QUERY_SITES = {
    "src/crud/document.py": 16,
    "src/crud/representation.py": 2,
    "src/deriver/scope_backfill.py": 5,
    "src/dreamer/dream_scheduler.py": 1,
    "src/dreamer/surprisal.py": 2,
    "src/reconciler/sync_vectors.py": 3,
}


def test_every_document_query_site_has_a_recorded_verdict():
    counts: dict[str, int] = {}
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        found = path.read_text().count("select(models.Document")
        if found:
            counts[path.relative_to(REPO_ROOT).as_posix()] = found

    assert counts == _DOCUMENT_QUERY_SITES

    # The three modules an agent's reasoning reaches must filter unconditionally.
    for module in (crud_representation, surprisal, dream_scheduler):
        source = Path(module.__file__).read_text()
        assert source.count("select(models.Document") == source.count(
            "models.Document.deleted_at.is_(None)"
        )


def test_nothing_hard_deletes_a_retired_conclusion():
    """A retired row carries the ledger; reaping it destroys the record of why it went."""
    reconciler = Path(sync_vectors.__file__).read_text()

    assert "_cleanup_soft_deleted_documents_pgvector" not in reconciler
    assert "delete(models.Document)" not in reconciler
    assert "delete(models.Document)" not in Path(crud_document.__file__).read_text()


def test_both_sdks_declare_the_same_version():
    python_sdk = tomllib.loads(
        (REPO_ROOT / "sdks" / "python" / "pyproject.toml").read_text()
    )
    typescript_sdk = json.loads(
        (REPO_ROOT / "sdks" / "typescript" / "package.json").read_text()
    )

    assert python_sdk["project"]["version"] == typescript_sdk["version"]


def test_dreamer_prompts_keep_anti_fabrication_and_enrich():
    source = (REPO_ROOT / "src/dreamer/specialists.py").read_text()
    assert "delete outdated" not in source
    assert "Use `enrich`; do not separately delete its target" in source
    anti = (
        "Copy source_ids exactly from the [id:xxx] shown in observation results"
        " - `search_messages` results have no ID and cannot be cited; "
        "invented IDs are discarded"
    )
    assert source.count(anti) == 2


def test_source_grounding_holds_the_schema_minimum_for_inductive():
    """Upstream grounds inductive on one source; ConclusionCreate refuses fewer than two,
    so upstream's minimum hands the write path an observation it will reject wholesale."""
    minimum = 'min_sources = 2 if obs.level in ("contradiction", "inductive") else 1'
    assert minimum in Path(agent_tools.__file__).read_text()

    with pytest.raises(ValidationError, match="inductive conclusions require"):
        schemas.ConclusionCreate(
            content="the user reads at night",
            observer_id="observer",
            observed_id="observed",
            level="inductive",
            action="create",
            reason_for_entry="Distinct durable fact",
            search_query="night",
            searched_conclusion_ids=[],
            source_ids=["a" * 21],
            sources=["one", "two"],
            pattern_type="behavior",
            entry_origin="dreamer_agent",
            agent_trace_id="test",
            agent_model="test",
        )
