"""End-to-end admission contract coverage through the public Python SDK."""

from __future__ import annotations

import pytest
from nanoid import generate as generate_nanoid

from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.conclusions import ConclusionCreateParams


@pytest.mark.asyncio
async def test_public_sdk_search_create_enrich_levels_and_history(
    client_fixture: tuple[Honcho, str],
) -> None:
    client, client_type = client_fixture
    suffix = generate_nanoid(size=10)
    observer_id = f"admission-observer-{suffix}"
    target_id = f"admission-target-{suffix}"
    session_id = f"admission-session-{suffix}"

    if client_type == "async":
        observer = await client.aio.peer(id=observer_id)
        target = await client.aio.peer(id=target_id)
        session = await client.aio.session(id=session_id)
        await session.aio.add_messages(
            [
                observer.message("I will record a durable preference."),
                target.message("I prefer dark mode on every device."),
            ]
        )
    else:
        observer = client.peer(id=observer_id)
        target = client.peer(id=target_id)
        session = client.session(id=session_id)
        session.add_messages(
            [
                observer.message("I will record a durable preference."),
                target.message("I prefer dark mode on every device."),
            ]
        )

    scope = observer.conclusions_of(target)
    query = "dark mode preference"
    if client_type == "async":
        searched = await scope.aio.query(query, top_k=10)
    else:
        searched = scope.query(query, top_k=10)

    explicit = ConclusionCreateParams(
        content="The user prefers dark mode on every device.",
        session_id=session.id,
        action="create",
        reason_for_entry="Explicit durable preference absent from searched candidates.",
        search_query=query,
        searched_conclusion_ids=[row.id for row in searched],
        source_tool_call_id="integration-tool-explicit",
        entry_origin="explicit_agent",
        agent_trace_id=f"explicit-trace-{suffix}",
        agent_model="integration-test-model",
    )
    if client_type == "async":
        created = await scope.aio.create([explicit])
    else:
        created = scope.create([explicit])

    assert len(created) == 1
    original = created[0]
    assert original.admission["entry_origin"] == "explicit_agent"
    assert original.admission["searched_conclusion_ids"] == [row.id for row in searched]

    enriched_query = "cross-device dark mode preference"
    if client_type == "async":
        enrich_candidates = await scope.aio.query(enriched_query, top_k=10)
    else:
        enrich_candidates = scope.query(enriched_query, top_k=10)
    searched_ids = [row.id for row in enrich_candidates]
    if original.id not in searched_ids:
        searched_ids.append(original.id)

    enrichment = ConclusionCreateParams(
        content="The user prefers dark mode across desktop and mobile devices.",
        session_id=session.id,
        action="enrich",
        target_id=original.id,
        reason_for_entry="Adds the explicit desktop and mobile scope.",
        search_query=enriched_query,
        searched_conclusion_ids=searched_ids,
        source_tool_call_id="integration-tool-enrich",
        entry_origin="explicit_agent",
        agent_trace_id=f"enrich-trace-{suffix}",
        agent_model="integration-test-model",
    )
    if client_type == "async":
        enriched = await scope.aio.create([enrichment])
    else:
        enriched = scope.create([enrichment])

    assert enriched[0].admission["supersedes_id"] == original.id
    assert enriched[0].admission_history[-1]["document_id"] == original.id
    assert enriched[0].admission_history[-1]["content"] == original.content

    dreamer_batch = [
        ConclusionCreateParams(
            content="The user likely prefers quiet workspaces.",
            level="deductive",
            action="create",
            reason_for_entry="Novel deduction after semantic candidate review.",
            search_query="quiet workspace preference",
            searched_conclusion_ids=[],
            source_ids=[enriched[0].id],
            premises=["The user repeatedly chooses library workspaces."],
            entry_origin="dreamer_agent",
            agent_trace_id=f"dream-trace-{suffix}",
            agent_model="integration-dreamer-model",
        ),
        ConclusionCreateParams(
            content="The user usually schedules focus work before 09:00.",
            level="inductive",
            action="create",
            reason_for_entry="Novel temporal pattern after semantic candidate review.",
            search_query="morning focus work pattern",
            searched_conclusion_ids=[],
            source_ids=[original.id, enriched[0].id],
            sources=["Focus block at 07:00", "Focus block at 08:00"],
            pattern_type="tendency",
            confidence="high",
            entry_origin="dreamer_agent",
            agent_trace_id=f"dream-trace-{suffix}",
            agent_model="integration-dreamer-model",
        ),
    ]
    operator_decision = ConclusionCreateParams(
        content="Operator-confirmed durable fact.",
        action="create",
        reason_for_entry="Operator explicitly requested durable storage.",
        search_query="Operator-confirmed durable fact.",
        searched_conclusion_ids=[],
        source_tool_call_id="integration-tool-operator",
        entry_origin="operator_sdk",
        agent_trace_id=f"operator-trace-{suffix}",
        agent_model="integration-test-model",
    )
    if client_type == "async":
        dreamed = await scope.aio.create(dreamer_batch)
        operator = await scope.aio.create([operator_decision])
    else:
        dreamed = scope.create(dreamer_batch)
        operator = scope.create([operator_decision])

    assert [row.level for row in dreamed] == ["deductive", "inductive"]
    assert dreamed[0].admission["source_ids"] == [enriched[0].id]
    assert dreamed[1].admission["pattern_type"] == "tendency"
    assert dreamed[1].admission["confidence"] == "high"
    assert operator[0].admission["entry_origin"] == "operator_sdk"
    assert operator[0].admission["searched_conclusion_ids"] is not None
