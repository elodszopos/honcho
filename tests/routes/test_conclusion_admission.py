from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.models import Peer, Workspace


def _admission_payload(
    *,
    content: str,
    observer: str,
    observed: str,
    searched_ids: list[str],
    action: str = "create",
    target_id: str | None = None,
) -> dict[str, Any]:
    return {
        "content": content,
        "observer_id": observer,
        "observed_id": observed,
        "action": action,
        "target_id": target_id,
        "reason_for_entry": "The agent judged this to be a durable user fact",
        "search_query": content if action == "create" else "durable preference",
        "searched_conclusion_ids": searched_ids,
        "source_message_ids": [],
        "source_tool_call_id": "tool-call-1",
        "entry_origin": "explicit_agent",
        "agent_trace_id": "agent-trace-1",
        "agent_model": "test-agent",
    }


def _search(
    client: TestClient,
    *,
    workspace: str,
    query: str,
    observer: str,
    observed: str,
) -> list[dict[str, Any]]:
    response = client.post(
        f"/v3/workspaces/{workspace}/conclusions/query",
        json={
            "query": query,
            "top_k": 10,
            "filters": {"observer": observer, "observed": observed},
        },
    )
    assert response.status_code == 200
    return response.json()


@pytest.fixture
async def conclusion_scope(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
) -> tuple[Workspace, Peer, Peer]:
    workspace, observer = sample_data
    observed = models.Peer(
        name=str(generate_nanoid()),
        workspace_name=workspace.name,
    )
    db_session.add(observed)
    await db_session.commit()
    return workspace, observer, observed


@pytest.fixture
async def conclusion_messages(
    db_session: AsyncSession,
    conclusion_scope: tuple[Workspace, Peer, Peer],
) -> tuple[models.Session, models.Message, models.Message]:
    workspace, observer, observed = conclusion_scope
    session = models.Session(
        name=str(generate_nanoid()),
        workspace_name=workspace.name,
    )
    db_session.add(session)
    await db_session.flush()
    observed_message = models.Message(
        session_name=session.name,
        workspace_name=workspace.name,
        peer_name=observed.name,
        content="I prefer dark mode",
        seq_in_session=1,
    )
    wrong_peer_message = models.Message(
        session_name=session.name,
        workspace_name=workspace.name,
        peer_name=observer.name,
        content="The other peer prefers dark mode",
        seq_in_session=2,
    )
    db_session.add_all([observed_message, wrong_peer_message])
    await db_session.commit()
    await db_session.refresh(observed_message)
    await db_session.refresh(wrong_peer_message)
    return session, observed_message, wrong_peer_message


def test_raw_create_without_agent_admission_is_rejected(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
):
    workspace, observer, observed = conclusion_scope
    response = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={
            "conclusions": [
                {
                    "content": "The user prefers dark mode",
                    "observer_id": observer.name,
                    "observed_id": observed.name,
                }
            ]
        },
    )
    assert response.status_code == 422


def test_rejects_source_message_authored_by_wrong_peer(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
    conclusion_messages: tuple[models.Session, models.Message, models.Message],
):
    workspace, observer, observed = conclusion_scope
    session, _, wrong_peer_message = conclusion_messages
    payload = _admission_payload(
        content="The user prefers dark mode",
        observer=observer.name,
        observed=observed.name,
        searched_ids=[],
    )
    payload["session_id"] = session.name
    payload["source_message_ids"] = [wrong_peer_message.id]

    response = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={"conclusions": [payload]},
    )

    assert response.status_code == 422
    assert "wrong-peer" in response.text


@pytest.fixture
async def cross_session_source_messages(
    db_session: AsyncSession,
    conclusion_scope: tuple[Workspace, Peer, Peer],
    conclusion_messages: tuple[models.Session, models.Message, models.Message],
) -> tuple[models.Message, models.Message]:
    workspace, _, observed = conclusion_scope
    _, first_message, _ = conclusion_messages
    second_session = models.Session(
        name=str(generate_nanoid()),
        workspace_name=workspace.name,
    )
    db_session.add(second_session)
    await db_session.flush()
    second_message = models.Message(
        session_name=second_session.name,
        workspace_name=workspace.name,
        peer_name=observed.name,
        content="I also prefer keyboard navigation",
        seq_in_session=1,
    )
    db_session.add(second_message)
    await db_session.commit()
    await db_session.refresh(second_message)
    return first_message, second_message


def test_global_conclusion_accepts_same_peer_sources_across_sessions(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
    cross_session_source_messages: tuple[models.Message, models.Message],
):
    workspace, observer, observed = conclusion_scope
    first_message, second_message = cross_session_source_messages
    payload = _admission_payload(
        content="The user prefers dark mode and keyboard navigation",
        observer=observer.name,
        observed=observed.name,
        searched_ids=[],
    )
    payload["source_message_ids"] = [first_message.id, second_message.id]

    response = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={"conclusions": [payload]},
    )

    assert response.status_code == 201


def test_batch_accepts_independently_searched_agent_decisions(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
):
    workspace, observer, observed = conclusion_scope
    payloads = [
        _admission_payload(
            content="The user prefers dark mode",
            observer=observer.name,
            observed=observed.name,
            searched_ids=[],
        ),
        _admission_payload(
            content="The user prefers keyboard navigation",
            observer=observer.name,
            observed=observed.name,
            searched_ids=[],
        ),
    ]
    response = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={"conclusions": payloads},
    )
    assert response.status_code == 201
    assert {row["content"] for row in response.json()} == {
        "The user prefers dark mode",
        "The user prefers keyboard navigation",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("searched_conclusion_ids", ["fabricated-search-result"]),
        ("source_ids", ["fabricated-source"]),
        ("source_message_ids", [999_999]),
    ],
)
def test_rejects_fabricated_or_out_of_scope_provenance(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
    field: str,
    value: list[str] | list[int],
):
    workspace, observer, observed = conclusion_scope
    payload = _admission_payload(
        content="The user prefers dark mode",
        observer=observer.name,
        observed=observed.name,
        searched_ids=[],
    )
    payload[field] = value

    response = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={"conclusions": [payload]},
    )

    assert response.status_code == 422


def test_enrichment_preserves_predecessor_and_records_revision(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
):
    workspace, observer, observed = conclusion_scope
    old_content = "The user prefers dark mode"
    initial = _admission_payload(
        content=old_content,
        observer=observer.name,
        observed=observed.name,
        searched_ids=[],
    )
    created = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={"conclusions": [initial]},
    )
    assert created.status_code == 201
    old_id = created.json()[0]["id"]

    candidates = _search(
        client,
        workspace=workspace.name,
        query="durable preference",
        observer=observer.name,
        observed=observed.name,
    )
    candidate_ids = [row["id"] for row in candidates]
    assert old_id in candidate_ids

    enriched_content = (
        old_content + "; the preference applies across desktop and mobile"
    )
    enriched = _admission_payload(
        content=enriched_content,
        observer=observer.name,
        observed=observed.name,
        searched_ids=candidate_ids,
        action="enrich",
        target_id=old_id,
    )
    response = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={"conclusions": [enriched]},
    )
    assert response.status_code == 201
    row = response.json()[0]
    assert row["content"] == enriched_content
    assert row["admission"]["supersedes_id"] == old_id
    assert row["admission_history"][-1]["document_id"] == old_id

    active = _search(
        client,
        workspace=workspace.name,
        query="durable preference",
        observer=observer.name,
        observed=observed.name,
    )
    active_ids = [item["id"] for item in active]
    assert old_id not in active_ids
    assert row["id"] in active_ids


def test_batch_enrichment_conflict_rolls_back_every_replacement(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
):
    workspace, observer, observed = conclusion_scope
    workspace_name = workspace.name
    observer_name = observer.name
    observed_name = observed.name
    old_content = "The user prefers dark mode"
    created = client.post(
        f"/v3/workspaces/{workspace_name}/conclusions",
        json={
            "conclusions": [
                _admission_payload(
                    content=old_content,
                    observer=observer_name,
                    observed=observed_name,
                    searched_ids=[],
                )
            ]
        },
    )
    assert created.status_code == 201
    old_id = created.json()[0]["id"]

    replacements = [
        _admission_payload(
            content=content,
            observer=observer_name,
            observed=observed_name,
            searched_ids=[old_id],
            action="enrich",
            target_id=old_id,
        )
        for content in [
            "The user prefers dark mode on desktop",
            "The user prefers dark mode on mobile",
        ]
    ]
    response = client.post(
        f"/v3/workspaces/{workspace_name}/conclusions",
        json={"conclusions": replacements},
    )
    assert response.status_code == 422
    assert "changed during admission" in response.text

    active = _search(
        client,
        workspace=workspace_name,
        query="dark mode preference",
        observer=observer_name,
        observed=observed_name,
    )
    assert [row["content"] for row in active] == [old_content]


def test_enrichment_can_rewrite_while_history_retains_predecessor(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
):
    workspace, observer, observed = conclusion_scope
    old_content = "The user prefers dark mode with high contrast"
    initial = _admission_payload(
        content=old_content,
        observer=observer.name,
        observed=observed.name,
        searched_ids=[],
    )
    created = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={"conclusions": [initial]},
    )
    assert created.status_code == 201
    old_id = created.json()[0]["id"]

    candidates = _search(
        client,
        workspace=workspace.name,
        query="durable preference",
        observer=observer.name,
        observed=observed.name,
    )
    rewritten = _admission_payload(
        content="The user likes dark themes",
        observer=observer.name,
        observed=observed.name,
        searched_ids=[row["id"] for row in candidates],
        action="enrich",
        target_id=old_id,
    )
    response = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={"conclusions": [rewritten]},
    )
    assert response.status_code == 201
    replacement = response.json()[0]
    assert replacement["content"] == "The user likes dark themes"
    assert replacement["admission_history"][-1]["content"] == old_content

    active = _search(
        client,
        workspace=workspace.name,
        query="durable preference",
        observer=observer.name,
        observed=observed.name,
    )
    assert old_id not in [row["id"] for row in active]
    assert replacement["id"] in [row["id"] for row in active]


def test_dreamer_deductive_and_inductive_admissions_preserve_level_evidence(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
):
    workspace, observer, observed = conclusion_scope
    source_response = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={
            "conclusions": [
                _admission_payload(
                    content=content,
                    observer=observer.name,
                    observed=observed.name,
                    searched_ids=[],
                )
                for content in [
                    "The user repeatedly works in libraries",
                    "The user scheduled a focus block at 07:00",
                    "The user scheduled a focus block at 08:00",
                ]
            ]
        },
    )
    assert source_response.status_code == 201
    source_ids = [row["id"] for row in source_response.json()]

    common: dict[str, Any] = {
        "observer_id": observer.name,
        "observed_id": observed.name,
        "action": "create",
        "reason_for_entry": "Novel result after semantic candidate review",
        "searched_conclusion_ids": [],
        "entry_origin": "dreamer_agent",
        "agent_trace_id": "dream-run-1",
        "agent_model": "test-dreamer-model",
    }
    payloads = [
        {
            **common,
            "content": "The user likely prefers quiet workspaces",
            "level": "deductive",
            "source_ids": [source_ids[0]],
            "premises": ["The user repeatedly works in libraries"],
            "search_query": "quiet workspace preference",
        },
        {
            **common,
            "content": "The user usually schedules focus work before 09:00",
            "level": "inductive",
            "source_ids": source_ids[1:],
            "sources": ["Focus block at 07:00", "Focus block at 08:00"],
            "pattern_type": "tendency",
            "confidence": "high",
            "search_query": "morning focus work pattern",
        },
    ]

    response = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={"conclusions": payloads},
    )

    assert response.status_code == 201
    rows = response.json()
    assert [row["level"] for row in rows] == ["deductive", "inductive"]
    assert all(row["admission"]["entry_origin"] == "dreamer_agent" for row in rows)
    assert all(row["admission"]["agent_trace_id"] == "dream-run-1" for row in rows)
    assert rows[0]["admission"]["source_ids"] == [source_ids[0]]
    assert rows[0]["admission"]["premises"] == [
        "The user repeatedly works in libraries"
    ]
    assert rows[1]["admission"]["source_ids"] == source_ids[1:]
    assert rows[1]["admission"]["pattern_type"] == "tendency"
    assert rows[1]["admission"]["confidence"] == "high"


def test_provenance_contract_rejects_missing_agent_evidence(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
):
    workspace, observer, observed = conclusion_scope
    payload = _admission_payload(
        content="The user prefers dark mode",
        observer=observer.name,
        observed=observed.name,
        searched_ids=[],
    )
    payload["source_message_ids"] = []
    payload["source_tool_call_id"] = None

    response = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={"conclusions": [payload]},
    )

    assert response.status_code == 422
    assert "source_message_ids or source_tool_call_id" in response.text


def test_enrichment_requires_target_in_searched_candidates(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
):
    workspace, observer, observed = conclusion_scope
    payload = _admission_payload(
        content="Richer durable preference",
        observer=observer.name,
        observed=observed.name,
        searched_ids=["different-id"],
        action="enrich",
        target_id="target-id",
    )

    response = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={"conclusions": [payload]},
    )

    assert response.status_code == 422
    assert "searched_conclusion_ids" in response.text


def test_explicit_create_decisions_are_not_automatically_deduplicated(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
):
    workspace, observer, observed = conclusion_scope
    payload = _admission_payload(
        content="The user prefers dark mode",
        observer=observer.name,
        observed=observed.name,
        searched_ids=[],
    )

    first = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={"conclusions": [payload]},
    )
    second = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={"conclusions": [payload]},
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()[0]["id"] != second.json()[0]["id"]
