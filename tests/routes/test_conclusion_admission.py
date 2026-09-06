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
    times_derived: int | None = None,
) -> dict[str, Any]:
    payload = {
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
    if times_derived is not None:
        payload["times_derived"] = times_derived
    return payload


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


def _retire(
    client: TestClient,
    *,
    workspace: str,
    conclusion_id: str,
    category: str = "misderived",
    reason: str = "The extraction misread the message",
    absorbed_into: str | None = None,
    entry_origin: str = "explicit_agent",
    agent_trace_id: str | None = "agent-trace-1",
    agent_model: str | None = "test-agent",
):
    body: dict[str, Any] = {
        "category": category,
        "reason": reason,
        "entry_origin": entry_origin,
    }
    if absorbed_into is not None:
        body["absorbed_into"] = absorbed_into
    if agent_trace_id is not None:
        body["agent_trace_id"] = agent_trace_id
    if agent_model is not None:
        body["agent_model"] = agent_model
    return client.request(
        "DELETE",
        f"/v3/workspaces/{workspace}/conclusions/{conclusion_id}",
        json=body,
    )


def _lineage(
    client: TestClient, *, workspace: str, conclusion_id: str
) -> dict[str, Any]:
    """Prior formulations and absorptions live here, never on the list or create response."""
    response = client.get(
        f"/v3/workspaces/{workspace}/conclusions/{conclusion_id}/lineage"
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
    assert "admission_history" not in row
    lineage = _lineage(client, workspace=workspace.name, conclusion_id=row["id"])
    assert lineage["admission_history"][-1]["document_id"] == old_id

    retired = _lineage(client, workspace=workspace.name, conclusion_id=old_id)
    assert retired["removal"]["category"] == "superseded_by_enrichment"
    assert retired["removal"]["entry_origin"] == "system"
    assert "agent_trace_id" not in retired["removal"]

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


def _enrich_a_reinforced_conclusion(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
    *,
    predecessor_count: int,
    supplied_count: int | None,
) -> int:
    """Admit a conclusion at a known count, enrich it, return the replacement's count."""
    workspace, observer, observed = conclusion_scope
    created = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={
            "conclusions": [
                _admission_payload(
                    content="The user prefers dark mode",
                    observer=observer.name,
                    observed=observed.name,
                    searched_ids=[],
                    times_derived=predecessor_count,
                )
            ]
        },
    )
    assert created.status_code == 201
    old_id = created.json()[0]["id"]
    assert created.json()[0]["times_derived"] == predecessor_count

    response = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={
            "conclusions": [
                _admission_payload(
                    content="The user prefers dark mode on desktop and mobile",
                    observer=observer.name,
                    observed=observed.name,
                    searched_ids=[old_id],
                    action="enrich",
                    target_id=old_id,
                    times_derived=supplied_count,
                )
            ]
        },
    )
    assert response.status_code == 201
    row = response.json()[0]
    lineage = _lineage(client, workspace=workspace.name, conclusion_id=row["id"])
    assert lineage["admission_history"][-1]["times_derived"] == predecessor_count
    return row["times_derived"]


def test_enrichment_carries_the_predecessor_reinforcement_forward(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
):
    """An admitted enrich is a re-derivation: the count rises, it does not reset to 1."""
    assert (
        _enrich_a_reinforced_conclusion(
            client,
            conclusion_scope,
            predecessor_count=5,
            supplied_count=None,
        )
        == 6
    )


def test_enrichment_cannot_lower_a_reinforcement_count(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
):
    """A caller that supplies too little is corrected by the store, not obeyed."""
    assert (
        _enrich_a_reinforced_conclusion(
            client,
            conclusion_scope,
            predecessor_count=5,
            supplied_count=2,
        )
        == 6
    )


def test_enrichment_honours_a_supplied_count_above_the_predecessor(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
):
    """A consolidation carrying several sources' counts keeps its own arithmetic."""
    assert (
        _enrich_a_reinforced_conclusion(
            client,
            conclusion_scope,
            predecessor_count=5,
            supplied_count=9,
        )
        == 9
    )


def _admit(
    client: TestClient,
    workspace: str,
    observer: str,
    observed: str,
    content: str,
    *,
    times_derived: int | None = None,
) -> str:
    response = client.post(
        f"/v3/workspaces/{workspace}/conclusions",
        json={
            "conclusions": [
                _admission_payload(
                    content=content,
                    observer=observer,
                    observed=observed,
                    searched_ids=[],
                    times_derived=times_derived,
                )
            ]
        },
    )
    assert response.status_code == 201
    return response.json()[0]["id"]


def test_absorption_moves_the_count_onto_the_survivor(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
):
    """A duplicate's derivations are the survivor's derivations; nothing is lost."""
    workspace, observer, observed = conclusion_scope
    survivor = _admit(
        client,
        workspace.name,
        observer.name,
        observed.name,
        "The user keeps bees",
        times_derived=3,
    )
    duplicate = _admit(
        client,
        workspace.name,
        observer.name,
        observed.name,
        "The user is a beekeeper",
        times_derived=4,
    )

    response = _retire(
        client,
        workspace=workspace.name,
        conclusion_id=duplicate,
        category="duplicate_absorbed",
        reason="The survivor states this more fully",
        absorbed_into=survivor,
    )
    assert response.status_code == 204

    lineage = _lineage(client, workspace=workspace.name, conclusion_id=survivor)
    assert lineage["times_derived"] == 7
    absorbed = lineage["absorbed"][0]
    assert absorbed["document_id"] == duplicate
    assert absorbed["times_derived"] == 4
    assert absorbed["survivor_count_before"] == 3
    assert absorbed["survivor_count_after"] == 7
    assert absorbed["content"] == "The user is a beekeeper"

    retired = _lineage(client, workspace=workspace.name, conclusion_id=duplicate)
    assert retired["removal"]["absorbed_into"] == survivor
    assert retired["deleted_at"] is not None


def test_an_absorption_survives_the_next_enrichment(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
):
    """A revision inherits the whole ledger, or the first rewrite erases what it swallowed."""
    workspace, observer, observed = conclusion_scope
    survivor = _admit(
        client, workspace.name, observer.name, observed.name, "The user keeps bees"
    )
    duplicate = _admit(
        client, workspace.name, observer.name, observed.name, "The user is a beekeeper"
    )
    assert (
        _retire(
            client,
            workspace=workspace.name,
            conclusion_id=duplicate,
            category="duplicate_absorbed",
            reason="The survivor states this more fully",
            absorbed_into=survivor,
        ).status_code
        == 204
    )

    enriched = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={
            "conclusions": [
                _admission_payload(
                    content="The user keeps bees on the roof",
                    observer=observer.name,
                    observed=observed.name,
                    searched_ids=[survivor],
                    action="enrich",
                    target_id=survivor,
                )
            ]
        },
    )
    assert enriched.status_code == 201

    lineage = _lineage(
        client, workspace=workspace.name, conclusion_id=enriched.json()[0]["id"]
    )
    assert [row["document_id"] for row in lineage["absorbed"]] == [duplicate]
    assert lineage["times_derived"] == 3


def test_a_conclusion_cannot_be_absorbed_into_itself_or_a_stranger(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
):
    workspace, observer, observed = conclusion_scope
    conclusion = _admit(
        client, workspace.name, observer.name, observed.name, "The user keeps bees"
    )

    same = _retire(
        client,
        workspace=workspace.name,
        conclusion_id=conclusion,
        category="duplicate_absorbed",
        reason="Absorbing into itself",
        absorbed_into=conclusion,
    )
    assert same.status_code == 422
    assert "absorbed into itself" in same.text

    missing = _retire(
        client,
        workspace=workspace.name,
        conclusion_id=conclusion,
        category="duplicate_absorbed",
        reason="Absorbing into a conclusion that is not there",
        absorbed_into="does-not-exist",
    )
    assert missing.status_code == 404


def test_a_search_receipt_may_name_a_retired_conclusion_but_evidence_may_not(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
):
    """A receipt is history; evidence a new conclusion rests on has to still be live."""
    workspace, observer, observed = conclusion_scope
    retired = _admit(
        client, workspace.name, observer.name, observed.name, "The user keeps bees"
    )
    assert (
        _retire(
            client,
            workspace=workspace.name,
            conclusion_id=retired,
            category="contradicted",
            reason="The user has never kept bees",
        ).status_code
        == 204
    )

    receipt = _admission_payload(
        content="The user keeps chickens",
        observer=observer.name,
        observed=observed.name,
        searched_ids=[retired],
    )
    accepted = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={"conclusions": [receipt]},
    )
    assert accepted.status_code == 201

    evidence = _admission_payload(
        content="The user likely enjoys smallholding",
        observer=observer.name,
        observed=observed.name,
        searched_ids=[],
    )
    evidence["level"] = "deductive"
    evidence["source_ids"] = [retired]
    evidence["premises"] = ["The user keeps bees"]
    rejected = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions",
        json={"conclusions": [evidence]},
    )
    assert rejected.status_code == 422
    assert "retired" in rejected.text


def test_lineage_returns_prior_formulations_newest_first(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
):
    """Enrichment appends, so the stored order is oldest-first and must be flipped."""
    workspace, observer, observed = conclusion_scope
    current = _admit(
        client, workspace.name, observer.name, observed.name, "The user keeps bees"
    )
    wordings = [
        "The user keeps bees in the garden",
        "The user keeps bees on the roof",
    ]
    for wording in wordings:
        response = client.post(
            f"/v3/workspaces/{workspace.name}/conclusions",
            json={
                "conclusions": [
                    _admission_payload(
                        content=wording,
                        observer=observer.name,
                        observed=observed.name,
                        searched_ids=[current],
                        action="enrich",
                        target_id=current,
                    )
                ]
            },
        )
        assert response.status_code == 201
        current = response.json()[0]["id"]

    lineage = _lineage(client, workspace=workspace.name, conclusion_id=current)
    assert [entry["content"] for entry in lineage["admission_history"]] == [
        "The user keeps bees in the garden",
        "The user keeps bees",
    ]
    assert lineage["times_derived"] == 3


@pytest.fixture
async def second_observed_peer(
    db_session: AsyncSession,
    conclusion_scope: tuple[Workspace, Peer, Peer],
) -> Peer:
    workspace, _, _ = conclusion_scope
    stranger = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(stranger)
    await db_session.commit()
    return stranger


def test_a_conclusion_cannot_be_absorbed_across_collections(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
    second_observed_peer: Peer,
):
    """A survivor in someone else's collection would move the count out of scope."""
    workspace, observer, observed = conclusion_scope

    mine = _admit(
        client, workspace.name, observer.name, observed.name, "The user keeps bees"
    )
    theirs = _admit(
        client,
        workspace.name,
        observer.name,
        second_observed_peer.name,
        "The user keeps bees",
    )

    response = _retire(
        client,
        workspace=workspace.name,
        conclusion_id=mine,
        category="duplicate_absorbed",
        reason="Absorbing into a conclusion in another collection",
        absorbed_into=theirs,
    )
    assert response.status_code == 422
    assert "its own collection" in response.text

    survivor = _lineage(client, workspace=workspace.name, conclusion_id=theirs)
    assert survivor["times_derived"] == 1
    assert survivor["absorbed"] == []


def test_a_removal_needs_a_reason_and_an_agent_category(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
):
    workspace, observer, observed = conclusion_scope
    conclusion = _admit(
        client, workspace.name, observer.name, observed.name, "The user keeps bees"
    )

    blank = _retire(
        client, workspace=workspace.name, conclusion_id=conclusion, reason="   "
    )
    assert blank.status_code == 422

    unattributed = _retire(
        client,
        workspace=workspace.name,
        conclusion_id=conclusion,
        agent_trace_id=None,
        agent_model=None,
    )
    assert unattributed.status_code == 422

    system_claim = _retire(
        client,
        workspace=workspace.name,
        conclusion_id=conclusion,
        category="scope_removed",
        reason="Pretending to be the scope reconciler",
        entry_origin="system",
        agent_trace_id=None,
        agent_model=None,
    )
    assert system_claim.status_code == 422
    assert "recorded by Honcho itself" in system_claim.text


def test_a_retired_conclusion_leaves_search_but_stays_listable(
    client: TestClient,
    conclusion_scope: tuple[Workspace, Peer, Peer],
):
    workspace, observer, observed = conclusion_scope
    conclusion = _admit(
        client, workspace.name, observer.name, observed.name, "The user keeps bees"
    )
    assert (
        _retire(
            client,
            workspace=workspace.name,
            conclusion_id=conclusion,
            category="contradicted",
            reason="The user has never kept bees",
        ).status_code
        == 204
    )

    live = _search(
        client,
        workspace=workspace.name,
        query="bees",
        observer=observer.name,
        observed=observed.name,
    )
    assert conclusion not in [row["id"] for row in live]

    listed = client.post(
        f"/v3/workspaces/{workspace.name}/conclusions/list",
        json={"filters": {"observer": observer.name, "observed": observed.name}},
        params={"include_deleted": "true"},
    )
    assert listed.status_code == 200
    retired = next(row for row in listed.json()["items"] if row["id"] == conclusion)
    assert retired["removal"]["category"] == "contradicted"
    assert "admission_history" not in retired


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
    lineage = _lineage(
        client, workspace=workspace.name, conclusion_id=replacement["id"]
    )
    assert lineage["admission_history"][-1]["content"] == old_content

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
