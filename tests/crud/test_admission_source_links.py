"""The admission path refuses a citation it cannot resolve to a live conclusion.

Upstream validates nothing on this path and filters ungrounded ids in the agent
tool instead, so a merge that adopts its write path drops the guard entirely.
"""

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.exceptions import ValidationException


def _explicit(observer: str, observed: str) -> schemas.ConclusionCreate:
    return schemas.ConclusionCreate(
        content="the user messages after midnight",
        observer_id=observer,
        observed_id=observed,
        action="create",
        reason_for_entry="Distinct durable fact",
        search_query="midnight",
        searched_conclusion_ids=[],
        source_tool_call_id="test-call",
        agent_trace_id="test-trace",
        agent_model="test-model",
    )


def _deductive(
    observer: str, observed: str, source_ids: list[str]
) -> schemas.ConclusionCreate:
    return schemas.ConclusionCreate(
        content="the user works late",
        observer_id=observer,
        observed_id=observed,
        level="deductive",
        action="create",
        reason_for_entry="Distinct durable fact",
        search_query="late",
        searched_conclusion_ids=[],
        source_ids=source_ids,
        premises=["the user messages after midnight"],
        entry_origin="dreamer_agent",
        agent_trace_id="test-trace",
        agent_model="test-model",
    )


async def _observed_peer(db: AsyncSession, workspace: str) -> str:
    peer = models.Peer(name=str(generate_nanoid()), workspace_name=workspace)
    db.add(peer)
    await db.commit()
    return peer.name


@pytest.mark.asyncio
async def test_an_unresolvable_cited_id_is_refused(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    workspace, observer = sample_data
    observed = await _observed_peer(db_session, workspace.name)

    with pytest.raises(ValidationException, match="source_ids contain missing"):
        await crud.create_observations(
            db_session,
            observations=[
                _deductive(observer.name, observed, [str(generate_nanoid())])
            ],
            workspace_name=workspace.name,
            embeddings=[[0.11] * 1536],
        )


@pytest.mark.asyncio
async def test_a_retired_source_cannot_support_a_new_conclusion(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    workspace, observer = sample_data
    observed = await _observed_peer(db_session, workspace.name)

    source = (
        await crud.create_observations(
            db_session,
            observations=[_explicit(observer.name, observed)],
            workspace_name=workspace.name,
            embeddings=[[0.11] * 1536],
        )
    )[0]
    await crud.soft_delete_documents(
        db_session,
        workspace.name,
        [source.id],
        removal=schemas.ConclusionRemoval(
            category="misderived",
            reason="The test retired this conclusion",
            entry_origin="operator_sdk",
            agent_trace_id="test-trace",
            agent_model="test-model",
        ),
        observer=observer.name,
        observed=observed,
    )
    await db_session.commit()

    with pytest.raises(ValidationException, match="retired"):
        await crud.create_observations(
            db_session,
            observations=[_deductive(observer.name, observed, [source.id])],
            workspace_name=workspace.name,
            embeddings=[[0.11] * 1536],
        )
