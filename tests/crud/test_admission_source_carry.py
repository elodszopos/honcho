"""A live conclusion lists every message it rests on, across enrichment and absorption."""

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas

_DIMENSIONS = 1536


async def _collection_with_messages(
    db_session: AsyncSession, workspace_name: str, observer: str
) -> tuple[str, str, list[int]]:
    observed = models.Peer(name=str(generate_nanoid()), workspace_name=workspace_name)
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace_name)
    db_session.add_all([observed, session])
    await db_session.flush()
    messages = [
        models.Message(
            workspace_name=workspace_name,
            session_name=session.name,
            peer_name=observed.name,
            content=f"statement {position}",
            token_count=2,
            seq_in_session=position,
        )
        for position in (1, 2, 3)
    ]
    db_session.add_all(
        [
            *messages,
            models.Collection(
                workspace_name=workspace_name, observer=observer, observed=observed.name
            ),
        ]
    )
    await db_session.commit()
    return observed.name, session.name, [message.id for message in messages]


def _conclusion(
    content: str,
    observer: str,
    observed: str,
    session: str,
    message_ids: list[int],
    **admission: object,
) -> schemas.ConclusionCreate:
    return schemas.ConclusionCreate(
        content=content,
        observer_id=observer,
        observed_id=observed,
        session_id=session,
        reason_for_entry="Probe",
        search_query="probe",
        source_message_ids=message_ids,
        entry_origin="explicit_agent",
        agent_trace_id="probe-trace",
        agent_model="probe-model",
        **admission,
    )


async def _live_sources(
    db_session: AsyncSession, workspace_name: str, content: str
) -> tuple[list[int], list[int]]:
    metadata = (
        await db_session.execute(
            select(models.Document.internal_metadata).where(
                models.Document.workspace_name == workspace_name,
                models.Document.content == content,
                models.Document.deleted_at.is_(None),
            )
        )
    ).scalar_one()
    return metadata["admission"]["source_message_ids"], metadata["message_ids"]


@pytest.mark.asyncio
async def test_an_enrichment_lists_its_predecessors_source_messages(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    workspace, observer = sample_data
    observed, session, (first, second, _) = await _collection_with_messages(
        db_session, workspace.name, observer.name
    )
    [original] = await crud.create_observations(
        db_session,
        [
            _conclusion(
                "The user keeps bees",
                observer.name,
                observed,
                session,
                [first],
                action="create",
                searched_conclusion_ids=[],
            )
        ],
        workspace.name,
        embeddings=[[0.2] * _DIMENSIONS],
    )
    await crud.create_observations(
        db_session,
        [
            _conclusion(
                "The user keeps bees on the roof",
                observer.name,
                observed,
                session,
                [second],
                action="enrich",
                target_id=original.id,
                searched_conclusion_ids=[original.id],
            )
        ],
        workspace.name,
        embeddings=[[0.21] * _DIMENSIONS],
    )
    await db_session.commit()

    assert await _live_sources(
        db_session, workspace.name, "The user keeps bees on the roof"
    ) == ([second, first], [second, first])

    traced = (
        (
            await db_session.execute(
                crud.get_documents_with_filters(
                    workspace.name,
                    filters={
                        "metadata": {"admission": {"source_message_ids": [first]}}
                    },
                )
            )
        )
        .scalars()
        .all()
    )
    assert [row.content for row in traced] == ["The user keeps bees on the roof"]


@pytest.mark.asyncio
async def test_an_absorbing_survivor_lists_the_absorbed_source_messages(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    workspace, observer = sample_data
    observed, session, (first, second, _) = await _collection_with_messages(
        db_session, workspace.name, observer.name
    )
    survivor, duplicate = await crud.create_observations(
        db_session,
        [
            _conclusion(
                "The user keeps bees",
                observer.name,
                observed,
                session,
                [first],
                action="create",
                searched_conclusion_ids=[],
            ),
            _conclusion(
                "The user is a beekeeper",
                observer.name,
                observed,
                session,
                [second],
                action="create",
                searched_conclusion_ids=[],
            ),
        ],
        workspace.name,
        embeddings=[[0.2] * _DIMENSIONS, [0.22] * _DIMENSIONS],
    )
    await crud.soft_delete_documents(
        db_session,
        workspace.name,
        [duplicate.id],
        removal=schemas.ConclusionRemoval(
            category="duplicate_absorbed",
            reason="The survivor states the same memory",
            absorbed_into=survivor.id,
            entry_origin="operator_sdk",
            agent_trace_id="probe-trace",
            agent_model="probe-model",
        ),
    )
    await db_session.commit()

    assert await _live_sources(db_session, workspace.name, "The user keeps bees") == (
        [first, second],
        [first, second],
    )


@pytest.mark.asyncio
async def test_an_enrichment_keeps_a_legacy_rows_message_ids(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    workspace, observer = sample_data
    observed, session, (first, second, _) = await _collection_with_messages(
        db_session, workspace.name, observer.name
    )
    legacy = models.Document(
        workspace_name=workspace.name,
        observer=observer.name,
        observed=observed,
        content="The user keeps bees",
        session_name=session,
        internal_metadata={"message_ids": [first]},
        embedding=[0.2] * _DIMENSIONS,
    )
    db_session.add(legacy)
    await db_session.commit()

    await crud.create_observations(
        db_session,
        [
            _conclusion(
                "The user keeps bees on the roof",
                observer.name,
                observed,
                session,
                [second],
                action="enrich",
                target_id=legacy.id,
                searched_conclusion_ids=[legacy.id],
            )
        ],
        workspace.name,
        embeddings=[[0.21] * _DIMENSIONS],
    )
    await db_session.commit()

    assert await _live_sources(
        db_session, workspace.name, "The user keeps bees on the roof"
    ) == ([second, first], [second, first])
