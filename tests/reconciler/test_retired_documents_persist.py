"""A retired conclusion carries the ledger. Reaping it destroys the record of why it went."""

import datetime
from unittest.mock import patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.reconciler.sync_vectors import run_vector_reconciliation_cycle


async def _retired_document(
    db_session: AsyncSession,
    workspace: models.Workspace,
    observer: models.Peer,
    *,
    aged_minutes: int,
) -> str:
    """A conclusion retired far enough in the past that any reaper would be eligible."""
    observed = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(observed)
    await db_session.flush()
    db_session.add(
        models.Collection(
            workspace_name=workspace.name,
            observer=observer.name,
            observed=observed.name,
        )
    )
    await db_session.flush()
    document = models.Document(
        workspace_name=workspace.name,
        observer=observer.name,
        observed=observed.name,
        content="The user keeps bees",
        embedding=[0.25] * 1536,
        session_name=None,
    )
    db_session.add(document)
    await db_session.commit()

    await crud.soft_delete_documents(
        db_session,
        workspace.name,
        [document.id],
        removal=schemas.ConclusionRemoval(
            category="contradicted",
            reason="The user has never kept bees",
            entry_origin="operator_sdk",
            agent_trace_id="test-trace",
            agent_model="test-model",
        ),
    )
    await db_session.execute(
        update(models.Document)
        .where(models.Document.id == document.id)
        .values(
            deleted_at=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(minutes=aged_minutes)
        )
    )
    await db_session.commit()
    return document.id


async def _row(db_session: AsyncSession, document_id: str):
    return (
        await db_session.execute(
            select(
                models.Document.id,
                models.Document.deleted_at,
                models.Document.embedding,
                models.Document.sync_state,
                models.Document.internal_metadata,
            ).where(models.Document.id == document_id)
        )
    ).one_or_none()


@pytest.mark.asyncio
async def test_pgvector_reconciliation_leaves_a_retired_row_and_its_vector(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    """The autouse fixture patches in an external store, so pgvector mode has to be
    forced or this exercises the wrong branch entirely."""
    workspace, observer = sample_data
    document_id = await _retired_document(
        db_session, workspace, observer, aged_minutes=60
    )

    with patch(
        "src.reconciler.sync_vectors.get_external_vector_store", return_value=None
    ):
        await run_vector_reconciliation_cycle()

    row = await _row(db_session, document_id)
    assert row is not None
    assert row.deleted_at is not None
    assert row.embedding is not None
    assert row.internal_metadata["removal"]["category"] == "contradicted"


@pytest.mark.asyncio
async def test_external_cleanup_purges_the_vector_once_and_keeps_the_row(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    mock_vector_store,
):
    """The row survives, and a second cycle must not re-select work already done."""
    workspace, observer = sample_data
    document_id = await _retired_document(
        db_session, workspace, observer, aged_minutes=60
    )

    await run_vector_reconciliation_cycle()

    row = await _row(db_session, document_id)
    assert row is not None
    assert row.deleted_at is not None
    assert row.sync_state == "purged"

    calls_after_first = mock_vector_store.delete_many.await_count
    await run_vector_reconciliation_cycle()

    assert mock_vector_store.delete_many.await_count == calls_after_first
    assert (await _row(db_session, document_id)) is not None
