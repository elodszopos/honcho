"""A retired conclusion carries the ledger. Reaping it destroys the record of why it went."""

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.reconciler.sync_vectors import run_vector_reconciliation_cycle


@pytest.mark.asyncio
async def test_a_reconciliation_cycle_leaves_a_retired_row_and_its_vector(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    workspace, observer = sample_data
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
    document_id = document.id

    await crud.soft_delete_documents(
        db_session,
        workspace.name,
        [document_id],
        removal=schemas.ConclusionRemoval(
            category="contradicted",
            reason="The user has never kept bees",
            entry_origin="operator_sdk",
            agent_trace_id="test-trace",
            agent_model="test-model",
        ),
    )
    await db_session.commit()

    await run_vector_reconciliation_cycle()

    row = (
        await db_session.execute(
            select(
                models.Document.id,
                models.Document.deleted_at,
                models.Document.embedding,
                models.Document.internal_metadata,
            ).where(models.Document.id == document_id)
        )
    ).one_or_none()

    assert row is not None
    assert row.deleted_at is not None
    assert row.embedding is not None
    assert row.internal_metadata["removal"]["category"] == "contradicted"
