"""Whichever of an absorb into a conclusion and an enrich of it lands first, no derivation is lost."""

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src import crud, models, schemas

_DIMENSIONS = 1536


async def _collection(
    db_session: AsyncSession, workspace_name: str, observer: str
) -> tuple[str, str]:
    observed = models.Peer(name=str(generate_nanoid()), workspace_name=workspace_name)
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace_name)
    db_session.add_all([observed, session])
    await db_session.flush()
    db_session.add(
        models.Collection(
            workspace_name=workspace_name, observer=observer, observed=observed.name
        )
    )
    await db_session.commit()
    return observed.name, session.name


async def _attempt(step: Coroutine[Any, Any, None]) -> BaseException | None:
    try:
        await step
    except Exception as exc:  # noqa: BLE001
        return exc
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["enrich_first", "absorb_first", "concurrent"])
async def test_absorb_and_enrich_of_one_survivor_keep_every_derivation(
    order: str,
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    sample_data: tuple[models.Workspace, models.Peer],
):
    workspace, observer_peer = sample_data
    observer = observer_peer.name
    observed, session_name = await _collection(db_session, workspace.name, observer)
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)

    async def enrich(survivor_id: str, marker: str) -> None:
        async with factory() as db:
            await crud.create_observations(
                db,
                observations=[
                    schemas.ConclusionCreate(
                        content=f"{marker} survivor, restated",
                        observer_id=observer,
                        observed_id=observed,
                        session_id=session_name,
                        action="enrich",
                        target_id=survivor_id,
                        reason_for_entry="Race probe",
                        search_query="probe",
                        searched_conclusion_ids=[survivor_id],
                        source_tool_call_id="probe",
                        entry_origin="explicit_agent",
                        agent_trace_id="probe-trace",
                        agent_model="probe-model",
                    )
                ],
                workspace_name=workspace.name,
                embeddings=[[0.11] * _DIMENSIONS],
            )

    async def absorb(duplicate_id: str, survivor_id: str) -> None:
        async with factory() as db:
            await crud.soft_delete_documents(
                db,
                workspace.name,
                [duplicate_id],
                removal=schemas.ConclusionRemoval(
                    category="duplicate_absorbed",
                    reason="The survivor states the same memory",
                    absorbed_into=survivor_id,
                    entry_origin="operator_sdk",
                    agent_trace_id="probe-trace",
                    agent_model="probe-model",
                ),
            )
            await db.commit()

    for _ in range(5 if order == "concurrent" else 1):
        marker = generate_nanoid()
        survivor = models.Document(
            workspace_name=workspace.name,
            observer=observer,
            observed=observed,
            content=f"{marker} survivor",
            session_name=session_name,
            times_derived=2,
            embedding=[0.31] * _DIMENSIONS,
        )
        duplicate = models.Document(
            workspace_name=workspace.name,
            observer=observer,
            observed=observed,
            content=f"{marker} duplicate",
            session_name=session_name,
            times_derived=3,
            embedding=[0.32] * _DIMENSIONS,
        )
        db_session.add_all([survivor, duplicate])
        await db_session.commit()

        if order == "enrich_first":
            enriched = await _attempt(enrich(survivor.id, marker))
            await _attempt(absorb(duplicate.id, survivor.id))
        elif order == "absorb_first":
            await _attempt(absorb(duplicate.id, survivor.id))
            enriched = await _attempt(enrich(survivor.id, marker))
        else:
            enriched, _ = await asyncio.gather(
                _attempt(enrich(survivor.id, marker)),
                _attempt(absorb(duplicate.id, survivor.id)),
            )

        assert enriched is None, enriched
        live_counts = (
            await db_session.execute(
                select(models.Document.times_derived).where(
                    models.Document.workspace_name == workspace.name,
                    models.Document.content.startswith(marker, autoescape=True),
                    models.Document.deleted_at.is_(None),
                )
            )
        ).scalars()
        # The survivor's 2, the duplicate's 3, and the enrichment's own derivation.
        assert sum(live_counts) == 6
