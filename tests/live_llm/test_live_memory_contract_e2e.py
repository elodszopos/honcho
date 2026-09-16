"""One journey through the memory contract against real models.

Every other test in the suite mocks the LLM, so nothing else proves that a real
deriver run produces a complete admission envelope, that a cited source becomes a
reasoning-tree edge, or that retirement keeps the ledger. Spends real extraction,
admission and embedding calls.
"""

from __future__ import annotations

import datetime

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.deriver.deriver import process_representation_tasks_batch
from src.exceptions import ValidationException
from src.schemas.configuration import (
    ResolvedConfiguration,
    ResolvedDreamConfiguration,
    ResolvedPeerCardConfiguration,
    ResolvedReasoningConfiguration,
    ResolvedSummaryConfiguration,
)
from src.utils import agent_tools
from tests.tracked_db_patch import patch_tracked_db

pytestmark = pytest.mark.live_llm


@pytest.fixture
def live_db(db_engine: object):
    """`tests/live_llm/` opts out of the autouse runtime mocks, which also unhooks
    `tracked_db`. Background code would otherwise reach the configured database."""
    with patch_tracked_db(db_engine):  # pyright: ignore[reportArgumentType]
        yield

DURABLE_FACT = "From now on, always give me engineering answers in metric units only."


def _configuration() -> ResolvedConfiguration:
    return ResolvedConfiguration(
        reasoning=ResolvedReasoningConfiguration(enabled=True),
        peer_card=ResolvedPeerCardConfiguration(use=False, create=False),
        summary=ResolvedSummaryConfiguration(
            enabled=False, messages_per_short_summary=20, messages_per_long_summary=60
        ),
        dream=ResolvedDreamConfiguration(enabled=False),
    )


async def _seed(db: AsyncSession) -> tuple[str, str, str, models.Message]:
    suffix = generate_nanoid(size=8)
    workspace = models.Workspace(name=f"e2e-{suffix}")
    db.add(workspace)
    await db.commit()

    observer = models.Peer(name=f"assistant-{suffix}", workspace_name=workspace.name)
    observed = models.Peer(name=f"user-{suffix}", workspace_name=workspace.name)
    session = models.Session(name=f"session-{suffix}", workspace_name=workspace.name)
    db.add_all([observer, observed, session])
    await db.commit()

    message = models.Message(
        session_name=session.name,
        workspace_name=workspace.name,
        peer_name=observed.name,
        content=DURABLE_FACT,
        token_count=16,
        seq_in_session=1,
        created_at=datetime.datetime.now(datetime.UTC),
    )
    db.add(message)
    await db.commit()
    return workspace.name, observer.name, observed.name, message


async def _documents(
    db: AsyncSession, workspace: str, observer: str, observed: str
) -> list[models.Document]:
    result = await db.execute(
        select(models.Document).where(
            models.Document.workspace_name == workspace,
            models.Document.observer == observer,
            models.Document.observed == observed,
            models.Document.deleted_at.is_(None),
        )
    )
    return list(result.scalars().all())


@pytest.mark.asyncio
async def test_live_memory_contract_end_to_end(
    db_session: AsyncSession,
    live_db: None,  # pyright: ignore[reportUnusedParameter]
) -> None:
    workspace, observer, observed, message = await _seed(db_session)

    # 1. A real extraction pass and a real admission pass.
    await process_representation_tasks_batch(
        [message],
        _configuration(),
        observers=[observer],
        observed=observed,
        queue_item_message_ids=[message.id],
        session_id=message.session_name,
    )

    stored = await _documents(db_session, workspace, observer, observed)
    assert stored, (
        "the live deriver admitted nothing from an explicit durable preference"
    )
    explicit = stored[0]
    admission = explicit.internal_metadata["admission"]
    assert explicit.level == "explicit"
    assert explicit.times_derived == 1
    assert admission["entry_origin"] == "deriver_agent"
    assert admission["source_message_ids"] == [message.id]
    assert admission["reason_for_entry"].strip()
    assert admission["search_query"].strip()
    assert admission["action"] == "create"

    # 2. A derived conclusion citing it, with one fabricated id alongside.
    result = await agent_tools.create_observations(
        [
            schemas.ObservationInput(
                content="The user works in metric units by default.",
                level="deductive",
                source_ids=[explicit.id, str(generate_nanoid())],
                premises=[explicit.content],
                action="create",
                reason_for_entry="Rests on the stored unit preference.",
                search_query="units",
                searched_conclusion_ids=[explicit.id],
            )
        ],
        observer,
        observed,
        message.session_name,
        workspace,
        [message.id],
        str(message.created_at),
        run_id=f"live-e2e-{generate_nanoid(size=6)}",
        agent_model="live-e2e",
        source_tool_call_id="live-e2e-call",
        entry_origin="dreamer_agent",
    )
    assert result.created_count == 1
    assert result.failed == []

    derived = next(
        doc
        for doc in await _documents(db_session, workspace, observer, observed)
        if doc.level == "deductive"
    )

    # 3. The fabricated id is gone; the real one became an edge.
    assert derived.source_ids == [explicit.id]
    edges = (
        (
            await db_session.execute(
                select(models.DocumentSource.source_id).where(
                    models.DocumentSource.derived_id == derived.id
                )
            )
        )
        .scalars()
        .all()
    )
    assert list(edges) == [explicit.id]

    children = (
        (
            await db_session.execute(
                crud.get_child_observations(
                    workspace, explicit.id, observer=observer, observed=observed
                )
            )
        )
        .scalars()
        .all()
    )
    assert [child.id for child in children] == [derived.id]

    # 4. Retirement keeps the row and its ledger, and closes it to new derivations.
    await crud.soft_delete_documents(
        db_session,
        workspace,
        [explicit.id],
        removal=schemas.ConclusionRemoval(
            category="superseded",
            reason="The live end-to-end run retired this conclusion.",
            entry_origin="operator_sdk",
            agent_trace_id="live-e2e",
            agent_model="live-e2e",
        ),
        observer=observer,
        observed=observed,
    )
    await db_session.commit()

    retired = await crud.get_document(db_session, workspace, explicit.id)
    assert retired is not None
    assert retired.internal_metadata["removal"]["category"] == "superseded"
    assert explicit.id not in [
        doc.id for doc in await _documents(db_session, workspace, observer, observed)
    ]

    with pytest.raises(ValidationException, match="retired"):
        await crud.create_observations(
            db_session,
            observations=[
                schemas.ConclusionCreate(
                    content="A conclusion resting on a retired source.",
                    observer_id=observer,
                    observed_id=observed,
                    level="deductive",
                    action="create",
                    reason_for_entry="Should not be admitted.",
                    search_query="units",
                    searched_conclusion_ids=[],
                    source_ids=[explicit.id],
                    premises=[explicit.content],
                    entry_origin="dreamer_agent",
                    agent_trace_id="live-e2e",
                    agent_model="live-e2e",
                )
            ],
            workspace_name=workspace,
            embeddings=[[0.11] * settings.EMBEDDING.VECTOR_DIMENSIONS],
        )
