"""An admitted conclusion is dated by when its source message was sent, not when it was stored."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.crud.representation import RepresentationManager
from src.schemas.configuration import (
    ResolvedConfiguration,
    ResolvedDreamConfiguration,
    ResolvedPeerCardConfiguration,
    ResolvedReasoningConfiguration,
    ResolvedSummaryConfiguration,
)
from src.utils.formatting import format_datetime_utc
from src.utils.representation import ExplicitObservation, Representation

_DIMENSIONS = 1536
_SENT_AT = datetime(2026, 1, 2, 3, 4, 5, 678901, tzinfo=UTC)
_BATCH_LATEST_AT = _SENT_AT + timedelta(minutes=25)


def _configuration() -> ResolvedConfiguration:
    return ResolvedConfiguration(
        reasoning=ResolvedReasoningConfiguration(enabled=True),
        peer_card=ResolvedPeerCardConfiguration(use=False, create=False),
        summary=ResolvedSummaryConfiguration(
            enabled=False,
            messages_per_short_summary=20,
            messages_per_long_summary=60,
        ),
        dream=ResolvedDreamConfiguration(enabled=False),
    )


async def _pair(
    db_session: AsyncSession, workspace_name: str
) -> tuple[models.Peer, models.Session]:
    observed = models.Peer(name=str(generate_nanoid()), workspace_name=workspace_name)
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace_name)
    db_session.add_all([observed, session])
    await db_session.flush()
    return observed, session


async def _stored_metadata(
    db_session: AsyncSession, workspace_name: str, content: str
) -> dict[str, object]:
    return (
        await db_session.execute(
            select(models.Document.internal_metadata).where(
                models.Document.workspace_name == workspace_name,
                models.Document.content == content,
            )
        )
    ).scalar_one()


@pytest.mark.asyncio
async def test_deriver_admission_records_the_message_send_time(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    workspace, observer = sample_data
    observed, session = await _pair(db_session, workspace.name)
    message = models.Message(
        workspace_name=workspace.name,
        session_name=session.name,
        peer_name=observed.name,
        content="I keep bees on the roof.",
        token_count=6,
        seq_in_session=1,
        created_at=_SENT_AT,
    )
    db_session.add(message)
    await db_session.commit()

    manager = RepresentationManager(
        workspace.name, observer=observer.name, observed=observed.name
    )
    admitted = Representation(
        explicit=[
            ExplicitObservation(
                content="The user keeps bees",
                created_at=_SENT_AT,
                message_ids=[message.id],
                session_name=session.name,
                action="create",
                reason_for_entry="A durable hobby",
                searched_conclusion_ids=[],
                search_query="bees",
                trace_id="probe-trace",
                model="probe-model",
            )
        ]
    )

    with patch(
        "src.crud.representation.embedding_client.simple_batch_embed",
        new=AsyncMock(return_value=[[0.2] * _DIMENSIONS]),
    ):
        saved = await manager.save_representation(
            admitted,
            [message.id],
            session.name,
            _BATCH_LATEST_AT,
            _configuration(),
        )

    assert saved == 1
    metadata = await _stored_metadata(db_session, workspace.name, "The user keeps bees")
    assert metadata["message_created_at"] == format_datetime_utc(_SENT_AT)


@pytest.mark.asyncio
async def test_an_enrichment_keeps_the_original_send_time(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    workspace, observer = sample_data
    observed, session = await _pair(db_session, workspace.name)
    await db_session.commit()

    def _conclusion(content: str, **admission: object) -> schemas.ConclusionCreate:
        return schemas.ConclusionCreate(
            content=content,
            observer_id=observer.name,
            observed_id=observed.name,
            session_id=session.name,
            reason_for_entry="Probe",
            search_query="probe",
            source_tool_call_id="probe",
            entry_origin="explicit_agent",
            agent_trace_id="probe-trace",
            agent_model="probe-model",
            **admission,
        )

    [original] = await crud.create_observations(
        db_session,
        [
            _conclusion(
                "The user keeps bees", action="create", searched_conclusion_ids=[]
            )
        ],
        workspace.name,
        embeddings=[[0.2] * _DIMENSIONS],
        message_created_at=format_datetime_utc(_SENT_AT),
    )
    await crud.create_observations(
        db_session,
        [
            _conclusion(
                "The user keeps bees on the roof",
                action="enrich",
                target_id=original.id,
                searched_conclusion_ids=[original.id],
            )
        ],
        workspace.name,
        embeddings=[[0.21] * _DIMENSIONS],
    )

    metadata = await _stored_metadata(
        db_session, workspace.name, "The user keeps bees on the roof"
    )
    assert metadata["message_created_at"] == format_datetime_utc(_SENT_AT)
