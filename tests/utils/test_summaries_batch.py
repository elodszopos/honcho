"""One query returns the short summary of every named session that has one."""

from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.utils import summarizer


async def _session(
    db: AsyncSession, workspace: models.Workspace, peer: models.Peer
) -> str:
    created = await crud.get_or_create_session(
        db,
        schemas.SessionCreate(
            name=str(generate_nanoid()),
            peers={peer.name: schemas.SessionPeerConfig(observe_me=True)},
        ),
        workspace.name,
    )
    return created.resource.name


async def _summarize(
    db: AsyncSession, workspace: str, session: str, content: str
) -> None:
    await summarizer._save_summary(
        db,
        {
            "content": content,
            "message_id": 1,
            "summary_type": summarizer.SummaryType.SHORT.value,
            "created_at": "2026-01-01T00:00:00Z",
            "token_count": 4,
            "message_public_id": "m",
        },
        workspace,
        session,
    )


@pytest.mark.asyncio
async def test_get_summaries_returns_only_sessions_with_a_summary(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
) -> None:
    workspace, peer = sample_data
    summarized = await _session(db_session, workspace, peer)
    bare = await _session(db_session, workspace, peer)
    await _summarize(
        db_session, workspace.name, summarized, "Moved the digest to nine."
    )
    await db_session.commit()

    found = await summarizer.get_summaries(
        db_session, workspace.name, [summarized, bare, "no-such-session"]
    )

    assert set(found) == {summarized}
    assert found[summarized]["content"] == "Moved the digest to nine."


@pytest.mark.asyncio
async def test_get_summaries_with_no_names_runs_no_query(
    db_session: AsyncSession,
) -> None:
    assert await summarizer.get_summaries(db_session, "any-workspace", []) == {}


@pytest.mark.asyncio
async def test_get_summaries_matches_get_summary_per_session(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
) -> None:
    workspace, peer = sample_data
    names = [await _session(db_session, workspace, peer) for _ in range(2)]
    for index, name in enumerate(names):
        await _summarize(db_session, workspace.name, name, f"summary {index}")
    await db_session.commit()

    batched = await summarizer.get_summaries(db_session, workspace.name, names)
    single: dict[str, Any] = {
        name: await summarizer.get_summary(db_session, workspace.name, name)
        for name in names
    }

    assert batched == single
