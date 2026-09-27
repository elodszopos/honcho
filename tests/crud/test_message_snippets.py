"""Per-thought message search: filters run inside the query and the store must be pgvector."""

from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.crud.message import search_message_snippets
from src.exceptions import VectorStoreError

NEAR = "the reddit digest job now runs at nine every morning"
FAR = "the kitchen tap drips when the boiler is cold"


@pytest.fixture
async def threads(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Two sessions, one message each, embedded at different distances from the query vector."""
    monkeypatch.setattr(settings.VECTOR_STORE, "MIGRATED", False)
    workspace, peer = sample_data
    dims = settings.EMBEDDING.VECTOR_DIMENSIONS
    vectors = {NEAR: [1.0] + [0.0] * (dims - 1), FAR: [0.0, 1.0] + [0.0] * (dims - 2)}
    sessions: dict[str, str] = {}
    for content, vector in vectors.items():
        session = (
            await crud.get_or_create_session(
                db_session,
                schemas.SessionCreate(
                    name=str(generate_nanoid()),
                    peers={peer.name: schemas.SessionPeerConfig(observe_me=True)},
                ),
                workspace.name,
            )
        ).resource
        message = models.Message(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer.name,
            content=content,
            seq_in_session=1,
            token_count=10,
        )
        db_session.add(message)
        await db_session.flush()
        db_session.add(
            models.MessageEmbedding(
                content=content,
                message_id=message.public_id,
                workspace_name=workspace.name,
                session_name=session.name,
                peer_name=peer.name,
                sync_state="synced",
                embedding=vector,
            )
        )
        sessions[content] = session.name
    await db_session.commit()
    query_vector = [1.0] + [0.0] * (dims - 1)
    return workspace, sessions, query_vector


def _contents(snippets: list[Any]) -> set[str]:
    return {message.content for _, context in snippets for message in context}


@pytest.mark.asyncio
async def test_distance_floor_drops_far_messages_inside_the_query(threads: Any):
    workspace, _sessions, query_vector = threads

    unfloored, distances = await search_message_snippets(
        workspace.name, [query_vector], limit=5
    )
    floored, floored_distances = await search_message_snippets(
        workspace.name, [query_vector], limit=5, max_distance=0.5
    )

    assert _contents(unfloored) == {NEAR, FAR}
    assert _contents(floored) == {NEAR}
    assert max(floored_distances.values()) <= 0.5
    assert min(distances.values()) == min(floored_distances.values())


@pytest.mark.asyncio
async def test_excluded_session_never_appears(threads: Any):
    workspace, sessions, query_vector = threads

    snippets, _ = await search_message_snippets(
        workspace.name, [query_vector], limit=5, exclude_session_name=sessions[NEAR]
    )

    assert _contents(snippets) == {FAR}


@pytest.mark.asyncio
async def test_excluded_messages_are_neither_hits_nor_context(
    threads: Any, db_session: AsyncSession
):
    workspace, sessions, query_vector = threads
    near_session = sessions[NEAR]
    near = (
        await db_session.execute(
            select(models.Message).where(models.Message.content == NEAR)
        )
    ).scalar_one()
    neighbour = models.Message(
        workspace_name=workspace.name,
        session_name=near_session,
        peer_name=near.peer_name,
        content="and the digest bills to the google account",
        seq_in_session=2,
        token_count=8,
    )
    db_session.add(neighbour)
    await db_session.flush()
    db_session.add(
        models.MessageEmbedding(
            content=neighbour.content,
            message_id=neighbour.public_id,
            workspace_name=workspace.name,
            session_name=near_session,
            peer_name=near.peer_name,
            sync_state="synced",
            embedding=[0.0, 0.0, 1.0]
            + [0.0] * (settings.EMBEDDING.VECTOR_DIMENSIONS - 3),
        )
    )
    await db_session.commit()

    plain, _ = await search_message_snippets(workspace.name, [query_vector], limit=5)
    assert neighbour.content in _contents(plain)

    excluded, _ = await search_message_snippets(
        workspace.name, [query_vector], limit=5, exclude_message_ids={neighbour.id}
    )
    assert neighbour.content not in _contents(excluded)
    assert NEAR in _contents(excluded)

    without_near, _ = await search_message_snippets(
        workspace.name, [query_vector], limit=5, exclude_message_ids={near.id}
    )
    assert NEAR not in _contents(without_near)


@pytest.mark.asyncio
async def test_migrated_external_store_is_refused(
    threads: Any, monkeypatch: pytest.MonkeyPatch
):
    workspace, _sessions, query_vector = threads
    monkeypatch.setattr(settings.VECTOR_STORE, "TYPE", "turbopuffer")
    monkeypatch.setattr(settings.VECTOR_STORE, "MIGRATED", True)

    with pytest.raises(VectorStoreError, match="pgvector"):
        await search_message_snippets(workspace.name, [query_vector], limit=5)
