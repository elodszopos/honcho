"""A per-thought query searches each thought of the text and ranks a conclusion at its best distance."""

from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings

CONTENTS = [
    "For the upstream gap jobs, each daily job reads its own repository's PATCHES.md.",
    "The user's sister, Maya, lives in Lisbon.",
    "The user runs a media server on their homelab that handles their streaming.",
]


@pytest.fixture
async def conclusions(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    monkeypatch.setattr(settings.VECTOR_STORE, "MIGRATED", False)
    workspace, peer = sample_data
    dims = settings.EMBEDDING.VECTOR_DIMENSIONS
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(session)
    db_session.add(
        models.Collection(
            workspace_name=workspace.name, observer=peer.name, observed=peer.name
        )
    )
    await db_session.flush()
    documents = []
    for index, content in enumerate(CONTENTS):
        document = models.Document(
            workspace_name=workspace.name,
            observer=peer.name,
            observed=peer.name,
            content=content,
            embedding=[0.1 * (index + 1)] * dims,
            session_name=session.name,
            level="explicit",
        )
        db_session.add(document)
        documents.append(document)
    await db_session.commit()
    return workspace, peer, documents


@pytest.mark.asyncio
async def test_per_thought_results_carry_distance_closest_first(
    db_session: AsyncSession, conclusions: Any
):
    workspace, peer, documents = conclusions

    results = await crud.query_documents(
        db_session,
        workspace.name,
        "what about the gap jobs?\nwhere does my sister live these days?",
        observer=peer.name,
        observed=peer.name,
        top_k=2,
        per_thought=True,
    )

    assert len(results) == 2
    distances = [doc.distance for doc in results]
    assert distances == sorted(distances)
    assert {doc.id for doc in results} <= {doc.id for doc in documents}


@pytest.mark.asyncio
async def test_per_thought_honours_the_distance_floor(
    db_session: AsyncSession, conclusions: Any
):
    workspace, peer, _documents = conclusions

    results = await crud.query_documents(
        db_session,
        workspace.name,
        "what about the gap jobs?",
        observer=peer.name,
        observed=peer.name,
        top_k=3,
        max_distance=0.0001,
        per_thought=True,
    )

    assert results == []


@pytest.mark.asyncio
async def test_blank_per_thought_query_returns_nothing(
    db_session: AsyncSession, conclusions: Any
):
    workspace, peer, _documents = conclusions

    results = await crud.query_documents(
        db_session,
        workspace.name,
        "   ",
        observer=peer.name,
        observed=peer.name,
        top_k=3,
        per_thought=True,
    )

    assert results == []
