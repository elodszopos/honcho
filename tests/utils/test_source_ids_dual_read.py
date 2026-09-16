"""Runtime coverage for mid-drain source_ids dual-read.

Compile tests cannot see the legacy JSONB `@>` fallback. These execute it.
Metadata-only rows are GET-visible via the ORM property and invisible to
SQL reverse walks until the drain copies them; that gap is left as-is so
the walk stays in upstream's table-or-column shape.
"""

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.utils.filter import apply_filter


async def _pair(db: AsyncSession) -> tuple[str, str]:
    workspace = models.Workspace(name=str(generate_nanoid()))
    db.add(workspace)
    await db.commit()
    peer = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    db.add(peer)
    await db.commit()
    db.add(
        models.Collection(
            workspace_name=workspace.name, observer=peer.name, observed=peer.name
        )
    )
    await db.commit()
    return workspace.name, peer.name


def _doc(
    workspace_name: str,
    peer: str,
    *,
    content: str,
    legacy: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> models.Document:
    return models.Document(
        workspace_name=workspace_name,
        observer=peer,
        observed=peer,
        content=content,
        level="deductive",
        embedding=[0.1] * 1536,
        legacy_source_ids=legacy,
        internal_metadata=metadata or {},
    )


async def _ids(db: AsyncSession, stmt) -> set[str]:
    return set((await db.execute(stmt)).scalars().all())


@pytest.mark.asyncio
async def test_undrained_column_is_visible_to_reverse_walks(
    db_session: AsyncSession,
) -> None:
    ws, peer = await _pair(db_session)
    parent = _doc(ws, peer, content="parent")
    db_session.add(parent)
    await db_session.commit()

    child = _doc(ws, peer, content="column-child", legacy=[parent.id])
    db_session.add(child)
    await db_session.commit()
    child_id = child.id

    found = (
        await db_session.execute(crud.get_child_observations(ws, parent.id))
    ).scalars()
    assert {doc.id for doc in found} == {child_id}

    filtered = apply_filter(
        select(models.Document.id).where(models.Document.workspace_name == ws),
        models.Document,
        {"source_ids": {"contains": parent.id}},
    )
    assert child_id in await _ids(db_session, filtered)


@pytest.mark.asyncio
async def test_undrained_metadata_is_property_only_until_drain(
    db_session: AsyncSession,
) -> None:
    ws, peer = await _pair(db_session)
    parent_id = generate_nanoid()
    child = _doc(
        ws,
        peer,
        content="metadata-child",
        metadata={"source_ids": [parent_id]},
    )
    db_session.add(child)
    await db_session.commit()
    child_id = child.id
    db_session.expire_all()

    loaded = (
        await db_session.execute(
            select(models.Document).where(models.Document.id == child_id)
        )
    ).scalar_one()
    assert loaded.source_ids == [parent_id]

    found = (
        await db_session.execute(crud.get_child_observations(ws, parent_id))
    ).scalars()
    assert {doc.id for doc in found} == set()

    filtered = apply_filter(
        select(models.Document.id).where(models.Document.workspace_name == ws),
        models.Document,
        {"source_ids": {"contains": parent_id}},
    )
    assert child_id not in await _ids(db_session, filtered)
