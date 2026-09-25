"""A scope copy restored after its session rejoins the scope comes back into semantic search."""

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.deriver.scope_backfill import process_scope_backfill, process_scope_removal
from src.utils.queue_payload import ScopeBackfillPayload, ScopeRemovalPayload
from src.utils.scopes import scope_peer_name

_EMBEDDING = [0.5] * 1536


@pytest.mark.asyncio
async def test_a_restored_scope_copy_is_found_by_semantic_search(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    workspace, sender = sample_data
    scope_peer = models.Peer(
        name=scope_peer_name(str(generate_nanoid())),
        workspace_name=workspace.name,
        internal_metadata={"kind": "scope"},
        configuration={"observe_me": False},
    )
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add_all([scope_peer, session])
    await db_session.flush()
    db_session.add_all(
        [
            models.SessionPeer(
                workspace_name=workspace.name,
                session_name=session.name,
                peer_name=scope_peer.name,
            ),
            models.Collection(
                workspace_name=workspace.name,
                observer=sender.name,
                observed=sender.name,
            ),
            models.Collection(
                workspace_name=workspace.name,
                observer=scope_peer.name,
                observed=sender.name,
            ),
        ]
    )
    await db_session.flush()
    db_session.add(
        models.Document(
            workspace_name=workspace.name,
            observer=sender.name,
            observed=sender.name,
            content="The user keeps bees",
            level="explicit",
            session_name=session.name,
            embedding=_EMBEDDING,
        )
    )
    await db_session.commit()

    backfill = ScopeBackfillPayload(
        scope_peer=scope_peer.name, session_name=session.name
    )
    removal = ScopeRemovalPayload(scope_peer=scope_peer.name, session_name=session.name)
    await process_scope_backfill(backfill, workspace.name)
    await process_scope_removal(removal, workspace.name)
    await process_scope_backfill(backfill, workspace.name)

    found = await crud.query_documents(
        db_session,
        workspace.name,
        "beekeeping",
        observer=scope_peer.name,
        observed=sender.name,
        embedding=_EMBEDDING,
    )

    assert [doc.content for doc in found] == ["The user keeps bees"]
