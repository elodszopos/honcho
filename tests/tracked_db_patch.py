"""Bind every `tracked_db` import site to one engine.

Background code opens its own sessions instead of taking the request's, so a test
that drives the deriver, the dreamer or the agent tools reaches whatever database
settings resolve to unless every import site is redirected. Lives outside
`conftest` so a module excluded from the autouse runtime mocks can still ask for
the redirect on its own.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, asynccontextmanager, contextmanager
from unittest.mock import patch

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

# Each module imports tracked_db by name, so every import site needs its own patch.
TRACKED_DB_TARGETS: tuple[str, ...] = (
    "src.dependencies.tracked_db",
    "src.deriver.queue_manager.tracked_db",
    "src.deriver.consumer.tracked_db",
    "src.deriver.deriver.tracked_db",
    "src.deriver.enqueue.tracked_db",
    "src.routers.peers.tracked_db",
    "src.routers.workspaces.tracked_db",
    "src.crud.representation.tracked_db",
    "src.dreamer.orchestrator.tracked_db",
    "src.dreamer.dream_scheduler.tracked_db",
    "src.dialectic.chat.tracked_db",
    "src.utils.summarizer.tracked_db",
    "src.webhooks.events.tracked_db",
    "src.webhooks.webhook_delivery.tracked_db",
    "src.utils.agent_tools.tracked_db",
    "src.utils.search.tracked_db",
    "src.crud.document.tracked_db",
    "src.crud.message.tracked_db",
    "src.reconciler.sync_vectors.tracked_db",
    "src.reconciler.embed_now.tracked_db",
    "src.dialectic.core.tracked_db",
    "src.dialectic.automatic.tracked_db",
    "src.dreamer.specialists.tracked_db",
    "src.dreamer.surprisal.tracked_db",
    "src.deriver.scope_backfill.tracked_db",
)


@contextmanager
def patch_tracked_db(engine: AsyncEngine) -> Iterator[None]:
    """Point every `tracked_db` import site at `engine`, one session per call.

    A session factory rather than a shared session: concurrent calls under
    `asyncio.gather` raise on a session already in use.
    """
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    @asynccontextmanager
    async def _tracked_db(_: str | None = None, *, read_only: bool = False):
        # read_only is accepted and ignored: both engines resolve to the same
        # per-test database here.
        del read_only
        async with session_factory() as session:
            try:
                yield session
            finally:
                await session.rollback()

    # ExitStack rather than a parenthesized `with`: the target list is longer than
    # CPython's 20-statically-nested-block limit.
    with ExitStack() as stack:
        for target in TRACKED_DB_TARGETS:
            stack.enter_context(patch(target, _tracked_db))
        yield
