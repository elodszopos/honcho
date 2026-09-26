"""Transaction-local HNSW settings for filtered vector queries."""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

ITERATIVE_SCAN_SQL = "SET LOCAL hnsw.iterative_scan = relaxed_order"
EF_SEARCH_SQL = "SET LOCAL hnsw.ef_search = 100"


async def widen_hnsw_scan(db: AsyncSession) -> None:
    """Let a filtered HNSW query keep scanning until the limit is met, within this transaction."""
    await db.execute(text(ITERATIVE_SCAN_SQL))
    await db.execute(text(EF_SEARCH_SQL))
