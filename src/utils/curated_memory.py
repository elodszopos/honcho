"""The user's curated memory file, appended at the end of a prompt's static block."""

import logging
import os
import threading

from src.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cache: dict[str, tuple[float, str]] = {}


def curated_memory_block(workspace_name: str) -> str:
    """The workspace's USER.md as a labeled block, re-read when the file changes; "" when unset."""
    path = settings.CURATED_MEMORY.PATHS.get(workspace_name)
    if not path:
        return ""
    try:
        mtime = os.stat(path).st_mtime
    except OSError as exc:
        logger.warning(
            "curated memory unreadable: workspace=%s path=%s error=%s",
            workspace_name,
            path,
            exc,
        )
        return ""
    with _lock:
        cached = _cache.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read().strip()
        except OSError as exc:
            logger.warning(
                "curated memory unreadable: workspace=%s path=%s error=%s",
                workspace_name,
                path,
                exc,
            )
            return ""
        block = f"## USER.md\n{text}" if text else ""
        _cache[path] = (mtime, block)
        logger.info(
            "curated memory loaded: workspace=%s path=%s chars=%d",
            workspace_name,
            path,
            len(block),
        )
        return block
