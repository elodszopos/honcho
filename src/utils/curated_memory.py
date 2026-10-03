"""The files the assistant already carries, appended at the end of a prompt's static block."""

import logging
import os
import threading

from src.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cache: dict[str, tuple[float, str]] = {}


def curated_memory_block(workspace_name: str) -> str:
    """The workspace's curated files as labeled blocks in configured order, each re-read when it changes."""
    blocks = (
        _file_block(workspace_name, path)
        for path in settings.CURATED_MEMORY.PATHS.get(workspace_name, [])
    )
    return "\n\n".join(block for block in blocks if block)


def _file_block(workspace_name: str, path: str) -> str:
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
        block = (
            f'<file name="{os.path.basename(path)}">\n{text}\n</file>' if text else ""
        )
        _cache[path] = (mtime, block)
        logger.info(
            "curated memory loaded: workspace=%s path=%s chars=%d",
            workspace_name,
            path,
            len(block),
        )
        return block
