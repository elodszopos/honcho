"""The curated memory block follows the configured files in order and re-reads each when it changes."""

import os
from pathlib import Path

import pytest

from src.config import settings
from src.utils import curated_memory


@pytest.fixture(autouse=True)
def clean_queue_tables():
    """File reads only; override the package's DB fixture."""
    yield


@pytest.fixture
def user_md(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "USER.md"
    path.write_text("# USER\n\n- vegetarian\n", encoding="utf-8")
    monkeypatch.setattr(settings.CURATED_MEMORY, "PATHS", {"ws": [str(path)]})
    curated_memory._cache.clear()
    return path


def test_block_wraps_the_file_in_a_tag_named_after_it(user_md: Path):
    assert (
        curated_memory.curated_memory_block("ws")
        == '<file name="USER.md">\n# USER\n\n- vegetarian\n</file>'
    )


def test_files_keep_their_configured_order_and_a_missing_one_is_skipped(
    user_md: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    memory_md = tmp_path / "MEMORY.md"
    memory_md.write_text("- the home server runs the assistant\n", encoding="utf-8")
    monkeypatch.setattr(
        settings.CURATED_MEMORY,
        "PATHS",
        {"ws": [str(memory_md), str(tmp_path / "gone.md"), str(user_md)]},
    )

    assert curated_memory.curated_memory_block("ws") == (
        '<file name="MEMORY.md">\n- the home server runs the assistant\n</file>\n\n'
        '<file name="USER.md">\n# USER\n\n- vegetarian\n</file>'
    )


def test_unconfigured_workspace_and_missing_file_yield_nothing(
    user_md: Path, monkeypatch: pytest.MonkeyPatch
):
    assert curated_memory.curated_memory_block("other") == ""
    monkeypatch.setattr(
        settings.CURATED_MEMORY, "PATHS", {"ws": [str(user_md) + ".gone"]}
    )
    assert curated_memory.curated_memory_block("ws") == ""


def test_edits_are_picked_up_by_modification_time(user_md: Path):
    first = curated_memory.curated_memory_block("ws")
    user_md.write_text("# USER\n\n- vegetarian\n- runs at dawn\n", encoding="utf-8")
    os.utime(user_md, (user_md.stat().st_atime, user_md.stat().st_mtime + 5))

    second = curated_memory.curated_memory_block("ws")

    assert first != second and second.endswith("- runs at dawn\n</file>")
