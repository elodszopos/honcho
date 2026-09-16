"""`conclusion list` and `conclusion search` carry the attribution fields.

The CLI is the fifth client surface and had no conclusion coverage at all, so a
field that reaches the SDK and stops here is invisible to anything scripted
against it.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from honcho_cli.main import app


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Isolated config file + clean HONCHO_* env."""
    f = tmp_path / "config.json"
    monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("honcho_cli.config.CONFIG_FILE", f)
    for k in [k for k in os.environ if k.startswith("HONCHO_")]:
        monkeypatch.delenv(k)
    f.write_text(
        json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"})
    )
    return f


@pytest.fixture
def runner():
    return CliRunner()


def _conclusion(index: int) -> MagicMock:
    return MagicMock(
        id=f"c{index}",
        content=f"conclusion {index}",
        observer_id="alice",
        observed_id="bob",
        session_id="s1",
        times_derived=index + 1,
        source_ids=[f"p{index}"],
        created_at="2026-01-01T00:00:00Z",
    )


def _patched_client(conclusions: list[MagicMock]):
    scope = MagicMock()
    scope.list.return_value = MagicMock(items=conclusions)
    scope.query.return_value = conclusions
    peer = MagicMock()
    peer.conclusions_of.return_value = scope
    peer.conclusions = scope
    client = MagicMock()
    client.peer.return_value = peer
    config = MagicMock(workspace_id="ws1")
    return patch(
        "honcho_cli.commands.conclusion.get_client", return_value=(client, config)
    )


@pytest.mark.parametrize(
    ("argv", "label"),
    [
        (["conclusion", "list", "--observer", "alice", "--observed", "bob"], "list"),
        (
            ["conclusion", "search", "dark mode", "--observer", "alice", "--observed", "bob"],
            "search",
        ),
    ],
)
def test_conclusion_rows_carry_attribution(cfg, runner, argv: list[str], label: str):
    conclusions = [_conclusion(0), _conclusion(1)]

    with _patched_client(conclusions):
        result = runner.invoke(app, argv)

    assert result.exit_code == 0, f"{label} failed: {result.output}"
    rows = json.loads(result.output)
    assert [row["id"] for row in rows] == ["c0", "c1"]
    assert [row["times_derived"] for row in rows] == [1, 2]
    assert [row["source_ids"] for row in rows] == [["p0"], ["p1"]]
