import ast
from pathlib import Path

from tests.tracked_db_patch import TRACKED_DB_TARGETS

_SRC = Path(__file__).resolve().parent.parent / "src"


def _module_level_tracked_db_importers() -> set[str]:
    importers: set[str] = set()
    for path in _SRC.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "src.dependencies"
                and any(alias.name == "tracked_db" for alias in node.names)
            ):
                relative = path.relative_to(_SRC.parent).with_suffix("")
                importers.add(".".join(relative.parts))
    return importers


def test_every_module_level_tracked_db_import_is_redirected_in_tests():
    """An unpatched import site opens sessions on whatever database settings resolve to."""
    importers = _module_level_tracked_db_importers()
    assert importers

    missing = sorted(
        module
        for module in importers
        if f"{module}.tracked_db" not in TRACKED_DB_TARGETS
    )
    assert missing == []
