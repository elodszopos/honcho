from datetime import datetime, timezone
from types import SimpleNamespace

from src.schemas.api import Conclusion


def test_conclusion_response_exposes_times_derived() -> None:
    document = SimpleNamespace(
        id="conclusion-1",
        content="The user prefers focused tests.",
        observer="observer",
        observed="observed",
        session_name="session-1",
        level="explicit",
        internal_metadata={},
        times_derived=7,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    payload = Conclusion.model_validate(document).model_dump(by_alias=True)

    assert payload["times_derived"] == 7