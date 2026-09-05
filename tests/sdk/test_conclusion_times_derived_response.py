from datetime import datetime, timezone

from sdks.python.src.honcho.api_types import ConclusionResponse
from sdks.python.src.honcho.conclusions import Conclusion


def test_sdk_conclusion_exposes_times_derived() -> None:
    response = ConclusionResponse(
        id="conclusion-1",
        content="The user prefers focused tests.",
        observer_id="observer",
        observed_id="observed",
        session_id="session-1",
        times_derived=7,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    conclusion = Conclusion.from_api_response(response)

    assert conclusion.times_derived == 7