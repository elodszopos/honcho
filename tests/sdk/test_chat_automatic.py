"""Automatic chat options reach the dialectic intact; invalid ones never leave the SDK."""

from typing import Any

import pytest
from pydantic import ValidationError

from sdks.python.src.honcho import AutomaticChatParams
from sdks.python.src.honcho.client import Honcho
from src import schemas

QUERY = "What did we decide about the reminder job?"
OPTIONS: dict[str, Any] = {
    "search_text": "reminder job | fire at 9",
    "exclude_conclusion_ids": ["c-1", "c-2"],
    "exclude_session_id": "thread-1",
    "conclusion_limit": 7,
    "excerpt_limit": 3,
    "max_answer_chars": 600,
}


def _forwarded(mock: Any) -> Any:
    return mock.await_args.kwargs["automatic"]


@pytest.mark.asyncio
async def test_peer_chat_forwards_every_automatic_option(
    client_fixture: tuple[Honcho, str],
    mock_llm_call_functions: dict[str, Any],
):
    honcho, client_type = client_fixture

    if client_type == "async":
        peer = await honcho.aio.peer("alice")
        answer = await peer.aio.chat(QUERY, automatic=OPTIONS)
    else:
        answer = honcho.peer("alice").chat(QUERY, automatic=OPTIONS)

    assert isinstance(answer, str)
    forwarded = _forwarded(mock_llm_call_functions["agentic_chat"])
    assert isinstance(forwarded, schemas.AutomaticChatOptions)
    assert forwarded.model_dump() == OPTIONS


def test_typed_params_arrive_with_their_defaults(
    honcho_sync_test_client: Honcho,
    mock_llm_call_functions: dict[str, Any],
):
    params = AutomaticChatParams(search_text="reminder job")

    honcho_sync_test_client.peer("alice").chat(QUERY, automatic=params)

    forwarded = _forwarded(mock_llm_call_functions["agentic_chat"])
    assert forwarded.model_dump() == params.model_dump()


def test_chat_without_automatic_forwards_none(
    honcho_sync_test_client: Honcho,
    mock_llm_call_functions: dict[str, Any],
):
    honcho_sync_test_client.peer("alice").chat(QUERY)

    assert _forwarded(mock_llm_call_functions["agentic_chat"]) is None


@pytest.mark.parametrize(
    "bad_options",
    [
        {"search_text": "reminder job", "unknown_field": 1},
        {"search_text": ""},
        {"search_text": "reminder job", "conclusion_limit": 51},
        {"search_text": "reminder job", "exclude_conclusion_ids": ["c"] * 201},
    ],
)
def test_invalid_options_are_refused_before_the_request(
    honcho_sync_test_client: Honcho,
    mock_llm_call_functions: dict[str, Any],
    bad_options: dict[str, Any],
):
    with pytest.raises(ValidationError):
        honcho_sync_test_client.peer("alice").chat(QUERY, automatic=bad_options)

    mock_llm_call_functions["agentic_chat"].assert_not_awaited()
