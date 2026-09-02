from datetime import datetime, timezone
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.deriver.deriver import process_representation_tasks_batch
from src.llm import HonchoLLMCallResponse
from src.models import Message
from src.utils.representation import (
    AdmissionDecision,
    AdmissionRepresentation,
    ExtractedObservation,
    ExtractedRepresentation,
    Representation,
)


def _message(message_id: int, peer_name: str, content: str) -> Message:
    return cast(
        Message,
        Mock(
            id=message_id,
            public_id=f"msg_{message_id}",
            session_name="session-1",
            workspace_name="workspace-1",
            peer_name=peer_name,
            content=content,
            token_count=5,
            created_at=datetime.now(timezone.utc),
        ),
    )


def _configuration() -> Mock:
    configuration = Mock()
    configuration.reasoning.enabled = True
    configuration.reasoning.custom_instructions = None
    return configuration


def _llm_response(content: object, *, input_tokens: int, output_tokens: int):
    return HonchoLLMCallResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reasons=["STOP"],
    )


@pytest.mark.asyncio
async def test_admission_is_batched_and_persisted_once_with_candidate_source_ids() -> (
    None
):
    messages = [
        _message(11, "alice", "I prefer dark mode."),
        _message(12, "bob", "Do you also prefer keyboard navigation?"),
        _message(13, "alice", "I always use keyboard navigation."),
    ]
    extraction = _llm_response(
        ExtractedRepresentation(
            explicit=[
                ExtractedObservation(content="The user prefers dark mode"),
                ExtractedObservation(content="The user prefers keyboard navigation"),
            ]
        ),
        input_tokens=100,
        output_tokens=20,
    )
    admission = _llm_response(
        AdmissionRepresentation(
            explicit=[
                AdmissionDecision(
                    admission_case_id=0,
                    content="The user prefers dark mode",
                    action="create",
                    reason_for_entry="Durable interface preference",
                    source_message_ids=[11],
                ),
                AdmissionDecision(
                    admission_case_id=1,
                    content="The user prefers keyboard navigation",
                    action="create",
                    reason_for_entry="Durable interaction preference",
                    source_message_ids=[13],
                ),
            ]
        ),
        input_tokens=60,
        output_tokens=15,
    )

    with (
        patch(
            "src.deriver.deriver.honcho_llm_call",
            new=AsyncMock(side_effect=[extraction, admission]),
        ) as llm_call,
        patch(
            "src.deriver.deriver.RepresentationManager.get_working_representation",
            new=AsyncMock(side_effect=[Representation(), Representation()]),
        ) as search,
        patch(
            "src.deriver.deriver.RepresentationManager.save_representation",
            new=AsyncMock(return_value=2),
        ) as save,
        patch("src.deriver.deriver.emit") as emit,
    ):
        await process_representation_tasks_batch(
            messages=messages,
            message_level_configuration=_configuration(),
            observers=["bob"],
            observed="alice",
            queue_item_message_ids=[11, 13],
        )

    assert llm_call.await_count == 2
    assert search.await_count == 2
    assert (
        llm_call.await_args_list[1].kwargs["response_model"] is AdmissionRepresentation
    )
    admission_prompt = llm_call.await_args_list[1].kwargs["prompt"]
    assert "[message_id:11]" in admission_prompt
    assert "[message_id:12]" in admission_prompt
    assert "[message_id:13]" in admission_prompt
    assert '"admission_case_id": 0' in admission_prompt
    assert '"admission_case_id": 1' in admission_prompt

    save.assert_awaited_once()
    save_args = save.await_args
    if save_args is None:
        raise AssertionError("Expected one atomic representation save")
    persisted = save_args.args[0]
    assert [item.message_ids for item in persisted.explicit] == [[11], [13]]

    event = emit.call_args.args[0]
    assert event.total_input_tokens == 160
    assert event.output_tokens == 35
    assert event.llm_call_ms >= 0


@pytest.mark.asyncio
async def test_invalid_late_admission_fails_before_atomic_write() -> None:
    messages = [
        _message(21, "alice", "I prefer dark mode."),
        _message(22, "bob", "Context from the other peer."),
        _message(23, "alice", "I prefer keyboard navigation."),
    ]
    extraction = _llm_response(
        ExtractedRepresentation(
            explicit=[
                ExtractedObservation(content="The user prefers dark mode"),
                ExtractedObservation(content="The user prefers keyboard navigation"),
            ]
        ),
        input_tokens=100,
        output_tokens=20,
    )
    admission = _llm_response(
        AdmissionRepresentation(
            explicit=[
                AdmissionDecision(
                    admission_case_id=0,
                    content="The user prefers dark mode",
                    action="create",
                    reason_for_entry="Durable interface preference",
                    source_message_ids=[21],
                ),
                AdmissionDecision(
                    admission_case_id=1,
                    content="The user prefers keyboard navigation",
                    action="create",
                    reason_for_entry="Purported durable preference",
                    source_message_ids=[22],
                ),
            ]
        ),
        input_tokens=60,
        output_tokens=15,
    )

    with (
        patch(
            "src.deriver.deriver.honcho_llm_call",
            new=AsyncMock(side_effect=[extraction, admission]),
        ),
        patch(
            "src.deriver.deriver.RepresentationManager.get_working_representation",
            new=AsyncMock(side_effect=[Representation(), Representation()]),
        ),
        patch(
            "src.deriver.deriver.RepresentationManager.save_representation",
            new=AsyncMock(return_value=2),
        ) as save,
        pytest.raises(ValueError, match="wrong-peer message IDs"),
    ):
        await process_representation_tasks_batch(
            messages=messages,
            message_level_configuration=_configuration(),
            observers=["bob"],
            observed="alice",
            queue_item_message_ids=[21, 23],
        )

    save.assert_not_awaited()
