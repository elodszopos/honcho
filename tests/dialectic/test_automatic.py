"""The automatic dialectic answers from one prefetched block with no tools."""

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.dialectic.automatic import AutomaticDialecticAgent
from src.llm import HonchoLLMCallResponse
from src.utils import summarizer
from src.utils.evidence import EvidenceAccumulator

KNOWN = "User has a Google billing account"
ON_TOPIC = "User monitors Reddit with scheduled digests"
OTHER = "User reviews pull requests first thing in the morning"
EARLIER_MESSAGE = "the reddit digest job runs at 9 every morning now"
CURRENT_MESSAGE = "current thread text about the reddit digest job"
EARLIER_SUMMARY = "Reviewed the reddit digest job and moved it to 9."


def _llm_response(content: str) -> HonchoLLMCallResponse[str]:
    return HonchoLLMCallResponse(
        content=content,
        input_tokens=10,
        output_tokens=5,
        finish_reasons=["end_turn"],
        tool_calls_made=[],
    )


@pytest.fixture
async def automatic_data(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """A peer with conclusions and two threads: an earlier one with a summary, and the current one.

    Returns (workspace, peer, earlier_session, current_session, documents).
    """
    monkeypatch.setattr(settings.VECTOR_STORE, "MIGRATED", False)
    workspace, peer = sample_data
    dims = settings.EMBEDDING.VECTOR_DIMENSIONS

    sessions: list[models.Session] = []
    for content in (EARLIER_MESSAGE, CURRENT_MESSAGE):
        session = (
            await crud.get_or_create_session(
                db_session,
                schemas.SessionCreate(
                    name=str(generate_nanoid()),
                    peers={peer.name: schemas.SessionPeerConfig(observe_me=True)},
                ),
                workspace.name,
            )
        ).resource
        message = models.Message(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer.name,
            content=content,
            seq_in_session=1,
            token_count=8,
        )
        db_session.add(message)
        await db_session.flush()
        db_session.add(
            models.MessageEmbedding(
                content=content,
                message_id=message.public_id,
                workspace_name=workspace.name,
                session_name=session.name,
                peer_name=peer.name,
                sync_state="synced",
                embedding=[0.1] * dims,
            )
        )
        sessions.append(session)
    earlier, current = sessions
    await summarizer._save_summary(
        db_session,
        {
            "content": EARLIER_SUMMARY,
            "message_id": 1,
            "summary_type": summarizer.SummaryType.SHORT.value,
            "created_at": "2026-01-01T00:00:00Z",
            "token_count": 9,
            "message_public_id": "m",
        },
        workspace.name,
        earlier.name,
    )

    db_session.add(
        models.Collection(
            workspace_name=workspace.name, observer=peer.name, observed=peer.name
        )
    )
    await db_session.flush()

    documents: list[models.Document] = []
    for i, content in enumerate([ON_TOPIC, KNOWN, OTHER]):
        document = models.Document(
            workspace_name=workspace.name,
            observer=peer.name,
            observed=peer.name,
            content=content,
            embedding=[0.1 * (i + 1)] * dims,
            session_name=earlier.name,
            level="explicit",
        )
        db_session.add(document)
        documents.append(document)
    await db_session.flush()
    for row in (*documents, *sessions):
        await db_session.refresh(row)
    await db_session.commit()
    return workspace, peer, earlier, current, documents


def make_agent(automatic_data: Any, **overrides: Any) -> AutomaticDialecticAgent:
    workspace, peer, _earlier, current, documents = automatic_data
    known = next(d for d in documents if d.content == KNOWN)
    options = {
        "search_text": "reddit digest",
        "exclude_conclusion_ids": [known.id],
        "exclude_session_id": current.name,
        "excerpt_limit": 5,
    }
    options.update(overrides.pop("options", {}))
    return AutomaticDialecticAgent(
        workspace_name=workspace.name,
        session_name=None,
        observer=peer.name,
        observed=peer.name,
        options=schemas.AutomaticChatOptions(**options),
        **overrides,
    )


async def run_answer(
    agent: AutomaticDialecticAgent, content: str = "- The reddit digest job runs at 9."
) -> tuple[str, AsyncMock]:
    mock_llm_call = AsyncMock(return_value=_llm_response(content))
    with patch("src.dialectic.core.honcho_llm_call", new=mock_llm_call):
        answer = await agent.answer("## Current user message\nlook at the reddit job")
    return answer, mock_llm_call


@pytest.mark.asyncio
class TestAutomaticAgent:
    async def test_one_model_call_with_no_tools(self, automatic_data: Any):
        answer, mock_llm_call = await run_answer(make_agent(automatic_data))

        assert answer == "- The reddit digest job runs at 9."
        assert mock_llm_call.await_count == 1
        kwargs = mock_llm_call.await_args.kwargs
        assert kwargs["tools"] == []
        assert kwargs["tool_choice"] is None

    async def test_excerpt_trace_records_ids_sessions_and_distances(
        self, automatic_data: Any, caplog: pytest.LogCaptureFixture
    ):
        _workspace, _peer, earlier, _current, _documents = automatic_data
        with caplog.at_level(logging.INFO, logger="src.dialectic.automatic"):
            await run_answer(make_agent(automatic_data))
        lines = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("dialectic.automatic excerpts:")
        ]
        assert len(lines) == 1
        hits = json.loads(lines[0].split("hits=", 1)[1])
        assert [hit["session"] for hit in hits] == [earlier.name]
        assert hits[0]["rank"] == 1
        assert hits[0]["ids"] and hits[0]["distances"][0] is not None

    async def test_prefetch_block_carries_known_and_earlier_threads_only(
        self, automatic_data: Any
    ):
        agent = make_agent(automatic_data)
        await run_answer(agent)

        prompt = agent.messages[-1]["content"]
        known_block = prompt.split("## Known to the assistant")[1].split("##")[0]
        threads_block = prompt.split("## Earlier threads")[1]

        assert KNOWN in known_block
        assert "## Conclusions on topic" not in prompt
        assert ON_TOPIC not in prompt and OTHER not in prompt
        assert EARLIER_MESSAGE in threads_block
        assert f"Summary: {EARLIER_SUMMARY}" in threads_block
        assert CURRENT_MESSAGE not in prompt

    async def test_each_thought_of_the_search_text_is_searched(
        self, automatic_data: Any, caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(logging.INFO, logger="src.dialectic.automatic")
        agent = make_agent(
            automatic_data,
            options={
                "search_text": "what about the reddit digest?\nand the morning job timing?"
            },
        )
        await run_answer(agent)

        line = next(
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith("dialectic.automatic prefetch:")
        )
        assert "thoughts=2" in line
        assert EARLIER_MESSAGE in agent.messages[-1]["content"]

    async def test_short_messages_never_become_excerpts(
        self, automatic_data: Any, db_session: AsyncSession
    ):
        workspace, peer, earlier, _current, _documents = automatic_data
        short = models.Message(
            workspace_name=workspace.name,
            session_name=earlier.name,
            peer_name=peer.name,
            content="ok reddit",
            seq_in_session=2,
            token_count=2,
        )
        db_session.add(short)
        await db_session.flush()
        db_session.add(
            models.MessageEmbedding(
                content=short.content,
                message_id=short.public_id,
                workspace_name=workspace.name,
                session_name=earlier.name,
                peer_name=peer.name,
                sync_state="synced",
                embedding=[0.1] * settings.EMBEDDING.VECTOR_DIMENSIONS,
            )
        )
        await db_session.commit()

        agent = make_agent(automatic_data)
        await run_answer(agent)

        threads_block = agent.messages[-1]["content"].split("## Earlier threads")[1]
        assert EARLIER_MESSAGE in threads_block
        assert "- ok reddit" not in threads_block

    async def test_system_prompt_is_the_automatic_one(self, automatic_data: Any):
        agent = make_agent(automatic_data, options={"max_answer_chars": 450})

        system = agent.messages[0]["content"]
        assert "no tools are available" in system
        assert "under 450 characters" in system
        assert "single word NONE" in system
        assert "## TOOLS" not in system

    async def test_curated_memory_closes_the_system_prompt(
        self, automatic_data: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ):
        from src.utils import curated_memory

        workspace = automatic_data[0]
        path = tmp_path / "USER.md"
        path.write_text("- the user is vegetarian\n", encoding="utf-8")
        monkeypatch.setattr(
            settings.CURATED_MEMORY, "PATHS", {workspace.name: str(path)}
        )
        curated_memory._cache.clear()

        agent = make_agent(automatic_data)

        system = agent.messages[0]["content"]
        assert system.rstrip().endswith("## USER.md\n- the user is vegetarian")
        assert "a fact `USER.md` states" in system

    async def test_none_answer_becomes_empty(self, automatic_data: Any):
        answer, _ = await run_answer(make_agent(automatic_data), content="NONE.")

        assert answer == ""

    async def test_evidence_records_excerpt_messages_and_no_conclusions(
        self, automatic_data: Any
    ):
        evidence = EvidenceAccumulator()
        await run_answer(make_agent(automatic_data, evidence=evidence))

        assert not evidence.conclusions
        assert evidence.build().messages

    async def test_prefetch_and_answer_are_logged_with_their_counts(
        self, automatic_data: Any, caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(logging.INFO, logger="src.dialectic.automatic")
        await run_answer(make_agent(automatic_data))
        await run_answer(make_agent(automatic_data), content="NONE")

        prefetch = [
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith("dialectic.automatic prefetch:")
        ]
        answers = [
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith("dialectic.automatic answer:")
        ]
        assert prefetch, "no prefetch line"
        line = prefetch[0]
        for fragment in (
            "thoughts=1",
            "known=1",
            "excerpts=1",
            "summaries=1",
            "exclude_session=True",
        ):
            assert fragment in line, f"{fragment} missing from {line}"
        assert "outcome=answer" in answers[0]
        assert "outcome=none chars=0" in answers[1]

    async def test_zero_excerpts_prefetch_nothing_but_known(self, automatic_data: Any):
        agent = make_agent(automatic_data, options={"excerpt_limit": 0})
        await run_answer(agent)

        prompt = agent.messages[-1]["content"]
        assert "## Known to the assistant" in prompt
        assert "## Earlier threads" not in prompt
