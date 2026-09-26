"""Automatic dialectic: one prefetch on the caller's search text, one model call, no tools;
conclusions the caller already holds come back to the model as known."""

import logging
from typing import Any

from src import models, schemas
from src.config import DialecticLevelSettings, ReasoningLevel
from src.crud.document import fetch_documents_by_ids
from src.crud.message import search_messages
from src.dependencies import tracked_db
from src.dialectic import prompts
from src.dialectic.core import DialecticAgent
from src.embedding_client import embedding_client
from src.telemetry.events import EmbeddingCallPurpose
from src.utils import summarizer
from src.utils.agent_tools import search_memory
from src.utils.evidence import EvidenceAccumulator
from src.utils.formatting import format_new_turn_with_timestamp
from src.utils.types import embedding_call_purpose

logger = logging.getLogger(__name__)

_EXCERPT_MESSAGE_CHARS = 600
_NONE_ANSWER = "NONE"

Snippet = tuple[list[models.Message], list[models.Message]]


class AutomaticDialecticAgent(DialecticAgent):
    """Answers from one prefetched context block; the model gets no tools."""

    def __init__(
        self,
        *,
        workspace_name: str,
        session_name: str | None,
        observer: str,
        observed: str,
        options: schemas.AutomaticChatOptions,
        metric_key: str | None = None,
        reasoning_level: ReasoningLevel = "low",
        session_id: str | None = None,
        session_allowlist: list[str] | None = None,
        evidence: EvidenceAccumulator | None = None,
    ) -> None:
        super().__init__(
            workspace_name=workspace_name,
            session_name=session_name,
            observer=observer,
            observed=observed,
            metric_key=metric_key,
            reasoning_level=reasoning_level,
            session_id=session_id,
            session_allowlist=session_allowlist,
            evidence=evidence,
        )
        self.options: schemas.AutomaticChatOptions = options
        self.messages[0] = {
            "role": "system",
            "content": prompts.automatic_system_prompt(
                observed, options.max_answer_chars
            ),
        }

    def _select_tools(self) -> list[dict[str, Any]]:
        return []

    def _tool_choice(
        self, level_settings: DialecticLevelSettings
    ) -> str | dict[str, Any] | None:
        return None

    def _prefetch_heading(self) -> str:
        return "Context"

    def _prefetch_intro(self) -> str:
        return "Answer from this context only; no tools are available."

    async def _prefetch_relevant_observations(self, query: str) -> str | None:
        options = self.options
        try:
            with embedding_call_purpose(
                EmbeddingCallPurpose.DIALECTIC_PREFETCH.value,
                workspace_name=self.workspace_name,
                run_id=self._run_id,
                parent_category="dialectic",
                session_id=self.session_id,
            ):
                embedding = await embedding_client.embed(options.search_text)
            on_topic = await self._on_topic_conclusions(embedding)
            known = await self._known_conclusions()
            snippets = await self._excerpts(embedding)
            summaries = await self._summaries_for(snippets)
        except Exception as e:
            logger.warning(f"Failed to prefetch automatic context: {e}")
            return None

        self._prefetched_conclusion_count = len(on_topic)
        if self.evidence is not None:
            self.evidence.add_documents(on_topic)
            for _, context in snippets:
                self.evidence.add_messages(context)
        return _render(known, on_topic, snippets, summaries)

    async def _on_topic_conclusions(
        self, embedding: list[float]
    ) -> list[models.Document]:
        options = self.options
        if options.conclusion_limit <= 0:
            return []
        excluded = set(options.exclude_conclusion_ids)
        found: list[models.Document] = []
        await search_memory(
            workspace_name=self.workspace_name,
            observer=self.observer,
            observed=self.observed,
            query=options.search_text,
            limit=options.conclusion_limit + len(excluded),
            embedding=embedding,
            session_allowlist=self.session_allowlist,
            documents_out=found,
        )
        return [doc for doc in found if doc.id not in excluded][
            : options.conclusion_limit
        ]

    async def _known_conclusions(self) -> list[models.Document]:
        ids = self.options.exclude_conclusion_ids
        if not ids:
            return []
        async with tracked_db("dialectic.automatic.known", read_only=True) as db:
            documents = await fetch_documents_by_ids(
                db, self.workspace_name, self.observer, self.observed, list(ids)
            )
            for doc in documents:
                db.expunge(doc)
        return documents

    async def _excerpts(self, embedding: list[float]) -> list[Snippet]:
        options = self.options
        if options.excerpt_limit <= 0:
            return []
        fetch = options.excerpt_limit + (3 if options.exclude_session_id else 0)
        snippets = await search_messages(
            self.workspace_name,
            None,
            options.search_text,
            limit=fetch,
            context_window=1,
            embedding=embedding,
            observer=self.observer,
            session_allowlist=self.session_allowlist,
        )
        kept: list[Snippet] = []
        for matches, context in snippets:
            session = context[0].session_name if context else None
            if options.exclude_session_id and session == options.exclude_session_id:
                continue
            kept.append((matches, context))
        return kept[: options.excerpt_limit]

    async def _summaries_for(self, snippets: list[Snippet]) -> dict[str, str]:
        names = {context[0].session_name for _, context in snippets if context}
        if not names:
            return {}
        found: dict[str, str] = {}
        async with tracked_db("dialectic.automatic.summaries", read_only=True) as db:
            for name in sorted(names):
                summary = await summarizer.get_summary(
                    db, self.workspace_name, name, summarizer.SummaryType.SHORT
                )
                if summary:
                    found[name] = summary["content"]
        return found

    async def answer(self, query: str, response_model: type[Any] | None = None) -> str:
        content = await super().answer(query, response_model=response_model)
        if (
            response_model is None
            and content.strip().strip(".").upper() == _NONE_ANSWER
        ):
            return ""
        return content


def _render(
    known: list[models.Document],
    on_topic: list[models.Document],
    snippets: list[Snippet],
    summaries: dict[str, str],
) -> str | None:
    parts: list[str] = []
    if known:
        parts.append(
            "## Known to the assistant\n"
            + "\n".join(f"- {doc.content}" for doc in known)
        )
    if on_topic:
        parts.append(
            "## Conclusions on topic\n"
            + "\n".join(f"- {doc.content}" for doc in on_topic)
        )
    if snippets:
        blocks: list[str] = []
        for index, (_, context) in enumerate(snippets, 1):
            session = context[0].session_name if context else "unknown"
            head = f"### Thread {index} ({session})"
            summary = summaries.get(session)
            if summary:
                head += f"\nSummary: {summary}"
            lines = [
                format_new_turn_with_timestamp(
                    message.content[:_EXCERPT_MESSAGE_CHARS],
                    message.created_at,
                    message.peer_name,
                )
                for message in context
            ]
            blocks.append(head + "\n" + "\n".join(lines))
        parts.append("## Earlier threads\n" + "\n\n".join(blocks))
    return "\n\n".join(parts) if parts else None
