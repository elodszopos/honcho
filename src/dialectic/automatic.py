"""Automatic dialectic: one prefetch on the caller's search text, one model call, no tools;
the messages behind conclusions the caller already holds never become excerpts."""

import json
import logging
import re
import time
from typing import Any

from src import models, schemas
from src.config import DialecticLevelSettings, ReasoningLevel, settings
from src.crud.document import embed_thoughts, source_message_ids
from src.crud.message import search_message_snippets
from src.dependencies import tracked_db
from src.dialectic import prompts
from src.dialectic.core import DialecticAgent
from src.telemetry.events import EmbeddingCallPurpose
from src.utils import summarizer
from src.utils.curated_memory import curated_memory_block
from src.utils.evidence import EvidenceAccumulator
from src.utils.formatting import format_new_turn_with_timestamp
from src.utils.types import embedding_call_purpose

logger = logging.getLogger(__name__)

_NONE_ANSWER = "NONE"
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)|\n")

Snippet = tuple[list[models.Message], list[models.Message]]


def cut_at_sentence(text: str, limit: int) -> str:
    """``text`` whole when it fits, else its longest prefix ending a sentence within ``limit``,
    else a hard cut at ``limit``."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    ends = [match.end() for match in _SENTENCE_END.finditer(head)]
    return head[: ends[-1]].rstrip() if ends else head


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
        self._prefetch_stats: dict[str, float] = {}
        self.messages[0] = {
            "role": "system",
            "content": prompts.automatic_system_prompt(
                observed,
                options.max_answer_chars,
                curated_memory_block(workspace_name),
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
        """The context block; a failed embed or read raises out of the run."""
        options = self.options
        started = time.perf_counter()
        with embedding_call_purpose(
            EmbeddingCallPurpose.DIALECTIC_PREFETCH.value,
            workspace_name=self.workspace_name,
            run_id=self._run_id,
            parent_category="dialectic",
            session_id=self.session_id,
        ):
            embeddings = await embed_thoughts(options.search_text)
        self._prefetch_stats["embed_ms"] = (time.perf_counter() - started) * 1000
        excluded = await self._excluded_message_ids()
        snippets = await self._excerpts(embeddings, excluded)
        summaries = await self._summaries_for(snippets)

        self._prefetched_conclusion_count = 0
        self._context_empty = not snippets
        if self.evidence is not None:
            for _, context in snippets:
                self.evidence.add_messages(context)
        rendered, cuts = _render(snippets, summaries)
        logger.info(
            "dialectic.automatic prefetch: run_id=%s search_chars=%d thoughts=%d excluded_conclusions=%d excluded_messages=%d excerpts=%d summaries=%d excerpt_limit=%d excerpt_max_distance=%s excerpt_messages=%d excerpts_cut=%d excerpt_chars=%d/%d max_answer_chars=%d exclude_session=%s embed_ms=%.1f elapsed_ms=%.1f rendered_chars=%d",
            self._run_id,
            len(options.search_text),
            len(embeddings),
            len(options.exclude_conclusion_ids),
            len(excluded),
            len(snippets),
            len(summaries),
            options.excerpt_limit,
            options.excerpt_max_distance,
            cuts.messages,
            cuts.cut,
            cuts.rendered_chars,
            cuts.raw_chars,
            options.max_answer_chars,
            bool(options.exclude_session_id),
            self._prefetch_stats.get("embed_ms", 0.0),
            (time.perf_counter() - started) * 1000,
            len(rendered or ""),
        )
        return rendered

    async def _excluded_message_ids(self) -> set[int]:
        ids = self.options.exclude_conclusion_ids
        if not ids:
            return set()
        async with tracked_db("dialectic.automatic.excluded", read_only=True) as db:
            return await source_message_ids(
                db, self.workspace_name, self.observer, self.observed, list(ids)
            )

    async def _excerpts(
        self, embeddings: list[list[float]], excluded: set[int]
    ) -> list[Snippet]:
        options = self.options
        if options.excerpt_limit <= 0 or not embeddings:
            return []
        snippets, distances = await search_message_snippets(
            self.workspace_name,
            embeddings,
            limit=options.excerpt_limit,
            context_window=1,
            observer=self.observer,
            session_allowlist=self.session_allowlist,
            exclude_session_name=options.exclude_session_id,
            min_chars=settings.DIALECTIC.AUTOMATIC_EXCERPT_MIN_MESSAGE_CHARS,
            max_distance=options.excerpt_max_distance,
            exclude_message_ids=excluded,
        )
        hits = [
            {
                "rank": rank,
                "session": context[0].session_name if context else None,
                "ids": [message.public_id for message in matches],
                "distances": [
                    _rounded(distances.get(message.public_id)) for message in matches
                ],
            }
            for rank, (matches, context) in enumerate(snippets, 1)
        ]
        logger.info(
            "dialectic.automatic excerpts: run_id=%s hits=%s",
            self._run_id,
            json.dumps(hits),
        )
        return snippets

    async def _summaries_for(self, snippets: list[Snippet]) -> dict[str, str]:
        names = {context[0].session_name for _, context in snippets if context}
        if not names:
            return {}
        async with tracked_db("dialectic.automatic.summaries", read_only=True) as db:
            found = await summarizer.get_summaries(
                db, self.workspace_name, names, summarizer.SummaryType.SHORT
            )
        return {name: summary["content"] for name, summary in found.items()}

    async def _prepare_query(self, query: str) -> Any:
        prepared = await super()._prepare_query(query)
        if getattr(self, "_context_empty", False):
            raise _NoContext
        return prepared

    async def answer(self, query: str, response_model: type[Any] | None = None) -> str:
        """The model answers from the excerpts; with none there is no call and the answer is empty."""
        try:
            content = await super().answer(query, response_model=response_model)
        except _NoContext:
            logger.info(
                "dialectic.automatic answer: run_id=%s outcome=no_context chars=0",
                self._run_id,
            )
            return ""
        is_none = (
            response_model is None
            and content.strip().strip(".").upper() == _NONE_ANSWER
        )
        logger.info(
            "dialectic.automatic answer: run_id=%s outcome=%s chars=%d",
            self._run_id,
            "none" if is_none else "answer",
            0 if is_none else len(content),
        )
        return "" if is_none else content


class _NoContext(Exception):
    """Raised inside prepare when the prefetch found no excerpt to answer from."""


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


class _ExcerptCuts:
    def __init__(self) -> None:
        self.messages = 0
        self.cut = 0
        self.raw_chars = 0
        self.rendered_chars = 0

    def take(self, content: str) -> str:
        rendered = cut_at_sentence(
            content, settings.DIALECTIC.AUTOMATIC_EXCERPT_MESSAGE_CHARS
        )
        self.messages += 1
        self.raw_chars += len(content)
        self.rendered_chars += len(rendered)
        if len(rendered) < len(content):
            self.cut += 1
        return rendered


def _render(
    snippets: list[Snippet],
    summaries: dict[str, str],
) -> tuple[str | None, _ExcerptCuts]:
    parts: list[str] = []
    cuts = _ExcerptCuts()
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
                    cuts.take(message.content),
                    message.created_at,
                    message.peer_name,
                )
                for message in context
            ]
            blocks.append(head + "\n" + "\n".join(lines))
        parts.append("## Earlier threads\n" + "\n\n".join(blocks))
    return ("\n\n".join(parts) if parts else None), cuts
