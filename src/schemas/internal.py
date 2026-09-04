"""Internal schemas used by the deriver, dreamer, and other background systems.

These are not part of the public API contract and may change without notice.
"""

from enum import Enum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from src.schemas.api import MessageCreate
from src.schemas.configuration import SessionPeerConfig
from src.utils.types import DocumentLevel
from src.writing_contract import MAX_CONCLUSION_CHARS


class ReconcilerType(str, Enum):
    """Types of reconciler tasks that can be performed."""

    SYNC_VECTORS = "sync_vectors"
    CLEANUP_QUEUE = "cleanup_queue"


# ---------------------------------------------------------------------------
# Document / observation schemas (vector storage internals)
# ---------------------------------------------------------------------------


class DocumentBase(BaseModel):
    pass


class DocumentMetadata(BaseModel):
    message_ids: list[int] = Field(
        description="The ID range(s) of the messages that this document was derived from. Acts as a link to the primary source of the document. Note that as a document gets deduplicated, additional ranges will be added, because the same document could be derived from completely separate message ranges."
    )
    message_created_at: str = Field(
        description="The timestamp of the message that this document was derived from. Note that this is not the same as the created_at timestamp of the document. This timestamp is usually only saved with second-level precision."
    )
    source_ids: list[str] | None = Field(
        default=None,
        description="Document IDs of source documents for tree traversal -- required for deductive and inductive documents",
    )
    premises: list[str] | None = Field(
        default=None,
        description="Human-readable premise text for display -- only applicable for deductive documents",
    )
    sources: list[str] | None = Field(
        default=None,
        description="Human-readable source text for display -- only applicable for inductive documents",
    )
    pattern_type: str | None = Field(
        default=None,
        description="Type of pattern identified (preference, behavior, personality, tendency, correlation) -- only applicable for inductive documents",
    )
    confidence: str | None = Field(
        default=None,
        description="Confidence level (high, medium, low) -- only applicable for inductive documents",
    )
    admission: dict[str, Any] | None = Field(
        default=None,
        description="Mandatory admission decision and deterministic provenance for new conclusions.",
    )
    admission_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Prior content and admission envelopes retained across enrichment revisions.",
    )


class DocumentCreate(DocumentBase):
    content: Annotated[str, Field(min_length=1, max_length=100000)]
    session_name: str | None = Field(
        default=None,
        description="The session from which the document was derived (NULL for global observations)",
    )
    level: DocumentLevel = Field(
        default="explicit",
        description="The level of the document (explicit, deductive, inductive, or contradiction)",
    )
    times_derived: int = Field(
        default=1,
        ge=1,
        description="The number of times that a semantic duplicate document to this one has been derived",
    )
    metadata: DocumentMetadata = Field()
    embedding: list[float] = Field()
    # Tree linkage field
    source_ids: list[str] | None = Field(
        default=None,
        description="Document IDs of source/premise documents -- for deductive and inductive documents",
    )


class ObservationInput(BaseModel):
    """Validated agent observation plus its search-backed admission decision."""

    content: Annotated[str, Field(min_length=1, max_length=MAX_CONCLUSION_CHARS)]
    level: DocumentLevel = "explicit"
    source_ids: list[str] = Field(default_factory=list)
    premises: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    pattern_type: (
        Literal["preference", "behavior", "personality", "tendency", "correlation"]
        | None
    ) = None
    confidence: Literal["high", "medium", "low"] | None = None
    action: Literal["create", "enrich"]
    target_id: str | None = None
    reason_for_entry: Annotated[str, Field(min_length=1)]
    search_query: Annotated[str, Field(min_length=1)]
    searched_conclusion_ids: list[str]

    @field_validator("content", "reason_for_entry", "search_query", mode="after")
    @classmethod
    def sanitize_required_text(cls, value: str) -> str:
        cleaned = value.replace("\x00", "").strip()
        if not cleaned:
            raise ValueError("value must contain non-whitespace text")
        return cleaned

    @field_validator("target_id", mode="after")
    @classmethod
    def sanitize_target_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("source_ids", "searched_conclusion_ids", mode="after")
    @classmethod
    def sanitize_id_list(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @model_validator(mode="after")
    def validate_admission_and_level_fields(self) -> Self:
        if self.action == "enrich" and not self.target_id:
            raise ValueError("target_id is required when action is enrich")
        if self.action == "create" and self.target_id is not None:
            raise ValueError("target_id is forbidden when action is create")
        if self.target_id and self.target_id not in self.searched_conclusion_ids:
            raise ValueError("target_id must be present in searched_conclusion_ids")

        if self.level == "deductive" and (not self.source_ids or not self.premises):
            raise ValueError("deductive observations require source_ids and premises")
        if self.level == "inductive" and (
            len(self.source_ids) < 2 or len(self.sources) < 2 or not self.pattern_type
        ):
            raise ValueError(
                "inductive observations require at least two source_ids, two sources, and pattern_type"
            )
        if self.level == "contradiction" and (
            len(self.source_ids) < 2 or len(self.sources) < 2
        ):
            raise ValueError(
                "contradiction observations require at least two source_ids and two sources"
            )
        return self


# ---------------------------------------------------------------------------
# Queue internals
# ---------------------------------------------------------------------------


class SessionCounts(BaseModel):
    """Counts for a specific session in queue processing."""

    completed: int
    in_progress: int
    pending: int


class QueueCounts(BaseModel):
    """Aggregated counts for queue processing status."""

    total: int
    completed: int
    in_progress: int
    pending: int
    sessions: dict[str, SessionCounts]


class QueueStatusRow(BaseModel):
    """Represents a row from the queue status SQL query result."""

    session_id: str | None
    total: int
    completed: int
    in_progress: int
    pending: int
    session_total: int
    session_completed: int
    session_in_progress: int
    session_pending: int


# ---------------------------------------------------------------------------
# Internal data containers
# ---------------------------------------------------------------------------


class SessionPeerData(BaseModel):
    """Data for managing session peer relationships."""

    peer_names: dict[str, SessionPeerConfig]


class MessageBulkData(BaseModel):
    """Data for bulk message operations."""

    messages: list[MessageCreate]
    session_name: str
    workspace_name: str
