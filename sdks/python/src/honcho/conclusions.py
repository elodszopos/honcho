# pyright: reportPrivateUsage=false
"""Conclusion types and scoped access for the Honcho SDK."""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal
from pydantic import BaseModel, Field, model_validator

from .api_types import (
    ConclusionLevel,
    ConclusionLineageResponse,
    ConclusionRemovalParams,
    ConclusionResponse,
    RemovalCategory,
    RepresentationResponse,
)
from .base import SessionBase
from .http import routes
from .pagination import SyncPage
from .utils import resolve_id

if TYPE_CHECKING:
    from .aio import ConclusionsViewAio
    from .client import Honcho

__all__ = [
    "Conclusion",
    "ConclusionsView",
    "ConclusionCreateParams",
]

# Filter keys that define a conclusions view (the observer/observed peer pair).
# They are set from the view itself, so a caller must not pass them in `filters`.
_VIEW_RESERVED = ("observer", "observed", "observer_id", "observed_id")


def _reject_reserved_filter_keys(
    filters: dict[str, Any] | None, reserved: tuple[str, ...]
) -> None:
    """Raise if ``filters`` contains keys managed by the conclusions view.

    The observer/observed peer pair (and, on ``list``, the session) is fixed by
    the view, so letting a user filter override it would silently return data
    from a different pair than requested. Fail loud instead.
    """
    if not filters:
        return
    clash = sorted(k for k in reserved if k in filters)
    if clash:
        guidance = (
            "Choose the peer pair via peer.conclusions / peer.conclusions_of(target)"
        )
        if "session" in reserved or "session_id" in reserved:
            guidance += "; use the session= parameter to filter by session"
        raise ValueError(
            f"Filter key(s) {clash} are managed by this conclusions view and "
            + f"cannot be passed in filters. {guidance}."
        )


class ConclusionCreateParams(BaseModel):
    content: str = Field(min_length=1, max_length=800)
    session_id: str | None = None
    level: ConclusionLevel = "explicit"
    action: Literal["create", "enrich"]
    target_id: str | None = None
    reason_for_entry: str
    search_query: str
    searched_conclusion_ids: list[str]
    source_message_ids: list[int] = Field(default_factory=list)
    source_tool_call_id: str | None = None
    entry_origin: Literal[
        "deriver_agent",
        "dreamer_agent",
        "explicit_agent",
        "operator_cli",
        "operator_sdk",
        "operator_import",
    ]
    agent_trace_id: str
    agent_model: str
    times_derived: int | None = None
    source_ids: list[str] | None = None
    premises: list[str] | None = None
    sources: list[str] | None = None
    pattern_type: (
        Literal["preference", "behavior", "personality", "tendency", "correlation"]
        | None
    ) = None
    confidence: Literal["high", "medium", "low"] | None = None

    @model_validator(mode="after")
    def validate_admission(self) -> "ConclusionCreateParams":
        if self.action == "create" and self.target_id is not None:
            raise ValueError("create decisions cannot set target_id")
        if self.action == "enrich":
            if self.target_id is None:
                raise ValueError("enrich decisions require target_id")
            if self.target_id not in self.searched_conclusion_ids:
                raise ValueError("target_id must be present in searched_conclusion_ids")
        if self.entry_origin == "deriver_agent" and not self.source_message_ids:
            raise ValueError("deriver_agent conclusions require source_message_ids")
        if self.entry_origin == "dreamer_agent" and not self.source_ids:
            raise ValueError("dreamer_agent conclusions require source_ids")
        if self.entry_origin == "explicit_agent" and not (
            self.source_message_ids or self.source_tool_call_id
        ):
            raise ValueError(
                "explicit_agent conclusions require source_message_ids or source_tool_call_id"
            )
        if self.entry_origin.startswith("operator_") and not self.source_tool_call_id:
            raise ValueError("operator conclusions require source_tool_call_id")
        return self


class Conclusion:
    """
    A conclusion from Honcho's reasoning system.

    Conclusions are facts derived from messages that help build a representation
    of a peer.

    Attributes:
        id: Unique identifier for this conclusion
        content: The conclusion content/text
        observer_id: The peer ID who made this conclusion
        observed_id: The peer ID this conclusion is about
        session_id: The session this conclusion relates to
        level: Reasoning level ("explicit", "deductive", "inductive",
            "contradiction"). "explicit" conclusions are extracted directly
            from messages; the others are derived during dreaming.
        times_derived: Number of times this conclusion has been independently
            derived, including counts inherited from conclusions it absorbed.
        removal: Why this conclusion was retired, or None while it is live.
        created_at: Timestamp for when the conclusion was created
        deleted_at: When it was retired, or None while it is live
    """

    id: str
    content: str
    observer_id: str
    observed_id: str
    session_id: str | None = None
    level: ConclusionLevel = "explicit"
    admission: dict[str, Any]
    removal: dict[str, Any] | None = None
    times_derived: int = 1
    created_at: datetime.datetime
    deleted_at: datetime.datetime | None = None

    def __init__(
        self,
        id: str,
        content: str,
        observer_id: str,
        observed_id: str,
        session_id: str | None,
        created_at: datetime.datetime,
        level: ConclusionLevel = "explicit",
        admission: dict[str, Any] | None = None,
        removal: dict[str, Any] | None = None,
        times_derived: int = 1,
        deleted_at: datetime.datetime | None = None,
    ) -> None:
        self.id = id
        self.content = content
        self.observer_id = observer_id
        self.observed_id = observed_id
        self.session_id = session_id
        self.level = level
        self.admission = admission or {}
        self.removal = removal
        self.times_derived = times_derived
        self.created_at = created_at
        self.deleted_at = deleted_at

    @classmethod
    def from_api_response(cls, data: ConclusionResponse) -> "Conclusion":
        """Create a Conclusion from an API response."""
        return cls(
            id=data.id,
            content=data.content,
            observer_id=data.observer_id,
            observed_id=data.observed_id,
            session_id=data.session_id,
            level=data.level,
            admission=data.admission,
            removal=data.removal,
            times_derived=data.times_derived,
            created_at=data.created_at,
            deleted_at=data.deleted_at,
        )

    def __repr__(self) -> str:
        truncated = (
            f"{self.content[:50]}..." if len(self.content) > 50 else self.content
        )
        return f"{type(self).__name__}(id='{self.id}', content='{truncated}')"

    def __str__(self) -> str:
        return self.content


class ConclusionLineage(Conclusion):
    """A conclusion with its full ledger: prior formulations and absorptions."""

    admission_history: list[dict[str, Any]]
    absorbed: list[dict[str, Any]]

    def __init__(
        self,
        *,
        admission_history: list[dict[str, Any]] | None = None,
        absorbed: list[dict[str, Any]] | None = None,
        **conclusion_fields: Any,
    ) -> None:
        super().__init__(**conclusion_fields)
        self.admission_history = admission_history or []
        self.absorbed = absorbed or []

    @classmethod
    def from_lineage_response(
        cls, data: ConclusionLineageResponse
    ) -> "ConclusionLineage":
        """Create a ConclusionLineage from a lineage API response."""
        return cls(
            id=data.id,
            content=data.content,
            observer_id=data.observer_id,
            observed_id=data.observed_id,
            session_id=data.session_id,
            level=data.level,
            admission=data.admission,
            removal=data.removal,
            times_derived=data.times_derived,
            created_at=data.created_at,
            deleted_at=data.deleted_at,
            admission_history=data.admission_history,
            absorbed=data.absorbed,
        )


class ConclusionsView:
    """
    Scoped access to conclusions for a specific observer/observed relationship.

    This class provides convenient methods to list, query, create, and delete conclusions
    that are automatically scoped to a specific observer/observed pair.

    Typically accessed via `peer.conclusions` (for self-conclusions) or
    `peer.conclusions_of(target)` (for conclusions about another peer).

    Example:
        ```python
        # Get self-conclusions
        conclusions = peer.conclusions
        obs_list = conclusions.list()
        search_results = conclusions.query("preferences")

        # Get conclusions about another peer
        bob_conclusions = peer.conclusions_of("bob")
        bob_list = bob_conclusions.list()

        # Async operations via .aio accessor
        obs_list = await peer.conclusions.aio.list()
        ```
    """

    _honcho: "Honcho"
    workspace_id: str
    observer: str
    observed: str

    def __init__(
        self,
        honcho: "Honcho",
        workspace_id: str,
        observer: str,
        observed: str,
    ):
        """
        Initialize a ConclusionsView.

        Args:
            honcho: The Honcho client instance
            workspace_id: The workspace ID
            observer: The observer peer ID
            observed: The observed peer ID
        """
        self._honcho = honcho
        self.workspace_id = workspace_id
        self.observer = observer
        self.observed = observed

    @property
    def aio(self) -> "ConclusionsViewAio":
        """
        Access async versions of all ConclusionsView methods.

        Returns a ConclusionsViewAio view that provides async versions of all methods
        while sharing state with this ConclusionsView instance.

        Example:
            ```python
            # Async operations
            obs_list = await scope.aio.list()
            results = await scope.aio.query("preferences")
            ```
        """
        # Import here to avoid circular import (aio.py imports from this module)
        from .aio import ConclusionsViewAio

        return ConclusionsViewAio(self)

    def list(
        self,
        page: int = 1,
        size: int = 50,
        session: str | SessionBase | None = None,
        *,
        filters: dict[str, Any] | None = None,
        reverse: bool = False,
        include_deleted: bool = False,
    ) -> SyncPage[ConclusionResponse, Conclusion]:
        """
        List conclusions in this scope.

        Args:
            page: Page number (1-indexed)
            size: Number of results per page
            session: Optional session (ID string or Session object) to filter by
            filters: Optional dictionary of additional filter criteria, merged
                with this scope's observer/observed (and session, if given).
                Supports the same operators as other list endpoints — e.g.
                ``{"level": "explicit"}`` to get only conclusions extracted
                directly from messages (i.e. not derived during dreaming). See
                https://honcho.dev/docs/v3/documentation/features/advanced/using-filters
            reverse: If True, reverses the default ordering. Default: False.
            include_deleted: If True, retired conclusions are included, each carrying
                the `removal` record saying why it went. Default: False.

        Returns:
            Paginated response containing Conclusion objects
        """
        _reject_reserved_filter_keys(
            filters, _VIEW_RESERVED + ("session", "session_id")
        )
        self._honcho._ensure_workspace()
        resolved_session_id = resolve_id(session)
        filters = {
            "observer_id": self.observer,
            "observed_id": self.observed,
            **({"session_id": resolved_session_id} if resolved_session_id else {}),
            **(filters or {}),
        }

        query: dict[str, Any] = {"page": page, "size": size}
        if reverse:
            query["reverse"] = "true"
        if include_deleted:
            query["include_deleted"] = "true"
        data = self._honcho._http.post(
            routes.conclusions_list(self.workspace_id),
            body={"filters": filters},
            query=query,
        )

        def transform(response: ConclusionResponse) -> Conclusion:
            return Conclusion.from_api_response(response)

        def fetch_next(
            next_page: int,
        ) -> SyncPage[ConclusionResponse, Conclusion]:
            next_query: dict[str, Any] = {"page": next_page, "size": size}
            if reverse:
                next_query["reverse"] = "true"
            if include_deleted:
                next_query["include_deleted"] = "true"
            next_data = self._honcho._http.post(
                routes.conclusions_list(self.workspace_id),
                body={"filters": filters},
                query=next_query,
            )
            return SyncPage(next_data, ConclusionResponse, transform, fetch_next)

        return SyncPage(data, ConclusionResponse, transform, fetch_next)

    def query(
        self,
        query: str,
        top_k: int = 10,
        distance: float | None = None,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[Conclusion]:
        """
        Semantic search for conclusions in this scope.

        Args:
            query: The search query string
            top_k: Maximum number of results to return
            distance: Maximum cosine distance threshold (0.0-1.0)
            filters: Optional dictionary of additional filter criteria, merged
                with this scope's observer/observed. Supports the same operators
                as the list endpoint — e.g. ``{"level": "deductive"}`` to search
                only conclusions derived during dreaming. See
                https://honcho.dev/docs/v3/documentation/features/advanced/using-filters

        Returns:
            List of matching Conclusion objects
        """
        _reject_reserved_filter_keys(filters, _VIEW_RESERVED)
        self._honcho._ensure_workspace()
        filters = {
            "observer_id": self.observer,
            "observed_id": self.observed,
            **(filters or {}),
        }

        body: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "filters": filters,
        }
        if distance is not None:
            body["distance"] = distance

        data = self._honcho._http.post(
            routes.conclusions_query(self.workspace_id),
            body=body,
        )
        return [
            Conclusion.from_api_response(ConclusionResponse.model_validate(item))
            for item in data
        ]

    def delete(
        self,
        conclusion_id: str,
        *,
        category: RemovalCategory,
        reason: str,
        agent_trace_id: str,
        agent_model: str,
        absorbed_into: str | None = None,
        entry_origin: str = "operator_sdk",
    ) -> None:
        """
        Retire a conclusion by ID, recording why.

        The row and its ledger are kept; the conclusion stops being returned by search
        and stops reaching any deriving agent.

        Args:
            conclusion_id: The ID of the conclusion to retire
            category: Why it is going. One of api_types.REMOVAL_CATEGORIES
            reason: Specific justification, naming what carries the memory now
            agent_trace_id: The run or trace that decided
            agent_model: The exact model that decided
            absorbed_into: Required for "duplicate_absorbed" — the surviving conclusion,
                which inherits this one's derivation count
            entry_origin: Which kind of caller decided
        """
        self._honcho._ensure_workspace()
        removal = ConclusionRemovalParams(
            category=category,
            reason=reason,
            absorbed_into=absorbed_into,
            entry_origin=entry_origin,
            agent_trace_id=agent_trace_id,
            agent_model=agent_model,
        )
        self._honcho._http.delete(
            routes.conclusion(self.workspace_id, conclusion_id),
            body=removal.model_dump(exclude_none=True),
        )

    def lineage(self, conclusion_id: str) -> ConclusionLineage:
        """
        Get one conclusion's full ledger, whether it is live or retired.

        Returns how it was admitted, every prior formulation it was rewritten from,
        everything it absorbed with the count each brought, and why it was removed if
        it was. Retired conclusions are only reachable here, never through search.
        """
        self._honcho._ensure_workspace()
        data = self._honcho._http.get(
            routes.conclusion_lineage(self.workspace_id, conclusion_id)
        )
        return ConclusionLineage.from_lineage_response(
            ConclusionLineageResponse.model_validate(data)
        )

    def create(
        self,
        conclusions: Sequence[ConclusionCreateParams | dict[str, Any]],
    ) -> list[Conclusion]:
        """Create or enrich conclusions through the public admission API."""
        self._honcho._ensure_workspace()

        def build_conclusion_payload(
            item: ConclusionCreateParams | dict[str, Any],
        ) -> dict[str, Any]:
            params = (
                item
                if isinstance(item, ConclusionCreateParams)
                else ConclusionCreateParams.model_validate(item)
            )
            raw = params.model_dump(exclude_none=True)
            payload: dict[str, Any] = {
                "content": raw["content"],
                "observer_id": self.observer,
                "observed_id": self.observed,
            }
            session_id = raw.pop("session_id", None)
            raw.pop("content")
            if session_id is not None:
                payload["session_id"] = session_id
            payload.update(raw)
            return payload

        conclusion_params = [build_conclusion_payload(item) for item in conclusions]
        data = self._honcho._http.post(
            routes.conclusions(self.workspace_id),
            body={"conclusions": conclusion_params},
        )
        return [
            Conclusion.from_api_response(ConclusionResponse.model_validate(item))
            for item in data
        ]

    def representation(
        self,
        search_query: str | None = None,
        search_top_k: int | None = None,
        search_max_distance: float | None = None,
        include_most_frequent: bool | None = None,
        max_conclusions: int | None = None,
    ) -> str:
        """
        Get the computed representation for this scope.

        This returns the working representation (narrative) built from the
        conclusions in this scope.

        Args:
            search_query: Optional semantic search query to curate the representation
            search_top_k: Number of semantically relevant facts to return
            search_max_distance: Maximum semantic distance for search results (0.0-1.0)
            include_most_frequent: Whether to include the most frequent conclusions
            max_conclusions: Maximum number of conclusions to include

        Returns:
            A Representation string
        """
        self._honcho._ensure_workspace()
        body: dict[str, Any] = {"target": self.observed}
        if search_query is not None:
            body["search_query"] = search_query
        if search_top_k is not None:
            body["search_top_k"] = search_top_k
        if search_max_distance is not None:
            body["search_max_distance"] = search_max_distance
        if include_most_frequent is not None:
            body["include_most_frequent"] = include_most_frequent
        if max_conclusions is not None:
            body["max_conclusions"] = max_conclusions

        data = self._honcho._http.post(
            routes.peer_representation(self.workspace_id, self.observer),
            body=body,
        )
        response = RepresentationResponse.model_validate(data)
        return response.representation

    def __repr__(self) -> str:
        return (
            f"ConclusionsView(workspace_id={self.workspace_id!r}, "
            f"observer={self.observer!r}, observed={self.observed!r})"
        )
