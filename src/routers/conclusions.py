import logging

from fastapi import APIRouter, Body, Depends, Path, Query
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import apaginate
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas
from src.dependencies import db, read_db
from src.exceptions import ResourceNotFoundException, ValidationException
from src.security import require_auth
from src.telemetry.events import EmbeddingCallPurpose
from src.utils.types import embedding_call_purpose

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/conclusions",
    tags=["conclusions"],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)


@router.post(
    "",
    response_model=list[schemas.Conclusion],
    status_code=201,
)
async def create_conclusions(
    workspace_id: str = Path(...),
    body: schemas.ConclusionBatchCreate = Body(
        ...,
        description="Batch of Conclusions to create",
    ),
    db: AsyncSession = db,
) -> list[schemas.Conclusion]:
    """
    Create one or more Conclusions.

    Conclusions are logical certainties derived from interactions between Peers. They form the basis of a Peer's Representation.
    """
    documents = await crud.create_observations(
        db,
        observations=body.conclusions,
        workspace_name=workspace_id,
    )

    logger.debug(
        "Created %d conclusions in workspace %s",
        len(documents),
        workspace_id,
    )
    return [schemas.Conclusion.model_validate(doc) for doc in documents]


@router.post(
    "/list",
    response_model=Page[schemas.Conclusion],
)
async def list_conclusions(
    workspace_id: str = Path(...),
    options: schemas.ConclusionGet | None = Body(
        None,
        description="Filtering options for the Conclusions list",
    ),
    reverse: bool | None = Query(
        False,
        description="Whether to reverse the order of results",
    ),
    include_deleted: bool = Query(
        False,
        description="Include retired Conclusions, each carrying the reason it was removed",
    ),
    db: AsyncSession = read_db,
):
    """
    List Conclusions using optional filters, ordered by recency unless `reverse` is true. Results are paginated.
    """
    filters = None
    if options and hasattr(options, "filters"):
        filters = options.filters
        if filters == {}:
            filters = None

    stmt = crud.get_documents_with_filters(
        workspace_name=workspace_id,
        filters=filters,
        reverse=reverse or False,
        include_deleted=include_deleted,
    )

    return await apaginate(db, stmt)


@router.post(
    "/query",
    response_model=list[schemas.Conclusion],
)
async def query_conclusions(
    workspace_id: str = Path(...),
    body: schemas.ConclusionQuery = Body(
        ...,
        description="Semantic search parameters for Conclusions",
    ),
    db: AsyncSession = read_db,
) -> list[schemas.Conclusion]:
    """
    Query Conclusions using semantic search. Use `top_k` to control the number of results returned.
    """
    observer = None
    observed = None
    if body.filters:
        observer = body.filters.get("observer") or body.filters.get("observer_id")
        observed = body.filters.get("observed") or body.filters.get("observed_id")

    if not observer or not observed:
        raise ValidationException(
            "observer and observed must be specified for semantic search. "
            + "Pass them inside the 'filters' object, e.g. "
            + '{"query": "...", "filters": {"observer": "alice", "observed": "bob"}}. '
            + "Both 'observer'/'observer_id' and 'observed'/'observed_id' are accepted."
        )

    with embedding_call_purpose(
        EmbeddingCallPurpose.GENERIC_DOCUMENT_SEARCH.value,
        workspace_name=workspace_id,
        parent_category="api",
    ):
        documents = await crud.query_documents(
            db,
            workspace_name=workspace_id,
            query=body.query,
            observer=observer,
            observed=observed,
            filters=body.filters,
            max_distance=body.distance,
            top_k=body.top_k,
        )
    return [schemas.Conclusion.model_validate(doc) for doc in documents]


@router.get(
    "/{conclusion_id}/lineage",
    response_model=schemas.ConclusionDetail,
)
async def get_conclusion_lineage(
    workspace_id: str = Path(...),
    conclusion_id: str = Path(...),
    db: AsyncSession = read_db,
) -> schemas.ConclusionDetail:
    """
    Get one Conclusion's full ledger: how it was admitted, every prior formulation it
    has been rewritten from, everything it absorbed, and why it was removed if it was.

    Retired Conclusions are returned here; they are never returned by search.
    """
    document = await crud.get_document(
        db,
        workspace_name=workspace_id,
        document_id=conclusion_id,
    )
    if document is None:
        raise ResourceNotFoundException("Conclusion not found")
    return schemas.ConclusionDetail.model_validate(document)


@router.delete(
    "/{conclusion_id}",
    status_code=204,
    response_model=None,
)
async def delete_conclusion(
    workspace_id: str = Path(...),
    conclusion_id: str = Path(...),
    body: schemas.ConclusionRemoval = Body(
        ...,
        description="Why this Conclusion is being retired",
    ),
    db: AsyncSession = db,
):
    """
    Retire a single Conclusion by ID, recording why.

    The row and its ledger are kept; the Conclusion stops being returned by search and
    stops reaching any deriving agent. Use `duplicate_absorbed` with `absorbed_into` when
    another Conclusion carries the memory, so its derivation count moves with it.
    """
    if body.category not in schemas.AGENT_REMOVAL_CATEGORIES:
        raise ValidationException(
            f"'{body.category}' is recorded by Honcho itself and cannot be supplied. "
            + f"Choose one of: {', '.join(sorted(schemas.AGENT_REMOVAL_CATEGORIES))}."
        )
    try:
        await crud.delete_document_by_id(
            db,
            workspace_name=workspace_id,
            document_id=conclusion_id,
            removal=body,
        )

        logger.debug("Conclusion %s retired successfully", conclusion_id)
    except ResourceNotFoundException:
        raise
    except ValueError as e:
        logger.warning(f"Failed to retire conclusion {conclusion_id}: {str(e)}")
        raise ResourceNotFoundException("Conclusion not found") from e
