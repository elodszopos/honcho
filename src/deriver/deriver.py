import asyncio
import json
import logging
import time
from typing import Any, cast

from nanoid import generate as generate_nanoid

from src import crud
from src.config import ConfiguredModelSettings, settings
from src.crud.representation import RepresentationManager
from src.dependencies import tracked_db
from src.exceptions import RepresentationSaveError
from src.llm import honcho_llm_call
from src.llm.types import LLMTelemetryContext
from src.models import Message
from src.schemas import ResolvedConfiguration
from src.telemetry import prometheus_metrics
from src.telemetry.events import RepresentationCompletedEvent, emit
from src.telemetry.events.llm import CallPurpose
from src.telemetry.logging import accumulate_metric, log_performance_metrics
from src.telemetry.prometheus.metrics import (
    DeriverComponents,
    DeriverTaskTypes,
    TokenTypes,
)
from src.telemetry.sentry import with_sentry_transaction
from src.utils.config_helpers import get_configuration
from src.utils.curated_memory import curated_memory_block
from src.utils.representation import (
    AdmissionRepresentation,
    ExplicitObservation,
    ExtractedRepresentation,
    Representation,
)
from src.utils.retryable_errors import is_retryable_error
from src.utils.tokens import estimate_tokens, track_deriver_input_tokens

from .prompts import (
    deriver_messages,
    estimate_deriver_prompt_tokens,
    format_deriver_message,
)

logger = logging.getLogger(__name__)


def _get_deriver_model_config() -> ConfiguredModelSettings:
    return settings.DERIVER.MODEL_CONFIG


def _model_config_log_fields(config: ConfiguredModelSettings) -> dict[str, Any]:
    """Return non-secret model-routing fields for operational logs."""
    overrides = getattr(config, "overrides", None)
    fallback = getattr(config, "fallback", None)
    raw_provider_params = getattr(overrides, "provider_params", None)
    provider_params = (
        cast(dict[str, Any], raw_provider_params)
        if isinstance(raw_provider_params, dict)
        else {}
    )
    return {
        "transport": config.transport,
        "model": config.model,
        "thinking_effort": config.thinking_effort,
        "thinking_budget_tokens": config.thinking_budget_tokens,
        "max_output_tokens": config.max_output_tokens,
        "structured_output_mode": config.structured_output_mode,
        "base_url": getattr(overrides, "base_url", None),
        "provider_param_keys": sorted(provider_params.keys()),
        "fallback": (
            None
            if fallback is None
            else {
                "transport": fallback.transport,
                "model": fallback.model,
                "thinking_effort": fallback.thinking_effort,
            }
        ),
    }


@with_sentry_transaction("minimal_deriver_batch", op="deriver")
async def process_representation_tasks_batch(
    messages: list[Message],
    message_level_configuration: ResolvedConfiguration | None,
    *,
    observers: list[str],
    observed: str,
    queue_item_message_ids: list[int],
    session_id: str | None = None,
    queue_item_ids: list[int] | None = None,
    hit_batch_token_cap: bool = False,
    was_flush_enabled: bool = False,
    batch_max_tokens: int = 0,
) -> None:
    """
    Process messages with minimal overhead - single LLM call, save to multiple collections.

    Args:
        messages: List of messages to process (includes interleaving context).
        message_level_configuration: Optional configuration override.
        observers: List of observer peer IDs (collections to save to).
        observed: The observed peer ID.
        queue_item_message_ids: Message IDs from queue items being processed
        session_id: Canonical Session.id from the queue, resolved if not provided.
        queue_item_ids: Queue rows that triggered this batch, when available.
        hit_batch_token_cap: queue batcher clamped this batch to fit
        was_flush_enabled: DERIVER.FLUSH_ENABLED snapshot at batch time
        batch_max_tokens: DERIVER.REPRESENTATION_BATCH_TARGET_INPUT_TOKENS snapshot
    """
    if not messages:
        return

    overall_start = time.perf_counter()

    messages.sort(key=lambda x: x.id)
    latest_message = messages[-1]
    earliest_message = messages[0]
    logger.info(
        "deriver.batch start: workspace=%s session=%s observed=%s observers=%d queued_items=%d prompt_messages=%d message_id_range=%s:%s hit_batch_token_cap=%s flush_enabled=%s batch_max_tokens=%d",
        latest_message.workspace_name,
        latest_message.session_name,
        observed,
        len(observers),
        len(queue_item_message_ids),
        len(messages),
        earliest_message.id,
        latest_message.id,
        hit_batch_token_cap,
        was_flush_enabled,
        batch_max_tokens,
    )

    # Reuse the queue's canonical session ID, resolving it for direct/legacy callers.
    # TODO: this appears to be a very rare edge case coming out of `get_queue_item_batch` in queue_manager.py,
    # possible that we can remove this and require configuration to come through with the payload.
    if message_level_configuration is None or session_id is None:
        async with tracked_db("minimal_deriver.get_config") as db:
            session = await crud.get_session(
                db, latest_message.session_name, latest_message.workspace_name
            )
            session_id = session.id
            if message_level_configuration is None:
                message_level_configuration = get_configuration(
                    None,
                    session,
                    await crud.get_workspace(
                        db, workspace_name=latest_message.workspace_name
                    ),
                )

    # Skip if disabled
    if message_level_configuration.reasoning.enabled is False:
        logger.info(
            "deriver.batch skipped: workspace=%s session=%s observed=%s reason=reasoning_disabled elapsed_ms=%.1f",
            latest_message.workspace_name,
            latest_message.session_name,
            observed,
            (time.perf_counter() - overall_start) * 1000,
        )
        return

    custom_instructions = message_level_configuration.reasoning.custom_instructions

    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "starting_message_id",
        earliest_message.id,
        "id",
    )
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "ending_message_id",
        latest_message.id,
        "id",
    )

    formatted_messages = "\n".join(
        format_deriver_message(
            idx, msg.id, msg.peer_name, observed, msg.created_at, msg.content
        )
        for idx, msg in enumerate(messages)
    )

    # Track token usage - count only tokens from messages being processed
    curated_memory = curated_memory_block(latest_message.workspace_name)
    prompt_tokens = estimate_deriver_prompt_tokens(custom_instructions, curated_memory)
    queue_item_message_ids_set = set(queue_item_message_ids)
    messages_tokens = sum(
        msg.token_count for msg in messages if msg.id in queue_item_message_ids_set
    )
    track_deriver_input_tokens(
        task_type=DeriverTaskTypes.INGESTION,
        components={
            DeriverComponents.PROMPT: prompt_tokens,
            DeriverComponents.MESSAGES: messages_tokens,
        },
    )
    estimated_input_tokens = prompt_tokens + sum(msg.token_count for msg in messages)
    if estimated_input_tokens > settings.DERIVER.MAX_INPUT_TOKENS:
        logger.warning(
            "deriver.batch over_cap: workspace=%s session=%s observed=%s estimated_input_tokens=%d max_input_tokens=%d prompt_scaffold_tokens=%d prompt_messages=%d",
            latest_message.workspace_name,
            latest_message.session_name,
            observed,
            estimated_input_tokens,
            settings.DERIVER.MAX_INPUT_TOKENS,
            prompt_tokens,
            len(messages),
        )

    extraction_messages = deriver_messages(
        peer_id=observed,
        messages=formatted_messages,
        custom_instructions=custom_instructions,
        curated_memory=curated_memory,
    )
    prompt = "\n\n".join(message["content"] for message in extraction_messages)

    context_prep_duration = (time.perf_counter() - overall_start) * 1000
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "context_preparation",
        context_prep_duration,
        "ms",
    )

    # validation on settings means max_tokens will always be > 0
    base_model_config = _get_deriver_model_config()
    max_tokens = base_model_config.max_output_tokens or settings.LLM.DEFAULT_MAX_TOKENS
    model_config = base_model_config

    # First LLM call: extract durable-memory candidates without writing them.
    trace_id = generate_nanoid()
    llm_start = time.perf_counter()
    logger.info(
        "deriver.batch llm_start: trace_id=%s workspace=%s session=%s observed=%s model=%s max_tokens=%d max_input_tokens=%d prompt_chars=%d prompt_scaffold_tokens=%d queued_message_tokens=%d prompt_message_tokens=%d queued_items=%d prompt_messages=%d",
        trace_id,
        latest_message.workspace_name,
        latest_message.session_name,
        observed,
        _model_config_log_fields(model_config),
        max_tokens,
        settings.DERIVER.MAX_INPUT_TOKENS,
        len(prompt),
        prompt_tokens,
        messages_tokens,
        sum(msg.token_count for msg in messages),
        len(queue_item_message_ids),
        len(messages),
    )
    try:
        response = await honcho_llm_call(
            model_config=model_config,
            prompt=extraction_messages[-1]["content"],
            messages=extraction_messages,
            max_tokens=max_tokens,
            response_model=ExtractedRepresentation,
            json_mode=True,
            max_input_tokens=settings.DERIVER.MAX_INPUT_TOKENS,
            enable_retry=True,
            retry_attempts=3,
            trace_name="minimal_deriver",
            telemetry=LLMTelemetryContext(
                workspace_name=latest_message.workspace_name,
                session_id=session_id,
                call_purpose=CallPurpose.DERIVER_REPRESENTATION.value,
                parent_category="representation",
                agent_type="deriver",
                observers=observers,
                observed=observed,
                source_message_ids=[
                    m.public_id for m in messages if m.id in queue_item_message_ids_set
                ],
                queue_item_ids=queue_item_ids or [],
                track_name="Minimal Deriver",
                trace_id=trace_id,
                span_id=trace_id,
            ),
        )
    except Exception:
        logger.exception(
            "deriver.batch llm_failed: trace_id=%s workspace=%s session=%s observed=%s elapsed_ms=%.1f",
            trace_id,
            latest_message.workspace_name,
            latest_message.session_name,
            observed,
            (time.perf_counter() - llm_start) * 1000,
        )
        raise
    llm_duration = (time.perf_counter() - llm_start) * 1000
    logger.info(
        "deriver.batch llm_done: trace_id=%s workspace=%s session=%s observed=%s elapsed_ms=%.1f input_tokens=%d output_tokens=%d cache_read=%d cache_creation=%d hit_input_token_cap=%s response_type=%s",
        trace_id,
        latest_message.workspace_name,
        latest_message.session_name,
        observed,
        llm_duration,
        response.input_tokens,
        response.output_tokens,
        response.cache_read_input_tokens or 0,
        response.cache_creation_input_tokens or 0,
        response.hit_input_token_cap,
        type(response.content).__name__,
    )

    message_ids = [m.id for m in messages if m.peer_name == observed]
    target_message_ids = set(message_ids)
    eligible_message_ids = {m.id for m in messages}
    observations = Representation()
    successful_observer_count = 0
    admission_response = None
    admission_llm_duration = 0.0
    save_errors: list[tuple[str, Exception]] = []

    if not response.content.explicit or not message_ids:
        logger.info(
            "Deriver generated zero admission candidates for messages %s:%s in %s/%s",
            earliest_message.id,
            latest_message.id,
            latest_message.workspace_name,
            latest_message.session_name,
        )
    else:
        managers = {
            observer: RepresentationManager(
                workspace_name=latest_message.workspace_name,
                observer=observer,
                observed=observed,
            )
            for observer in observers
        }
        case_inputs = [
            (observer, candidate)
            for observer in observers
            for candidate in response.content.explicit
        ]
        candidate_representations = await asyncio.gather(
            *(
                managers[observer].get_working_representation(
                    include_semantic_query=candidate.content,
                    semantic_search_top_k=10,
                    semantic_search_max_distance=None,
                    include_most_derived=False,
                    max_observations=10,
                    parent_category="representation_admission",
                )
                for observer, candidate in case_inputs
            )
        )

        admission_cases: list[dict[str, Any]] = []
        case_context: dict[int, tuple[str, Any, list[str]]] = {}
        for case_id, ((observer, candidate), candidate_representation) in enumerate(
            zip(case_inputs, candidate_representations, strict=True)
        ):
            candidate_ids = [
                item.id
                for item in (
                    candidate_representation.explicit
                    + candidate_representation.deductive
                    + candidate_representation.inductive
                    + candidate_representation.contradiction
                )
                if item.id
            ]
            case_context[case_id] = (observer, candidate, candidate_ids)
            admission_cases.append(
                {
                    "admission_case_id": case_id,
                    "observer_id": observer,
                    "candidate_observation": candidate.content,
                    "searched_conclusion_ids": candidate_ids,
                }
            )

        admission_messages = deriver_messages(
            peer_id=observed,
            messages=formatted_messages,
            existing_conclusions=json.dumps(
                [
                    {
                        "admission_case_id": case_id,
                        "conclusions": candidate_representation.format_as_markdown(
                            include_ids=True
                        ),
                    }
                    for case_id, candidate_representation in enumerate(
                        candidate_representations
                    )
                ],
                indent=2,
            ),
            candidate_observation=json.dumps(admission_cases, indent=2),
            custom_instructions=custom_instructions,
            curated_memory=curated_memory,
        )
        admission_prompt = "\n\n".join(
            message["content"] for message in admission_messages
        )
        track_deriver_input_tokens(
            task_type=DeriverTaskTypes.INGESTION,
            components={DeriverComponents.ADMISSION: estimate_tokens(admission_prompt)},
        )
        admission_trace_id = generate_nanoid()
        admission_llm_start = time.perf_counter()
        try:
            admission_response = await honcho_llm_call(
                model_config=model_config,
                prompt=admission_messages[-1]["content"],
                messages=admission_messages,
                max_tokens=max_tokens,
                response_model=AdmissionRepresentation,
                json_mode=True,
                max_input_tokens=settings.DERIVER.MAX_INPUT_TOKENS,
                enable_retry=True,
                retry_attempts=3,
                trace_name="conclusion_admission",
                telemetry=LLMTelemetryContext(
                    workspace_name=latest_message.workspace_name,
                    session_id=session_id,
                    call_purpose=CallPurpose.DERIVER_REPRESENTATION.value,
                    parent_category="representation_admission",
                    agent_type="deriver",
                    observers=observers,
                    observed=observed,
                    source_message_ids=[
                        m.public_id
                        for m in messages
                        if m.id in queue_item_message_ids_set
                    ],
                    queue_item_ids=queue_item_ids or [],
                    track_name="Conclusion Admission",
                    trace_id=admission_trace_id,
                    span_id=admission_trace_id,
                ),
            )
        except Exception:
            logger.exception(
                "deriver.batch admission_failed: trace_id=%s workspace=%s session=%s observed=%s elapsed_ms=%.1f",
                admission_trace_id,
                latest_message.workspace_name,
                latest_message.session_name,
                observed,
                (time.perf_counter() - admission_llm_start) * 1000,
            )
            raise
        admission_llm_duration = (time.perf_counter() - admission_llm_start) * 1000

        admitted_by_observer: dict[str, Representation] = {}
        decided_case_ids: set[int] = set()
        for decision in admission_response.content.explicit:
            if decision.admission_case_id in decided_case_ids:
                raise ValueError(
                    f"Admission agent returned duplicate case {decision.admission_case_id}"
                )
            decided_case_ids.add(decision.admission_case_id)
            if decision.admission_case_id not in case_context:
                raise ValueError(
                    f"Admission agent returned unknown case {decision.admission_case_id}"
                )
            observer, candidate, candidate_ids = case_context[
                decision.admission_case_id
            ]
            cited = set(decision.source_message_ids)
            if (
                len(cited) != len(decision.source_message_ids)
                or not cited.issubset(eligible_message_ids)
                or not cited & target_message_ids
            ):
                raise ValueError(
                    "Admission agent cited duplicate or non-prompt message IDs, or none from the target peer"
                )
            if decision.action == "enrich" and decision.target_id not in candidate_ids:
                raise ValueError(
                    f"Admission agent targeted unsearched conclusion {decision.target_id!r}"
                )
            admitted_by_observer.setdefault(observer, Representation()).explicit.append(
                ExplicitObservation(
                    content=decision.content,
                    action=decision.action,
                    target_id=decision.target_id,
                    reason_for_entry=decision.reason_for_entry,
                    created_at=latest_message.created_at,
                    message_ids=decision.source_message_ids,
                    session_name=latest_message.session_name,
                    searched_conclusion_ids=candidate_ids,
                    search_query=candidate.content,
                    trace_id=admission_trace_id,
                    model=model_config.model,
                )
            )

        # Persist each observer's complete candidate set in one transactional call.
        for observer, admitted in admitted_by_observer.items():
            try:
                saved = await managers[observer].save_representation(
                    admitted,
                    message_ids,
                    latest_message.session_name,
                    latest_message.created_at,
                    message_level_configuration,
                )
            except Exception as e:  # noqa: BLE001
                logger.exception(
                    "Failed to save representation for observer %s", observer
                )
                save_errors.append((observer, e))
                continue
            if saved:
                successful_observer_count += 1
                observations.explicit.extend(admitted.explicit)

    total_llm_duration = llm_duration + admission_llm_duration
    total_input_tokens = response.input_tokens + (
        admission_response.input_tokens if admission_response is not None else 0
    )
    total_output_tokens = response.output_tokens + (
        admission_response.output_tokens if admission_response is not None else 0
    )
    hit_input_token_cap = response.hit_input_token_cap or (
        admission_response.hit_input_token_cap
        if admission_response is not None
        else False
    )
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "llm_call_duration",
        total_llm_duration,
        "ms",
    )

    if settings.METRICS.ENABLED:
        prometheus_metrics.record_deriver_tokens(
            count=total_output_tokens,
            task_type=DeriverTaskTypes.INGESTION.value,
            token_type=TokenTypes.OUTPUT.value,
            component=DeriverComponents.OUTPUT_TOTAL.value,
        )

    # Log metrics
    overall_duration = (time.perf_counter() - overall_start) * 1000
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "total_processing_time",
        overall_duration,
        "ms",
    )

    total_observations = len(observations.explicit) + len(observations.deductive)
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "observation_count",
        total_observations,
        "count",
    )

    if settings.DERIVER.LOG_OBSERVATIONS:
        # Log messages fed into deriver
        accumulate_metric(
            f"minimal_deriver_{latest_message.id}_{observed}",
            "messages",
            formatted_messages,
            "blob",
        )
        # Log actual observations created as blob metrics
        accumulate_metric(
            f"minimal_deriver_{latest_message.id}_{observed}",
            "explicit_observations",
            "\n".join(f" • {obs}" for obs in observations.explicit),
            "blob",
        )

    log_performance_metrics("minimal_deriver", f"{latest_message.id}_{observed}")

    # token-breakdown fields derived from messages + cap snapshots.
    queued_message_count = len(queue_item_message_ids)
    prompt_message_count = len(messages)
    prompt_message_tokens = sum(msg.token_count for msg in messages)
    extra_context_message_count = max(prompt_message_count - queued_message_count, 0)
    extra_context_tokens = max(prompt_message_tokens - messages_tokens, 0)

    logger.info(
        "deriver.batch done: trace_id=%s workspace=%s session=%s observed=%s observers_saved=%d/%d explicit=%d deductive=%d total_observations=%d queued_items=%d prompt_messages=%d extra_context_messages=%d total_elapsed_ms=%.1f llm_elapsed_ms=%.1f",
        trace_id,
        latest_message.workspace_name,
        latest_message.session_name,
        observed,
        successful_observer_count,
        len(observers),
        len(observations.explicit),
        len(observations.deductive),
        total_observations,
        queued_message_count,
        prompt_message_count,
        extra_context_message_count,
        overall_duration,
        total_llm_duration,
    )

    # Data-quality invariants. Best-effort — telemetry never bleeds into the
    # deriver path — but log loudly when violated so analytics alerting catches
    # silent estimator failures (provider tokenization drift, scaffold helper
    # returning 0) at the source instead of as drift in BigQuery later.
    if response.input_tokens < messages_tokens:
        logger.warning(
            "token-breakdown invariant violated: response.input_tokens (%d) < messages_tokens (%d) for observed=%s, latest=%s — provider tokenization drift or wrong messages_tokens computation?",
            response.input_tokens,
            messages_tokens,
            observed,
            latest_message.public_id,
        )
    if prompt_tokens <= 0:
        logger.warning(
            "prompt_scaffold_tokens estimated as %d for observed=%s, latest=%s — estimate_deriver_prompt_tokens may have failed silently",
            prompt_tokens,
            observed,
            latest_message.public_id,
        )

    # Emit telemetry event
    emit(
        RepresentationCompletedEvent(
            workspace_name=latest_message.workspace_name,
            session_name=latest_message.session_name,
            observed=observed,
            queue_items_processed=len(queue_item_message_ids),
            earliest_message_id=earliest_message.public_id,
            latest_message_id=latest_message.public_id,
            message_count=len(messages),
            explicit_conclusion_count=len(observations.explicit),
            context_preparation_ms=context_prep_duration,
            llm_call_ms=total_llm_duration,
            total_duration_ms=overall_duration,
            input_tokens=messages_tokens,
            total_input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            # additive fields
            queued_message_count=queued_message_count,
            prompt_message_count=prompt_message_count,
            prompt_message_tokens=prompt_message_tokens,
            extra_context_message_count=extra_context_message_count,
            extra_context_tokens=extra_context_tokens,
            prompt_scaffold_tokens=prompt_tokens,
            batch_max_tokens=batch_max_tokens,
            max_input_tokens=settings.DERIVER.MAX_INPUT_TOKENS,
            was_flush_enabled=was_flush_enabled,
            hit_batch_token_cap=hit_batch_token_cap,
            hit_input_token_cap=hit_input_token_cap,
            observer_count=successful_observer_count,
            failed_observer_count=len(save_errors),
        )
    )

    retryable = next(
        (exc for _, exc in save_errors if is_retryable_error(exc)),
        None,
    )
    if retryable is not None:
        raise retryable
    if save_errors and successful_observer_count == 0:
        details = "; ".join(
            f"{observer}: {exc.__class__.__name__}: {exc}"
            for observer, exc in save_errors
        )
        raise RepresentationSaveError(
            f"save_representation failed for all {len(save_errors)} observer(s): "
            + details
        ) from save_errors[0][1]
