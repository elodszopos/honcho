from __future__ import annotations

import pytest

from src.config import settings
from src.deriver.prompts import minimal_deriver_prompt
from src.llm import honcho_llm_call
from src.utils.representation import ExtractedRepresentation

pytestmark = pytest.mark.live_llm


async def _extract(message: str) -> ExtractedRepresentation:
    model_config = settings.DERIVER.MODEL_CONFIG
    response = await honcho_llm_call(
        model_config=model_config,
        prompt=minimal_deriver_prompt(peer_id="user", messages=f"user: {message}"),
        max_tokens=model_config.max_output_tokens or settings.LLM.DEFAULT_MAX_TOKENS,
        response_model=ExtractedRepresentation,
        json_mode=True,
        max_input_tokens=settings.DERIVER.MAX_INPUT_TOKENS,
        enable_retry=True,
        retry_attempts=3,
        trace_name="live_travel_memory_scope",
    )
    return response.content


@pytest.mark.asyncio
async def test_live_deriver_rejects_representative_travel_persona_doctrine() -> None:
    result = await _extract(
        " ".join(
            [
                "For flexible base-camp travel days, prioritize forecast-weighted experience quality over route efficiency",
                "and accept reasonable backtracking. I do not want guided hikes included. Choose parking by proximity to",
                "the actual planned attractions, not generic venue labels.",
            ]
        )
    )

    assert result.explicit == []


@pytest.mark.asyncio
async def test_live_deriver_still_extracts_cross_domain_user_preferences() -> None:
    result = await _extract(
        "From now on, keep my engineering answers concise and evidence-based."
    )

    assert len(result.explicit) == 1
    extracted = result.explicit[0].content.lower()
    assert "concise" in extracted
    assert "evidence" in extracted
