from typing import Any
from unittest.mock import patch

import pytest

from src.llm import tool_loop as tool_loop_module
from src.llm.runtime import AttemptPlan
from src.llm.tool_loop import execute_tool_loop


class _QuotaRejection(RuntimeError):
    status_code: int = 429


async def test_a_spent_quota_is_attempted_once_inside_the_tool_loop() -> None:
    calls = 0

    async def _quota_spent(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise _QuotaRejection("usage_limit_reached: the plan's quota is spent")

    plan = AttemptPlan(
        provider="openai",
        model="gpt-6-sol",
        client=object(),
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=None,
        attempt=1,
        retry_attempts=3,
        is_fallback=False,
    )
    tools: list[dict[str, Any]] = [
        {"name": "noop", "description": "no-op", "input_schema": {"type": "object"}}
    ]

    with (
        patch.object(tool_loop_module, "honcho_llm_call_inner", new=_quota_spent),
        pytest.raises(_QuotaRejection),
    ):
        await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            tool_choice="auto",
            tool_executor=lambda _name, _input: "",
            max_tool_iterations=5,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=True,
            retry_attempts=3,
            max_input_tokens=None,
            get_attempt_plan=lambda: plan,
            before_retry_callback=lambda _r: None,
            stream_final=False,
            telemetry=None,
        )

    assert calls == 1
