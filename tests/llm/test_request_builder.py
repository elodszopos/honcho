import pytest
from pydantic import BaseModel

from src.config import ModelConfig, settings
from src.exceptions import ValidationException
from src.llm.caching import PromptCachePolicy
from src.llm.request_builder import execute_completion
from tests.llm.conftest import FakeBackend


class SampleResponse(BaseModel):
    answer: str


async def test_last_user_message_ends_with_the_character_limit_when_no_cap_is_sent(
    fake_backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings.LLM, "SEND_OUTPUT_CAP", False)
    config = ModelConfig(model="gpt-4.1", transport="openai")
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Again"},
    ]

    await execute_completion(fake_backend, config, messages=messages, max_tokens=100)

    sent = fake_backend.calls[0]["messages"]
    assert sent[:3] == messages[:3]
    assert sent[3] == {
        "role": "user",
        "content": "Again\n\nTry not to go beyond 400 characters.",
    }
    assert messages[3]["content"] == "Again"


async def test_messages_are_untouched_when_the_cap_goes_on_the_wire(
    fake_backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings.LLM, "SEND_OUTPUT_CAP", True)
    config = ModelConfig(model="gpt-4.1", transport="openai")
    messages = [{"role": "user", "content": "Hello"}]

    await execute_completion(fake_backend, config, messages=messages, max_tokens=100)

    assert fake_backend.calls[0]["messages"] == [{"role": "user", "content": "Hello"}]


async def test_gemini_explicit_budget_passes_tokens_through_without_adjustment(
    fake_backend: FakeBackend,
) -> None:
    config = ModelConfig(
        model="gemini-2.5-flash",
        transport="gemini",
        thinking_budget_tokens=256,
    )

    await execute_completion(
        fake_backend,
        config,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )

    call = fake_backend.calls[0]
    # No auto-adjustment — operators set explicit values
    assert call["max_output_tokens"] == 100
    assert call["max_tokens"] == 100
    assert call["thinking_budget_tokens"] == 256


async def test_thinking_params_are_passed_through_without_capability_dropping(
    fake_backend: FakeBackend,
) -> None:
    config = ModelConfig(
        model="claude-haiku-4-5",
        transport="anthropic",
        thinking_effort="high",
        thinking_budget_tokens=1024,
    )

    await execute_completion(
        fake_backend,
        config,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )

    call = fake_backend.calls[0]
    assert call["thinking_effort"] == "high"
    assert call["thinking_budget_tokens"] == 1024


async def test_cache_policy_is_passed_through_extra_params(
    fake_backend: FakeBackend,
) -> None:
    config = ModelConfig(model="gpt-4.1-mini", transport="openai")
    cache_policy = PromptCachePolicy(mode="prefix", ttl_seconds=300)

    await execute_completion(
        fake_backend,
        config,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=SampleResponse,
        cache_policy=cache_policy,
    )

    call = fake_backend.calls[0]
    assert call["response_format"] is SampleResponse
    assert call["extra_params"]["cache_policy"] == cache_policy


async def test_provider_params_are_merged_into_extra_params(
    fake_backend: FakeBackend,
) -> None:
    config = ModelConfig(
        model="gpt-4.1-mini",
        transport="openai",
        top_p=0.9,
        provider_params={"custom_flag": True},
    )

    await execute_completion(
        fake_backend,
        config,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )

    call = fake_backend.calls[0]
    assert call["extra_params"]["top_p"] == 0.9
    assert call["extra_params"]["custom_flag"] is True


async def test_provider_timeout_is_normalized_into_extra_params(
    fake_backend: FakeBackend,
) -> None:
    """Numeric-string provider timeout values are normalized before backends."""
    config = ModelConfig(
        model="gpt-4.1-mini",
        transport="openai",
        provider_params={"timeout": "42.5"},
    )

    await execute_completion(
        fake_backend,
        config,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )

    call = fake_backend.calls[0]
    assert call["extra_params"]["timeout"] == 42.5


@pytest.mark.parametrize(
    "timeout",
    ["slow", "", 0, -1, True, float("nan"), float("inf"), "nan", "inf"],
)
async def test_provider_timeout_rejects_invalid_values(
    fake_backend: FakeBackend,
    timeout: object,
) -> None:
    """Invalid provider timeout values fail before provider SDK calls."""
    config = ModelConfig(
        model="gpt-4.1-mini",
        transport="openai",
        provider_params={"timeout": timeout},
    )

    with pytest.raises(
        ValidationException,
        match=r"provider_params\.timeout must be a positive number of seconds",
    ):
        await execute_completion(
            fake_backend,
            config,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )
