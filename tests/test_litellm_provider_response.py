from types import SimpleNamespace

import pytest

from pico.providers.litellm_provider import LiteLLMProvider


def test_provider_does_not_own_cache_control_by_default() -> None:
    provider = LiteLLMProvider(api_key="test", default_model="anthropic/claude-sonnet-4-5")

    assert provider.disable_auto_cache_control is True


@pytest.mark.asyncio
@pytest.mark.parametrize(("disable_auto", "expected_markers"), [(True, 0), (False, 2)])
async def test_chat_payload_respects_cache_control_owner(
    monkeypatch: pytest.MonkeyPatch,
    disable_auto: bool,
    expected_markers: int,
) -> None:
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            model="anthropic/claude-sonnet-4-5",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="ok",
                        tool_calls=[],
                        reasoning_content=None,
                        thinking_blocks=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                prompt_tokens_details=None,
            ),
        )

    monkeypatch.setattr("pico.providers.litellm_provider.acompletion", fake_acompletion)
    provider = LiteLLMProvider(
        api_key="test",
        default_model="anthropic/claude-sonnet-4-5",
        disable_auto_cache_control=disable_auto,
    )
    tools = [{"type": "function", "function": {"name": "read"}}]

    await provider.chat(
        messages=[{"role": "system", "content": "stable"}],
        tools=tools,
    )

    payload = {"messages": captured["messages"], "tools": captured["tools"]}
    assert _cache_markers(payload) == expected_markers


def _cache_markers(value) -> int:
    if isinstance(value, dict):
        return int("cache_control" in value) + sum(_cache_markers(item) for item in value.values())
    if isinstance(value, list):
        return sum(_cache_markers(item) for item in value)
    return 0


def test_parse_response_preserves_actual_model_identity() -> None:
    provider = LiteLLMProvider.__new__(LiteLLMProvider)
    response = SimpleNamespace(
        model="deepseek-v4-flash",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="ready",
                    tool_calls=[],
                    reasoning_content=None,
                    thinking_blocks=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
            prompt_tokens_details=None,
        ),
    )

    parsed = provider._parse_response(response)

    assert parsed.model == "deepseek-v4-flash"


def test_parse_response_normalizes_deepseek_cache_usage() -> None:
    provider = LiteLLMProvider.__new__(LiteLLMProvider)
    response = SimpleNamespace(
        model="deepseek-v4-flash",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="ready",
                    tool_calls=[],
                    reasoning_content=None,
                    thinking_blocks=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=4090,
            completion_tokens=8,
            total_tokens=4098,
            prompt_cache_hit_tokens=3968,
            prompt_cache_miss_tokens=122,
            prompt_tokens_details=None,
        ),
    )

    parsed = provider._parse_response(response)

    assert parsed.usage == {
        "prompt_tokens": 4090,
        "completion_tokens": 8,
        "total_tokens": 4098,
        "cache_read_input_tokens": 3968,
        "cache_miss_input_tokens": 122,
    }


def test_parse_response_preserves_explicit_zero_cache_counts() -> None:
    provider = LiteLLMProvider.__new__(LiteLLMProvider)
    response = SimpleNamespace(
        model="anthropic/claude-sonnet-4-5",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="ready",
                    tool_calls=[],
                    reasoning_content=None,
                    thinking_blocks=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
            cache_read_input_tokens=0,
            _cache_read_input_tokens=9,
            cache_creation_input_tokens=0,
            _cache_creation_input_tokens=7,
            prompt_tokens_details=None,
        ),
    )

    parsed = provider._parse_response(response)

    assert parsed.usage["cache_read_input_tokens"] == 0
    assert parsed.usage["cache_creation_input_tokens"] == 0


# --- R1: 解析层保留 Provider 原始 tool_call_id ----------------------------------

_LONG_TOOL_ID = "call_00_ofc2MSK9qptdZfbIIIKK7328"


def _raw_tool_call(call_id: str | None, name: str = "read_file", arguments: str = "{}") -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _tool_response(*choices: tuple[list[SimpleNamespace], str | None]) -> SimpleNamespace:
    return SimpleNamespace(
        model="deepseek-v4-pro",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=raw_calls,
                    reasoning_content=None,
                    thinking_blocks=None,
                ),
                finish_reason=finish_reason,
            )
            for raw_calls, finish_reason in choices
        ],
        usage=None,
    )


def test_parse_response_preserves_provider_tool_call_id() -> None:
    """Provider 原始 ID 必须逐字保留：改写会让校验来源的网关在回放时 400。"""
    provider = LiteLLMProvider.__new__(LiteLLMProvider)

    parsed = provider._parse_response(_tool_response(([_raw_tool_call(_LONG_TOOL_ID)], "tool_calls")))

    assert [tc.id for tc in parsed.tool_calls] == [_LONG_TOOL_ID]
    assert parsed.finish_reason == "tool_calls"


@pytest.mark.parametrize("missing_id", [None, ""])
def test_parse_response_falls_back_to_short_id_when_provider_omits_id(missing_id: str | None) -> None:
    """Provider 不给 ID 时仍生成 9 位 alnum 兜底 ID，保持旧行为。"""
    provider = LiteLLMProvider.__new__(LiteLLMProvider)

    parsed = provider._parse_response(_tool_response(([_raw_tool_call(missing_id)], "tool_calls")))

    (tool_call,) = parsed.tool_calls
    assert len(tool_call.id) == 9
    assert tool_call.id.isalnum()


def test_parse_response_keeps_ids_unique_when_choices_merge_duplicate_ids() -> None:
    """多 choice 合并遇到重复 ID 时重新生成，响应内 ID 仍然唯一。"""
    provider = LiteLLMProvider.__new__(LiteLLMProvider)
    duplicate = "call_duplicate"

    parsed = provider._parse_response(
        _tool_response(([_raw_tool_call(duplicate)], "tool_calls"), ([_raw_tool_call(duplicate)], None))
    )

    ids = [tc.id for tc in parsed.tool_calls]
    assert len(ids) == 2
    assert len(set(ids)) == 2
    assert ids[0] == duplicate
    assert len(ids[1]) == 9
    assert ids[1].isalnum()
