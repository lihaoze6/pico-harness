"""Streaming tests for `LiteLLMProvider.chat_stream`.

Covers:
- happy-path: chat_stream yields StreamDelta sequence matching mock chunks
- _normalize_stream_chunk default OpenAI shape extraction
- None-content chunks (e.g. final stop chunk) are skipped (return None → no yield)
- signature parity with chat() (messages/tools/model/max_tokens/temperature/
  reasoning_effort/tool_choice all accepted; stream=True forwarded to acompletion)

Mocks patch `pico.providers.litellm_provider.acompletion` because the
provider module imports `from litellm import acompletion` at top level, so
patching `litellm.acompletion` after import would not be picked up.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from litellm.types.utils import Usage

from pico.call_efficiency import CallEfficiency
from pico.providers.base import LLMResponse, StreamDelta
from pico.providers.litellm_provider import LiteLLMProvider


@dataclass
class _FakeDelta:
    content: str | None = None
    tool_calls: list[Any] | None = None


@dataclass
class _FakeChoice:
    delta: _FakeDelta
    finish_reason: str | None = None
    index: int = 0


@dataclass
class _FakeChunk:
    choices: list[_FakeChoice]
    usage: Any | None = None
    model: str | None = None


def _chunk(content: str | None) -> _FakeChunk:
    return _FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(content=content))])


async def _fake_stream(chunks: list[_FakeChunk]):
    """Async generator standing in for litellm's streamed response."""
    for ch in chunks:
        yield ch


def _make_provider() -> LiteLLMProvider:

    return LiteLLMProvider(api_key="test-key", default_model="openai/gpt-4o")


@pytest.mark.asyncio
async def test_chat_stream_yields_stream_deltas_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """chat_stream yields StreamDelta sequence matching mock OpenAI-shape chunks."""
    chunks = [_chunk("Hel"), _chunk("lo"), _chunk(" world")]

    captured_kwargs: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any):
        captured_kwargs.update(kwargs)
        return _fake_stream(chunks)

    monkeypatch.setattr(
        "pico.providers.litellm_provider.acompletion",
        fake_acompletion,
    )

    provider = _make_provider()
    out: list[StreamDelta] = []
    async for delta in provider.chat_stream(
        messages=[{"role": "user", "content": "hi"}],
        model="openai/gpt-4o",
    ):
        out.append(delta)

    assert [d.content for d in out] == ["Hel", "lo", " world"]
    assert all(isinstance(d, StreamDelta) for d in out)

    assert captured_kwargs.get("stream") is True

    assert captured_kwargs.get("stream_options") == {"include_usage": True}


@pytest.mark.asyncio
async def test_chat_stream_default_payload_has_no_provider_cache_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any):
        captured.update(kwargs)
        return _fake_stream([_chunk("ok")])

    monkeypatch.setattr("pico.providers.litellm_provider.acompletion", fake_acompletion)
    provider = LiteLLMProvider(api_key="test", default_model="anthropic/claude-sonnet-4-5")
    messages = [{"role": "system", "content": "stable"}]
    tools = [{"type": "function", "function": {"name": "read"}}]

    _ = [delta async for delta in provider.chat_stream(messages=messages, tools=tools)]

    assert "cache_control" not in str({"messages": captured["messages"], "tools": captured["tools"]})


def test_normalize_stream_chunk_openai_shape_default() -> None:
    """_normalize_stream_chunk default path extracts OpenAI-shape content."""
    provider = _make_provider()
    chunk = _chunk("token")
    delta = provider._normalize_stream_chunk(chunk)
    assert delta is not None
    assert delta.content == "token"
    assert delta.tool_call_delta is None
    assert delta.usage is None


def test_normalize_stream_chunk_returns_none_for_empty_payload() -> None:
    """Chunks with no content/tool_calls/usage return None — chat_stream skips them."""
    provider = _make_provider()

    chunk = _FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(content=None), finish_reason="stop")])
    assert provider._normalize_stream_chunk(chunk) is None


def test_normalize_stream_chunk_preserves_usage_only_terminal_chunk() -> None:
    provider = _make_provider()
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    chunk = _FakeChunk(choices=[], usage=usage, model="openai/gpt-4o")

    delta = provider._normalize_stream_chunk(chunk)

    assert delta is not None
    assert delta.content is None
    assert delta.model == "openai/gpt-4o"
    assert delta.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
    }


@pytest.mark.parametrize(
    ("model", "usage", "expected_fresh", "expected_read", "expected_write"),
    [
        (
            "deepseek/deepseek-v4-flash",
            Usage(
                prompt_tokens=100,
                completion_tokens=5,
                total_tokens=105,
                prompt_cache_hit_tokens=80,
                prompt_cache_miss_tokens=20,
            ),
            20,
            80,
            0,
        ),
        (
            "openai/gpt-4o",
            Usage(
                prompt_tokens=100,
                completion_tokens=5,
                total_tokens=105,
                prompt_tokens_details={"cached_tokens": 80},
            ),
            20,
            80,
            0,
        ),
        (
            "anthropic/claude-sonnet-4-5",
            Usage(
                prompt_tokens=20,
                completion_tokens=5,
                total_tokens=115,
                cache_read_input_tokens=80,
                cache_creation_input_tokens=10,
            ),
            20,
            80,
            10,
        ),
    ],
)
def test_stream_terminal_usage_uses_canonical_cache_fields(
    tmp_path,
    model: str,
    usage: Usage,
    expected_fresh: int,
    expected_read: int,
    expected_write: int,
) -> None:
    provider = _make_provider()
    delta = provider._normalize_stream_chunk(_FakeChunk(choices=[], usage=usage, model=model))

    assert delta is not None
    controller = CallEfficiency(mode="observe", telemetry_dir=tmp_path, persist=False)
    record = controller.record(
        LLMResponse(content="ok", usage=delta.usage or {}, model=model),
        requested_model=model,
        session_key="session-1",
    )

    assert record.usage.input_tokens == expected_fresh
    assert record.usage.cache_read_tokens == expected_read
    assert record.usage.cache_write_tokens == expected_write
    assert record.usage.complete is True


@pytest.mark.asyncio
async def test_chat_stream_skips_none_content_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mixed sequence with a None-content chunk: normalizer returns None → no yield."""
    chunks = [
        _chunk("a"),
        _FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(content=None), finish_reason=None)]),
        _chunk("b"),
    ]

    async def fake_acompletion(**_kwargs: Any):
        return _fake_stream(chunks)

    monkeypatch.setattr(
        "pico.providers.litellm_provider.acompletion",
        fake_acompletion,
    )

    provider = _make_provider()
    out = [d async for d in provider.chat_stream(messages=[{"role": "user", "content": "hi"}])]

    assert [d.content for d in out] == ["a", "b"]


@pytest.mark.asyncio
async def test_chat_stream_signature_parity_with_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    """chat_stream accepts every chat() kwarg without raising.

    Smoke check: pass the full chat() parameter set and verify kwargs hit
    acompletion (stream=True, model/messages/tools/tool_choice present;
    reasoning_effort forwarded; max_tokens/temperature forwarded).
    """
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any):
        captured.update(kwargs)
        return _fake_stream([_chunk("ok")])

    monkeypatch.setattr(
        "pico.providers.litellm_provider.acompletion",
        fake_acompletion,
    )

    provider = _make_provider()
    tools = [{"type": "function", "function": {"name": "noop", "parameters": {}}}]
    out: list[StreamDelta] = []
    async for delta in provider.chat_stream(
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        model="openai/gpt-4o-mini",
        max_tokens=128,
        temperature=0.3,
        reasoning_effort="medium",
        tool_choice="auto",
    ):
        out.append(delta)

    assert [d.content for d in out] == ["ok"]
    assert captured["stream"] is True
    assert captured["max_tokens"] == 128
    assert captured["temperature"] == 0.3
    assert captured["reasoning_effort"] == "medium"
    assert captured["tool_choice"] == "auto"
    assert captured["tools"] == tools

    assert "gpt-4o-mini" in captured["model"]


@pytest.mark.asyncio
async def test_deepseek_replays_reasoning_field_for_assistant_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any):
        captured.update(kwargs)
        return _fake_stream([_chunk("ok")])

    monkeypatch.setattr(
        "pico.providers.litellm_provider.acompletion",
        fake_acompletion,
    )

    provider = LiteLLMProvider(api_key="test-key", default_model="deepseek/deepseek-v4-flash")
    messages = [
        {"role": "user", "content": "inspect the repository"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call12345",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "contents", "tool_call_id": "call12345"},
    ]

    _ = [delta async for delta in provider.chat_stream(messages=messages)]

    assert captured["messages"][1]["reasoning_content"] == ""


@pytest.mark.asyncio
async def test_openai_does_not_receive_deepseek_reasoning_replay_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any):
        captured.update(kwargs)
        return _fake_stream([_chunk("ok")])

    monkeypatch.setattr(
        "pico.providers.litellm_provider.acompletion",
        fake_acompletion,
    )

    provider = LiteLLMProvider(api_key="test-key", default_model="openai/gpt-4o")
    messages = [
        {"role": "user", "content": "inspect the repository"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call12345",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
    ]

    _ = [delta async for delta in provider.chat_stream(messages=messages)]

    assert "reasoning_content" not in captured["messages"][1]


@pytest.mark.asyncio
async def test_custom_deepseek_preserves_original_tool_call_id_for_thinking_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom OpenAI-compatible thinking gateways associate thinking by tool id.

    The gateway rejects a replayed assistant tool call when the original provider
    id is shortened, so this path must keep the exact id on both the assistant
    tool call and the matching tool result.
    """
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any):
        captured.update(kwargs)
        return _fake_stream([_chunk("ok")])

    monkeypatch.setattr(
        "pico.providers.litellm_provider.acompletion",
        fake_acompletion,
    )

    provider = LiteLLMProvider(
        api_key="test-key",
        api_base="http://127.0.0.1:9/v1",
        default_model="deepseek-v4-pro",
        provider_name="custom",
    )
    long_id = "call_00_ofc2MSK9qptdZfbIIIKK7328"
    messages = [
        {"role": "user", "content": "inspect the repository"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should inspect the repository.",
            "tool_calls": [
                {
                    "id": long_id,
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "contents", "tool_call_id": long_id},
    ]

    _ = [delta async for delta in provider.chat_stream(messages=messages)]

    assert captured["messages"][1]["tool_calls"][0]["id"] == long_id
    assert captured["messages"][2]["tool_call_id"] == long_id
    assert captured["messages"][1]["reasoning_content"] == "I should inspect the repository."


# --- R1: 发送期按 Provider 能力决定是否保留原始 tool_call_id ----------------------

_LONG_TOOL_ID = "call_00_ofc2MSK9qptdZfbIIIKK7328"
_CUSTOM_GATEWAY = {
    "api_key": "test-key",
    "api_base": "http://127.0.0.1:9/v1",
    "provider_name": "custom",
}


def _replay_messages() -> list[dict[str, Any]]:
    """模拟 AgentLoop 回放：assistant tool_calls[].id 与 tool.tool_call_id 同源。"""
    return [
        {"role": "user", "content": "inspect the repository"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should inspect the repository.",
            "tool_calls": [
                {
                    "id": _LONG_TOOL_ID,
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "contents", "tool_call_id": _LONG_TOOL_ID},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("default_model", "provider_kwargs", "expect_preserved"),
    [
        # Gateway 级能力：gateway 声明 preserve_tool_call_ids 时所有模型都保留原始 ID。
        ("deepseek-v4-pro", _CUSTOM_GATEWAY, True),
        # R2 回归：显式 `openai/` 前缀先命中 Standard Spec，Gateway 能力仍必须生效。
        ("openai/deepseek-v4-pro", _CUSTOM_GATEWAY, True),
        ("glm-5.3-flash", _CUSTOM_GATEWAY, True),
        # 非网关直连 Provider 行为不变：仍然确定性缩短，两侧同步。
        ("deepseek/deepseek-v4-flash", {"api_key": "test-key"}, False),
    ],
)
async def test_tool_call_id_replay_follows_provider_capability(
    monkeypatch: pytest.MonkeyPatch,
    default_model: str,
    provider_kwargs: dict[str, Any],
    expect_preserved: bool,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any):
        captured.update(kwargs)
        return _fake_stream([_chunk("ok")])

    monkeypatch.setattr("pico.providers.litellm_provider.acompletion", fake_acompletion)

    provider = LiteLLMProvider(default_model=default_model, **provider_kwargs)
    _ = [delta async for delta in provider.chat_stream(messages=_replay_messages())]

    assistant_id = captured["messages"][1]["tool_calls"][0]["id"]
    tool_id = captured["messages"][2]["tool_call_id"]

    # assistant 侧与 tool 侧必须始终一致，否则 Provider 会拒绝断裂的关联。
    assert assistant_id == tool_id
    if expect_preserved:
        assert assistant_id == _LONG_TOOL_ID
    else:
        assert assistant_id != _LONG_TOOL_ID
        assert assistant_id == hashlib.sha1(_LONG_TOOL_ID.encode()).hexdigest()[:9]


@pytest.mark.asyncio
async def test_chat_non_streaming_replays_provider_tool_call_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """非流式端到端（gateway/cron/subagent 路径）：解析保留 ID，回放不再被改写。

    第一轮 chat() 拿到 tool_call 后按 AgentLoop 方式回放，捕获的 messages 中 assistant
    tool_calls[].id 与对应 tool 消息的 tool_call_id 必须都是 Provider 原始长 ID。
    """
    captured: dict[str, Any] = {}
    response = SimpleNamespace(
        model="deepseek-v4-pro",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id=_LONG_TOOL_ID,
                            function=SimpleNamespace(name="read_file", arguments="{}"),
                        )
                    ],
                    reasoning_content=None,
                    thinking_blocks=None,
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=None,
    )

    async def fake_acompletion(**kwargs: Any):
        captured.update(kwargs)
        return response

    monkeypatch.setattr("pico.providers.litellm_provider.acompletion", fake_acompletion)

    provider = LiteLLMProvider(default_model="deepseek-v4-pro", **_CUSTOM_GATEWAY)
    first = await provider.chat(
        messages=[{"role": "user", "content": "inspect the repository"}],
        tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
    )

    assert [tc.id for tc in first.tool_calls] == [_LONG_TOOL_ID]

    replay = [
        {"role": "user", "content": "inspect the repository"},
        {
            "role": "assistant",
            "content": first.content or "",
            "reasoning_content": first.reasoning_content,
            "tool_calls": [tc.to_openai_tool_call() for tc in first.tool_calls],
        },
        {"role": "tool", "content": "contents", "tool_call_id": first.tool_calls[0].id},
    ]
    await provider.chat(messages=replay)

    assert captured["messages"][1]["tool_calls"][0]["id"] == _LONG_TOOL_ID
    assert captured["messages"][2]["tool_call_id"] == _LONG_TOOL_ID
