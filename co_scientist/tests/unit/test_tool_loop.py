"""Tests for the tool-use loop, especially terminal-tool short-circuit.

The terminal-tool short-circuit matters for any provider whose model does NOT
reliably emit `stop_reason="end_turn"` after calling a recording tool — that
includes most OpenAI-compat models (Gemini, OpenAI o-series via tool_calls,
Llama through OpenRouter, etc.). Without the short-circuit they loop until
max_iters and ToolLoopExhausted, even though a perfectly valid record was
emitted on the first call.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from co_scientist.llm.anthropic_client import (
    AgentCallSpec,
    AnthropicResponse,
    CachedBlock,
    CallContext,
)
from co_scientist.llm.routing import ModelRoute
from co_scientist.llm.tool_loop import (
    _compact_tool_content,
    _tool_result_block,
    _trim_tool_history,
    run_tool_loop,
)


def _fake_response(*, stop_reason: str, blocks: list[dict]) -> AnthropicResponse:
    """Build an AnthropicResponse whose .raw quacks like an anthropic Message."""
    content = []
    for b in blocks:
        # Each block must expose .type, .name, .input, .text
        content.append(
            SimpleNamespace(
                type=b.get("type", "text"),
                name=b.get("name", ""),
                input=b.get("input", {}),
                text=b.get("text", ""),
                id=b.get("id", ""),
            )
        )
    raw = SimpleNamespace(stop_reason=stop_reason, content=content)
    return AnthropicResponse(
        raw=raw,
        transcript_id="trn_x",
        cost_usd=0.0,
        input_tokens=0,
        output_tokens=0,
        cache_read=0,
        cache_write=0,
    )


def _spec() -> AgentCallSpec:
    return AgentCallSpec(
        route=ModelRoute(agent="generation", mode="literature", model="x"),
        user_blocks=[CachedBlock("go")],
        tools=[{"name": "search", "description": "", "input_schema": {}}],
        max_output_tokens=512,
    )


def _ctx() -> CallContext:
    return CallContext(session_id="s", task_id="t", agent="generation", action="a")


@pytest.mark.asyncio
async def test_loop_ends_on_record_hypothesis_even_when_stop_reason_is_tool_use() -> None:
    """The bug we hit on Gemini: model emits record_hypothesis but keeps
    stop_reason=tool_use, so without short-circuit the loop runs to
    max_iters."""
    client = MagicMock()
    client.call = AsyncMock(
        side_effect=[
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "record_hypothesis",
                        "input": {"title": "t", "statement": "s"},
                    }
                ],
            ),
        ]
    )
    registry = MagicMock()

    result = await run_tool_loop(
        client,
        spec=_spec(),
        ctx=_ctx(),
        registry=registry,
        max_iters=8,
        parallel_cap=4,
        tool_timeout_s=1.0,
    )
    assert result.iterations == 1
    # The terminal tool_use is logged but never dispatched.
    assert client.call.await_count == 1
    assert result.tool_calls[0]["name"] == "record_hypothesis"


@pytest.mark.asyncio
async def test_loop_ends_normally_on_end_turn() -> None:
    client = MagicMock()
    client.call = AsyncMock(
        return_value=_fake_response(
            stop_reason="end_turn",
            blocks=[{"type": "text", "text": "all done"}],
        )
    )
    registry = MagicMock()
    result = await run_tool_loop(
        client,
        spec=_spec(),
        ctx=_ctx(),
        registry=registry,
        max_iters=8,
        parallel_cap=4,
        tool_timeout_s=1.0,
    )
    assert result.iterations == 1


@pytest.mark.asyncio
async def test_loop_dispatches_non_terminal_tools_then_continues() -> None:
    """Search tool calls should still be dispatched and the loop continues."""
    from co_scientist.tools.base import ToolResult

    client = MagicMock()
    client.call = AsyncMock(
        side_effect=[
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "call_search",
                        "name": "search",
                        "input": {"q": "foo"},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "call_record",
                        "name": "record_hypothesis",
                        "input": {"title": "t", "statement": "s"},
                    }
                ],
            ),
        ]
    )

    registry = MagicMock()
    registry._cfg = SimpleNamespace()
    registry.call = AsyncMock(
        return_value=ToolResult(
            is_error=False,
            content={"ok": True},
            duration_ms=1,
        )
    )

    result = await run_tool_loop(
        client,
        spec=_spec(),
        ctx=_ctx(),
        registry=registry,
        max_iters=8,
        parallel_cap=4,
        tool_timeout_s=1.0,
    )
    assert result.iterations == 2
    # search was dispatched; record_hypothesis was NOT (terminal)
    assert registry.call.await_count == 1
    names = [tc["name"] for tc in result.tool_calls]
    assert names == ["search", "record_hypothesis"]


@pytest.mark.asyncio
async def test_loop_terminates_on_record_review() -> None:
    client = MagicMock()
    client.call = AsyncMock(
        return_value=_fake_response(
            stop_reason="tool_use",
            blocks=[
                {
                    "type": "tool_use",
                    "id": "c",
                    "name": "record_review",
                    "input": {"verdict": "accept"},
                }
            ],
        )
    )
    result = await run_tool_loop(
        client,
        spec=_spec(),
        ctx=_ctx(),
        registry=MagicMock(),
        max_iters=8,
        parallel_cap=4,
        tool_timeout_s=1.0,
    )
    assert result.iterations == 1


@pytest.mark.asyncio
async def test_loop_terminates_on_custom_terminal_tool() -> None:
    """The terminal-tool set is configurable via kwarg."""
    client = MagicMock()
    client.call = AsyncMock(
        return_value=_fake_response(
            stop_reason="tool_use",
            blocks=[{"type": "tool_use", "id": "c", "name": "my_done_signal", "input": {}}],
        )
    )
    result = await run_tool_loop(
        client,
        spec=_spec(),
        ctx=_ctx(),
        registry=MagicMock(),
        max_iters=8,
        parallel_cap=4,
        tool_timeout_s=1.0,
        terminal_tool_names=("my_done_signal",),
    )
    assert result.iterations == 1


@pytest.mark.asyncio
async def test_force_terminal_tool_on_final_iteration() -> None:
    """A model that only ever searches should be forced to record on the last
    iteration when force_terminal_tool is set — instead of exhausting the loop."""
    from co_scientist.tools.base import ToolResult

    # The model keeps searching forever unless tool_choice forces a specific
    # tool — exactly how a real provider behaves under a forced tool_choice.
    def _respect_tool_choice(spec, *_a, **_k):
        tc = spec.tool_choice or {}
        if tc.get("type") == "tool" and tc.get("name") == "record_hypothesis":
            return _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "r",
                        "name": "record_hypothesis",
                        "input": {"title": "t", "statement": "s"},
                    }
                ],
            )
        return _fake_response(
            stop_reason="tool_use",
            blocks=[{"type": "tool_use", "id": "s", "name": "search", "input": {"q": "x"}}],
        )

    client = MagicMock()
    client.call = AsyncMock(side_effect=_respect_tool_choice)
    registry = MagicMock()
    registry._cfg = SimpleNamespace()
    registry.call = AsyncMock(
        return_value=ToolResult(
            is_error=False,
            content={"n": 0, "results": []},
            duration_ms=1,
        )
    )

    result = await run_tool_loop(
        client,
        spec=_spec(),
        ctx=_ctx(),
        registry=registry,
        max_iters=3,
        parallel_cap=4,
        tool_timeout_s=1.0,
        force_terminal_tool="record_hypothesis",
    )
    # The loop committed on the forced final iteration instead of exhausting.
    assert result.iterations == 3
    assert result.tool_calls[-1]["name"] == "record_hypothesis"
    # The final (3rd) call forced tool_choice to record_hypothesis.
    final_spec = client.call.await_args_list[-1].args[0]
    assert final_spec.tool_choice == {"type": "tool", "name": "record_hypothesis"}
    # Earlier calls used the original auto tool_choice (no forcing).
    first_spec = client.call.await_args_list[0].args[0]
    assert first_spec.tool_choice != {"type": "tool", "name": "record_hypothesis"}


@pytest.mark.asyncio
async def test_dashscope_terminal_output_uses_reason_then_extract() -> None:
    """DashScope thinking models cannot combine thinking with forced tools."""
    draft = _fake_response(
        stop_reason="tool_use",
        blocks=[
            {
                "type": "tool_use",
                "id": "draft",
                "name": "record_hypothesis",
                "input": {"title": "draft", "statement": "draft statement"},
            }
        ],
    )
    reasoning = _fake_response(
        stop_reason="end_turn",
        blocks=[{"type": "text", "text": "Deep synthesis with corrected mechanism."}],
    )
    extracted = _fake_response(
        stop_reason="tool_use",
        blocks=[
            {
                "type": "tool_use",
                "id": "final",
                "name": "record_hypothesis",
                "input": {"title": "final", "statement": "corrected statement"},
            }
        ],
    )
    client = MagicMock()
    client._base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    client.call = AsyncMock(side_effect=[draft, reasoning, extracted])
    registry = MagicMock()
    registry._cfg = SimpleNamespace()

    result = await run_tool_loop(
        client,
        spec=_spec(),
        ctx=_ctx(),
        registry=registry,
        max_iters=3,
        parallel_cap=4,
        tool_timeout_s=1.0,
        force_terminal_tool="record_hypothesis",
    )

    assert client.call.await_count == 3
    reasoning_spec = client.call.await_args_list[1].args[0]
    extraction_spec = client.call.await_args_list[2].args[0]
    assert reasoning_spec.tools == []
    assert reasoning_spec.tool_choice is None
    assert extraction_spec.tool_choice == {
        "type": "tool",
        "name": "record_hypothesis",
    }
    assert result.tool_calls[-1]["args"]["statement"] == "corrected statement"


def test_tool_result_compaction_bounds_items_and_abstracts() -> None:
    body = {"results": [{"title": f"paper-{i}", "abstract": "x" * 5000} for i in range(10)]}
    compact = _compact_tool_content(
        body,
        item_max_chars=100,
        max_items=3,
    )
    assert len(compact["results"]) == 3
    assert len(compact["results"][0]["abstract"]) < 130


def test_tool_result_is_bounded_and_quoted_as_untrusted() -> None:
    tool_use = SimpleNamespace(id="call_1")
    block = _tool_result_block(
        tool_use,
        {"content": {"results": [{"abstract": "x" * 5000}]}, "is_error": False},
        max_chars=500,
        item_max_chars=200,
        max_items=2,
    )
    assert block["content"].startswith("<UNTRUSTED_SOURCE")
    assert "tool result truncated" not in block["content"]
    assert len(block["content"]) < 800


def test_tool_history_drops_oldest_complete_pairs() -> None:
    messages = []
    for i in range(4):
        messages.extend(
            [
                {"role": "assistant", "content": "a" * 100},
                {"role": "user", "content": f"{i}" + "b" * 100},
            ]
        )
    trimmed = _trim_tool_history(messages, max_chars=500)
    assert len(trimmed) < len(messages)
    assert len(trimmed) % 2 == 0
    assert trimmed[0]["role"] == "assistant"
