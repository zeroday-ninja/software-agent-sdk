"""The Antigravity adapter, tested against recordings of the real CLI.

The stream format is undocumented and was learned by running the tool, so
these fixtures are transcripts of actual turns rather than an imagined shape.
A format change therefore shows up here as a failing test rather than as an
agent that silently says nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from openhands.sdk.acp_servers.antigravity import (
    AntigravityAgent,
    _tool_title,
    _turn_failure,
    _user_message,
)
from openhands.sdk.agent.acp_usage_limit import (
    parse_reset_hint,
    text_indicates_usage_limit,
)


# Recorded from `agy --print="" --input-format=stream-json
# --output-format=stream-json --model gemini-3.8-flash-low`, 2026-10-24.
RECORDED_TURN = [
    {
        "event": "init",
        "conversation_id": "1fb0968b",
        "init": {"model": "gemini-3.8-flash-low", "cwd": "/tmp", "tools": []},
    },
    {
        "event": "step_update",
        "step_update": {
            "step_index": 0,
            "state": "DONE",
            "step_type": "user_input",
        },
    },
    {
        "event": "step_update",
        "step_update": {
            "step_index": 1,
            "state": "ACTIVE",
            "step_type": "agent_response",
            "text_delta": "ok",
        },
    },
    {
        "event": "step_update",
        "step_update": {
            "step_index": 2,
            "state": "ACTIVE",
            "step_type": "tool",
            "tool_name": "write_file",
            "tool_info": {
                "name": "write_file",
                "parameters": {"Path": "/tmp/probe.txt"},
            },
        },
    },
    {
        "event": "step_update",
        "step_update": {
            "step_index": 2,
            "state": "DONE",
            "step_type": "tool",
            "tool_name": "write_file",
            "tool_info": {
                "name": "write_file",
                "parameters": {"Path": "/tmp/probe.txt"},
            },
        },
    },
    {
        "event": "result",
        "result": {
            "status": "SUCCESS",
            "response": "ok\n",
            "duration_seconds": 2.2,
            "num_turns": 1,
            "usage": {"input_tokens": 6768, "output_tokens": 1},
        },
    },
]


# Recorded from the same flags against an account whose weekly quota was
# spent, 2026-10-24. The `error_message` step carries no text; the words are
# on the result, and the status code is only ever on stderr.
RECORDED_REFUSAL = [
    {
        "event": "step_update",
        "step_update": {"step_index": 0, "state": "DONE", "step_type": "user_input"},
    },
    {
        "event": "step_update",
        "step_update": {"step_index": 1, "state": "DONE", "step_type": "error_message"},
    },
    {
        "event": "result",
        "result": {
            "conversation_id": "879ba243",
            "status": "ERROR",
            "response": "",
            "error": (
                "Individual quota reached. Please upgrade your subscription to "
                "increase your limits. Resets in 132h6m57s."
            ),
            "num_turns": 1,
        },
    },
]

RECORDED_STDERR = (
    "error: Individual quota reached. Please upgrade your subscription to "
    "increase your limits. Resets in 132h6m57s.\n"
    'AGY_ERROR: {"short_error":"RESOURCE_EXHAUSTED (code 429): Individual '
    "quota reached. Please upgrade your subscription to increase your limits. "
    'Resets in 132h6m57s.","status":"RESOURCE_EXHAUSTED","error_code":429,'
    '"code_kind":"http","retryable":true,'
    '"error_id":"799feefc-c28c-464b-8260-9da757494384-1"}'
)


class RecordingConnection:
    """Collects what the adapter would have sent to a client."""

    def __init__(self) -> None:
        self.updates: list[Any] = []

    async def session_update(self, session_id: str, update: Any) -> None:
        self.updates.append(update)


@pytest.fixture
def agent() -> tuple[AntigravityAgent, RecordingConnection]:
    adapter = AntigravityAgent()
    connection = RecordingConnection()
    adapter.on_connect(connection)
    return adapter, connection


async def _replay(
    adapter: AntigravityAgent, events: list[dict], session_id: str = "s1"
) -> tuple[str | None, str | None]:
    session = adapter._sessions[session_id]  # noqa: SLF001 - the unit under test
    stop: str | None = None
    failure: str | None = None
    for event in events:
        reason, reported = await adapter._handle(session, event)  # noqa: SLF001
        if reason is not None:
            stop = reason
        if reported is not None:
            failure = reported
    return stop, failure


@pytest.mark.asyncio
async def test_a_recorded_turn_becomes_text_and_one_tool_call(agent) -> None:
    # Arrange
    adapter, connection = agent
    await adapter.new_session(cwd="/tmp")
    adapter._sessions["s1"] = adapter._sessions.pop("antigravity-1")  # noqa: SLF001
    adapter._sessions["s1"].session_id = "s1"  # noqa: SLF001

    # Act
    stop, failure = await _replay(adapter, RECORDED_TURN)

    # Assert: the answer arrives as text, and the file write is announced once
    # and then completed — not twice, which would read as two writes.
    kinds = [type(update).__name__ for update in connection.updates]
    assert "AgentMessageChunk" in kinds
    assert kinds.count("ToolCallStart") == 1
    assert kinds.count("ToolCallProgress") == 1
    assert stop == "end_turn"
    assert failure is None


@pytest.mark.asyncio
async def test_a_refusal_is_reported_for_raising_not_streamed_as_an_answer(
    agent,
) -> None:
    # Arrange
    adapter, connection = agent
    await adapter.new_session(cwd="/tmp")
    adapter._sessions["s1"] = adapter._sessions.pop("antigravity-1")  # noqa: SLF001
    adapter._sessions["s1"].session_id = "s1"  # noqa: SLF001

    # Act
    stop, failure = await _replay(adapter, RECORDED_REFUSAL)

    # Assert: the words come back to be raised, and nothing was streamed as
    # agent text. A usage limit delivered as an answer finishes the turn
    # successfully, and the conversation never fails over.
    assert stop == "refusal"
    assert failure is not None and "quota reached" in failure
    assert connection.updates == []


def test_a_refusal_reads_as_a_usage_limit_with_its_reset() -> None:
    # Act: what `prompt` raises, assembled from both streams.
    detail = _turn_failure(RECORDED_REFUSAL[-1]["result"]["error"], RECORDED_STDERR)

    # Assert: the two facts the scheduler needs. The status code only ever
    # appears on stderr, and the reset only in the sentence.
    assert text_indicates_usage_limit(detail)
    hint = parse_reset_hint(detail, now=datetime(2026, 10, 24, 8, 0, tzinfo=UTC))
    assert hint.at == datetime(2026, 10, 24, 8, 0, tzinfo=UTC) + timedelta(
        hours=132, minutes=6, seconds=57
    )
    assert hint.confidence == "exact"


@pytest.mark.asyncio
async def test_a_step_type_this_build_has_never_seen_is_reported(agent) -> None:
    # Arrange
    adapter, connection = agent
    await adapter.new_session(cwd="/tmp")
    adapter._sessions["s1"] = adapter._sessions.pop("antigravity-1")  # noqa: SLF001
    adapter._sessions["s1"].session_id = "s1"  # noqa: SLF001

    # Act
    await _replay(
        adapter,
        [{"event": "step_update", "step_update": {"step_type": "plan_revision"}}],
    )

    # Assert: a step type that grew is visible. `user_input` and
    # `error_message` stay silent on purpose -- the first is our own prompt
    # echoed back, the second is the prose half of the result event.
    assert "plan_revision" in connection.updates[0].content.text


@pytest.mark.asyncio
async def test_an_event_this_build_has_never_seen_is_reported(agent) -> None:
    # Arrange
    adapter, connection = agent
    await adapter.new_session(cwd="/tmp")
    adapter._sessions["s1"] = adapter._sessions.pop("antigravity-1")  # noqa: SLF001
    adapter._sessions["s1"].session_id = "s1"  # noqa: SLF001

    # Act: a format that grew a new event kind.
    await _replay(adapter, [{"event": "plan_update", "plan_update": {}}])

    # Assert: a format that grew should look like a format that grew, not like
    # a runtime that said nothing.
    assert "plan_update" in connection.updates[0].content.text


def test_a_prompt_becomes_the_input_message_the_cli_accepts() -> None:
    # Act
    message = _user_message([{"type": "text", "text": "hello"}])

    # Assert: the event name is the one the CLI accepts — every other
    # spelling is ignored with a warning nobody reads.
    assert message["event"] == "user"
    assert message["message"]["content"] == [{"type": "text", "text": "hello"}]


def test_content_the_stream_cannot_carry_is_named_rather_than_dropped() -> None:
    # Act
    message = _user_message([{"type": "image", "data": "…"}])

    # Assert
    text = message["message"]["content"][0]["text"]
    assert "image" in text and "omitted" in text


def test_a_tool_call_reads_as_one_line() -> None:
    assert _tool_title("write_file", {"parameters": {"Path": "/tmp/x"}}) == (
        "write_file: /tmp/x"
    )
    assert _tool_title("run_command", {}) == "run_command"
