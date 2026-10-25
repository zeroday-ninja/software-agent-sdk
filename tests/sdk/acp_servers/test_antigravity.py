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
    AGY_BINARY_ENV,
    AntigravityAgent,
    AntigravityUnavailable,
    _sign_in_detail,
    _signed_out,
    _tool_title,
    _turn_failure,
    _unavailable_detail,
    _user_message,
)
from openhands.sdk.agent.acp_agent import _classify_acp_turn_error
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
    assert connection.updates[0].field_meta == {
        "openhands.unrecognised": {
            "what": "step",
            "kind": "plan_revision",
            "adapter": "antigravity",
        }
    }


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
    # And as a field, not only a sentence. The milestone asks for "any
    # unrecognised event surfaced as a typed error rather than dropped", and
    # a line in the assistant's own message stream is something nothing can
    # count and nothing can tell from the model's words.
    assert connection.updates[0].field_meta == {
        "openhands.unrecognised": {
            "what": "event",
            "kind": "plan_update",
            "adapter": "antigravity",
        }
    }


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


class TestWhenTheCliCannotRunAtAll:
    """Two failures that are not the model answering badly.

    `FileNotFoundError: [Errno 2] No such file or directory: 'agy'` is what a
    sandbox produced, and it is true and useless: it names a binary nobody
    chose and gives no next step. "You are not logged into Antigravity"
    arrives as an ordinary turn failure and reads like the model declining.
    Both are the runtime not being reachable, and both are fixed by a person
    doing something specific.
    """

    def test_a_missing_cli_says_what_is_missing_and_how_to_point_at_it(self) -> None:
        # Arrange / Act
        detail = _unavailable_detail(
            "agy", FileNotFoundError(2, "No such file or directory")
        )

        # Assert: the binary, why a sandbox would not have it, and the way out.
        assert "not installed here" in detail
        assert "npm ACP providers" in detail
        assert AGY_BINARY_ENV in detail

    def test_a_cli_that_cannot_be_executed_is_told_apart_from_a_missing_one(
        self,
    ) -> None:
        # A binary that is present and not executable is a different morning's
        # work from one that was never installed.
        detail = _unavailable_detail("/opt/agy", PermissionError(13, "Denied"))

        assert "not executable" in detail
        assert "not installed" not in detail

    def test_the_sign_in_message_names_the_sandbox_as_the_thing_to_sign_in(
        self,
    ) -> None:
        # The login lives in the desktop keyring, which a sandbox has not got,
        # so "sign in" without saying *where* sends somebody to sign in again
        # on the machine that already is.
        detail = _sign_in_detail("agy")

        assert "keyring" in detail
        assert "sandbox" in detail
        # And in the wording the SDK's turn classifier reads as an auth
        # condition, so a client is offered the fix rather than a generic
        # prompt failure.
        assert _classify_acp_turn_error(AntigravityUnavailable(detail)) == (
            "ACPAuthRequired"
        )

    def test_a_missing_cli_is_not_reported_as_an_auth_problem(self) -> None:
        # Nothing is wrong with anybody's credentials; the tool is not there.
        # Offering re-authentication would send somebody to fix the one thing
        # that is already fine.
        detail = _unavailable_detail("agy", FileNotFoundError(2, "nope"))

        assert _classify_acp_turn_error(AntigravityUnavailable(detail)) == (
            "ACPPromptError"
        )

    def test_a_signed_out_cli_is_recognised_from_either_stream(self) -> None:
        # The CLI says it on stdout as a refusal and on stderr as a log line,
        # and which one arrives depends on how far it got.
        assert _signed_out("Error: Please sign in to view available models.")
        assert _signed_out(
            "error getting token source: You are not logged into Antigravity."
        )
        # Not everything mentioning a sign-in is one: a usage limit is a
        # different answer and must still fail over as a turn error.
        assert not _signed_out("Individual quota reached. Resets in 121h4m23s.")
        assert not _signed_out("")

    @pytest.mark.asyncio
    async def test_a_missing_cli_is_raised_rather_than_answered(
        self, agent, monkeypatch
    ) -> None:
        # Raised, not streamed: the SDK reads a turn *error* to decide whether
        # to fail over, and the same words as an answer would end the turn
        # successfully with nothing having happened.
        adapter, _connection = agent
        monkeypatch.setenv(AGY_BINARY_ENV, "/nonexistent/agy-for-this-test")
        await adapter.new_session(cwd="/tmp")

        with pytest.raises(AntigravityUnavailable) as raised:
            await adapter.prompt("antigravity-1", [{"type": "text", "text": "hello"}])

        assert "not installed here" in str(raised.value)
