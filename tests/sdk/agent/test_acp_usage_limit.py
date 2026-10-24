"""Usage-limit recognition, reset-time parsing, and the failover hand-off."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from acp.exceptions import RequestError as ACPRequestError

from openhands.sdk.agent.acp_agent import (
    ACPAgent,
    _classify_acp_turn_error,
    _OpenHandsACPBridge,
)
from openhands.sdk.agent.acp_usage_limit import (
    ACPUsageLimitError,
    parse_reset_hint,
    response_indicates_usage_limit,
    text_indicates_usage_limit,
)
from openhands.sdk.conversation.state import (
    ConversationExecutionStatus,
    ConversationState,
)
from openhands.sdk.event import MessageEvent
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.event.error_classification import FailureKind, classify_error
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.workspace import LocalWorkspace


CLAUDE_NOTICE = "You've hit your limit · resets 3pm (America/New_York)"
CLAUDE_SESSION_NOTICE = (
    "You've hit your session limit · resets 1:30pm (America/New_York)"
)
CODEX_NOTICE = (
    "You've hit your usage limit. Visit https://chatgpt.com/codex/settings/usage"
)
# 08:00 in America/New_York, so both notices above still reset later today.
#: Captured verbatim from `agy --print` on an exhausted account. Antigravity
#: states the reset as a duration, which is the only machine-readable reset any
#: of these runtimes gives.
ANTIGRAVITY_NOTICE = (
    "RESOURCE_EXHAUSTED (code 429): Individual quota reached. Please upgrade "
    "your subscription to increase your limits. Resets in 132h9m47s."
)

MORNING = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "text",
    [
        CLAUDE_NOTICE,
        "You're out of extra usage",
        "Fable 5 requires usage credits",
        CODEX_NOTICE,
        "Internal error: {'codexErrorInfo': 'usageLimitExceeded'}",
        "API error 429: rate_limit_error",
    ],
)
def test_text_indicates_usage_limit(text: str) -> None:
    assert text_indicates_usage_limit(text)


@pytest.mark.parametrize(
    "text", ["401 Unauthorized", "disk full", "", "Model not found"]
)
def test_text_without_a_limit_marker(text: str) -> None:
    assert not text_indicates_usage_limit(text)


def test_response_notice_must_be_short_and_lead_with_the_marker() -> None:
    assert response_indicates_usage_limit(CLAUDE_NOTICE)
    assert not response_indicates_usage_limit(
        "Sure. Note that you've hit your limit of three retries in the code."
    )
    assert not response_indicates_usage_limit(CLAUDE_NOTICE + " x" * 400)


@pytest.mark.parametrize(
    ("notice", "expected", "fragment"),
    [
        (
            CLAUDE_NOTICE,
            datetime(2026, 10, 24, 19, 0, tzinfo=UTC),
            "resets 3pm (America/New_York)",
        ),
        (
            CLAUDE_SESSION_NOTICE,
            datetime(2026, 10, 24, 17, 30, tzinfo=UTC),
            "resets 1:30pm (America/New_York)",
        ),
    ],
)
def test_a_zoned_reset_time_becomes_an_absolute_utc_instant(
    notice: str, expected: datetime, fragment: str
) -> None:
    hint = parse_reset_hint(notice, now=MORNING)

    assert hint.at == expected
    assert hint.confidence == "exact"
    assert hint.source == fragment


def test_a_reset_time_already_past_means_the_next_days_occurrence() -> None:
    # 16:00 in America/New_York: today's 3pm has been and gone.
    afternoon = datetime(2026, 10, 24, 20, 0, tzinfo=UTC)

    hint = parse_reset_hint(CLAUDE_NOTICE, now=afternoon)

    assert hint.at == datetime(2026, 10, 25, 19, 0, tzinfo=UTC)


def test_a_reset_time_without_a_zone_is_read_locally_and_says_so() -> None:
    hint = parse_reset_hint("You've hit your limit · resets 3pm", now=MORNING)

    assert hint.confidence == "approximate"
    assert hint.at is not None
    # Whatever this host's zone is, the next local 3pm is within a day.
    assert MORNING < hint.at <= MORNING + timedelta(days=1)
    assert hint.source == "resets 3pm"


@pytest.mark.parametrize(
    ("notice", "ahead"),
    [
        # The compact form the CLI actually prints: no separator between a
        # unit and the next number, which is where a word-boundary rule fails.
        (ANTIGRAVITY_NOTICE, timedelta(hours=132, minutes=9, seconds=47)),
        ("Rate limit reached; resets in 45m", timedelta(minutes=45)),
        (
            "Quota exceeded, resets in 2 hours 30 minutes",
            timedelta(hours=2, minutes=30),
        ),
    ],
)
def test_a_reset_stated_as_a_duration_is_exact_from_when_it_was_read(
    notice: str, ahead: timedelta
) -> None:
    hint = parse_reset_hint(notice, now=MORNING)

    # Exact with no zone involved: a duration is anchored to the instant it
    # was read, so there is nothing to be wrong about.
    assert hint.at == MORNING + ahead
    assert hint.confidence == "exact"


@pytest.mark.parametrize(
    "text",
    [
        CODEX_NOTICE,
        # A duration no usage window could plausibly have: acting on it would
        # park a runtime for weeks on a misparse.
        "Quota exceeded, resets in 90 days",
        # "m" followed by a letter is a month, not a minute.
        "Quota exceeded, resets in 3 months",
        "Internal error: {'codexErrorInfo': 'usageLimitExceeded'}",
        "The retry counter resets 3 times an hour",
        "You've hit your limit · resets 25pm (America/New_York)",
        None,
    ],
)
def test_an_unreadable_reset_time_is_a_hint_with_no_instant(text: str | None) -> None:
    hint = parse_reset_hint(text, now=MORNING)

    assert hint.at is None
    assert hint.confidence == "unknown"


def test_turn_error_classification_and_failure_kind() -> None:
    exc = ACPRequestError(-32603, f"Internal error: {CLAUDE_NOTICE}")
    assert _classify_acp_turn_error(exc) == "ACPUsageLimit"
    assert classify_error("ACPUsageLimit", CLAUDE_NOTICE).kind == FailureKind.QUOTA
    # A limit notice must not be mistaken for the auth or policy lanes.
    assert _classify_acp_turn_error(ACPRequestError(-32000, "auth")) != "ACPUsageLimit"


def _live_agent(tmp_path, **fields) -> tuple[ACPAgent, MagicMock, ConversationState]:
    """An ACP agent with a faked live session whose prompt fails."""
    agent = ACPAgent(acp_command=["echo", "test"], acp_server="claude-code", **fields)
    state = ConversationState.create(
        id=uuid.uuid4(),
        agent=agent,
        workspace=LocalWorkspace(working_dir=str(tmp_path)),
    )
    state.events.append(
        MessageEvent(
            source="user",
            llm_message=Message(role="user", content=[TextContent(text="hi")]),
        )
    )
    agent._client = _OpenHandsACPBridge()
    agent._conn = MagicMock()
    agent._session_id = "sess"
    executor = MagicMock()
    executor.run_async = MagicMock(
        side_effect=ACPRequestError(-32603, f"Internal error: {CLAUDE_NOTICE}")
    )
    agent._executor = executor
    conversation = MagicMock()
    conversation.state = state
    return agent, conversation, state


def test_step_hands_the_limit_to_the_conversation_when_a_fallback_exists(
    tmp_path,
) -> None:
    agent, conversation, state = _live_agent(tmp_path, acp_fallback_server="codex")
    events: list = []

    with patch("openhands.sdk.agent.acp_agent.time.sleep") as sleep:
        with pytest.raises(ACPUsageLimitError) as excinfo:
            agent.step(conversation, on_event=events.append)

    # No retry back-off, no error events, and the run is still live: the
    # conversation decides what happens next.
    sleep.assert_not_called()
    assert CLAUDE_NOTICE in excinfo.value.detail
    assert excinfo.value.acp_server == "claude-code"
    assert not any(isinstance(e, ConversationErrorEvent) for e in events)
    assert state.execution_status != ConversationExecutionStatus.ERROR


def test_step_without_a_fallback_fails_fast_as_a_usage_limit_error(
    tmp_path,
) -> None:
    agent, conversation, state = _live_agent(tmp_path)
    events: list = []

    with patch("openhands.sdk.agent.acp_agent.time.sleep") as sleep:
        with pytest.raises(ACPRequestError):
            agent.step(conversation, on_event=events.append)

    sleep.assert_not_called()
    errors = [e for e in events if isinstance(e, ConversationErrorEvent)]
    assert [e.code for e in errors] == ["ACPUsageLimit"]
    assert CLAUDE_NOTICE in errors[0].detail
    assert state.execution_status == ConversationExecutionStatus.ERROR


def test_a_fallback_equal_to_the_runtime_itself_is_ignored(tmp_path) -> None:
    agent, conversation, _state = _live_agent(
        tmp_path, acp_fallback_server="claude-code"
    )
    with patch("openhands.sdk.agent.acp_agent.time.sleep"):
        with pytest.raises(ACPRequestError):
            agent.step(conversation, on_event=lambda _e: None)
