"""Usage-limit recognition and the step-level failover hand-off."""

from __future__ import annotations

import uuid
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
CODEX_NOTICE = (
    "You've hit your usage limit. Visit https://chatgpt.com/codex/settings/usage"
)


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
