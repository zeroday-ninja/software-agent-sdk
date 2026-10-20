"""Runtime failover and manual runtime switching on an ACP conversation.

Both runtimes are the stdio fake from ``tests/sdk/agent``: the primary runs as a
``custom`` server that refuses every prompt with a usage-limit error, and the
fallback is the ``claude-code`` provider resolved to a wrapper on ``PATH`` that
runs the same fake without the error.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

from openhands.sdk.agent.acp_agent import ACPAgent
from openhands.sdk.conversation.impl.local_conversation import (
    ACP_FAILOVER_FROM_KEY,
    LocalConversation,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, MessageEvent
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.llm import content_to_str


FAKE_SERVER = Path(__file__).parents[1] / "agent" / "fake_acp_models_server.py"
LIMIT_NOTICE = "You've hit your limit · resets 3pm (America/New_York)"


@pytest.fixture
def fallback_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a ``claude-agent-acp`` wrapper on PATH that runs the fake, answering."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "claude-agent-acp"
    wrapper.write_text(
        "#!/bin/sh\n"
        "unset FAKE_ACP_PROMPT_ERROR\n"
        f'FAKE_ACP_REPLY=fallback exec "{sys.executable}" "{FAKE_SERVER}" "$@"\n'
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    assert shutil.which("claude-agent-acp") == str(wrapper)
    for var in ("FAKE_ACP_MODELS", "FAKE_ACP_RESUME", "FAKE_ACP_EFFORT"):
        monkeypatch.delenv(var, raising=False)
    return wrapper


def _conversation(tmp_path: Path, **fields) -> LocalConversation:
    agent = ACPAgent(
        acp_command=[sys.executable, str(FAKE_SERVER)],
        acp_server="custom",
        acp_startup_timeout=30.0,
        acp_prompt_timeout=30.0,
        **fields,
    )
    return LocalConversation(
        agent=agent,
        workspace=str(tmp_path / "ws"),
        persistence_dir=str(tmp_path / "persist"),
    )


def _finish_texts(conv: LocalConversation) -> list[str]:
    return [
        e.action.message  # type: ignore[attr-defined]
        for e in conv.state.events
        if isinstance(e, ActionEvent) and e.tool_name == "finish"
    ]


def _agent_texts(conv: LocalConversation) -> list[str]:
    return [
        " ".join(content_to_str(e.llm_message.content))
        for e in conv.state.events
        if isinstance(e, MessageEvent) and e.source == "agent"
    ]


def _assert_failed_over(conv: LocalConversation) -> None:
    assert conv.state.execution_status == ConversationExecutionStatus.FINISHED
    assert not any(isinstance(e, ConversationErrorEvent) for e in conv.state.events)
    # The user saw why and where the conversation went...
    notices = [t for t in _agent_texts(conv) if LIMIT_NOTICE in t]
    assert len(notices) == 1
    assert "Switching to Claude Code" in notices[0]
    # ...and the fallback answered the same prompt with the history in hand.
    assert _finish_texts(conv) == ["fallback (resumed)"]
    agent = conv.agent
    assert isinstance(agent, ACPAgent)
    assert agent.acp_server == "claude-code"
    assert agent.acp_fallback_server is None
    assert conv.state.agent_state[ACP_FAILOVER_FROM_KEY] == "custom"
    assert conv.state.agent_state["acp_agent_name"] == "fake-acp"


def test_run_fails_over_to_the_fallback_runtime(
    tmp_path: Path, fallback_on_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ACP_PROMPT_ERROR", LIMIT_NOTICE)
    conv = _conversation(tmp_path, acp_fallback_server="claude-code")
    try:
        conv.send_message("hello")
        conv.run()
        _assert_failed_over(conv)
    finally:
        conv.close()


def test_arun_fails_over_to_the_fallback_runtime(
    tmp_path: Path, fallback_on_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ACP_PROMPT_ERROR", LIMIT_NOTICE)
    conv = _conversation(tmp_path, acp_fallback_server="claude-code")
    try:
        conv.send_message("hello")
        asyncio.run(conv.arun())
        _assert_failed_over(conv)
    finally:
        conv.close()


def test_without_a_fallback_the_limit_ends_the_run_in_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ACP_PROMPT_ERROR", LIMIT_NOTICE)
    conv = _conversation(tmp_path)
    try:
        conv.send_message("hello")
        with pytest.raises(Exception):
            conv.run()
        assert conv.state.execution_status == ConversationExecutionStatus.ERROR
        errors = [e for e in conv.state.events if isinstance(e, ConversationErrorEvent)]
        assert [e.code for e in errors] == ["ACPUsageLimit"]
    finally:
        conv.close()


def test_switch_acp_server_replays_history_to_the_new_runtime(
    tmp_path: Path, fallback_on_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FAKE_ACP_PROMPT_ERROR", raising=False)
    conv = _conversation(tmp_path)
    try:
        conv.send_message("first")
        conv.run()
        assert _finish_texts(conv) == ["ok (fresh)"]

        conv.switch_acp_server("claude-code", "opus/high")

        agent = conv.agent
        assert isinstance(agent, ACPAgent)
        assert agent.acp_server == "claude-code"
        assert agent.acp_model == "opus/high"
        assert agent.llm.model == "opus/high"
        assert ACP_FAILOVER_FROM_KEY not in conv.state.agent_state
        # A fresh session on the new runtime, not a resume of the old one.
        assert agent._resumed_existing_session is False

        conv.send_message("second")
        conv.run()
        assert _finish_texts(conv) == ["ok (fresh)", "fallback (resumed)"]
        # The transcript rides only the first prompt after the switch.
        conv.send_message("third")
        conv.run()
        assert _finish_texts(conv)[-1] == "fallback (fresh)"
    finally:
        conv.close()


def test_switch_acp_server_rejects_unknown_and_non_acp_targets(tmp_path: Path) -> None:
    conv = _conversation(tmp_path)
    try:
        with pytest.raises(ValueError, match="Unsupported ACP server"):
            conv.switch_acp_server("no-such-runtime")
    finally:
        conv.close()
