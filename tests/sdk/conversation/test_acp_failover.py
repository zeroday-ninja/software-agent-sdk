"""Runtime failover and manual runtime switching on an ACP conversation.

Every runtime here is the stdio fake from ``tests/sdk/agent``. A built-in
provider is steered onto it the way the agent-server image steers one onto a
pinned CLI: a wrapper on ``PATH`` under the provider's ``binary_name``, which
``resolve_acp_command`` prefers over the ``npx`` default. ``FAKE_ACP_PROMPT_ERROR``
in the environment makes a prompt fail with a usage-limit error, and a wrapper
that unsets it is a runtime that answers.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from openhands.sdk.agent.acp_agent import ACPAgent
from openhands.sdk.conversation.impl.local_conversation import (
    ACP_FAILOVER_FROM_KEY,
    ACP_FAILOVER_TRIED_KEY,
    LocalConversation,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ACPRuntimeSwitchEvent, ActionEvent, MessageEvent
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.event.error_classification import FailureKind
from openhands.sdk.llm import content_to_str


FAKE_SERVER = Path(__file__).parents[1] / "agent" / "fake_acp_models_server.py"
LIMIT_NOTICE = "You've hit your limit · resets 3pm (America/New_York)"


InstallCLI = Callable[..., Path]


@pytest.fixture
def acp_cli_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> InstallCLI:
    """Return an installer for fake provider CLIs on ``PATH``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    for var in ("FAKE_ACP_MODELS", "FAKE_ACP_RESUME", "FAKE_ACP_EFFORT"):
        monkeypatch.delenv(var, raising=False)

    def install(binary: str, *, reply: str = "ok", refuses: bool = False) -> Path:
        wrapper = bin_dir / binary
        wrapper.write_text(
            "#!/bin/sh\n"
            f"{'' if refuses else 'unset FAKE_ACP_PROMPT_ERROR'}\n"
            f'FAKE_ACP_REPLY={reply} exec "{sys.executable}" "{FAKE_SERVER}" "$@"\n'
        )
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
        assert shutil.which(binary) == str(wrapper)
        return wrapper

    return install


@pytest.fixture
def fallback_on_path(acp_cli_on_path: InstallCLI) -> Path:
    """Put a ``claude-agent-acp`` wrapper on PATH that runs the fake, answering."""
    return acp_cli_on_path("claude-agent-acp", reply="fallback")


def _conversation(tmp_path: Path, **fields) -> LocalConversation:
    agent = ACPAgent(
        **{
            "acp_command": [sys.executable, str(FAKE_SERVER)],
            "acp_server": "custom",
            "acp_startup_timeout": 30.0,
            "acp_prompt_timeout": 30.0,
            **fields,
        }
    )
    return LocalConversation(
        agent=agent,
        workspace=str(tmp_path / "ws"),
        persistence_dir=str(tmp_path / "persist"),
    )


def _switches(conv: LocalConversation) -> list[ACPRuntimeSwitchEvent]:
    return [e for e in conv.state.events if isinstance(e, ACPRuntimeSwitchEvent)]


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
    # ...and so did anything reading events rather than prose.
    switches = _switches(conv)
    assert len(switches) == 1
    assert (switches[0].from_server, switches[0].to_server) == ("custom", "claude-code")
    assert switches[0].reason == "usage_limit"
    assert LIMIT_NOTICE in switches[0].detail
    assert switches[0].reset_hint is not None
    assert switches[0].reset_hint.at is not None
    assert switches[0].reset_hint.confidence == "exact"
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


def test_a_second_exhaustion_reports_the_chain_not_the_last_raw_refusal(
    tmp_path: Path, acp_cli_on_path: InstallCLI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ACP_PROMPT_ERROR", LIMIT_NOTICE)
    claude = acp_cli_on_path("claude-agent-acp", refuses=True)
    acp_cli_on_path("codex-acp", refuses=True)
    conv = _conversation(
        tmp_path,
        acp_server="claude-code",
        acp_command=[str(claude)],
        acp_fallback_server="codex",
    )
    try:
        conv.send_message("hello")
        with pytest.raises(Exception):
            conv.run()

        # The replacement carried a chain, so the run reached a second hop...
        assert [(s.from_server, s.to_server) for s in _switches(conv)] == [
            ("claude-code", "codex")
        ]
        assert conv.state.agent_state[ACP_FAILOVER_TRIED_KEY] == [
            "claude-code",
            "codex",
        ]
        # ...and stopping is reported as the whole chain being spent, naming
        # the runtimes, rather than as Codex's own notice.
        errors = [e for e in conv.state.events if isinstance(e, ConversationErrorEvent)]
        assert [e.code for e in errors] == ["ACPUsageLimitChainExhausted"]
        assert "Claude Code" in errors[0].detail
        assert "Codex" in errors[0].detail
        assert errors[0].classification is not None
        assert errors[0].classification.kind is FailureKind.QUOTA
        assert conv.state.execution_status == ConversationExecutionStatus.ERROR
    finally:
        conv.close()


def test_a_manually_switched_runtime_can_still_fail_over(
    tmp_path: Path, acp_cli_on_path: InstallCLI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ACP_PROMPT_ERROR", LIMIT_NOTICE)
    claude = acp_cli_on_path("claude-agent-acp", reply="claude")
    acp_cli_on_path("gemini", refuses=True)
    acp_cli_on_path("codex-acp", reply="codex")
    conv = _conversation(
        tmp_path,
        acp_server="claude-code",
        acp_command=[str(claude)],
        acp_fallback_server="codex",
    )
    try:
        conv.send_message("hello")
        conv.run()
        conv.switch_acp_server("gemini-cli")
        conv.send_message("again")
        conv.run()

        agent = conv.agent
        assert isinstance(agent, ACPAgent)
        assert agent.acp_server == "codex"
        assert _finish_texts(conv)[-1] == "codex (resumed)"
        assert [(s.from_server, s.to_server, s.reason) for s in _switches(conv)] == [
            ("claude-code", "gemini-cli", "manual"),
            ("gemini-cli", "codex", "usage_limit"),
        ]
        # The manual hop restarted the chain record, so it holds only the
        # runtime that actually refused.
        assert conv.state.agent_state[ACP_FAILOVER_TRIED_KEY] == ["gemini-cli"]
        assert conv.state.execution_status == ConversationExecutionStatus.FINISHED
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
