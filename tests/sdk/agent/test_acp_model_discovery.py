"""Model discovery and legacy model selection against a real stdio ACP server."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import httpx
import pytest

from openhands.sdk.agent import acp_model_discovery
from openhands.sdk.agent.acp_agent import ACPAgent
from openhands.sdk.agent.acp_model_discovery import discover_acp_models
from openhands.sdk.agent.acp_models import ACPModelDiscoveryError, ACPModelInfo
from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.settings.model import ACPAgentSettings
from openhands.sdk.workspace.local import LocalWorkspace


FAKE_SERVER = Path(__file__).with_name("fake_acp_models_server.py")
FAKE_COMMAND = [sys.executable, str(FAKE_SERVER)]


@pytest.fixture(autouse=True)
def _clean_fake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "FAKE_ACP_MODELS",
        "FAKE_ACP_KEY",
        "FAKE_ACP_REQUIRE_KEY",
        "FAKE_ACP_EFFORT",
        "FAKE_ACP_AGENT_NAME",
        "FAKE_ACP_RESUME",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def work_root(tmp_path: Path) -> Path:
    root = tmp_path / "work"
    root.mkdir()
    return root


def _settings(**overrides) -> ACPAgentSettings:
    return ACPAgentSettings(acp_server="custom", acp_command=FAKE_COMMAND, **overrides)


def test_reports_default_and_models_from_config_options(work_root: Path) -> None:
    result = discover_acp_models(_settings(), work_root=work_root)

    assert result.error is None
    assert (result.agent_name, result.agent_version) == ("fake-acp", "1.2.3")
    assert result.current_model_id == "m1"
    assert [m.model_id for m in result.available_models] == ["m1", "m2"]
    assert list(work_root.iterdir()) == []


def test_reports_models_from_the_legacy_models_block(
    work_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ACP_MODELS", "legacy")

    result = discover_acp_models(_settings(), work_root=work_root)

    assert result.error is None
    assert result.current_model_id == "m1"
    assert result.available_models == [
        ACPModelInfo(model_id="m1", name="M1", description="m1 model"),
        ACPModelInfo(model_id="m2", name="M2", description="m2 model"),
    ]


def test_secrets_reach_the_server(work_root: Path) -> None:
    result = discover_acp_models(
        _settings(), secrets={"FAKE_ACP_KEY": "key"}, work_root=work_root
    )

    assert [m.model_id for m in result.available_models] == ["m1", "m2", "premium"]


def test_reports_the_server_default_not_the_requested_model(work_root: Path) -> None:
    result = discover_acp_models(_settings(acp_model="m2"), work_root=work_root)

    assert result.current_model_id == "m1"


def test_reports_an_auth_failure(
    work_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ACP_REQUIRE_KEY", "1")

    result = discover_acp_models(_settings(), work_root=work_root)

    assert result.error == ACPModelDiscoveryError(
        code="ACPAuthRequired", detail="[-32000] Authentication required"
    )
    assert result.available_models == []
    assert list(work_root.iterdir()) == []


def test_reports_a_spawn_failure(work_root: Path, tmp_path: Path) -> None:
    settings = ACPAgentSettings(
        acp_server="custom", acp_command=[str(tmp_path / "no-such-acp-server")]
    )

    result = discover_acp_models(settings, work_root=work_root)

    assert result.error is not None
    assert result.error.code == "ACPSpawnError"
    assert list(work_root.iterdir()) == []


def test_reports_effort_levels_per_model(
    work_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ACP_EFFORT", "1")
    monkeypatch.setenv("FAKE_ACP_KEY", "unlocks-premium")

    result = discover_acp_models(_settings(), work_root=work_root)

    assert result.error is None
    assert result.efforts_by_model == {
        "m1": ["low", "high"],
        "m2": [],
        "premium": ["low", "high", "max"],
    }
    # Probing leaves the session on the server's default model.
    assert (result.current_model_id, result.current_effort) == ("m1", "low")
    assert result.available_efforts == ["low", "high"]


def _start(agent: ACPAgent, tmp_path: Path) -> ConversationState:
    state = ConversationState.create(
        id=uuid.uuid4(),
        agent=agent,
        workspace=LocalWorkspace(working_dir=str(tmp_path)),
    )
    with state:
        agent.init_state(state, on_event=lambda _: None)
    return state


def test_combined_id_selects_the_model_and_its_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ACP_EFFORT", "1")
    agent = ACPAgent(acp_command=FAKE_COMMAND, acp_model="m1/high")
    try:
        state = _start(agent, tmp_path)

        assert agent.current_model_id == "m1/high"
        assert (agent.current_effort, agent.available_efforts) == (
            "high",
            ["low", "high"],
        )
        assert {
            key: state.agent_state.get(key)
            for key in ("acp_current_effort", "acp_available_efforts")
        } == {"acp_current_effort": "high", "acp_available_efforts": ["low", "high"]}

        # A model without effort levels drops the option entirely.
        agent.set_acp_model("m2")
        assert (agent.current_effort, agent.available_efforts) == (None, [])
    finally:
        agent.close()


@pytest.mark.parametrize(
    ("requested", "running"),
    [("m2", "m2"), ("not-offered", "m1")],
)
def test_legacy_server_receives_the_initial_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requested: str, running: str
) -> None:
    monkeypatch.setenv("FAKE_ACP_MODELS", "legacy")
    agent = ACPAgent(acp_command=FAKE_COMMAND, acp_model=requested)
    try:
        _start(agent, tmp_path)
        assert agent.current_model_id == running
    finally:
        agent.close()


def test_legacy_server_switches_models_at_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ACP_MODELS", "legacy")
    agent = ACPAgent(acp_command=FAKE_COMMAND)
    try:
        _start(agent, tmp_path)
        agent.set_acp_model("m2")
        assert agent.current_model_id == "m2"
        with pytest.raises(Exception, match="Unknown model"):
            agent.set_acp_model("not-offered")
        assert agent.current_model_id == "m2"
    finally:
        agent.close()


def test_claude_switches_a_live_session_to_a_model_it_does_not_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """claude-agent-acp switches a live session only to its picker's models; an
    account model such as ``claude-mythos-5-1`` is applied by resuming the
    session with it in ``_meta``, which rebuilds the session on that model."""
    monkeypatch.setenv("FAKE_ACP_EFFORT", "1")
    monkeypatch.setenv("FAKE_ACP_RESUME", "1")
    monkeypatch.setenv("FAKE_ACP_AGENT_NAME", "claude-agent-acp")
    agent = ACPAgent(acp_command=FAKE_COMMAND, acp_model="m1")
    try:
        _start(agent, tmp_path)

        agent.set_acp_model("account-model/max")

        assert agent.current_model_id == "account-model/max"
        assert "account-model" in [m.model_id for m in agent.available_models]
        assert (agent.current_effort, agent.available_efforts) == (
            "max",
            ["low", "high", "max"],
        )
        # A listed model still switches through the config option.
        agent.set_acp_model("m1/high")
        assert (agent.current_model_id, agent.current_effort) == ("m1/high", "high")
    finally:
        agent.close()


def test_a_server_without_resume_rejects_a_model_it_does_not_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ACP_AGENT_NAME", "claude-agent-acp")
    agent = ACPAgent(acp_command=FAKE_COMMAND, acp_model="m1")
    try:
        _start(agent, tmp_path)

        with pytest.raises(ValueError, match="Unknown model"):
            agent.set_acp_model("account-model")
        assert agent.current_model_id == "m1"
    finally:
        agent.close()


def test_session_meta_carries_only_the_model_of_a_combined_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """claude-agent-acp lists an unknown ``_meta`` model as a custom model, so a
    combined id there would show up as a bogus ``m1/high`` model."""
    monkeypatch.setenv("FAKE_ACP_EFFORT", "1")
    monkeypatch.setenv("FAKE_ACP_AGENT_NAME", "claude-agent-acp")
    agent = ACPAgent(acp_command=FAKE_COMMAND, acp_model="m1/high")
    try:
        _start(agent, tmp_path)

        assert [m.model_id for m in agent.available_models] == ["m1", "m2"]
        assert (agent.current_model_id, agent.current_effort) == ("m1/high", "high")
    finally:
        agent.close()


def test_claude_discovery_adds_the_accounts_other_models(
    work_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """claude-agent-acp lists a few aliases; the account's other models (such as
    a research model) are appended, but not ones an alias already stands for."""
    monkeypatch.setenv("FAKE_ACP_AGENT_NAME", "claude-agent-acp")
    pages = {
        None: {
            "data": [
                {"id": "claude-m1", "display_name": "Claude M1"},
                {"id": "claude-mythos-5-1", "display_name": "Claude Mythos 5.1"},
            ],
            "has_more": True,
            "last_id": "claude-mythos-5-1",
        },
        "claude-mythos-5-1": {
            "data": [{"id": "claude-older", "display_name": "Claude Older"}],
            "has_more": False,
        },
    }
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=pages[request.url.params.get("after_id")])

    real_client = httpx.Client
    monkeypatch.setattr(
        acp_model_discovery.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )

    result = discover_acp_models(
        _settings(),
        secrets={
            "ANTHROPIC_API_KEY": "k-1",
            "ANTHROPIC_BASE_URL": "http://anthropic.test",
        },
        work_root=work_root,
    )

    assert result.error is None
    assert [m.model_id for m in result.available_models] == [
        "m1",
        "m2",
        "claude-mythos-5-1",
        "claude-older",
    ]
    assert {str(r.url.copy_with(params=None)) for r in requests} == {
        "http://anthropic.test/v1/models"
    }
    assert all(r.headers["x-api-key"] == "k-1" for r in requests)
