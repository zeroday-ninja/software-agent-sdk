"""Tests for ``POST /api/acp/models``."""

from __future__ import annotations

import json
import secrets
import sys
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from openhands.agent_server import acp_router as acp_router_module
from openhands.agent_server.acp_router import ACPModelDiscoveryRequest, acp_router
from openhands.agent_server.config import Config
from openhands.agent_server.credential_binding import LocalVersionedCredentialBinding
from openhands.agent_server.docker_runtime.registry import (
    ConversationContainer,
    DiscoveryRuntime,
    DockerConversationRegistry,
)
from openhands.agent_server.local_secret_resolver import local_secret_resolution
from openhands.agent_server.persistence import FileSecretsStore, get_secrets_store
from openhands.sdk.agent.acp_file_credentials import CODEX_AUTH_SECRET_NAME
from openhands.sdk.agent.acp_models import (
    ACPModelDiscovery,
    ACPModelDiscoveryError,
    ACPModelInfo,
)
from openhands.sdk.secret import StaticSecret
from openhands.sdk.utils.cipher import Cipher


FAKE_SERVER = Path(__file__).parents[1] / "sdk" / "agent" / "fake_acp_models_server.py"
FAKE_AGENT = {
    "agent_kind": "acp",
    "acp_server": "custom",
    "acp_command": [sys.executable, str(FAKE_SERVER)],
}


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(session_api_keys=[], conversations_path=tmp_path / "conversations")


@pytest.fixture
def client(config: Config) -> Iterator[TestClient]:
    app = FastAPI()
    app.state.config = config
    app.include_router(acp_router, prefix="/api")
    with local_secret_resolution(config):
        yield TestClient(app)


class _FakeDiscovery:
    def __init__(self, result: ACPModelDiscovery) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def __call__(self, settings, **kwargs: Any) -> ACPModelDiscovery:
        self.calls.append({"settings": settings, **kwargs})
        return self.result


@pytest.fixture
def fake_discovery(monkeypatch: pytest.MonkeyPatch) -> _FakeDiscovery:
    fake = _FakeDiscovery(
        ACPModelDiscovery(
            current_model_id="m1", available_models=[ACPModelInfo(model_id="m1")]
        )
    )
    monkeypatch.setattr(acp_router_module, "discover_acp_models", fake)
    return fake


def _discover(client: TestClient, **body: Any):
    return client.post("/api/acp/models", json={"agent_settings": FAKE_AGENT, **body})


def test_reports_a_live_servers_models(
    client: TestClient, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FAKE_ACP_KEY", raising=False)

    response = _discover(client)

    assert response.status_code == 200
    body = response.json()
    assert body["error"] is None
    assert body["agent_name"] == "fake-acp"
    assert body["current_model_id"] == "m1"
    assert [m["model_id"] for m in body["available_models"]] == ["m1", "m2"]
    assert list(config.conversations_path.iterdir()) == []


def test_resolves_stored_secrets_by_reference(
    client: TestClient, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FAKE_ACP_KEY", raising=False)
    get_secrets_store(config).set_secret("FAKE_ACP_KEY", "stored-key")
    lookup = {"kind": "LookupSecret", "url": "/api/settings/secrets/FAKE_ACP_KEY"}

    response = _discover(client, secrets={"FAKE_ACP_KEY": lookup})

    models = [m["model_id"] for m in response.json()["available_models"]]
    assert models == ["m1", "m2", "premium"]


def test_reuses_a_successful_result(
    client: TestClient, fake_discovery: _FakeDiscovery
) -> None:
    first = _discover(client)
    second = _discover(client)

    assert first.json() == second.json()
    assert len(fake_discovery.calls) == 1


def test_refresh_asks_the_server_again(
    client: TestClient, fake_discovery: _FakeDiscovery
) -> None:
    _discover(client)
    _discover(client, refresh=True)

    assert len(fake_discovery.calls) == 2


def test_new_credentials_ask_the_server_again(
    client: TestClient, fake_discovery: _FakeDiscovery
) -> None:
    def static(value: str) -> dict[str, str]:
        return {"kind": "StaticSecret", "value": value}

    _discover(client, secrets={"KEY": static("a")})
    _discover(client, secrets={"KEY": static("a")})
    _discover(client, secrets={"KEY": static("b")})

    assert [call["secrets"] for call in fake_discovery.calls] == [
        {"KEY": "a"},
        {"KEY": "b"},
    ]


def test_does_not_reuse_a_failure(
    client: TestClient, fake_discovery: _FakeDiscovery
) -> None:
    fake_discovery.result = ACPModelDiscovery(
        error=ACPModelDiscoveryError(code="ACPAuthRequired", detail="log in")
    )

    first = _discover(client)
    _discover(client)

    assert first.status_code == 200
    assert first.json()["error"] == {"code": "ACPAuthRequired", "detail": "log in"}
    assert len(fake_discovery.calls) == 2


def test_uses_the_stored_codex_login_like_a_conversation(
    client: TestClient, config: Config, fake_discovery: _FakeDiscovery
) -> None:
    get_secrets_store(config).set_secret(CODEX_AUTH_SECRET_NAME, '{"tokens": {}}')
    sent = {"kind": "StaticSecret", "value": "pasted"}

    client.post(
        "/api/acp/models",
        json={
            "agent_settings": {"agent_kind": "acp", "acp_server": "codex"},
            "secrets": {CODEX_AUTH_SECRET_NAME: sent},
        },
    )

    (call,) = fake_discovery.calls
    assert CODEX_AUTH_SECRET_NAME not in call["secrets"]
    binding = call["credential_bindings"][CODEX_AUTH_SECRET_NAME]
    assert isinstance(binding, LocalVersionedCredentialBinding)


def test_rejects_a_custom_server_without_a_command(client: TestClient) -> None:
    response = client.post(
        "/api/acp/models",
        json={"agent_settings": {"agent_kind": "acp", "acp_server": "custom"}},
    )

    assert response.status_code == 422


class _FakeSandbox:
    """Stands in for the registry's throwaway discovery container."""

    def __init__(self, store_dir: Path, result: ACPModelDiscovery | None) -> None:
        self.cipher = Cipher(secrets.token_urlsafe(32))
        self.store = FileSecretsStore(store_dir, self.cipher)
        self.result = result
        self.requests: list[httpx.Request] = []
        self.seen_codex_login: str | None = None
        self.refreshed_codex_login: str | None = None
        self.stopped = False

    @asynccontextmanager
    async def discovery_runtime(self):
        if self.result is None:
            raise RuntimeError("docker run failed")
        container = ConversationContainer(
            host="http://sandbox", api_key="sandbox-key", container_id="c1"
        )
        try:
            yield DiscoveryRuntime(
                container=container, cipher=self.cipher, secrets_store=self.store
            )
        finally:
            self.stopped = True

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert self.result is not None
        self.seen_codex_login = self.store.get_secret(CODEX_AUTH_SECRET_NAME)
        if self.refreshed_codex_login is not None:
            # What codex-acp's refresh does through the sandbox's store binding.
            self.store.set_secret(CODEX_AUTH_SECRET_NAME, self.refreshed_codex_login)
        return httpx.Response(200, json=self.result.model_dump(mode="json"))


@pytest.fixture
def sandbox(
    client: TestClient,
    config: Config,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> _FakeSandbox:
    fake = _FakeSandbox(
        tmp_path / "sandbox-persistence",
        ACPModelDiscovery(
            current_model_id="m1",
            available_models=[ACPModelInfo(model_id="m1")],
            efforts_by_model={"m1": ["low", "high"]},
        ),
    )
    registry = object.__new__(DockerConversationRegistry)
    registry.discovery_runtime = fake.discovery_runtime  # type: ignore[method-assign]
    app = client.app
    assert isinstance(app, FastAPI)
    app.state.config = config.model_copy(update={"conversation_runtime": "docker"})
    app.state.conversation_registry = registry
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        acp_router_module.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(fake.handle), **kwargs
        ),
    )
    return fake


def test_runs_in_a_sandbox_when_conversations_run_in_containers(
    client: TestClient, sandbox: _FakeSandbox, fake_discovery: _FakeDiscovery
) -> None:
    response = _discover(
        client, secrets={"FAKE_ACP_KEY": {"kind": "StaticSecret", "value": "k-123"}}
    )

    assert response.status_code == 200
    assert response.json()["efforts_by_model"] == {"m1": ["low", "high"]}
    assert fake_discovery.calls == []  # not on this host
    assert sandbox.stopped
    (forwarded,) = sandbox.requests
    assert forwarded.headers["X-Session-API-Key"] == "sandbox-key"
    payload = json.loads(forwarded.content)
    assert payload["secrets_encrypted"] is True
    assert "k-123" not in forwarded.content.decode()
    received = ACPModelDiscoveryRequest.model_validate(payload)
    (secret,) = received.secrets.values()
    decrypted = StaticSecret.model_validate(
        secret.model_dump(mode="json", context={"expose_secrets": "plaintext"}),
        context={"cipher": sandbox.cipher},
    )
    assert decrypted.get_value() == "k-123"


def _codex_login(refresh_token: str, last_refresh: str) -> str:
    return json.dumps(
        {
            "auth_mode": "chatgpt",
            "last_refresh": last_refresh,
            "tokens": {"refresh_token": refresh_token, "account_id": "acct-1"},
        }
    )


def test_a_sandbox_hands_back_the_codex_login_it_refreshed(
    client: TestClient, config: Config, sandbox: _FakeSandbox
) -> None:
    store = get_secrets_store(config)
    stored = _codex_login("r0", "2026-10-15T00:00:00Z")
    store.set_secret(CODEX_AUTH_SECRET_NAME, stored)
    refreshed = _codex_login("r1", "2026-10-15T01:00:00Z")
    sandbox.refreshed_codex_login = refreshed

    response = client.post(
        "/api/acp/models",
        json={
            "agent_settings": {"agent_kind": "acp", "acp_server": "codex"},
            "secrets": {
                CODEX_AUTH_SECRET_NAME: {"kind": "StaticSecret", "value": stored}
            },
        },
    )

    assert response.status_code == 200
    (forwarded,) = sandbox.requests
    assert CODEX_AUTH_SECRET_NAME not in json.loads(forwarded.content)["secrets"]
    assert sandbox.seen_codex_login == stored
    assert store.get_secret(CODEX_AUTH_SECRET_NAME) == refreshed


def test_reports_a_sandbox_failure_as_a_discovery_error(
    client: TestClient, sandbox: _FakeSandbox
) -> None:
    sandbox.result = None

    response = _discover(client)

    assert response.status_code == 200
    assert response.json()["error"]["code"] == "ACPSpawnError"


def test_decrypts_secrets_forwarded_by_a_host(
    client: TestClient, config: Config, fake_discovery: _FakeDiscovery
) -> None:
    key = secrets.token_urlsafe(32)
    app = client.app
    assert isinstance(app, FastAPI)
    app.state.config = config.model_copy(update={"secret_key": SecretStr(key)})
    encrypted = StaticSecret(value=SecretStr("k-123")).model_dump(
        mode="json", context={"cipher": Cipher(key)}
    )

    response = _discover(
        client, secrets={"FAKE_ACP_KEY": encrypted}, secrets_encrypted=True
    )

    assert response.status_code == 200
    (call,) = fake_discovery.calls
    assert call["secrets"] == {"FAKE_ACP_KEY": "k-123"}
