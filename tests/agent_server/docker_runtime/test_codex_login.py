"""Adopting the Codex ChatGPT login a Docker sandbox refreshed."""

from __future__ import annotations

import json
from pathlib import Path

from openhands.agent_server.docker_runtime.codex_login import (
    adopt_refreshed_codex_login,
    seed_codex_login,
)
from openhands.agent_server.persistence import FileSecretsStore
from openhands.sdk.agent.acp_file_credentials import CODEX_AUTH_SECRET_NAME


def _login(refresh_token: str, last_refresh: str, account: str = "acct-1") -> str:
    return json.dumps(
        {
            "auth_mode": "chatgpt",
            "last_refresh": last_refresh,
            "tokens": {"refresh_token": refresh_token, "account_id": account},
        }
    )


def _sandbox_copy(conversations: Path, conversation: str, value: str) -> None:
    path = conversations / conversation / "acp" / "codex" / "auth.json"
    path.parent.mkdir(parents=True)
    path.write_text(value, encoding="utf-8")


def test_adopts_the_most_recently_refreshed_sandbox_copy(tmp_path: Path) -> None:
    store = FileSecretsStore(tmp_path / "store")
    store.set_secret(
        CODEX_AUTH_SECRET_NAME, _login("r0", "2026-10-07T16:48:18.157230335Z")
    )
    conversations = tmp_path / "conversations"
    _sandbox_copy(conversations, "a", _login("r1", "2026-10-15T18:50:39.254259243Z"))
    _sandbox_copy(conversations, "b", _login("r2", "2026-10-15T18:50:39.3Z"))

    assert adopt_refreshed_codex_login(conversations, store) is True

    stored = json.loads(store.load_versioned_secret(CODEX_AUTH_SECRET_NAME)[0])
    assert stored["tokens"]["refresh_token"] == "r2"
    # Nothing newer is left, so a second pass changes nothing.
    assert adopt_refreshed_codex_login(conversations, store) is False


def test_keeps_the_store_when_no_sandbox_copy_is_newer_or_same_account(
    tmp_path: Path,
) -> None:
    store = FileSecretsStore(tmp_path / "store")
    current = _login("r0", "2026-10-15T00:00:00Z")
    store.set_secret(CODEX_AUTH_SECRET_NAME, current)
    conversations = tmp_path / "conversations"
    _sandbox_copy(conversations, "older", _login("r-old", "2026-10-01T00:00:00Z"))
    _sandbox_copy(
        conversations, "other", _login("r-other", "2026-10-16T00:00:00Z", "acct-2")
    )
    _sandbox_copy(conversations, "broken", "{not json")

    assert adopt_refreshed_codex_login(conversations, store) is False
    assert store.load_versioned_secret(CODEX_AUTH_SECRET_NAME)[0] == current


def test_does_nothing_without_a_stored_login(tmp_path: Path) -> None:
    conversations = tmp_path / "conversations"
    _sandbox_copy(conversations, "a", _login("r1", "2026-10-15T00:00:00Z"))

    store = FileSecretsStore(tmp_path / "store")
    sandbox = FileSecretsStore(tmp_path / "sandbox")

    assert adopt_refreshed_codex_login(conversations, store) is False
    assert seed_codex_login(store, sandbox) is False
    assert sandbox.get_secret(CODEX_AUTH_SECRET_NAME) is None
