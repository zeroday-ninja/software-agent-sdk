"""Tests for the ``/api/soul`` endpoints."""

from __future__ import annotations

import stat
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openhands.agent_server.api import create_app
from openhands.agent_server.config import Config
from openhands.agent_server.persistence import reset_stores
from openhands.agent_server.soul_router import MAX_SOUL_LENGTH
from openhands.sdk.context.soul import DEFAULT_SOUL, get_soul_path


@pytest.fixture
def persistence_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("OH_PERSISTENCE_DIR", str(tmp_path / "state"))
    yield tmp_path / "state"


@pytest.fixture
def client(persistence_dir: Path) -> Iterator[TestClient]:
    reset_stores()
    config = Config(static_files_path=None, session_api_keys=[], secret_key=None)
    yield TestClient(create_app(config))
    reset_stores()


def test_get_returns_the_builtin_default_when_no_file_exists(
    client: TestClient, persistence_dir: Path
) -> None:
    response = client.get("/api/soul")

    assert response.status_code == 200
    body = response.json()
    assert body["exists"] is False
    assert body["content"] == DEFAULT_SOUL
    assert body["path"] == str(persistence_dir / "SOUL.md")
    assert body["size_bytes"] == len(DEFAULT_SOUL.encode("utf-8"))


def test_put_then_get_round_trips_the_user_content(client: TestClient) -> None:
    written = client.put("/api/soul", json={"content": "You are smolpaws. Be brief."})

    assert written.status_code == 200
    assert written.json()["exists"] is True

    body = client.get("/api/soul").json()
    assert body["content"] == "You are smolpaws. Be brief."
    assert body["exists"] is True
    assert body["size_bytes"] == len(b"You are smolpaws. Be brief.")


def test_put_creates_missing_parent_dirs_with_owner_only_permissions(
    client: TestClient, persistence_dir: Path
) -> None:
    assert not persistence_dir.exists()

    response = client.put("/api/soul", json={"content": "You are smolpaws."})

    assert response.status_code == 200
    soul = persistence_dir / "SOUL.md"
    assert stat.S_IMODE(soul.stat().st_mode) == 0o600
    # Atomic write: no temp file is left behind next to the target.
    assert sorted(p.name for p in persistence_dir.iterdir()) == ["SOUL.md"]


def test_put_rejects_content_over_the_size_limit(client: TestClient) -> None:
    response = client.put("/api/soul", json={"content": "x" * (MAX_SOUL_LENGTH + 1)})

    assert response.status_code == 422
    assert client.get("/api/soul").json()["exists"] is False


def test_put_accepts_content_at_the_size_limit(client: TestClient) -> None:
    response = client.put("/api/soul", json={"content": "x" * MAX_SOUL_LENGTH})

    assert response.status_code == 200
    assert response.json()["exists"] is True


def test_put_empty_content_reverts_to_the_default_by_deleting_the_file(
    client: TestClient, persistence_dir: Path
) -> None:
    client.put("/api/soul", json={"content": "You are smolpaws."})

    response = client.put("/api/soul", json={"content": ""})

    assert response.status_code == 200
    body = response.json()
    assert body["exists"] is False
    assert body["content"] == DEFAULT_SOUL
    assert not (persistence_dir / "SOUL.md").exists()


def test_endpoint_and_sdk_loader_resolve_the_same_path(
    client: TestClient, persistence_dir: Path
) -> None:
    """The route must not re-derive ``~/.openhands``; both read one resolution."""
    reported = client.get("/api/soul").json()["path"]

    assert Path(reported) == get_soul_path() == persistence_dir / "SOUL.md"


def test_symlink_escaping_the_persistence_dir_is_rejected(
    client: TestClient, persistence_dir: Path, tmp_path: Path
) -> None:
    """SOUL.md must not become an arbitrary-file read/write primitive."""
    outside = tmp_path / "outside.md"
    outside.write_text("secrets")
    persistence_dir.mkdir(parents=True)
    (persistence_dir / "SOUL.md").symlink_to(outside)

    assert client.get("/api/soul").status_code == 403
    assert client.put("/api/soul", json={"content": "pwned"}).status_code == 403
    assert outside.read_text() == "secrets"


def test_soul_routes_require_the_session_api_key(
    persistence_dir: Path,
) -> None:
    reset_stores()
    config = Config(
        static_files_path=None, session_api_keys=["sekrit"], secret_key=None
    )
    try:
        with TestClient(create_app(config)) as client:
            assert client.get("/api/soul").status_code == 401
            assert (
                client.get(
                    "/api/soul", headers={"X-Session-API-Key": "sekrit"}
                ).status_code
                == 200
            )
    finally:
        reset_stores()
