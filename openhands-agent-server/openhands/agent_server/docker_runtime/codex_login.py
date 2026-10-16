"""Keep the stored Codex ChatGPT login current across Docker sandboxes.

codex-acp refreshes a login that is more than a few days old inside the
sandbox: a conversation sandbox writes it to its own ``CODEX_HOME``
(``<conversation dir>/acp/codex``, a directory this host mounts), a discovery
sandbox to the secret store it was seeded with. The refresh rotates the refresh
token, so the copy in this server's secret store stops working; the next
sandbox started from it can only ask for a new browser sign-in. Adopting the
most recently refreshed sandbox copy back into the store keeps every later
sandbox signed in.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from openhands.agent_server.persistence.store import FileSecretsStore
from openhands.sdk.agent.acp_file_credentials import (
    CODEX_AUTH_SECRET_NAME,
    is_valid_codex_auth,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

_SANDBOX_CODEX_AUTH_GLOB = "*/acp/codex/auth.json"


def _login(value: str) -> tuple[datetime, str | None] | None:
    """``(last_refresh, account_id)`` of a valid ChatGPT login, else ``None``."""
    if not is_valid_codex_auth(value):
        return None
    payload = json.loads(value)
    refreshed = payload.get("last_refresh")
    if not isinstance(refreshed, str):
        return None
    # Codex writes nanoseconds; datetime takes at most microseconds.
    stamp = refreshed.removesuffix("Z")
    whole, _, fraction = stamp.partition(".")
    try:
        when = datetime.fromisoformat(
            f"{whole}.{fraction[:6]}" if fraction else whole
        ).replace(tzinfo=UTC)
    except ValueError:
        return None
    account = payload["tokens"].get("account_id")
    return when, account if isinstance(account, str) else None


def _sandbox_copies(conversations_path: Path) -> Iterator[str]:
    for path in conversations_path.glob(_SANDBOX_CODEX_AUTH_GLOB):
        try:
            yield path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue


def _adopt_newest(store: FileSecretsStore, candidates: Iterable[str]) -> bool:
    try:
        stored_value, version = store.load_versioned_secret(CODEX_AUTH_SECRET_NAME)
    except KeyError:
        return False
    stored = _login(stored_value)
    if stored is None:
        return False
    newest_at, newest_value = stored[0], None
    for value in candidates:
        login = _login(value)
        if login is None or login[0] <= newest_at:
            continue
        if stored[1] and login[1] and login[1] != stored[1]:
            continue
        newest_at, newest_value = login[0], value
    if newest_value is None:
        return False
    try:
        store.replace_versioned_secret(CODEX_AUTH_SECRET_NAME, version, newest_value)
    except (KeyError, ValueError):
        return False
    logger.info("Adopted the Codex login a sandbox refreshed at %s", newest_at)
    return True


def adopt_refreshed_codex_login(
    conversations_path: Path, store: FileSecretsStore
) -> bool:
    """Store the most recently refreshed conversation-sandbox copy of the login.

    Only a copy of the same account that was refreshed after the stored one is
    adopted, through a versioned replace so a concurrent change wins. Returns
    whether the store was updated.
    """
    return _adopt_newest(store, _sandbox_copies(conversations_path))


def seed_codex_login(store: FileSecretsStore, sandbox_store: FileSecretsStore) -> bool:
    """Give a sandbox the stored login through its own secret store.

    A sandbox whose store holds the login refreshes it through that store (see
    ``LocalVersionedCredentialBinding``) instead of a throwaway file, so
    :func:`adopt_sandbox_codex_login` can bring the refresh back. Returns
    whether there was a login to seed.
    """
    value = store.get_secret(CODEX_AUTH_SECRET_NAME)
    if value is None:
        return False
    sandbox_store.set_secret(CODEX_AUTH_SECRET_NAME, value)
    return True


def adopt_sandbox_codex_login(
    sandbox_store: FileSecretsStore, store: FileSecretsStore
) -> bool:
    """Store the login a seeded sandbox refreshed, if it is newer."""
    value = sandbox_store.get_secret(CODEX_AUTH_SECRET_NAME)
    return _adopt_newest(store, () if value is None else (value,))
