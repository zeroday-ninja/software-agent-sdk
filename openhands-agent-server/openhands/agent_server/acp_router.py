"""Ask an ACP server which models it offers before a conversation starts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, SecretStr

from openhands.agent_server._secrets_exposure import get_config
from openhands.agent_server.config import Config
from openhands.agent_server.credential_binding import LocalVersionedCredentialBinding
from openhands.agent_server.docker_runtime.codex_login import (
    adopt_refreshed_codex_login,
    adopt_sandbox_codex_login,
    seed_codex_login,
)
from openhands.agent_server.persistence import get_secrets_store
from openhands.sdk.agent.acp_file_credentials import CODEX_AUTH_SECRET_NAME
from openhands.sdk.agent.acp_model_discovery import discover_acp_models
from openhands.sdk.agent.acp_models import ACPModelDiscovery, ACPModelDiscoveryError
from openhands.sdk.credential import VersionedCredentialBinding
from openhands.sdk.logger import get_logger
from openhands.sdk.secret import SecretSource, StaticSecret
from openhands.sdk.settings.model import ACPAgentSettings


logger = get_logger(__name__)

acp_router = APIRouter(prefix="/acp", tags=["ACP"])

_CACHE_TTL_SECONDS = 300.0
_MAX_CONCURRENT_DISCOVERIES = 2


class ACPModelDiscoveryRequest(BaseModel):
    agent_settings: ACPAgentSettings = Field(
        description=(
            "The ACP agent to ask. Only the launch fields are used; "
            "``acp_model``, MCP servers and agent context are ignored."
        ),
    )
    secrets: dict[str, SecretSource] = Field(
        default_factory=dict,
        description=(
            "Credentials for the server, in the same shape as a conversation "
            "start request's ``secrets`` (typically ``LookupSecret`` references "
            "to this server's stored secrets)."
        ),
    )
    refresh: bool = Field(
        default=False, description="Ask the server again instead of reusing a result."
    )
    secrets_encrypted: bool = Field(
        default=False,
        description=(
            "Set by a server that forwards discovery to a conversation container: "
            "``StaticSecret`` values in ``secrets`` are encrypted with the "
            "receiving server's key, as for a forwarded conversation start."
        ),
    )


@dataclass
class _DiscoveryCache:
    results: dict[str, tuple[float, ACPModelDiscovery]] = field(default_factory=dict)
    in_flight: dict[str, asyncio.Future[ACPModelDiscovery]] = field(
        default_factory=dict
    )
    slots: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(_MAX_CONCURRENT_DISCOVERIES)
    )

    async def get(
        self,
        key: str,
        discover: Callable[[], Awaitable[ACPModelDiscovery]],
        *,
        refresh: bool,
    ) -> ACPModelDiscovery:
        cached = self.results.get(key)
        if not refresh and cached is not None and cached[0] > time.monotonic():
            return cached[1]
        pending = self.in_flight.get(key)
        if pending is None:
            pending = asyncio.ensure_future(self._run(key, discover))
            self.in_flight[key] = pending
        return await asyncio.shield(pending)

    async def _run(
        self, key: str, discover: Callable[[], Awaitable[ACPModelDiscovery]]
    ) -> ACPModelDiscovery:
        try:
            async with self.slots:
                result = await discover()
            if result.error is None:
                self.results[key] = (time.monotonic() + _CACHE_TTL_SECONDS, result)
            return result
        finally:
            self.in_flight.pop(key, None)


def _discovery_cache(request: Request) -> _DiscoveryCache:
    cache = getattr(request.app.state, "acp_model_discovery_cache", None)
    if cache is None:
        cache = _DiscoveryCache()
        request.app.state.acp_model_discovery_cache = cache
    return cache


def _resolve_secrets(secrets: Mapping[str, SecretSource]) -> dict[str, str]:
    values: dict[str, str] = {}
    for name, source in secrets.items():
        try:
            value = source.get_value()
        except Exception:
            logger.warning("Could not resolve secret %r for ACP model discovery", name)
            continue
        if value is not None:
            values[name] = value
    return values


def _cache_key(
    settings: ACPAgentSettings, command: list[str], secrets: Mapping[str, str]
) -> str:
    payload = {
        "command": command,
        "args": list(settings.acp_args),
        "session_mode": settings.acp_session_mode,
        "isolate": settings.acp_isolate_data_dir,
        "file_secrets": [
            spec.model_dump(mode="json") for spec in settings.acp_file_secrets
        ],
        "secrets": sorted(secrets.items()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _uses_stored_codex_login(
    settings: ACPAgentSettings, config: Config, secrets: dict[str, str]
) -> bool:
    """Whether discovery signs in with the stored ChatGPT login.

    It then refreshes through the store, as in a conversation, rather than
    from the request's copy, which is dropped.
    """
    if settings.acp_server != "codex":
        return False
    if get_secrets_store(config).get_secret(CODEX_AUTH_SECRET_NAME) is None:
        return False
    secrets.pop(CODEX_AUTH_SECRET_NAME, None)
    return True


def _codex_bindings(
    settings: ACPAgentSettings, config: Config, secrets: dict[str, str]
) -> dict[str, VersionedCredentialBinding]:
    if not _uses_stored_codex_login(settings, config, secrets):
        return {}
    return {
        CODEX_AUTH_SECRET_NAME: LocalVersionedCredentialBinding(
            get_secrets_store(config), CODEX_AUTH_SECRET_NAME
        )
    }


def _decrypt_forwarded_secrets(
    secrets: Mapping[str, SecretSource], config: Config
) -> dict[str, SecretSource]:
    cipher = config.cipher
    if cipher is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Encrypted secrets need OH_SECRET_KEY on the receiving server",
        )
    return {
        name: (
            StaticSecret.model_validate(
                source.model_dump(mode="json", context={"expose_secrets": "plaintext"}),
                context={"cipher": cipher},
            )
            if isinstance(source, StaticSecret)
            else source
        )
        for name, source in secrets.items()
    }


async def _discover_in_sandbox(
    request: Request,
    settings: ACPAgentSettings,
    secrets: Mapping[str, str],
    *,
    codex_login: bool = False,
) -> ACPModelDiscovery:
    """Ask a throwaway conversation container to run the discovery.

    With ``codex_login`` the sandbox signs in with the stored ChatGPT login,
    seeded into its own secret store; a refresh there rotates the login, so it
    is adopted back before the sandbox is removed.
    """
    from openhands.agent_server.docker_runtime.routers import get_registry

    forwarded = ACPModelDiscoveryRequest(
        agent_settings=settings,
        secrets={
            name: StaticSecret(value=SecretStr(value))
            for name, value in secrets.items()
        },
        refresh=True,
        secrets_encrypted=True,
    )
    # Container start, then the ACP server's own startup and per-model probing.
    timeout = settings.acp_startup_timeout * 2 + 30
    try:
        async with get_registry(request).discovery_runtime() as runtime:
            store = get_secrets_store(get_config(request))
            seeded = codex_login and await asyncio.to_thread(
                seed_codex_login, store, runtime.secrets_store
            )
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        f"{runtime.container.host}/api/acp/models",
                        json=forwarded.model_dump(
                            mode="json", context={"cipher": runtime.cipher}
                        ),
                        headers={"X-Session-API-Key": runtime.container.api_key},
                    )
            finally:
                if seeded:
                    await asyncio.to_thread(
                        adopt_sandbox_codex_login, runtime.secrets_store, store
                    )
            response.raise_for_status()
            return ACPModelDiscovery.model_validate(response.json())
    except Exception as exc:
        logger.warning("ACP model discovery in a sandbox failed: %s", exc)
        return ACPModelDiscovery(
            error=ACPModelDiscoveryError(
                code="ACPSpawnError",
                detail=f"Could not run model discovery in a sandbox: {exc}",
            )
        )


@acp_router.post("/models", response_model=ACPModelDiscovery)
async def discover_models(
    body: ACPModelDiscoveryRequest, request: Request
) -> ACPModelDiscovery:
    """Start the ACP server in a throwaway session and report its models.

    The server answers with its default model and the models it offers for the
    given credentials; a launch or authentication failure comes back in
    ``error`` rather than as an HTTP error.
    """
    config: Config = get_config(request)
    settings = body.agent_settings
    try:
        command = settings.resolve_acp_command()
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    sources = (
        _decrypt_forwarded_secrets(body.secrets, config)
        if body.secrets_encrypted
        else body.secrets
    )
    if config.conversation_runtime == "docker":
        # A sandbox may have refreshed (and so rotated) the stored Codex login.
        await asyncio.to_thread(
            adopt_refreshed_codex_login,
            config.conversations_path,
            get_secrets_store(config),
        )
    secrets = await asyncio.to_thread(_resolve_secrets, sources)
    if config.conversation_runtime == "docker":
        codex_login = await asyncio.to_thread(
            _uses_stored_codex_login, settings, config, secrets
        )

        # Discovery runs where conversations run: the conversation image's ACP
        # servers and credential handling, not this host's.
        async def run_in_sandbox() -> ACPModelDiscovery:
            return await _discover_in_sandbox(
                request, settings, secrets, codex_login=codex_login
            )

        return await _discovery_cache(request).get(
            _cache_key(settings, command, secrets),
            run_in_sandbox,
            refresh=body.refresh,
        )

    bindings = await asyncio.to_thread(_codex_bindings, settings, config, secrets)
    key = _cache_key(settings, command, secrets)

    async def run() -> ACPModelDiscovery:
        config.conversations_path.mkdir(parents=True, exist_ok=True)
        return await asyncio.to_thread(
            discover_acp_models,
            settings,
            secrets=secrets,
            credential_bindings=bindings,
            # Shares the conversations' npx cache.
            work_root=config.conversations_path,
        )

    return await _discovery_cache(request).get(key, run, refresh=body.refresh)
