"""Ask an ACP server which models it offers, without starting a conversation."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path

import httpx

from openhands.sdk.agent.acp_agent import (
    ACPAgent,
    _acp_error_detail,
    _classify_acp_init_error,
)
from openhands.sdk.agent.acp_models import (
    ACPModelDiscovery,
    ACPModelDiscoveryError,
    ACPModelInfo,
)
from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.credential import VersionedCredentialBinding
from openhands.sdk.event import Event
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.logger import get_logger
from openhands.sdk.secret import SecretValue
from openhands.sdk.settings.acp_providers import detect_acp_provider_by_agent_name
from openhands.sdk.settings.model import ACPAgentSettings
from openhands.sdk.workspace.local import LocalWorkspace


logger = get_logger(__name__)

_PROBE_DIR_PREFIX = ".acp-model-discovery-"
_ANTHROPIC_API_URL = "https://api.anthropic.com"
_ANTHROPIC_API_VERSION = "2023-06-01"
_ANTHROPIC_OAUTH_BETA = "oauth-2025-04-20"


def _claude_account_models(env: Mapping[str, str]) -> list[ACPModelInfo]:
    """Models the Anthropic account behind Claude Code's credentials can use.

    Claude Code's picker -- and so claude-agent-acp's model list -- offers a
    few aliases, while the account's other models (earlier versions, research
    models) work by full id. Uses the credentials Claude Code would: the OAuth
    token (which ignores ``ANTHROPIC_BASE_URL``) or the API key. Best-effort:
    returns what it fetched before any failure.
    """
    if token := env.get("CLAUDE_CODE_OAUTH_TOKEN"):
        base_url = _ANTHROPIC_API_URL
        headers = {
            "Authorization": f"Bearer {token}",
            "anthropic-beta": _ANTHROPIC_OAUTH_BETA,
        }
    elif api_key := env.get("ANTHROPIC_API_KEY"):
        base_url = (env.get("ANTHROPIC_BASE_URL") or _ANTHROPIC_API_URL).rstrip("/")
        headers = {"x-api-key": api_key}
    else:
        return []
    headers["anthropic-version"] = _ANTHROPIC_API_VERSION
    models: list[ACPModelInfo] = []
    params: dict[str, str | int] = {"limit": 1000}
    try:
        with httpx.Client(timeout=10.0) as client:
            while True:
                response = client.get(
                    f"{base_url}/v1/models", headers=headers, params=params
                )
                response.raise_for_status()
                page = response.json()
                models.extend(
                    ACPModelInfo(model_id=item["id"], name=item.get("display_name"))
                    for item in page.get("data", [])
                    if isinstance(item, dict) and item.get("id")
                )
                if not page.get("has_more") or not page.get("last_id"):
                    break
                params = {"limit": 1000, "after_id": page["last_id"]}
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("Could not list the Claude account's models: %s", exc)
    return models


def _display_key(name: str | None) -> str:
    return (name or "").strip().lower().removeprefix("claude ").strip()


def _with_account_models(
    listed: list[ACPModelInfo], account: list[ACPModelInfo]
) -> list[ACPModelInfo]:
    """Append the account's models that no listed alias already stands for.

    claude-agent-acp names its aliases by model (``opus`` is "Opus 5.5"), and
    the API names models in full ("Claude Opus 5.5"), so matching display names
    keeps one entry per model.
    """
    covered = {m.model_id for m in listed} | {
        key for m in listed if (key := _display_key(m.name))
    }
    return listed + [
        m
        for m in account
        if m.model_id not in covered and _display_key(m.name) not in covered
    ]


def discover_acp_models(
    settings: ACPAgentSettings,
    *,
    secrets: Mapping[str, SecretValue] | None = None,
    credential_bindings: Mapping[str, VersionedCredentialBinding] | None = None,
    work_root: str | Path | None = None,
) -> ACPModelDiscovery:
    """Start the server in a throwaway session and report what it offers."""
    agent = ACPAgent(
        acp_command=settings.resolve_acp_command(),
        acp_server=settings.acp_server,
        acp_args=list(settings.acp_args),
        acp_session_mode=settings.acp_session_mode,
        acp_startup_timeout=settings.acp_startup_timeout,
        acp_isolate_data_dir=settings.acp_isolate_data_dir,
        acp_file_secrets=list(settings.acp_file_secrets),
    )
    probe_dir = Path(tempfile.mkdtemp(prefix=_PROBE_DIR_PREFIX, dir=work_root))
    state: ConversationState | None = None
    errors: list[ConversationErrorEvent] = []

    def on_event(event: Event) -> None:
        if isinstance(event, ConversationErrorEvent):
            errors.append(event)

    try:
        for name, binding in (credential_bindings or {}).items():
            agent.activate_file_credential_binding(name, binding)
        workspace = probe_dir / "workspace"
        workspace.mkdir()
        state = ConversationState.create(
            id=uuid.uuid4(),
            agent=agent,
            workspace=LocalWorkspace(working_dir=str(workspace)),
            persistence_dir=str(probe_dir),
        )
        state.secret_registry.update_secrets(dict(secrets or {}))
        with state:
            agent.init_state(state, on_event=on_event)
        available_models = agent.available_models
        provider = detect_acp_provider_by_agent_name(agent.agent_name or "")
        if provider is not None and provider.key == "claude-code":
            registry = state.secret_registry
            env = {**os.environ}
            for name in secrets or {}:
                if (value := registry.get_secret_value(name)) is not None:
                    env[name] = value
            available_models = _with_account_models(
                available_models, _claude_account_models(env)
            )
        return ACPModelDiscovery(
            agent_name=agent.agent_name or None,
            agent_version=agent.agent_version or None,
            current_model_id=agent.current_model_id,
            available_models=available_models,
            supports_runtime_model_switch=agent.supports_runtime_model_switch,
            efforts_by_model=agent.probe_model_efforts(),
            current_effort=agent.current_effort,
            available_efforts=agent.available_efforts,
        )
    except Exception as exc:
        if errors:
            error = ACPModelDiscoveryError(code=errors[0].code, detail=errors[0].detail)
        else:
            registry = state.secret_registry if state is not None else None
            error = ACPModelDiscoveryError(
                code=_classify_acp_init_error(exc),
                detail=_acp_error_detail(exc, registry),
            )
        logger.info(
            "ACP model discovery failed: server=%s code=%s",
            settings.acp_server,
            error.code,
        )
        return ACPModelDiscovery(error=error)
    finally:
        try:
            agent.close()
        except Exception:
            logger.warning(
                "Failed to stop the ACP model-discovery server", exc_info=True
            )
        shutil.rmtree(probe_dir, ignore_errors=True)
