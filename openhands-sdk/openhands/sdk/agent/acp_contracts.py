"""Typed ACP capability and session contracts.

Defines explicit typed models, protocols, and boundary normalization helpers
for ACP objects (session response options, model selection, MCP capabilities,
protocol errors, credential revisions, and serialization).

All dynamic attribute probing (for backward/forward protocol version
compatibility and UNSTABLE extensions) is confined to the normalization
boundary functions in this module. Downstream code in ACPAgent and its helpers
interacts strictly with typed dataclasses and protocols via direct attribute
access.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from openhands.sdk.agent.acp_models import ACPModelInfo


_MODEL_CONFIG_OPTION_ID = "model"
# ACP config-option category for a server's reasoning-effort select
# (claude-agent-acp ``effort``, codex-acp ``reasoning_effort``).
_THOUGHT_LEVEL_CATEGORY = "thought_level"
# Effort select ids recognized when a server omits the category.
EFFORT_CONFIG_OPTION_IDS: Final[frozenset[str]] = frozenset(
    {"effort", "reasoning_effort"}
)


# ---------------------------------------------------------------------------
# Auth Method Contracts
# ---------------------------------------------------------------------------


@runtime_checkable
class ACPAuthMethodProtocol(Protocol):
    """Protocol for an auth method returned by an ACP server."""

    id: str


@runtime_checkable
class ACPAuthMethodWithTypeProtocol(Protocol):
    """Protocol for an auth method specifying a login type (e.g. terminal)."""

    id: str
    type: str | None


@dataclass(frozen=True)
class ACPAuthMethod:
    """Normalized ACP auth method descriptor."""

    id: str
    type: str | None = None


def normalize_auth_method(raw: Any) -> ACPAuthMethod:
    """Normalize raw ACP auth method at the external-library boundary."""
    if isinstance(raw, ACPAuthMethod):
        return raw
    if isinstance(raw, Mapping):
        return ACPAuthMethod(
            id=str(raw.get("id", "")),
            type=raw.get("type") if isinstance(raw.get("type"), str) else None,
        )
    if isinstance(raw, ACPAuthMethodWithTypeProtocol):
        return ACPAuthMethod(
            id=str(raw.id),
            type=raw.type if isinstance(raw.type, str) else None,
        )
    if isinstance(raw, ACPAuthMethodProtocol):
        return ACPAuthMethod(id=str(raw.id), type=None)
    return ACPAuthMethod(id="", type=None)


# ---------------------------------------------------------------------------
# Model and Session Config Option Contracts
# ---------------------------------------------------------------------------


@runtime_checkable
class ACPConfigOptionBasicProtocol(Protocol):
    """Minimal protocol for any ACP config option."""

    id: str
    type: str


@runtime_checkable
class ACPConfigOptionValueProtocol(Protocol):
    """Protocol for an option carrying a current selection."""

    current_value: str | None


@runtime_checkable
class ACPConfigOptionOptionsProtocol(Protocol):
    """Protocol for a select option carrying available options."""

    options: list[Any]


@runtime_checkable
class ACPConfigOptionCategoryProtocol(Protocol):
    """Protocol for a config option carrying an ACP ``category``."""

    category: Any


@runtime_checkable
class ACPSelectOptionValueProtocol(Protocol):
    """Protocol for one selectable value of a select option."""

    value: Any


@runtime_checkable
class ACPRootModelProtocol(Protocol):
    """Protocol for Pydantic RootModel-wrapped options (agent-client-protocol 0.8.x)."""

    root: Any


@runtime_checkable
class ACPSessionConfigOptionsResponseProtocol(Protocol):
    """Protocol for session responses exposing config options."""

    config_options: list[Any] | None


@runtime_checkable
class ACPCurrentModelIdProtocol(Protocol):
    """Protocol for objects carrying a current_model_id attribute."""

    current_model_id: str | None


@runtime_checkable
class ACPAvailableModelsProtocol(Protocol):
    """Protocol for objects carrying an available_models list."""

    available_models: list[Any] | None


@runtime_checkable
class ACPSessionModelsResponseProtocol(Protocol):
    """Protocol for session responses exposing the UNSTABLE models block."""

    models: Any | None


class ACPLegacyModelInfo(BaseModel):
    """One entry of the UNSTABLE ``models`` block, as servers send it."""

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    model_id: str = Field(alias="modelId")
    name: str | None = None
    description: str | None = None


class ACPLegacySessionModels(BaseModel):
    """The UNSTABLE ``models`` block, which the ACP 0.12 schema no longer models."""

    model_config = ConfigDict(populate_by_name=True)

    current_model_id: str | None = Field(default=None, alias="currentModelId")
    available_models: list[ACPLegacyModelInfo] = Field(
        default_factory=list, alias="availableModels"
    )


@runtime_checkable
class ACPLegacyModelSwitchConnection(Protocol):
    """Protocol for connections supporting the legacy set_session_model RPC."""

    async def set_session_model(self, *, model_id: str, session_id: str) -> Any: ...


def supports_legacy_model_switch(conn: Any) -> bool:
    """Return True if conn supports the legacy set_session_model RPC."""
    return isinstance(conn, ACPLegacyModelSwitchConnection) or hasattr(
        conn, "set_session_model"
    )


@dataclass(frozen=True)
class ACPConfigSelectOption:
    """Normalized config select option from a session response."""

    id: str
    type: str
    current_value: str | None = None
    options: list[Any] = field(default_factory=list)
    category: str | None = None


@dataclass(frozen=True)
class ACPSessionModelState:
    """Model state extracted from a session response at the boundary."""

    current_model_id: str | None
    available_models: list[ACPModelInfo] | None
    via_config_option: bool


def _flatten_select_options(raw_options: Any) -> list[Any]:
    """Flatten grouped select options (``SessionConfigSelectGroup``) into a
    single list of ``{value, name, description}`` entries."""
    if not isinstance(raw_options, (list, tuple)):
        return []
    flat: list[Any] = []
    for raw in raw_options:
        entry = raw.root if isinstance(raw, ACPRootModelProtocol) else raw
        if isinstance(entry, Mapping):
            nested = entry.get("options")
        elif isinstance(entry, ACPConfigOptionOptionsProtocol):
            nested = entry.options
        else:
            nested = None
        if isinstance(nested, (list, tuple)):
            flat.extend(_flatten_select_options(nested))
        else:
            flat.append(entry)
    return flat


def _category_value(raw: Any) -> str | None:
    value = raw.value if isinstance(raw, Enum) else raw
    return value if isinstance(value, str) and value else None


def extract_config_select_options(response: Any) -> list[ACPConfigSelectOption] | None:
    """Normalize every ``select`` config option on a session response.

    Accepts ``session/new``, ``load_session`` and ``set_config_option``
    responses and ``config_option_update`` notifications, across
    agent-client-protocol 0.8.x (RootModel wrapped) and 0.10.x+ (direct union
    members) as well as dictionary payloads. Returns ``None`` when the response
    carries no config options at all, so callers can tell "not reported" from
    "reported, but this option is absent".
    """
    if response is None:
        return None

    raw_options: Any = None
    if isinstance(response, ACPSessionConfigOptionsResponseProtocol):
        raw_options = response.config_options
    elif isinstance(response, Mapping):
        raw_options = response.get("config_options")

    if not raw_options:
        return None

    selects: list[ACPConfigSelectOption] = []
    for raw in raw_options:
        opt = raw.root if isinstance(raw, ACPRootModelProtocol) else raw
        if isinstance(opt, Mapping):
            if opt.get("type") != "select" or not opt.get("id"):
                continue
            val = opt.get("current_value")
            selects.append(
                ACPConfigSelectOption(
                    id=str(opt.get("id")),
                    type="select",
                    current_value=val if isinstance(val, str) else None,
                    options=_flatten_select_options(opt.get("options")),
                    category=_category_value(opt.get("category")),
                )
            )
        elif isinstance(opt, ACPConfigOptionBasicProtocol):
            if opt.type != "select":
                continue
            val = (
                opt.current_value
                if isinstance(opt, ACPConfigOptionValueProtocol)
                else None
            )
            opts = (
                opt.options if isinstance(opt, ACPConfigOptionOptionsProtocol) else []
            )
            selects.append(
                ACPConfigSelectOption(
                    id=str(opt.id),
                    type=str(opt.type),
                    current_value=val if isinstance(val, str) else None,
                    options=_flatten_select_options(opts),
                    category=(
                        _category_value(opt.category)
                        if isinstance(opt, ACPConfigOptionCategoryProtocol)
                        else None
                    ),
                )
            )
    return selects


def extract_model_config_option(response: Any) -> ACPConfigSelectOption | None:
    """Extract the model select option from a session response at the boundary.

    Normalizes across agent-client-protocol 0.8.x (RootModel wrapped) and 0.10.x+
    (direct union members), as well as dictionary payloads.
    """
    for opt in extract_config_select_options(response) or []:
        if opt.id == _MODEL_CONFIG_OPTION_ID:
            return opt
    return None


def _option_value(raw: Any) -> str | None:
    if isinstance(raw, Mapping):
        value = raw.get("value")
    elif isinstance(raw, ACPSelectOptionValueProtocol):
        value = raw.value
    else:
        value = None
    return value if isinstance(value, str) and value else None


@dataclass(frozen=True)
class ACPSessionEffortState:
    """Reasoning-effort state extracted from a session response at the boundary.

    Like :class:`ACPSessionModelState`, ``available_efforts`` distinguishes
    absent from empty:

    - ``None`` -- the response carried no config options (state unknown).
    - ``[]``   -- it carried config options but none selects an effort, e.g.
      a model without effort levels (claude-agent-acp drops ``effort`` for
      Haiku).
    """

    config_id: str | None
    current_effort: str | None
    available_efforts: list[str] | None


def extract_session_efforts(response: Any) -> ACPSessionEffortState:
    """Extract the reasoning-effort select off a session response.

    The select is recognized by the ACP ``thought_level`` category, falling
    back to the known ids (``effort``, ``reasoning_effort``) for servers that
    omit the category.
    """
    selects = extract_config_select_options(response)
    if selects is None:
        return ACPSessionEffortState(None, None, None)
    for opt in selects:
        if (
            opt.category == _THOUGHT_LEVEL_CATEGORY
            or opt.id in EFFORT_CONFIG_OPTION_IDS
        ):
            values = [v for o in opt.options if (v := _option_value(o))]
            current = opt.current_value if opt.current_value else None
            return ACPSessionEffortState(opt.id, current, values)
    return ACPSessionEffortState(None, None, [])


def extract_session_models(
    response: Any,
    *,
    default_via_config_option: bool = False,
) -> ACPSessionModelState:
    """Extract model state off a session response into a typed contract.

    Handles both the newer ``configOptions`` model select mechanism and the
    UNSTABLE legacy ``models`` block, returning an :class:`ACPSessionModelState`.
    """
    if response is None:
        return ACPSessionModelState(
            current_model_id=None,
            available_models=None,
            via_config_option=default_via_config_option,
        )

    # 1. Prefer configOptions
    opt = extract_model_config_option(response)
    if opt is not None:
        current = opt.current_value
        current = current if isinstance(current, str) and current else None
        usable = [
            info
            for o in opt.options
            if (info := ACPModelInfo.from_protocol(o, id_attr="value")).model_id
        ]
        return ACPSessionModelState(
            current_model_id=current,
            available_models=usable,
            via_config_option=True,
        )

    # 2. Fall back to legacy models capability
    models_block = None
    if isinstance(response, ACPSessionModelsResponseProtocol):
        models_block = response.models
    elif isinstance(response, Mapping):
        models_block = response.get("models")

    if models_block is not None:
        current = None
        raw_list: list[Any] = []
        if isinstance(models_block, Mapping):
            current = models_block.get("current_model_id")
            raw_val = models_block.get("available_models")
            if isinstance(raw_val, (list, tuple)):
                raw_list = list(raw_val)
        else:
            if isinstance(models_block, ACPCurrentModelIdProtocol):
                current = models_block.current_model_id
            if isinstance(models_block, ACPAvailableModelsProtocol):
                raw_val = models_block.available_models
                if isinstance(raw_val, (list, tuple)):
                    raw_list = list(raw_val)

        current = current if isinstance(current, str) and current else None
        usable = [
            info for m in raw_list if (info := ACPModelInfo.from_protocol(m)).model_id
        ]
        return ACPSessionModelState(
            current_model_id=current,
            available_models=usable,
            via_config_option=False,
        )

    return ACPSessionModelState(
        current_model_id=None,
        available_models=None,
        via_config_option=default_via_config_option,
    )


# ---------------------------------------------------------------------------
# MCP Capabilities Contracts
# ---------------------------------------------------------------------------


@runtime_checkable
class ACPMcpHttpCapabilityProtocol(Protocol):
    """Protocol for MCP capabilities advertising http transport."""

    http: bool


@runtime_checkable
class ACPMcpSseCapabilityProtocol(Protocol):
    """Protocol for MCP capabilities advertising sse transport."""

    sse: bool


@dataclass(frozen=True)
class ACPMcpCapabilities:
    """Normalized MCP capabilities from an ACP initialize response."""

    http: bool = False
    sse: bool = False


def normalize_mcp_capabilities(raw: Any) -> ACPMcpCapabilities:
    """Normalize raw MCP capabilities at the boundary."""
    if isinstance(raw, ACPMcpCapabilities):
        return raw
    if isinstance(raw, Mapping):
        return ACPMcpCapabilities(
            http=bool(raw.get("http", False)),
            sse=bool(raw.get("sse", False)),
        )
    http_ok = bool(raw.http) if isinstance(raw, ACPMcpHttpCapabilityProtocol) else False
    sse_ok = bool(raw.sse) if isinstance(raw, ACPMcpSseCapabilityProtocol) else False
    return ACPMcpCapabilities(http=http_ok, sse=sse_ok)


# ---------------------------------------------------------------------------
# Error Info Contracts
# ---------------------------------------------------------------------------


@runtime_checkable
class ACPErrorCodeProtocol(Protocol):
    """Protocol for exceptions carrying a JSON-RPC error code."""

    code: int | None


@runtime_checkable
class ACPErrorDataProtocol(Protocol):
    """Protocol for exceptions carrying an error data payload."""

    data: Any


@dataclass(frozen=True)
class ACPErrorInfo:
    """Typed error details extracted from an ACP exception at the boundary."""

    code: int | None
    message: str
    data: Any


def normalize_acp_error(exc: BaseException) -> ACPErrorInfo:
    """Extract structured error information at the exception boundary."""
    code = (
        exc.code
        if isinstance(exc, ACPErrorCodeProtocol) and isinstance(exc.code, int)
        else None
    )
    data = exc.data if isinstance(exc, ACPErrorDataProtocol) else None
    return ACPErrorInfo(code=code, message=str(exc), data=data)


# ---------------------------------------------------------------------------
# File Credential Binding Revision Contracts
# ---------------------------------------------------------------------------


@runtime_checkable
class ACPRevisionedCredentialBinding(Protocol):
    """Protocol for credential bindings that expose authorization_revision."""

    @property
    def authorization_revision(self) -> int | None: ...


# ---------------------------------------------------------------------------
# Serialization / Tracing Contracts
# ---------------------------------------------------------------------------


@runtime_checkable
class ACPModelDumpable(Protocol):
    """Protocol for objects supporting Pydantic-style model_dump."""

    def model_dump(self, *args: Any, **kwargs: Any) -> Any: ...


def is_model_dumpable(obj: Any) -> bool:
    """Return True if obj is a BaseModel or implements model_dump()."""
    return isinstance(obj, (BaseModel, ACPModelDumpable)) or hasattr(obj, "model_dump")
