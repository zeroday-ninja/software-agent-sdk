"""The ACP runtime behind a conversation was replaced while it was running."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from rich.text import Text

from openhands.sdk.agent.acp_usage_limit import ResetHint
from openhands.sdk.event.base import Event
from openhands.sdk.event.types import SourceType


ACPRuntimeSwitchReason = Literal["usage_limit", "manual"]
"""Why the runtime changed. See :attr:`ACPRuntimeSwitchEvent.reason`."""


class ACPRuntimeSwitchEvent(Event):
    """One ACP runtime took over a conversation from another, in place.

    Emitted by ``LocalConversation`` once the new runtime is live, both when a
    usage-limit failover chose it and when the user asked for it through
    ``switch_acp_server``. The conversation keeps its id, workspace, and event
    history; only the subprocess answering prompts changed.

    A failover also narrates itself as assistant text so the user sees it in
    chat. This event is the machine-readable half of that: downstream
    consumers read these fields instead of scraping the prose.

    This event is not sent to the LLM.
    """

    source: SourceType = "environment"

    from_server: str = Field(
        description="Provider registry key of the runtime that was replaced "
        "('claude-code', 'codex', ..., or 'custom' for a raw command)."
    )
    to_server: str = Field(
        description="Provider registry key of the runtime now answering prompts."
    )
    to_model: str | None = Field(
        default=None,
        description="Model the new runtime was asked for, optionally "
        "'<model>/<effort>'. None means it picks its own default.",
    )
    reason: ACPRuntimeSwitchReason = Field(
        description="'usage_limit' for a failover off an exhausted runtime, "
        "'manual' when the user switched runtime themselves."
    )
    detail: str = Field(
        default="",
        description="The replaced runtime's own refusal text, already "
        "secret-redacted. Empty for a manual switch, which has no refusal.",
    )
    reset_hint: ResetHint | None = Field(
        default=None,
        description="When the replaced runtime's usage window reopens, as far "
        "as its notice could be read. None when nothing was parsed at all; a "
        "hint whose 'at' is None means the notice named no time.",
    )

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this runtime switch."""
        content = Text()
        content.append(f"{self.from_server} -> {self.to_server}", style="bold")
        if self.to_model:
            content.append(f" ({self.to_model})")
        content.append("\nReason: ", style="bold")
        content.append(self.reason)
        if self.detail:
            content.append("\n\nDetail:\n", style="bold")
            content.append(self.detail)
        if self.reset_hint is not None and self.reset_hint.at is not None:
            content.append("\n\nResets: ", style="bold")
            content.append(
                f"{self.reset_hint.at.isoformat()} ({self.reset_hint.confidence})"
            )
        return content
