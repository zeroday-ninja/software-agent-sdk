"""An ACP adapter met something it does not understand."""

from __future__ import annotations

from pydantic import Field
from rich.text import Text

from openhands.sdk.event.base import Event
from openhands.sdk.event.types import SourceType


#: The extension slot an adapter uses to mark a chunk as an incident report.
UNRECOGNISED_META_KEY = "openhands.unrecognised"


class ACPAdapterIncidentEvent(Event):
    """An adapter translating a CLI's stream met an event or step it has no
    translation for.

    The adapter says so twice: as a sentence in the assistant's own text, so
    the user sees it in chat, and as a typed field in the protocol's
    extension slot, so a machine can count it. Until this event existed the
    field was read by nothing — the bridge kept the chunk's text and let the
    field go, so "surfaced as a typed error rather than dropped" was true on
    the wire and false one hop later. This is the machine-readable half, in
    the conversation's own event stream, where a UI can file it as a lesson
    against the runtime and a later cycle can find it.

    This event is not sent to the LLM.
    """

    source: SourceType = "environment"

    adapter: str = Field(
        description="Provider registry key of the adapter that met it "
        "('antigravity', ...), as the adapter named itself."
    )
    what: str = Field(
        description="What was not understood: 'event' for a top-level stream "
        "event, 'step' for a step inside one."
    )
    unrecognised_kind: str | None = Field(
        default=None,
        description="The unrecognised event or step type, as the CLI named "
        "it. None when it carried no type at all. Not ``kind``: that is every "
        "event's own discriminator.",
    )
    detail: str = Field(
        default="",
        description="The sentence the adapter put in the assistant's text for "
        "the same incident, already secret-redacted. Empty when it said nothing.",
    )

    @property
    def visualize(self) -> Text:
        text = Text()
        text.append("Adapter incident: ", style="bold")
        text.append(f"{self.adapter} met a {self.what} it does not understand")
        if self.unrecognised_kind is not None:
            text.append(f": {self.unrecognised_kind!r}")
        return text


def unrecognised_incident(meta: object, detail: str = "") -> dict | None:
    """The incident an adapter's ``_meta`` describes, or None.

    Tolerant of shape, because the field is set by adapters this build may
    not know: anything that is a mapping under :data:`UNRECOGNISED_META_KEY`
    with a ``what`` counts, and missing names fall back rather than raise.
    """
    if not isinstance(meta, dict):
        return None
    marked = meta.get(UNRECOGNISED_META_KEY)
    if not isinstance(marked, dict):
        return None
    what = marked.get("what")
    if not isinstance(what, str) or not what:
        return None
    kind = marked.get("kind")
    adapter = marked.get("adapter")
    return {
        "adapter": adapter if isinstance(adapter, str) and adapter else "unknown",
        "what": what,
        "unrecognised_kind": None if kind is None else str(kind),
        "detail": detail,
    }
