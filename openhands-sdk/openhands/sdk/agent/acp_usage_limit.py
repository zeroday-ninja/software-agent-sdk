"""Recognise an ACP agent's "usage limit reached" failures.

Subscription-backed ACP CLIs stop answering when the account's usage window
is exhausted. Each reports it differently:

- claude-agent-acp ends the prompt with a JSON-RPC ``-32603`` whose message
  carries Claude Code's synthetic usage-limit text (``You've hit your limit ·
  resets 3pm``, ``You're out of extra usage``, ...). The prefixes below mirror
  ``USAGE_LIMIT_ERROR_PREFIXES`` in ``@anthropic-ai/claude-agent-sdk``.
- codex-acp ends the prompt with a ``-32603`` whose ``data`` names
  ``usageLimitExceeded`` and carries ``You've hit your usage limit ...``.

:class:`ACPUsageLimitError` is how :class:`~openhands.sdk.agent.ACPAgent`
hands such a failure to the conversation when a fallback runtime is
configured, so the conversation can switch runtime instead of ending in error.

Claude Code's notice also says when the window reopens. :func:`parse_reset_hint`
reads that out as a :class:`ResetHint` so clients can say "back at 3pm" instead
of re-parsing the prose.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field


# Text markers (casefolded) that identify an exhausted usage window. Kept
# conservative: a transient API 429 that the CLI already retried and gave up on
# is included, since to the user it is the same "this runtime won't answer".
USAGE_LIMIT_MARKERS: tuple[str, ...] = (
    # claude-agent-sdk USAGE_LIMIT_ERROR_PREFIXES
    "you've hit your",
    "you’ve hit your",
    "you've reached your",
    "you’ve reached your",
    "you're out of usage credits",
    "you’re out of usage credits",
    "out of usage",
    "doesn't include usage",
    "doesn’t include usage",
    "usage allocation has been disabled",
    "usage limit",
    "requires usage credits",
    "out of extra usage",
    # codex / OpenAI
    "usagelimitexceeded",
    "usage_limit_exceeded",
    "usage_limit_reached",
    "insufficient_quota",
    "insufficient quota",
    "credits_depleted",
    "quota exceeded",
    "exceeded your current quota",
    # generic
    "rate limit",
    "rate_limit",
)

_HTTP_429_RE = re.compile(r"\b429\b")

# Longer than this and the text is a real answer that happens to mention a
# limit, not the CLI's own short refusal.
_MAX_NOTICE_CHARS = 600


def text_indicates_usage_limit(text: str | None) -> bool:
    """True when ``text`` (an error message or data payload) names a usage limit."""
    if not text:
        return False
    folded = text.casefold()
    if any(marker in folded for marker in USAGE_LIMIT_MARKERS):
        return True
    return bool(_HTTP_429_RE.search(folded))


def response_indicates_usage_limit(text: str | None) -> bool:
    """True when a *successful* turn's whole answer is a usage-limit notice.

    Some CLIs forward the notice as ordinary agent text and end the turn
    normally. Only a short answer that *starts* with a known notice counts, so
    an agent that merely discusses rate limits is never mistaken for one.
    """
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) > _MAX_NOTICE_CHARS:
        return False
    folded = stripped.casefold()
    return any(folded.startswith(marker) for marker in USAGE_LIMIT_MARKERS)


ResetConfidence = Literal["exact", "approximate", "unknown"]
"""How much a :class:`ResetHint` can be trusted. See :attr:`ResetHint.confidence`."""


class ResetHint(BaseModel):
    """When an exhausted ACP runtime says its usage window reopens.

    A pydantic model rather than a dataclass because this rides along on
    :class:`~openhands.sdk.event.acp_runtime_switch.ACPRuntimeSwitchEvent` into
    the persisted event log and from there into the Agent Server's OpenAPI
    contract, which only serialises pydantic models.

    The default instance is the "we could not read a reset time" answer: that is
    a different statement from "there is no limit", so it is still a hint, with
    :attr:`at` unset.
    """

    model_config = ConfigDict(frozen=True)

    at: datetime | None = Field(
        default=None,
        description=(
            "UTC instant the usage window is expected to reopen, always in the "
            "future of when it was parsed. None means the notice carried no "
            "time this parser could read -- never that there is no limit."
        ),
    )
    source: str = Field(
        default="",
        description=(
            "Verbatim fragment the time was read from, for example "
            "'resets 3pm (America/New_York)'. Empty when nothing matched, so a "
            "client can show the runtime's own words rather than only an "
            "instant it has to re-render."
        ),
    )
    confidence: ResetConfidence = Field(
        default="unknown",
        description=(
            "'exact' when the notice named a time zone, so 'at' is a true "
            "instant; 'approximate' when it gave a bare wall-clock time that "
            "was read in the host's local zone and may be off by the "
            "difference; 'unknown' when 'at' is None."
        ),
    )


# Claude Code appends the reset time to its notice: "... · resets 3pm
# (America/New_York)", "... · resets 1:30pm". Anchoring on "reset" is what keeps
# a time mentioned elsewhere in a longer message from being read as one.
_RESET_TIME_RE = re.compile(
    r"""reset(?:s|ting)?\s+(?:at\s+)?
        (?P<hour>\d{1,2})
        (?::(?P<minute>\d{2}))?
        \s*(?P<meridiem>[ap]\.?m\.?)?
        (?:\s*\(\s*(?P<zone>[A-Za-z][A-Za-z0-9_+-]*(?:/[A-Za-z0-9_+-]+)*)\s*\))?
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _hour_24(hour_text: str, meridiem: str | None) -> int | None:
    """``hour_text`` as a 0-23 hour, or None when it is not an hour at all."""
    hour = int(hour_text)
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        return (hour % 12) + (12 if meridiem[0].lower() == "p" else 0)
    return hour if 0 <= hour <= 23 else None


def _named_zone(name: str | None) -> ZoneInfo | None:
    """The notice's zone, or None when it named none or named one we lack."""
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (KeyError, ValueError):
        # A zone this host has no data for is no more useful than no zone at
        # all; the caller falls back to local time and lowers its confidence.
        return None


def parse_reset_hint(text: str | None, *, now: datetime | None = None) -> ResetHint:
    """Read the reset time out of a usage-limit notice.

    Never raises: an unreadable or absent time comes back as a hint with
    :attr:`~ResetHint.at` unset, so a caller can tell "the window reopens then"
    apart from "we do not know when it reopens".

    ``now`` is the instant the "next occurrence" rule is measured from; it
    exists so that rule is testable and defaults to the current UTC instant.
    """
    if not text:
        return ResetHint()
    match = _RESET_TIME_RE.search(text)
    if match is None:
        return ResetHint()
    minute_text = match.group("minute")
    meridiem = match.group("meridiem")
    if minute_text is None and meridiem is None:
        # A bare number after "resets" is a count, not a clock: "resets 3
        # times" must not become 03:00.
        return ResetHint()
    hour = _hour_24(match.group("hour"), meridiem)
    minute = int(minute_text or 0)
    if hour is None or minute > 59:
        return ResetHint()

    zone = _named_zone(match.group("zone"))
    # astimezone(None) is local time, which is the best available reading of a
    # wall-clock time the runtime gave without a zone.
    reference = (now or datetime.now(UTC)).astimezone(zone)
    candidate = reference.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= reference:
        # "resets 3pm" at 4pm means tomorrow's 3pm: a reset hint only ever
        # points forward.
        candidate += timedelta(days=1)
    return ResetHint(
        at=candidate.astimezone(UTC),
        source=match.group(0).strip(),
        confidence="exact" if zone is not None else "approximate",
    )


class ACPUsageLimitError(Exception):
    """An ACP runtime refused a prompt because its usage limit is exhausted.

    Raised out of ``ACPAgent.step``/``astep`` *instead of* the usual error
    events when the agent has a fallback runtime, so ``LocalConversation`` can
    fail over. ``detail`` is already secret-redacted.
    """

    def __init__(self, detail: str, *, acp_server: str | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.acp_server = acp_server
