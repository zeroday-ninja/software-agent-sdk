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
"""

from __future__ import annotations

import re


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
