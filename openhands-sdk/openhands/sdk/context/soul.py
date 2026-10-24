"""The user's identity file (``SOUL.md``).

``SOUL.md`` sits at the root of the user persistence directory
(``~/.openhands/SOUL.md`` absent ``OH_PERSISTENCE_DIR``). When it has content it
replaces the built-in identity line at the top of the system prompt.

The path is resolved per call rather than frozen in a module constant, so a
process that sets ``OH_PERSISTENCE_DIR`` after importing the SDK still reads and
writes the same file, and so every reader (the prompt loader, the ACP adapter,
the agent-server endpoint) resolves one location.
"""

from pathlib import Path
from typing import Final

from openhands.sdk.context.identity import (
    IDENTITY_DOCUMENTS,
    identity_document_path,
    legacy_soul_path,
    read_identity_document,
    read_identity_prompt,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

__all__ = [
    "DEFAULT_SOUL",
    "get_soul_path",
    "load_soul",
    "read_user_soul",
    "render_soul_block",
]

DEFAULT_SOUL: Final[str] = (
    "You are OpenHands agent, a helpful AI assistant that can interact"
    " with a computer to solve tasks."
)


def get_soul_path() -> Path:
    """Absolute path of the user's ``SOUL.md``.

    ``SOUL.md`` is now one document in a set under ``soul/`` (see
    :mod:`openhands.sdk.context.identity`), but an install that predates the set
    has it at the old location and nothing has written since. So: the set's copy
    when it exists, the old path otherwise. The first write moves the old file
    into the set, which is what keeps there from being two files where only one
    is read.
    """
    document = IDENTITY_DOCUMENTS[0]
    in_set = identity_document_path(document)
    if in_set.exists():
        return in_set
    legacy = legacy_soul_path()
    return legacy if legacy.exists() else in_set


def read_user_soul() -> str | None:
    """The user's own identity text, or ``None`` when they have not written one.

    ``None`` covers a missing, unreadable, empty, or whitespace-only file, so
    callers that must not emit the built-in placeholder (ACP runtimes, a settings
    UI showing whether the user has customized anything) can tell the two apart.
    """
    return read_identity_document(IDENTITY_DOCUMENTS[0])


def load_soul() -> str:
    """The user's identity, falling back to :data:`DEFAULT_SOUL`.

    The whole document set, not only ``SOUL.md``: how someone writes and what
    has already been established with them change an answer at least as much
    as who they are, and a set that can be edited but never reaches a prompt
    is a settings page that lies.
    """
    return read_identity_prompt() or DEFAULT_SOUL


def render_soul_block(soul: str) -> str:
    """The ``<SOUL>`` prompt block. Shared by the static and dynamic tiers so an
    ACP runtime receives the identity text the regular agent does, verbatim."""
    return f"<SOUL>\n{soul}\n</SOUL>"
