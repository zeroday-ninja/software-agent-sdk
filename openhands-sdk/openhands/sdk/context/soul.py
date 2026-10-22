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

from openhands.sdk.logger import get_logger
from openhands.sdk.utils.path import get_user_persistence_dir


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
    """Absolute path of the user's ``SOUL.md``."""
    return get_user_persistence_dir() / "SOUL.md"


def read_user_soul() -> str | None:
    """The user's own identity text, or ``None`` when they have not written one.

    ``None`` covers a missing, unreadable, empty, or whitespace-only file, so
    callers that must not emit the built-in placeholder (ACP runtimes, a settings
    UI showing whether the user has customized anything) can tell the two apart.
    """
    path = get_soul_path()
    try:
        content = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        logger.debug("Could not read SOUL.md from %s: %s", path, exc)
        return None
    return content or None


def load_soul() -> str:
    """The user's identity text, falling back to :data:`DEFAULT_SOUL`."""
    return read_user_soul() or DEFAULT_SOUL


def render_soul_block(soul: str) -> str:
    """The ``<SOUL>`` prompt block. Shared by the static and dynamic tiers so an
    ACP runtime receives the identity text the regular agent does, verbatim."""
    return f"<SOUL>\n{soul}\n</SOUL>"
