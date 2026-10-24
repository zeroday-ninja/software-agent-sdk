"""The user's identity: a small set of documents, not one file.

``SOUL.md`` answers "who is this person"; it was never going to answer "how do
they write", "what have we already learned", or "what does good work look like
here". Those are different questions with different lifetimes — voice changes
slowly, memory is append-only, examples are calibration — so they are separate
documents under ``~/.openhands/soul/``:

    SOUL.md      identity, worldview, opinions
    STYLE.md     voice, syntax, patterns, anti-patterns
    MEMORY.md    continuity across sessions; entries are appended, never edited
    data/        raw source material, plus influences.md
    examples/    good-outputs.md and bad-outputs.md, for calibration

The load order is deliberate and is the one the upstream soul.md project
recommends: identity, then voice, then recent memory, then examples, and raw
data only when something needs grounding. Identity and voice go ahead of tool
definitions in a prompt, because a model that has read neither will answer in
nobody's voice.

Nothing here is required. A set with only ``SOUL.md`` behaves exactly as the
single file did, which is what every existing install has.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from openhands.sdk.logger import get_logger
from openhands.sdk.utils.files import atomic_write_text
from openhands.sdk.utils.path import get_user_persistence_dir, resolves_within


logger = get_logger(__name__)

__all__ = [
    "IDENTITY_DIR_NAME",
    "IDENTITY_DOCUMENTS",
    "IdentityDocument",
    "identity_dir",
    "identity_document_path",
    "known_identity_document",
    "legacy_soul_path",
    "migrate_legacy_soul",
    "read_identity_document",
    "read_identity_set",
    "write_identity_document",
]

IDENTITY_DIR_NAME: Final[str] = "soul"

#: Largest a single identity document may be, in characters.
#:
#: Per document rather than per set: the set is read in full on every
#: conversation, and one runaway file should not be able to crowd out the rest
#: of someone's identity — or the task.
MAX_DOCUMENT_LENGTH: Final[int] = 65536


@dataclass(frozen=True, slots=True)
class IdentityDocument:
    """One document in the set, and what it is for."""

    #: Stable id used in the API and on disk, e.g. ``"style"`` -> ``STYLE.md``.
    name: str
    #: Path relative to the identity directory.
    relative_path: str
    #: What a reader should put in it. Shown in the UI; never sent to a model.
    purpose: str
    #: Where this document sits in the prompt. Lower is earlier.
    order: int


#: The documents this build knows how to read, in load order.
#:
#: A closed set rather than "whatever is in the directory": each document has a
#: place in the prompt and a size budget, and a file nobody declared has
#: neither. Raw material under ``data/`` is deliberately not in here — it is
#: for grounding a specific question, not for every prompt.
IDENTITY_DOCUMENTS: Final[tuple[IdentityDocument, ...]] = (
    IdentityDocument(
        name="soul",
        relative_path="SOUL.md",
        purpose=(
            "Who you are: background, worldview, opinions by domain, "
            "interests, current focus, influences, vocabulary, tensions, "
            "boundaries."
        ),
        order=0,
    ),
    IdentityDocument(
        name="style",
        relative_path="STYLE.md",
        purpose=(
            "How you write and work: voice, words you use and never use, "
            "punctuation, rhetorical moves, anti-patterns."
        ),
        order=1,
    ),
    IdentityDocument(
        name="memory",
        relative_path="MEMORY.md",
        purpose=(
            "Continuity across sessions. Entries are appended with an id and "
            "a timestamp so two machines can merge without losing any."
        ),
        order=2,
    ),
    IdentityDocument(
        name="good-outputs",
        relative_path="examples/good-outputs.md",
        purpose="Work that sounds like you, for calibration.",
        order=3,
    ),
    IdentityDocument(
        name="bad-outputs",
        relative_path="examples/bad-outputs.md",
        purpose="Work that does not, and what is wrong with it.",
        order=4,
    ),
    IdentityDocument(
        name="influences",
        relative_path="data/influences.md",
        purpose="What shaped your thinking, and what you took from each.",
        order=5,
    ),
)


def known_identity_document(name: str) -> IdentityDocument | None:
    """The declared document with this id, or ``None``."""
    for document in IDENTITY_DOCUMENTS:
        if document.name == name:
            return document
    return None


def identity_dir() -> Path:
    """Absolute path of the identity directory."""
    return get_user_persistence_dir() / IDENTITY_DIR_NAME


def legacy_soul_path() -> Path:
    """Where ``SOUL.md`` lived before the set existed."""
    return get_user_persistence_dir() / "SOUL.md"


def identity_document_path(document: IdentityDocument) -> Path:
    """Absolute path of one document.

    Resolved per call rather than frozen at import, so a process that sets
    ``OH_PERSISTENCE_DIR`` late still reads and writes one location — the same
    reason the single-file module resolved its path per call.
    """
    return identity_dir() / document.relative_path


def _read(path: Path) -> str | None:
    """File contents, or ``None`` when there is nothing to read.

    ``None`` covers missing, unreadable, empty and whitespace-only alike, so a
    caller that must not emit a placeholder — an ACP runtime, a settings page
    showing whether anything has been written — can tell "nothing here" from
    "something here".
    """
    try:
        content = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        # Surfaced, not swallowed: a document that exists but cannot be read is
        # a different situation from one that was never written, and a prompt
        # that quietly drops someone's identity is the kind of failure nobody
        # notices until the output is wrong.
        logger.warning("Could not read identity document %s: %s", path, exc)
        return None
    return content or None


def read_identity_document(document: IdentityDocument) -> str | None:
    """One document's text, or ``None`` when it has not been written.

    ``SOUL.md`` falls back to the pre-set location, so an install that has only
    ever had the single file keeps working untouched until something writes.
    """
    content = _read(identity_document_path(document))
    if content is not None or document.name != "soul":
        return content
    return _read(legacy_soul_path())


def read_identity_set() -> dict[str, str]:
    """Every document that has been written, keyed by id, in load order."""
    found: dict[str, str] = {}
    for document in sorted(IDENTITY_DOCUMENTS, key=lambda item: item.order):
        content = read_identity_document(document)
        if content is not None:
            found[document.name] = content
    return found


def migrate_legacy_soul() -> Path | None:
    """Move a pre-set ``SOUL.md`` into the set. Returns where it went.

    Called before the first write. Reads already fall back to the old path, so
    this is about not leaving two files where only one is read: after the move
    there is exactly one ``SOUL.md`` and it is the one everything uses.
    """
    legacy = legacy_soul_path()
    target = identity_document_path(IDENTITY_DOCUMENTS[0])
    if target.exists() or not legacy.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    legacy.replace(target)
    logger.info("Moved %s into the identity set at %s", legacy, target)
    return target


def write_identity_document(document: IdentityDocument, content: str) -> Path:
    """Write one document, or delete it when the content is blank.

    Blank deletes rather than writing an empty file, so "never written" and
    "written, then emptied" stay one state — a settings form that always sends
    its whole textarea should not be able to create a document that exists and
    says nothing.

    The write is atomic, so an agent starting mid-write reads either the old
    document or the new one, never half of either. Raises ``ValueError`` when
    the resolved path would escape the identity directory, which a declared
    document's relative path should never do but is cheap to refuse rather
    than to trust.
    """
    migrate_legacy_soul()
    path = identity_document_path(document)
    root = identity_dir()
    root.mkdir(parents=True, exist_ok=True)
    # Argument order matters here: the child is checked against the root.
    if not resolves_within(path, root):
        raise ValueError(f"{path} is outside the identity directory")
    if not content.strip():
        path.unlink(missing_ok=True)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, content.strip() + "\n")
    return path
