"""``MEMORY.md`` as an append-only log that two machines can merge.

The other identity documents are edited: a person rewrites a sentence in
``STYLE.md`` and means the new one. Memory is not like that. It accumulates,
every entry stays true of the moment it was written, and nothing supersedes
anything. That difference is what makes memory safe to sync between backends
without a three-way merge: if entries are only ever added, a merge is the union
of two sets and nothing can be lost.

So an entry carries an id and a timestamp:

    ## 2026-10-24T08:15:00Z — Prefers a judge agent on a second runtime
    <!-- id: mem-3f9a2c -->

    Body, as much as it needs, in Markdown.

The id is what makes the union exact. An entry written by hand without one
still merges: its id is derived from its own text, so the same entry arriving
from two machines is recognised as one entry rather than duplicated.

What this module refuses to do is resolve a *conflict* silently. Two entries
sharing an id with different bodies means something edited history, which this
format says does not happen — so both are kept and the caller is told, rather
than one being dropped where nobody would see it go.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final


__all__ = [
    "MemoryEntry",
    "MemoryMerge",
    "derive_entry_id",
    "format_memory_log",
    "merge_memory_logs",
    "parse_memory_log",
]

_HEADING: Final[re.Pattern[str]] = re.compile(
    r"^##\s+(?P<timestamp>\S+)\s+—\s+(?P<title>.*)$"
)
_ID_COMMENT: Final[re.Pattern[str]] = re.compile(
    r"^<!--\s*id:\s*(?P<id>[A-Za-z0-9._-]+)\s*-->\s*$"
)

#: Length of a derived id's hash. Short enough to read, long enough that two
#: different entries colliding is not a thing that happens in one person's log.
_DERIVED_ID_CHARS: Final[int] = 8


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """One thing worth remembering, and when it was learned."""

    id: str
    timestamp: str
    title: str
    body: str

    def sort_key(self) -> tuple[str, str]:
        """Oldest first, with the id breaking ties so a merge is stable."""
        return (self.timestamp, self.id)


@dataclass(slots=True)
class MemoryMerge:
    """The union of two logs, and anything that did not merge cleanly."""

    entries: list[MemoryEntry] = field(default_factory=list)
    #: Ids that arrived twice with different text. Both are kept.
    conflicts: list[str] = field(default_factory=list)


def derive_entry_id(timestamp: str, title: str, body: str) -> str:
    """A stable id for an entry that was written without one."""
    digest = hashlib.sha256(
        "\n".join((timestamp, title, body)).encode("utf-8")
    ).hexdigest()
    return f"mem-{digest[:_DERIVED_ID_CHARS]}"


def _normalize_timestamp(raw: str) -> str:
    """An ISO instant, or the epoch when the heading's date is unreadable.

    A hand-written heading with a date nobody can parse still has to sort
    somewhere, and sorting it first keeps it visible rather than hiding it at
    the end of a long log.
    """
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return "1970-01-01T00:00:00Z"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_memory_log(text: str) -> list[MemoryEntry]:
    """Every entry in a log, in the order they appear.

    Anything before the first entry heading — a title, a paragraph explaining
    what the file is — is not an entry and is not returned. It is also not
    preserved by :func:`format_memory_log`, which is deliberate: the file is a
    log, and a merge that had to reconcile two different preambles would be
    exactly the three-way merge this format exists to avoid.
    """
    entries: list[MemoryEntry] = []
    timestamp: str | None = None
    title = ""
    entry_id: str | None = None
    body: list[str] = []

    def flush() -> None:
        if timestamp is None:
            return
        text_body = "\n".join(body).strip()
        entries.append(
            MemoryEntry(
                id=entry_id or derive_entry_id(timestamp, title, text_body),
                timestamp=timestamp,
                title=title,
                body=text_body,
            )
        )

    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading is not None:
            flush()
            timestamp = _normalize_timestamp(heading.group("timestamp"))
            title = heading.group("title").strip()
            entry_id = None
            body = []
            continue
        if timestamp is None:
            continue
        comment = _ID_COMMENT.match(line.strip())
        if comment is not None and entry_id is None and not body:
            entry_id = comment.group("id")
            continue
        body.append(line)
    flush()
    return entries


def format_memory_log(entries: list[MemoryEntry]) -> str:
    """The log as it goes on disk: oldest first, one entry per heading."""
    blocks: list[str] = []
    for entry in sorted(entries, key=MemoryEntry.sort_key):
        block = f"## {entry.timestamp} — {entry.title}\n<!-- id: {entry.id} -->"
        if entry.body:
            block = f"{block}\n\n{entry.body}"
        blocks.append(block)
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def merge_memory_logs(*logs: str) -> MemoryMerge:
    """The union of several logs of the same memory.

    Nothing is overwritten and nothing is dropped. An entry present in one log
    and not another is kept; an entry in both is kept once. An id carrying two
    different bodies is kept twice, under suffixed ids, and reported — because
    the alternative is deciding on the user's behalf which version of their own
    memory to throw away.
    """
    merged: dict[str, MemoryEntry] = {}
    conflicts: list[str] = []
    for log in logs:
        for entry in parse_memory_log(log):
            seen = merged.get(entry.id)
            if seen is None:
                merged[entry.id] = entry
                continue
            if seen.body == entry.body and seen.title == entry.title:
                # The same entry from two machines. Keep the earlier
                # timestamp: that is when it was actually learned.
                if entry.timestamp < seen.timestamp:
                    merged[entry.id] = entry
                continue
            conflicts.append(entry.id)
            suffix = 2
            while f"{entry.id}-{suffix}" in merged:
                suffix += 1
            merged[f"{entry.id}-{suffix}"] = MemoryEntry(
                id=f"{entry.id}-{suffix}",
                timestamp=entry.timestamp,
                title=entry.title,
                body=entry.body,
            )
    return MemoryMerge(
        entries=sorted(merged.values(), key=MemoryEntry.sort_key),
        conflicts=sorted(set(conflicts)),
    )
