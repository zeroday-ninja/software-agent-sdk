"""Memory as an append-only log two machines can merge."""

from openhands.sdk.context.memory_log import (
    format_memory_log,
    memory_log_preamble,
    merge_memory_logs,
    parse_memory_log,
)


LOG_A = """# Memory

## 2026-10-20T08:00:00Z — Prefers a judge on a second runtime
<!-- id: mem-aaa -->

A deletion attested by the runtime that wrote it is not attested.

## 2026-10-22T09:30:00Z — Never run suites at default parallelism
<!-- id: mem-bbb -->

The machine has other work to do.
"""

LOG_B = """## 2026-10-21T11:00:00Z — Commits go to the fork, never upstream
<!-- id: mem-ccc -->

Upstream is not ours to push to.

## 2026-10-22T09:30:00Z — Never run suites at default parallelism
<!-- id: mem-bbb -->

The machine has other work to do.
"""


def test_reads_every_entry_and_ignores_the_preamble() -> None:
    # Arrange / Act
    entries = parse_memory_log(LOG_A)

    # Assert: the file's title is not an entry, and each entry keeps its id.
    assert [entry.id for entry in entries] == ["mem-aaa", "mem-bbb"]
    assert entries[0].title == "Prefers a judge on a second runtime"
    assert "not attested" in entries[0].body


def test_an_entry_written_without_an_id_still_merges_as_one_entry() -> None:
    # Arrange: the same entry, by hand, on two machines.
    written_by_hand = (
        "## 2026-10-23T07:00:00Z — Likes short commits\n\nShorter is clearer."
    )

    # Act
    merged = merge_memory_logs(written_by_hand, written_by_hand)

    # Assert: derived from its own text, so it is recognised rather than
    # duplicated — which is what makes a sync safe for a file people edit.
    assert len(merged.entries) == 1
    assert merged.conflicts == []


def test_merging_two_logs_keeps_everything_and_converges() -> None:
    # Act
    one_way = merge_memory_logs(LOG_A, LOG_B)
    other_way = merge_memory_logs(LOG_B, LOG_A)

    # Assert: the union, in one order, whichever direction the sync ran.
    assert [entry.id for entry in one_way.entries] == [
        "mem-aaa",
        "mem-ccc",
        "mem-bbb",
    ]
    assert format_memory_log(one_way.entries) == format_memory_log(other_way.entries)
    assert one_way.conflicts == []


def test_merging_is_idempotent() -> None:
    # Arrange
    once = format_memory_log(merge_memory_logs(LOG_A, LOG_B).entries)

    # Act: the same sync again, which is what a periodic sync does.
    twice = format_memory_log(merge_memory_logs(once, LOG_A, LOG_B).entries)

    # Assert
    assert twice == once


def test_a_conflict_converges_and_then_stays_put() -> None:
    """The claim this format rests on, for the case that nearly broke it.

    Convergence was covered for logs that agree and never for logs that do
    not. Two backends merged the same pair of logs in opposite orders and
    filed the same two bodies under swapped ids, so they held different
    documents — and when they then synced *with each other*, each body
    arrived under an id the other side had never seen. The conflict doubled
    every round: three syncs, four copies of one memory.
    """
    # Arrange: the same entry, with two different bodies.
    edited = LOG_A.replace("not attested", "attested by nobody")

    # Act: both directions, then the two results syncing with each other.
    one_way = format_memory_log(merge_memory_logs(LOG_A, edited).entries)
    other_way = format_memory_log(merge_memory_logs(edited, LOG_A).entries)
    after = format_memory_log(merge_memory_logs(one_way, other_way).entries)

    # Assert: the same document whichever order it happened in, and a further
    # sync between them changes nothing.
    assert one_way == other_way
    assert after == one_way
    # Neither version keeps the ambiguous id: an id carrying two bodies
    # identifies neither of them, and awarding it to whichever arrived first
    # is what made the two documents differ.
    assert "<!-- id: mem-aaa -->" not in after
    assert after.count("<!-- id: mem-aaa-") == 2


def test_one_id_with_two_bodies_is_kept_twice_and_reported() -> None:
    # Arrange: history was edited somewhere, which this format says does not
    # happen.
    edited = LOG_A.replace("not attested", "attested by nobody")

    # Act
    merged = merge_memory_logs(LOG_A, edited)

    # Assert: both survive and the caller is told, because choosing which
    # version of someone's memory to discard is not a merge's decision.
    assert merged.conflicts == ["mem-aaa"]
    bodies = [entry.body for entry in merged.entries]
    assert any("not attested" in body for body in bodies)
    assert any("attested by nobody" in body for body in bodies)


def test_the_part_a_person_wrote_by_hand_is_readable() -> None:
    """A merge rewrites the whole file, so the preamble has to survive it.

    Whatever is above the first entry — a title, a sentence saying what the
    file is for — is the only part of this log a person writes by hand, and
    nothing could read it back. Filing one correction goes through the merge,
    so dropping it was a silent deletion on the first thumbs-down.
    """
    assert memory_log_preamble(LOG_A) == "# Memory\n\n"
    # A log that starts with its first entry has none, which is the usual case.
    assert memory_log_preamble(LOG_B) == ""
    # And a file with no entries yet is all preamble rather than nothing.
    assert memory_log_preamble("# Memory\n\nNothing yet.\n") == (
        "# Memory\n\nNothing yet.\n"
    )


def test_a_merge_that_keeps_the_preamble_still_converges() -> None:
    # The preamble is local to each machine and the entries are the union, so
    # merging twice changes nothing — which is the property the whole format
    # rests on.
    once = memory_log_preamble(LOG_A) + format_memory_log(
        merge_memory_logs(LOG_A, LOG_B).entries
    )
    twice = memory_log_preamble(once) + format_memory_log(
        merge_memory_logs(once, LOG_B).entries
    )

    assert once.startswith("# Memory\n\n")
    assert twice == once
