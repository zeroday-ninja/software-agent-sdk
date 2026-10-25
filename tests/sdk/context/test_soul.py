"""Unit tests for the shared ``SOUL.md`` loader.

Snapshot tests for the full rendered system prompt with default and custom
soul content live in ``tests/sdk/context/prompts/test_prompt_snapshot.py``
alongside the other prompt golden files.
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from openhands.sdk.context.identity import (
    IDENTITY_BUDGET_CHARS,
    IdentityDocument,
    known_identity_document,
    write_identity_document,
)
from openhands.sdk.context.soul import (
    DEFAULT_SOUL,
    get_soul_path,
    load_soul,
    read_user_soul,
)


@pytest.fixture
def soul_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """``OH_PERSISTENCE_DIR`` pointed at a clean tempdir, plus the soul path."""
    monkeypatch.setenv("OH_PERSISTENCE_DIR", str(tmp_path))
    # SOUL.md is one document in a set now, so a fresh install writes it under
    # `soul/`. The legacy location is still read; that has its own test.
    soul = tmp_path / "soul" / "SOUL.md"
    soul.parent.mkdir(parents=True, exist_ok=True)
    yield soul


def test_soul_path_follows_the_persistence_dir(soul_path: Path) -> None:
    assert get_soul_path() == soul_path


@pytest.mark.parametrize("body", ["", "   \n\n  \n"])
def test_blank_file_counts_as_no_user_soul(soul_path: Path, body: str) -> None:
    soul_path.write_text(body)

    assert read_user_soul() is None
    assert load_soul() == DEFAULT_SOUL


def test_missing_file_falls_back_to_the_default(soul_path: Path) -> None:
    assert not soul_path.exists()

    assert read_user_soul() is None
    assert load_soul() == DEFAULT_SOUL


def test_user_content_replaces_the_default_and_is_stripped(soul_path: Path) -> None:
    soul_path.write_text("\n  You are direct.  \n\n")

    assert read_user_soul() == "You are direct."
    assert load_soul() == "You are direct."


def test_internal_structure_is_preserved(soul_path: Path) -> None:
    soul_path.write_text(
        textwrap.dedent("""\
            # Identity
            You are smolpaws.

            # Style
            Be direct. Be concise.
        """)
    )

    result = load_soul()

    assert "# Identity" in result
    assert "# Style" in result
    assert "smolpaws" in result


def test_unreadable_file_falls_back_to_the_default(soul_path: Path) -> None:
    # A directory where the file should be: the read raises OSError, not
    # FileNotFoundError, and must not propagate into prompt assembly.
    soul_path.mkdir()

    assert read_user_soul() is None
    assert load_soul() == DEFAULT_SOUL


def test_a_soul_written_before_the_set_existed_is_still_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An install that predates the set keeps working untouched.

    Reads fall back to the old path, so nobody has to migrate anything to keep
    the identity they already wrote.
    """
    # Arrange
    monkeypatch.setenv("OH_PERSISTENCE_DIR", str(tmp_path))
    (tmp_path / "SOUL.md").write_text("I am who I was.\n")

    # Act / Assert
    assert read_user_soul() == "I am who I was."
    assert get_soul_path() == tmp_path / "SOUL.md"


def test_the_whole_set_reaches_the_prompt_in_load_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A set that can be edited but never reaches a prompt is a settings page
    that lies."""
    # Arrange
    monkeypatch.setenv("OH_PERSISTENCE_DIR", str(tmp_path))
    identity = tmp_path / "soul"
    (identity / "examples").mkdir(parents=True)
    (identity / "SOUL.md").write_text("I build tools for myself.\n")
    (identity / "STYLE.md").write_text("Short sentences. No hedging.\n")
    (identity / "MEMORY.md").write_text("Decided: fork, never upstream.\n")
    (identity / "examples" / "good-outputs.md").write_text("Like this.\n")

    # Act
    rendered = load_soul()

    # Assert: identity first, then voice, then what was established, then
    # calibration — and the headings say which is which.
    assert rendered.index("I build tools") < rendered.index("How I write")
    assert rendered.index("How I write") < rendered.index("already established")
    assert rendered.index("already established") < rendered.index("sounds like me")
    assert "fork, never upstream" in rendered


def test_a_set_with_only_a_soul_renders_exactly_that_document(
    soul_path: Path,
) -> None:
    """Every install that predates the set has exactly this, and must read the
    same as it did before."""
    # Arrange
    soul_path.write_text("I build tools for myself.\n")

    # Act / Assert: no headings, no scaffolding, just the document.
    assert load_soul() == "I build tools for myself."


def test_identity_stays_inside_its_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: far more identity than a prompt can afford.
    monkeypatch.setenv("OH_PERSISTENCE_DIR", str(tmp_path))
    identity = tmp_path / "soul"
    identity.mkdir(parents=True)
    (identity / "SOUL.md").write_text("S" * 40000)
    (identity / "STYLE.md").write_text("T" * 40000)
    (identity / "MEMORY.md").write_text("M" * 40000)

    # Act
    rendered = load_soul()

    # Assert: identity is context and the task is the task.
    assert len(rendered) <= IDENTITY_BUDGET_CHARS + 500
    assert "[truncated]" in rendered


def test_a_long_memory_keeps_what_was_learned_most_recently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setenv("OH_PERSISTENCE_DIR", str(tmp_path))
    identity = tmp_path / "soul"
    identity.mkdir(parents=True)
    (identity / "MEMORY.md").write_text("x" * 20000 + "\nlearned this today\n")

    # Act
    rendered = load_soul()

    # Assert: the oldest thing someone learned is rarely the most relevant.
    assert "learned this today" in rendered
    assert "earlier entries omitted" in rendered


def test_a_document_that_would_escape_the_identity_directory_is_refused(
    soul_path: Path,
) -> None:
    """The guard that had never been exercised.

    No live surface can reach it: a document is resolved by name from a closed
    declared set, so no caller's string ever becomes a path. What it protects
    against is a *declared* document whose relative path escapes — a mistake
    made in this file, not an attacker's input — and its own docstring says
    so: "which a declared document's relative path should never do but is
    cheap to refuse rather than to trust".

    Cheap to refuse, and it was free to believe. Nothing in the tree asserted
    the refusal, so a declared path with a `..` in it would have written
    wherever it pointed.
    """
    # Arrange: a document shaped exactly like a real one, aimed outside.
    escaping = IdentityDocument(
        name="escaping",
        relative_path="../../outside.md",
        purpose="Not a real document.",
        order=99,
    )
    outside = soul_path.parent.parent / "outside.md"

    # Act / Assert
    with pytest.raises(ValueError, match="outside the identity directory"):
        write_identity_document(escaping, "should never be written")
    assert not outside.exists()


def test_a_declared_document_writes_where_it_says(soul_path: Path) -> None:
    """The other half, so the guard cannot be satisfied by refusing everything.

    `examples/good-outputs.md` is a real declared document with a path
    separator in it, which is the case a careless guard would reject.
    """
    # Arrange
    examples = known_identity_document("good-outputs")
    assert examples is not None

    # Act
    written = write_identity_document(examples, "Work that sounds like them.")

    # Assert
    assert written.read_text().strip() == "Work that sounds like them."
    assert written.parent.name == "examples"
