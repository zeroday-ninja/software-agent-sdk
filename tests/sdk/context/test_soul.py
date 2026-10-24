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
