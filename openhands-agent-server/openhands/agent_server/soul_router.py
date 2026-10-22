"""Read and write the user's ``SOUL.md`` identity file.

``SOUL.md`` is a single user-level document (not a per-conversation resource),
so this router exposes it as one resource at ``/api/soul`` with ``GET`` and
``PUT``. There is no ``DELETE``: ``PUT`` with an empty ``content`` is the revert
action, documented on :func:`put_soul`, which keeps "what the agent will use" a
single round-trip for a settings form that always sends its whole textarea.

The handlers are sync ``def`` so FastAPI runs their blocking file I/O in its
worker threadpool rather than on the event loop.
"""

from pathlib import Path
from typing import Final

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from openhands.sdk.context.soul import (
    DEFAULT_SOUL,
    get_soul_path,
    read_user_soul,
)
from openhands.sdk.logger import get_logger
from openhands.sdk.utils.files import atomic_write_text
from openhands.sdk.utils.path import get_user_persistence_dir, resolves_within


logger = get_logger(__name__)

soul_router = APIRouter(prefix="/soul", tags=["Soul"])

# Relative to the router prefix: "" -> /api/soul.
SOUL_PATH: Final[str] = ""

MAX_SOUL_LENGTH: Final[int] = 65536


class SoulResponse(BaseModel):
    """The identity text the agent will use, and where it comes from."""

    content: str = Field(
        description=(
            "The identity text the agent will use. The built-in default when "
            "`exists` is false."
        )
    )
    exists: bool = Field(
        description=(
            "Whether the user has their own SOUL.md on disk. False means "
            "`content` is the built-in default."
        )
    )
    path: str = Field(
        description="Absolute path SOUL.md resolves to for this server.",
    )
    size_bytes: int = Field(
        description="UTF-8 byte length of `content`.",
    )


class SoulUpdateRequest(BaseModel):
    content: str = Field(
        max_length=MAX_SOUL_LENGTH,
        description=(
            "New identity text. An empty or whitespace-only value deletes "
            "SOUL.md and reverts the agent to its built-in default."
        ),
    )


def _soul_path() -> Path:
    """The resolved ``SOUL.md`` path, rejected if it escapes the state tree.

    A symlink planted at ``SOUL.md`` (or at the persistence dir itself) must not
    turn this endpoint into an arbitrary-file read/write primitive.
    """
    path = get_soul_path()
    if not resolves_within(path, get_user_persistence_dir()):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "SOUL.md resolves outside the persistence directory",
        )
    return path


def _soul_response() -> SoulResponse:
    user_soul = read_user_soul()
    content = user_soul if user_soul is not None else DEFAULT_SOUL
    return SoulResponse(
        content=content,
        exists=user_soul is not None,
        path=str(_soul_path()),
        size_bytes=len(content.encode("utf-8")),
    )


@soul_router.get(SOUL_PATH, response_model=SoulResponse)
def get_soul() -> SoulResponse:
    """Return the agent's identity text (``SOUL.md``, else the built-in default)."""
    return _soul_response()


@soul_router.put(SOUL_PATH, response_model=SoulResponse)
def put_soul(request: SoulUpdateRequest) -> SoulResponse:
    """Replace ``SOUL.md`` and return the identity text now in effect.

    An empty or whitespace-only ``content`` deletes the file instead of writing
    it, reverting the agent to its built-in default; the response then reports
    ``exists=false``. Content longer than 65536 characters is rejected with 422.

    The file is written atomically (temp file in the same directory, then
    ``os.replace``) with mode 0600, so a concurrent agent start reads either the
    old or the new identity, never a partial one. Content is stored stripped with
    a single trailing newline, which is what the prompt loader would use anyway.
    """
    path = _soul_path()
    if not request.content.strip():
        path.unlink(missing_ok=True)
        logger.info("Reverted %s to the built-in default", path)
        return _soul_response()

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, request.content.strip() + "\n")
    logger.info("Wrote %s (%d chars)", path, len(request.content))
    return _soul_response()
