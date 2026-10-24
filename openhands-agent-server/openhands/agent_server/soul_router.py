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

from openhands.sdk.context.identity import (
    IDENTITY_DOCUMENTS,
    MAX_DOCUMENT_LENGTH,
    IdentityDocument,
    identity_document_path,
    known_identity_document,
    read_identity_document,
    write_identity_document,
)
from openhands.sdk.context.memory_log import (
    format_memory_log,
    merge_memory_logs,
)
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


class IdentityDocumentResponse(BaseModel):
    """One document in the identity set."""

    name: str = Field(description="Stable id, e.g. `style`.")
    relative_path: str = Field(
        description="Where it sits under the identity directory."
    )
    purpose: str = Field(
        description="What belongs in it. For a settings UI, never for a model."
    )
    content: str = Field(description="What is written, or empty.")
    exists: bool = Field(description="Whether anything has been written.")
    path: str = Field(description="Absolute path on this server.")
    size_bytes: int = Field(description="UTF-8 byte length of `content`.")


class IdentitySetResponse(BaseModel):
    """Every document this build knows about, in load order."""

    documents: list[IdentityDocumentResponse]
    directory: str = Field(description="Absolute path of the identity set.")


class IdentityDocumentUpdateRequest(BaseModel):
    content: str = Field(
        max_length=MAX_DOCUMENT_LENGTH,
        description=(
            "New text. Empty or whitespace-only deletes the document, which "
            "is how a document is reverted to having never been written."
        ),
    )


class MemoryMergeRequest(BaseModel):
    """Logs from elsewhere, to be folded into this server's memory."""

    logs: list[str] = Field(
        description=(
            "Whole `MEMORY.md` bodies from other backends. Entries are "
            "unioned by id; nothing already here is removed."
        )
    )


class MemoryMergeResponse(BaseModel):
    content: str = Field(description="The merged log now on disk.")
    entry_count: int = Field(description="Entries after the merge.")
    added: int = Field(description="Entries this merge brought in.")
    conflicts: list[str] = Field(
        description=(
            "Ids that arrived with text differing from the copy already here. "
            "Both versions are kept, under suffixed ids: deciding which of two "
            "versions of someone's memory to discard is not this server's call."
        )
    )


def _document_response(document: IdentityDocument) -> IdentityDocumentResponse:
    content = read_identity_document(document)
    return IdentityDocumentResponse(
        name=document.name,
        relative_path=document.relative_path,
        purpose=document.purpose,
        content=content or "",
        exists=content is not None,
        path=str(identity_document_path(document)),
        size_bytes=len((content or "").encode("utf-8")),
    )


def _require_document(name: str) -> IdentityDocument:
    document = known_identity_document(name)
    if document is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No identity document named {name!r}",
        )
    return document


@soul_router.get("/documents", response_model=IdentitySetResponse)
def list_identity_documents() -> IdentitySetResponse:
    """Every identity document, written or not, in the order a prompt reads them."""
    ordered = sorted(IDENTITY_DOCUMENTS, key=lambda item: item.order)
    return IdentitySetResponse(
        documents=[_document_response(document) for document in ordered],
        directory=str(identity_document_path(IDENTITY_DOCUMENTS[0]).parent),
    )


@soul_router.get("/documents/{name}", response_model=IdentityDocumentResponse)
def get_identity_document(name: str) -> IdentityDocumentResponse:
    """One identity document. 404 when this build does not know that name."""
    return _document_response(_require_document(name))


@soul_router.put("/documents/{name}", response_model=IdentityDocumentResponse)
def put_identity_document(
    name: str, request: IdentityDocumentUpdateRequest
) -> IdentityDocumentResponse:
    """Replace one identity document and return it as it now stands.

    Empty content deletes the file rather than writing an empty one, so "never
    written" and "written, then emptied" stay the same state — a settings form
    that always sends its whole textarea should not be able to create a
    document that exists and says nothing.

    Writing anything moves a pre-set ``SOUL.md`` into the set first, so the
    install ends up with one copy in one place.
    """
    document = _require_document(name)
    path = write_identity_document(document, request.content)
    logger.info("Wrote %s (%d chars)", path, len(request.content.strip()))
    return _document_response(document)


@soul_router.post("/documents/memory/merge", response_model=MemoryMergeResponse)
def merge_memory(request: MemoryMergeRequest) -> MemoryMergeResponse:
    """Fold other backends' memory into this one, losing nothing.

    Memory is append-only, which is exactly what makes this safe: the merge is
    a union by entry id, so it converges whichever order the backends reach
    each other in, and an entry can only ever appear or stay. Conflicting text
    under one id is kept twice and reported rather than resolved here.
    """
    document = _require_document("memory")
    before = read_identity_document(document) or ""
    merged = merge_memory_logs(before, *request.logs)
    content = format_memory_log(merged.entries)
    write_identity_document(document, content)
    existing = len(merge_memory_logs(before).entries)
    return MemoryMergeResponse(
        content=content,
        entry_count=len(merged.entries),
        added=len(merged.entries) - existing,
        conflicts=merged.conflicts,
    )
