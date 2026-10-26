"""Drive Google's Antigravity CLI over the Agent Client Protocol.

Antigravity has no protocol mode. It has a headless streaming mode that reads
one JSON object per line on stdin and writes one per line to stdout, and this
translates between that and the protocol every other runtime here speaks.

Written rather than taken from the community adapters, which is cheaper than
it sounds: one of them needs a runtime this stack does not have, another warns
in its own README that using it may cost you your account, and both would be a
pinned dependency on an unofficial wrapper of a tool that may ship its own
support tomorrow. Everything needed to write one was already here — the
protocol library is a dependency of this package, and it ships the server side.

**The stream format below was learned by running the CLI**, not from a
specification, because Google publishes none. The recordings those probes
produced are what the tests are built from, so a format change shows up as a
failing test rather than as an agent that silently says nothing:

    in   {"event":"user",
          "message":{"role":"user","content":[{"type":"text","text":"…"}]}}
    out  {"event":"init","conversation_id":"…","init":{"model":…,"cwd":…,"tools":[…]}}
         {"event":"step_update","step_update":{"step_index":1,"state":"ACTIVE",
                                               "step_type":"agent_response","text_delta":"ok"}}
         {"event":"step_update","step_update":{"step_type":"tool","tool_name":"write_file",
                                               "tool_info":{"parameters":{…}}}}
         {"event":"result","result":{"status":"SUCCESS","response":"…","usage":{…}}}

A refused turn, captured from an account whose weekly quota was spent:

    out  {"event":"step_update","step_update":{"step_type":"error_message", …}}
         {"event":"result","result":{"status":"ERROR","response":"",
           "error":"Individual quota reached. … Resets in 132h6m57s."}}
    err  AGY_ERROR: {"short_error":"RESOURCE_EXHAUSTED (code 429): …",
           "status":"RESOURCE_EXHAUSTED","error_code":429,"retryable":true}

The refusal rides out of ``prompt`` as an exception rather than as agent text,
because that is the only path the SDK classifies: a usage limit delivered as
an answer finishes the turn, and the conversation never fails over. The CLI
states the reset as a duration, which is the most precise reset signal any
runtime here gives, so the text is carried through verbatim for the parser --
including the stderr line, which is where the status code lives.

An event this build does not recognise is reported, never dropped: a silent
drop is how a runtime ends up looking like it did nothing when it did
something nobody can see.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any

from acp import run_agent
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)


__all__ = ["AntigravityAgent", "AntigravityTurnError", "main"]

#: Protocol version this server speaks.
PROTOCOL_VERSION = 1

#: The CLI, and the flags that put it in the mode this adapter understands.
#:
#: ``--print`` carries its value attached. Passed as a separate argument the
#: CLI takes the *next flag* as the prompt, which fails in a way that looks
#: like a protocol bug rather than a command-line one.
AGY_BINARY_ENV = "OH_ANTIGRAVITY_BINARY"
DEFAULT_BINARY = "agy"
STREAM_FLAGS = (
    "--print=",
    "--input-format=stream-json",
    "--output-format=stream-json",
)

#: How long to wait for the CLI to exit after a cancel before killing it.
CANCEL_GRACE_SECONDS = 5.0

#: Step types this build knows about and deliberately says nothing for.
#:
#: ``user_input`` is the CLI echoing back the prompt we just sent it,
#: ``error_message`` is the prose half of a failure the ``result`` event
#: reports properly, and ``system_message`` is the CLI's own scaffolding --
#: the instructions it feeds its model, which are not the model's words and
#: not the user's. Seen seven times in one live turn, each surfaced as
#: ``[unrecognised step: 'system_message']`` in the transcript. Anything *not*
#: listed here is surfaced, so a step type the CLI grows is visible rather
#: than dropped.
IGNORED_STEP_TYPES = frozenset({"user_input", "error_message", "system_message"})

#: Longest stderr tail kept to describe a failure.
MAX_STDERR_CHARS = 2000


class AntigravityTurnError(RuntimeError):
    """The CLI refused or failed the turn.

    Raised out of :meth:`AntigravityAgent.prompt` rather than streamed as agent
    text. The SDK reads a turn *error* to decide whether a runtime is out of
    usage; the same words delivered as an answer finish the turn successfully
    and no failover happens.
    """


class AntigravityUnavailable(RuntimeError):
    """The CLI cannot run here at all, which is not a failed turn.

    Separate from :class:`AntigravityTurnError` because it means something
    different and is fixed differently: a turn error is the runtime answering
    badly, this is the runtime not being reachable. Failing over to another
    runtime is right for both, but only one of them is worth telling somebody
    how to repair.
    """


#: What the CLI says when it has never been signed in. Matched on the stable
#: half of the sentence: the second clause names a command that has already
#: changed shape once.
_NOT_SIGNED_IN = ("not logged into antigravity", "please sign in")


def _signed_out(text: str) -> bool:
    """Whether this failure is "nobody has signed this machine in"."""
    lowered = (text or "").casefold()
    return any(marker in lowered for marker in _NOT_SIGNED_IN)


def _unavailable_detail(binary: str, error: OSError) -> str:
    """Why the CLI could not be started, and what to do about it.

    `FileNotFoundError: [Errno 2] No such file or directory: 'agy'` is true
    and useless: it names a binary nobody chose and no next step. The sandbox
    case is the common one -- the image carries the npm-installed providers
    and Antigravity is a native download that is not among them.
    """
    if isinstance(error, FileNotFoundError):
        return (
            f"The Antigravity CLI ({binary!r}) is not installed here. It is a "
            "native binary rather than one of the npm ACP providers the "
            "sandbox image carries, so a sandbox does not have it unless it "
            f"was put there. Set {AGY_BINARY_ENV} to its path, or choose "
            "another runtime."
        )
    if isinstance(error, PermissionError):
        return f"The Antigravity CLI at {binary!r} is not executable by this user."
    return f"The Antigravity CLI at {binary!r} could not be started: {error}"


def _sign_in_detail(binary: str) -> str:
    """What to do about a CLI that is installed and signed out."""
    # "not logged in" is deliberate wording: the SDK's turn classifier reads
    # it as an auth condition, which is what lets a client offer to fix it
    # instead of reporting a generic prompt failure.
    return (
        "Antigravity is installed here but it is not logged in. Its login "
        "lives in the desktop keyring, which a sandbox does not have, so a "
        f"sandbox needs its own: run {binary!r} once in that environment and "
        "complete the Google sign-in. Until then this runtime cannot answer."
    )


async def _drain(stream: asyncio.StreamReader | None) -> str:
    """Everything the CLI wrote to stderr, bounded."""
    if stream is None:
        return ""
    data = await stream.read()
    return data.decode("utf-8", errors="replace")[-MAX_STDERR_CHARS:].strip()


def _turn_failure(reported: str, stderr_text: str) -> str:
    """What the CLI said went wrong, in the words a classifier can read.

    The stdout error is the sentence a person wants; the stderr line carries
    the status and the code -- ``RESOURCE_EXHAUSTED (code 429)`` -- which is
    what tells a usage limit apart from a crash. Both are kept, the readable
    one first, because dropping either loses something no other line has.
    """
    detail = reported
    for line in stderr_text.splitlines():
        line = line.strip()
        if not line.startswith("AGY_ERROR:"):
            continue
        try:
            payload = json.loads(line[len("AGY_ERROR:") :].strip())
        except json.JSONDecodeError:
            continue
        short = payload.get("short_error")
        if isinstance(short, str) and short and short != reported:
            detail = f"{detail} ({short})"
        break
    return detail


def _binary() -> str:
    """Where the CLI is.

    Resolved per call rather than at import so a sandbox that installs it late
    — or a test that points at a stub — still finds the right one.
    """
    override = os.environ.get(AGY_BINARY_ENV)
    if override:
        return override
    return shutil.which(DEFAULT_BINARY) or DEFAULT_BINARY


@dataclass
class _Session:
    """One conversation, and the process serving it."""

    session_id: str
    cwd: str
    model: str | None
    process: asyncio.subprocess.Process | None = None
    #: Step indices already announced, so a progress update is not a new call.
    announced_tools: set[int] = field(default_factory=set)
    #: (what, kind) pairs already reported as unrecognised, so a format that
    #: grew is said once per session rather than once per step.
    unrecognised_seen: set[tuple[str, str]] = field(default_factory=set)
    cancelled: bool = False


class AntigravityAgent:
    """An Agent Client Protocol server backed by the Antigravity CLI."""

    def __init__(self, *, model: str | None = None) -> None:
        self._sessions: dict[str, _Session] = {}
        self._connection: Any = None
        self._default_model = model
        self._next_session = 0

    # -- protocol plumbing -------------------------------------------------

    def on_connect(self, conn: Any) -> None:
        self._connection = conn

    async def initialize(
        self,
        protocol_version: int = PROTOCOL_VERSION,
        **_: Any,
    ) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=min(protocol_version, PROTOCOL_VERSION),
            agent_capabilities={"loadSession": False, "promptCapabilities": {}},
        )

    async def new_session(self, cwd: str, **kwargs: Any) -> NewSessionResponse:
        self._next_session += 1
        session_id = f"antigravity-{self._next_session}"
        model = kwargs.get("model") or self._default_model
        self._sessions[session_id] = _Session(
            session_id=session_id, cwd=cwd, model=model
        )
        return NewSessionResponse(session_id=session_id)

    async def close_session(self, session_id: str, **_: Any) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            await self._terminate(session)

    async def cancel(self, session_id: str, **_: Any) -> None:
        """Stop the turn in flight.

        The CLI has no cancel message, so the subprocess is ended. That is
        what cancelling means here, and saying so beats a cancel that returns
        immediately while the work carries on.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.cancelled = True
        await self._terminate(session)

    # -- the turn ----------------------------------------------------------

    async def prompt(
        self, session_id: str, prompt: list[Any], **_: Any
    ) -> PromptResponse:
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"unknown session {session_id!r}")
        session.cancelled = False
        session.announced_tools.clear()

        command = [_binary(), *STREAM_FLAGS]
        if session.model:
            command += ["--model", session.model]
        # Inside a sandbox there is nobody to answer a permission prompt, and
        # a turn that blocks forever waiting for one is worse than a turn that
        # acts. Outside a sandbox this would be the wrong default, which is
        # why it is the launcher's decision and not this file's.
        if os.environ.get("OH_ANTIGRAVITY_AUTO_APPROVE", "1") == "1":
            command.append("--dangerously-skip-permissions")

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=session.cwd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as error:
            # Raised, not streamed: the SDK reads a turn error to decide
            # whether to fail over, and the same words as an answer would end
            # the turn successfully with nothing having happened.
            raise AntigravityUnavailable(
                _unavailable_detail(command[0], error)
            ) from error
        session.process = process

        assert process.stdin is not None
        process.stdin.write((json.dumps(_user_message(prompt)) + "\n").encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()

        # Drained concurrently, for two reasons: an undrained pipe stops the
        # CLI dead once the kernel buffer fills, and the status code for a
        # refusal is only ever on stderr.
        errors = asyncio.create_task(_drain(process.stderr))
        try:
            stop_reason, failure = await self._pump(session, process)
        finally:
            await process.wait()
            session.process = None
        stderr_text = await errors
        if session.cancelled:
            return PromptResponse(stop_reason=stop_reason)
        # Checked before the general case: "not logged in" arrives as an
        # ordinary turn failure, and reported as one it reads like the model
        # declining rather than like a machine nobody has set up.
        if _signed_out(f"{failure or ''}\n{stderr_text}"):
            raise AntigravityUnavailable(_sign_in_detail(command[0]))
        if failure is not None:
            raise AntigravityTurnError(_turn_failure(failure, stderr_text))
        return PromptResponse(stop_reason=stop_reason)

    async def _pump(
        self, session: _Session, process: asyncio.subprocess.Process
    ) -> tuple[str, str | None]:
        """Translate the CLI's stream into session updates until it ends.

        Returns the stop reason and, when the CLI reported one, the failure it
        reported -- which the caller raises rather than streams.
        """
        assert process.stdout is not None
        stop_reason = "end_turn"
        failure: str | None = None
        while True:
            raw = await process.stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Not a protocol error on our side, but not nothing either:
                # the user should see that the runtime said something this
                # build could not read.
                await self._say(session, f"[unreadable output: {line[:200]}]")
                continue
            reason, reported = await self._handle(session, event)
            if reason is not None:
                stop_reason = reason
            if reported is not None:
                failure = reported
        if session.cancelled:
            return "cancelled", None
        return stop_reason, failure

    async def _handle(
        self, session: _Session, event: dict
    ) -> tuple[str | None, str | None]:
        """One stream event.

        Returns the stop reason when the turn ended, and the CLI's own words
        when it reported a failure.
        """
        kind = event.get("event")
        if kind == "init":
            return None, None
        if kind == "step_update":
            await self._step(session, event.get("step_update") or {})
            return None, None
        if kind == "result":
            result = event.get("result") or {}
            if result.get("status") == "ERROR":
                return "refusal", str(result.get("error") or "").strip() or "unknown"
            return "end_turn", None
        # An event kind this build has never seen. Reported rather than
        # dropped: a format that grew should look like a format that grew.
        await self._unrecognised(session, "event", kind)
        return None, None

    async def _step(self, session: _Session, step: dict) -> None:
        step_type = step.get("step_type")
        if step_type == "agent_response":
            delta = step.get("text_delta")
            if isinstance(delta, str) and delta:
                await self._say(session, delta)
            return
        if step_type == "thinking":
            delta = step.get("text_delta")
            if isinstance(delta, str) and delta:
                await self._update(
                    session,
                    AgentThoughtChunk(
                        session_update="agent_thought_chunk",
                        content=_text(delta),
                    ),
                )
            return
        if step_type != "tool":
            if step_type not in IGNORED_STEP_TYPES:
                await self._unrecognised(session, "step", step_type)
            return

        index = step.get("step_index")
        name = step.get("tool_name") or "tool"
        info = step.get("tool_info") or {}
        call_id = f"{session.session_id}:{index}"
        done = step.get("state") == "DONE"
        if index not in session.announced_tools:
            session.announced_tools.add(index)
            await self._update(
                session,
                ToolCallStart(
                    session_update="tool_call",
                    tool_call_id=call_id,
                    title=_tool_title(name, info),
                    status="completed" if done else "in_progress",
                    raw_input=info.get("parameters"),
                ),
            )
            if not done:
                return
        await self._update(
            session,
            ToolCallProgress(
                session_update="tool_call_update",
                tool_call_id=call_id,
                status="completed" if done else "in_progress",
            ),
        )

    async def _say(
        self, session: _Session, text: str, meta: dict | None = None
    ) -> None:
        await self._update(
            session,
            AgentMessageChunk(
                session_update="agent_message_chunk",
                content=_text(text),
                field_meta=meta,
            ),
        )

    async def _unrecognised(self, session: _Session, what: str, kind: Any) -> None:
        """Something this build has never seen, as a field and as a sentence.

        The milestone this adapter belongs to asks for "any unrecognised event
        surfaced as a *typed error* rather than dropped", and the execution
        rule beside it is blunter: anything a machine must act on is a typed
        field, never buried in a sentence. The sentence was all there was — it
        landed in the assistant's own message stream, where nothing can count
        it and nothing can tell it from the model's words.

        So the fact goes in the protocol's own extension slot, where a reader
        can find it by key, and the sentence stays for the person watching the
        transcript. A format that grew should look like a format that grew to
        both of them -- once. Said on every step of the turn it read as an
        error loop rather than a notice, and buried the answer under it.
        """
        key = (what, str(kind))
        if key in session.unrecognised_seen:
            return
        session.unrecognised_seen.add(key)
        await self._say(
            session,
            f"[unrecognised {what}: {kind!r}]",
            meta={
                "openhands.unrecognised": {
                    "what": what,
                    "kind": None if kind is None else str(kind),
                    "adapter": "antigravity",
                }
            },
        )

    async def _update(self, session: _Session, update: Any) -> None:
        if self._connection is None:
            return
        await self._connection.session_update(session.session_id, update)

    async def _terminate(self, session: _Session) -> None:
        process = session.process
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=CANCEL_GRACE_SECONDS)
        except TimeoutError:
            process.kill()
            await process.wait()


def _text(value: str) -> TextContentBlock:
    """A text block. The type is explicit because the union is discriminated."""
    return TextContentBlock(type="text", text=value)


def _user_message(prompt: list[Any]) -> dict:
    """The CLI's input message for a protocol prompt.

    Only text survives the translation. An image or an embedded resource has
    nowhere to go in this stream format, so it is named in the text rather
    than dropped without a word.
    """
    parts: list[dict] = []
    for block in prompt:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if isinstance(text, str) and text:
            parts.append({"type": "text", "text": text})
            continue
        kind = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        parts.append(
            {"type": "text", "text": f"[{kind or 'content'} omitted: unsupported]"}
        )
    return {"event": "user", "message": {"role": "user", "content": parts}}


def _tool_title(name: str, info: dict) -> str:
    """A tool call as one readable line."""
    parameters = info.get("parameters")
    if isinstance(parameters, dict) and parameters:
        first = next(iter(parameters.values()))
        if isinstance(first, (str, int)):
            return f"{name}: {str(first)[:120]}"
    return name


def main() -> None:
    """Run the adapter over stdio."""
    asyncio.run(
        run_agent(AntigravityAgent(model=os.environ.get("OH_ANTIGRAVITY_MODEL")))
    )


if __name__ == "__main__":  # pragma: no cover - module entry point
    main()
    sys.exit(0)
