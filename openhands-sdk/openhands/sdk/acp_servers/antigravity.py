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


__all__ = ["AntigravityAgent", "main"]

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

        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=session.cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        session.process = process

        assert process.stdin is not None
        process.stdin.write((json.dumps(_user_message(prompt)) + "\n").encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()

        stop_reason = await self._pump(session, process)
        await process.wait()
        session.process = None
        return PromptResponse(stop_reason=stop_reason)

    async def _pump(
        self, session: _Session, process: asyncio.subprocess.Process
    ) -> str:
        """Translate the CLI's stream into session updates until it ends."""
        assert process.stdout is not None
        stop_reason = "end_turn"
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
            reason = await self._handle(session, event)
            if reason is not None:
                stop_reason = reason
        if session.cancelled:
            return "cancelled"
        return stop_reason

    async def _handle(self, session: _Session, event: dict) -> str | None:
        """One stream event. Returns a stop reason when the turn ended."""
        kind = event.get("event")
        if kind == "init":
            return None
        if kind == "step_update":
            await self._step(session, event.get("step_update") or {})
            return None
        if kind == "result":
            result = event.get("result") or {}
            if result.get("status") == "ERROR":
                await self._say(
                    session,
                    f"[antigravity error: {result.get('error') or 'unknown'}]",
                )
                return "refusal"
            return "end_turn"
        # An event kind this build has never seen. Reported rather than
        # dropped: a format that grew should look like a format that grew.
        await self._say(session, f"[unrecognised event: {kind!r}]")
        return None

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

    async def _say(self, session: _Session, text: str) -> None:
        await self._update(
            session,
            AgentMessageChunk(
                session_update="agent_message_chunk",
                content=_text(text),
            ),
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
