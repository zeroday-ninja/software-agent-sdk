# OpenHands Software Agent SDK

What this repository is for, how it thinks, and what it will not do. The rest
of the set is beside it in `.openhands/identity/`.

> Drafted by an agent from the code and its history. **Current goals** is the
> owner's section to confirm or rewrite; everything above it describes what the
> repository already is.

## What it is

The canonical definition of what an agent *is*: the Python SDK, the Agent
Server, the tools, conversations, events, the REST and WebSocket contract, and
the generated TypeScript client. Everything downstream — the automation
service, Agent Canvas — consumes what is defined here.

## Worldview

- **The contract is the product.** A change to a request or response shape is
  a change every consumer has to survive, so it is designed before it is
  implemented and the client is regenerated with it.
- **Dependency direction is one way.** Agent Server contract, then TypeScript
  client, then Canvas. Nothing here knows what a dashboard is.
- **An agent is a loop over events, not a chat.** Events are the record; the
  transcript is a rendering of them.
- **Resolve paths per call, not at import.** A process that configures itself
  late must still read and write one location.

## Boundaries

- No scheduling. When work runs is the automation service's question.
- No UI, and no assumptions about one.
- No silent degradation of identity or context: a prompt that quietly drops
  what the user told it is a failure nobody notices until the output is wrong.

## Current goals

*Owner's section. The draft below is what the recent work implies.*

- Make an ACP runtime's usage limits a first-class, observable condition:
  failover that chains past one hop, a typed switch event, and the reset hint
  parsed out of the refusal rather than discarded.
- Give the user's identity room to be more than one file, and keep it readable
  by every runtime the server can drive.
