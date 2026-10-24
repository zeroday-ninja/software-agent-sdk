# Decisions, and approaches already tried

Append-only. Dated, with what was tried and what happened.

## 2026-10-16 — Live model switching cannot go through set_config_option

claude-agent-acp rejects the account's catalog models there. The session is
rebuilt with `session/resume` and a new `_meta` model instead; re-sending the
model id after that fails with -32603, so only the effort is sent.

## 2026-10-20 — A refreshed Codex login must be adopted back from the sandbox

Codex refreshes its token once the one-hour id_token expires, which is almost
every sandbox start. Writing it only inside the sandbox invalidated both the
stored secret and the host's copy. The host now adopts the newest same-account
copy before a sandbox starts, before a conversation is deleted, and after
discovery.

## 2026-10-24 — SOUL.md moved into a set, and reads still fall back

`~/.openhands/SOUL.md` predates the set. Reads fall back to it so an existing
install keeps working untouched; the first write moves it into `soul/` so
nobody ends up with two copies where only one is read.
