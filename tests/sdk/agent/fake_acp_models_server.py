"""Minimal stdio ACP server that reports models, for model-discovery tests.

``FAKE_ACP_MODELS`` picks how models are reported: ``config`` (a ``model``
configOptions select) or ``legacy`` (the UNSTABLE ``models`` block). A
``FAKE_ACP_KEY`` in the environment unlocks a ``premium`` model; with
``FAKE_ACP_REQUIRE_KEY`` set, ``session/new`` fails without it. With
``FAKE_ACP_EFFORT`` set, an ``effort`` select (category ``thought_level``)
follows the model: its levels depend on the model and it disappears for a model
without any, like claude-agent-acp's ``effort`` for Haiku. ``FAKE_ACP_AGENT_NAME``
overrides the reported agent name; like claude-agent-acp, a ``_meta`` model the
server doesn't know is added to its model list as a custom model. With
``FAKE_ACP_RESUME`` set the server advertises ``session/resume``, which, like
claude-agent-acp's, rebuilds the session on the ``_meta`` model it carries;
selecting such a model again by id then fails, as it can for claude-agent-acp.
``session/prompt`` answers with one ``agent_message_chunk`` saying whether the
prompt carried a resume transcript (``resumed``/``fresh``) and ends the turn;
with ``FAKE_ACP_PROMPT_ERROR`` set it instead fails the prompt with a ``-32603``
carrying that text, the way claude-agent-acp reports an exhausted usage limit.
"""

import json
import os
import sys
from typing import Any


MODELS = ["m1", "m2"] + (["premium"] if os.environ.get("FAKE_ACP_KEY") else [])
RESUMED_MODELS: set[str] = set()
EFFORTS = {"m1": ["low", "high"], "m2": [], "premium": ["low", "high", "max"]}
session = {"model": "m1", "effort": "low"}


def respond(request_id, result=None, error=None):
    message = {"jsonrpc": "2.0", "id": request_id}
    if error is None:
        message["result"] = result if result is not None else {}
    else:
        message["error"] = error
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def notify(method: str, params: dict[str, Any]) -> None:
    sys.stdout.write(
        json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n"
    )
    sys.stdout.flush()


def prompt(request_id, params: dict[str, Any]) -> None:
    error_text = os.environ.get("FAKE_ACP_PROMPT_ERROR")
    if error_text:
        respond(
            request_id,
            error={"code": -32603, "message": f"Internal error: {error_text}"},
        )
        return
    texts = [
        block.get("text", "")
        for block in params.get("prompt", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    resumed = any("<<RESUMED CONVERSATION>>" in text for text in texts)
    label = os.environ.get("FAKE_ACP_REPLY", "ok")
    reply = f"{label} ({'resumed' if resumed else 'fresh'})"
    notify(
        "session/update",
        {
            "sessionId": params.get("sessionId"),
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": reply},
            },
        },
    )
    respond(request_id, {"stopReason": "end_turn"})


def model_select(current: str) -> dict[str, Any]:
    return {
        "id": "model",
        "name": "Model",
        "type": "select",
        "currentValue": current,
        "options": [{"value": m, "name": m.upper()} for m in MODELS],
    }


def config_options() -> list[dict[str, Any]]:
    options = [model_select(session["model"])]
    levels = EFFORTS[session["model"]] if os.environ.get("FAKE_ACP_EFFORT") else []
    if levels:
        if session["effort"] not in levels:
            session["effort"] = levels[0]
        options.append(
            {
                "id": "effort",
                "name": "Effort",
                "category": "thought_level",
                "type": "select",
                "currentValue": session["effort"],
                "options": [{"value": level, "name": level} for level in levels],
            }
        )
    return options


def set_config_option(config_id: Any, value: Any) -> tuple[Any, Any]:
    if config_id == "effort":
        if value not in EFFORTS.get(session["model"], []):
            return None, {"code": -32602, "message": "Unknown effort"}
        session["effort"] = value
    elif value in RESUMED_MODELS:
        return None, {"code": -32603, "message": "Internal error"}
    elif value in MODELS:
        session["model"] = value
    else:
        return None, {"code": -32602, "message": "Unknown model"}
    return {"configOptions": config_options()}, None


def meta_model_of(params: dict[str, Any]) -> Any:
    return (
        ((params.get("_meta") or {}).get("claudeCode") or {}).get("options") or {}
    ).get("model")


def resume_session(params: dict[str, Any]) -> dict[str, Any]:
    meta_model = meta_model_of(params)
    if meta_model:
        if meta_model not in MODELS:
            MODELS.append(meta_model)
            EFFORTS[meta_model] = ["low", "high", "max"]
            RESUMED_MODELS.add(meta_model)
        session["model"] = meta_model
    return {"configOptions": config_options()}


def new_session(params: dict[str, Any]):
    meta_model = meta_model_of(params)
    if meta_model and meta_model not in MODELS:
        MODELS.append(meta_model)
        EFFORTS[meta_model] = []
    if os.environ.get("FAKE_ACP_REQUIRE_KEY") and not os.environ.get("FAKE_ACP_KEY"):
        return None, {"code": -32000, "message": "Authentication required"}
    result: dict[str, Any] = {"sessionId": "fake-session"}
    if os.environ.get("FAKE_ACP_MODELS") == "legacy":
        result["models"] = {
            "currentModelId": "m1",
            "availableModels": [
                {"modelId": m, "name": m.upper(), "description": f"{m} model"}
                for m in MODELS
            ],
        }
    else:
        result["configOptions"] = config_options()
    return result, None


def main():
    for line in sys.stdin:
        message = json.loads(line)
        request_id = message.get("id")
        if request_id is None:
            continue
        method = message["method"]
        params = message.get("params") or {}
        if method == "initialize":
            result = {
                "protocolVersion": 1,
                "agentInfo": {
                    "name": os.environ.get("FAKE_ACP_AGENT_NAME", "fake-acp"),
                    "version": "1.2.3",
                },
            }
            if os.environ.get("FAKE_ACP_RESUME"):
                result["agentCapabilities"] = {"sessionCapabilities": {"resume": {}}}
            respond(request_id, result)
        elif method == "session/new":
            result, error = new_session(params)
            respond(request_id, result, error)
        elif method == "session/resume" and os.environ.get("FAKE_ACP_RESUME"):
            respond(request_id, resume_session(params))
        elif method == "session/set_mode":
            respond(request_id)
        elif method == "session/prompt":
            prompt(request_id, params)
        elif method == "session/set_model":
            if params.get("modelId") not in MODELS:
                respond(request_id, error={"code": -32602, "message": "Unknown model"})
            else:
                respond(request_id)
        elif method == "session/set_config_option":
            result, error = set_config_option(
                params.get("configId"), params.get("value")
            )
            respond(request_id, result, error)
        else:
            respond(request_id, error={"code": -32601, "message": "Method not found"})


if __name__ == "__main__":
    main()
