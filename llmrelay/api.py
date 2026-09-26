"""Request validation and the deliberately small compatibility surface."""

import time
import uuid
from dataclasses import dataclass

from .providers import Provider, ProviderError, generate


MAX_BODY = 1_048_576
EFFORTS = {"minimal", "low", "medium", "high", "xhigh", "max"}


class RequestError(Exception):
    def __init__(self, message: str, status: int = 400, code: str = "invalid_request"):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class Selection:
    provider: Provider
    model: str | None
    label: str
    effort: str | None


def _text(value: object, field: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(p, dict) and p.get("type") == "text"
                                       and isinstance(p.get("text"), str) for p in value):
        return "\n".join(p["text"] for p in value)
    raise RequestError(f"{field} must contain text only; multimodal input is unsupported.")


def select(body: dict, providers: dict[str, Provider]) -> Selection:
    raw = body.get("model", "auto")
    if not isinstance(raw, str) or not raw or len(raw) > 128:
        raise RequestError("model must be a nonempty string of at most 128 characters.")
    explicit = body.get("provider")
    if explicit is not None and explicit not in ("codex", "claude"):
        raise RequestError("provider must be codex or claude.")
    prefix, slash, suffix = raw.partition("/")
    inferred = prefix if slash and prefix in ("codex", "claude") else None
    if inferred and explicit and inferred != explicit:
        raise RequestError("provider conflicts with model prefix.")
    name = explicit or inferred
    if not name and raw.startswith("claude-"):
        name = "claude"
    if not name and (raw.startswith("gpt-") or raw.startswith("o3")):
        name = "codex"
    if not name:
        name = "codex" if "codex" in providers else "claude"
    if name not in providers:
        raise RequestError(f"{name} CLI is not installed. Install it and sign in first.", 503,
                           "provider_unavailable")
    model = suffix if inferred else (None if raw == "auto" else raw)
    if model == "default":
        model = None
    if model is not None and (not model or any(c.isspace() for c in model)):
        raise RequestError("model name is invalid.")
    effort = body.get("reasoning_effort")
    reasoning = body.get("reasoning")
    if effort is None and isinstance(reasoning, dict):
        effort = reasoning.get("effort")
    if effort is not None and effort not in EFFORTS:
        raise RequestError("reasoning effort must be minimal, low, medium, high, xhigh, or max.")
    if name == "claude" and effort == "minimal":
        raise RequestError("Claude does not support minimal effort.")
    return Selection(providers[name], model, f"{name}/{model or 'default'}", effort)


def _reject(body: dict, *fields: str) -> None:
    used = [field for field in fields if field in body and body[field] not in (None, False, [], {})]
    if used:
        raise RequestError(f"Unsupported parameter: {', '.join(used)}.")


def _messages(body: dict, anthropic: bool) -> list[dict[str, str]]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a nonempty array.")
    normalized = []
    if anthropic and "system" in body:
        normalized.append({"role": "system", "content": _text(body["system"], "system")})
    roles = {"user", "assistant"} if anthropic else {"system", "developer", "user", "assistant"}
    for index, item in enumerate(messages):
        if not isinstance(item, dict) or item.get("role") not in roles:
            raise RequestError(f"messages[{index}].role is unsupported.")
        normalized.append({"role": item["role"],
                           "content": _text(item.get("content"), f"messages[{index}].content")})
    if normalized[-1]["role"] != "user":
        raise RequestError("The last message must have role user.")
    return normalized


def _prompt(messages: list[dict[str, str]]) -> str:
    import json
    return ("Answer the final user message in this conversation. Treat earlier assistant "
            "messages as context. Return only the answer text. Do not use tools or access "
            "local files.\n\nConversation (JSON):\n" +
            json.dumps(messages, ensure_ascii=False))


def complete(path: str, body: object, providers: dict[str, Provider], timeout: int) -> dict:
    if not isinstance(body, dict):
        raise RequestError("Request body must be a JSON object.")
    anthropic = path == "/v1/messages"
    if body.get("stream") is True:
        raise RequestError("Streaming is not supported yet.")
    if anthropic:
        _reject(body, "tools", "tool_choice", "thinking", "output_config")
    else:
        _reject(body, "tools", "tool_choice", "response_format", "functions",
                "function_call", "logprobs", "modalities", "audio", "n", "stop")
    selection = select(body, providers)
    messages = _messages(body, anthropic)
    result = generate(selection.provider, _prompt(messages), selection.model,
                      selection.effort, timeout)
    created = int(time.time())
    if anthropic:
        return {"id": f"msg_{uuid.uuid4().hex}", "type": "message", "role": "assistant",
                "model": selection.label, "content": [{"type": "text", "text": result.text}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": result.input_tokens or 0,
                          "output_tokens": result.output_tokens or 0}}
    return {"id": f"chatcmpl_{uuid.uuid4().hex}", "object": "chat.completion",
            "created": created, "model": selection.label,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": result.input_tokens or 0,
                      "completion_tokens": result.output_tokens or 0,
                      "total_tokens": (result.input_tokens or 0) + (result.output_tokens or 0)}}
