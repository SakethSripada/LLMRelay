"""Request validation and the deliberately small compatibility surface."""

import time
import uuid
import re
from dataclasses import dataclass

from .providers import Provider, ProviderError, find_cli, generate, login_state


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
    provider = providers.get(name)
    if provider is None:
        command = find_cli(name)
        if not command:
            raise RequestError(f"{name} CLI is not installed. Install it and sign in first.",
                               503, "provider_unavailable")
        state = login_state(name, command)
        if state == "unknown":
            raise RequestError(f"Could not determine {name} sign-in status. Run: "
                               f"python -m llmrelay status", 503, "provider_unavailable")
        if state != "subscription":
            raise RequestError(f"{name} subscription sign-in is required. Run: "
                               f"python -m llmrelay login {name}", 401,
                               "authentication_error")
        provider = Provider(name, command)
        providers[name] = provider
    model = suffix if inferred else (None if raw == "auto" else raw)
    if model == "default":
        model = None
    if model is not None and not re.fullmatch(r"[A-Za-z0-9._-]+", model):
        raise RequestError("model name may contain only letters, numbers, dot, underscore, and hyphen.")
    effort = body.get("reasoning_effort")
    reasoning = body.get("reasoning")
    if reasoning is not None and (not isinstance(reasoning, dict) or
                                  set(reasoning) - {"effort"}):
        raise RequestError("reasoning must be an object with an effort field.")
    if effort is None and isinstance(reasoning, dict):
        effort = reasoning.get("effort")
    if effort is not None and (not isinstance(effort, str) or effort not in EFFORTS):
        raise RequestError("reasoning effort must be minimal, low, medium, high, xhigh, or max.")
    if name == "claude" and effort == "minimal":
        raise RequestError("Claude does not support minimal effort.")
    return Selection(provider, model, f"{name}/{model or 'default'}", effort)


def _reject(body: dict, *fields: str) -> None:
    used = [field for field in fields if field in body and body[field] is not None
            and body[field] is not False and body[field] != [] and body[field] != {}]
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
        if set(item) - {"role", "content"}:
            raise RequestError(f"messages[{index}] contains unsupported fields.")
        normalized.append({"role": item["role"],
                           "content": _text(item.get("content"), f"messages[{index}].content")})
    if normalized[-1]["role"] != "user":
        raise RequestError("The last message must have role user.")
    return normalized


def _prompt(messages: list[dict[str, str]], max_tokens: int | None) -> str:
    import json
    limit = f" Aim for at most {max_tokens} output tokens." if max_tokens else ""
    return ("Answer the final user message in this conversation. Treat earlier assistant "
            "messages as context. Return only the answer text. Do not use tools or access "
            f"local files.{limit}\n\nConversation (JSON):\n" +
            json.dumps(messages, ensure_ascii=False))


def _response_messages(body: dict) -> list[dict[str, str]]:
    messages = []
    instructions = body.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise RequestError("instructions must be a string.")
        messages.append({"role": "system", "content": instructions})
    source = body.get("input")
    if isinstance(source, str):
        messages.append({"role": "user", "content": source})
        return messages
    if not isinstance(source, list) or not source:
        raise RequestError("input must be text or a nonempty array of text messages.")
    for index, item in enumerate(source):
        if not isinstance(item, dict) or item.get("role") not in ("system", "developer", "user", "assistant"):
            raise RequestError(f"input[{index}].role is unsupported.")
        content = item.get("content")
        if isinstance(content, list):
            allowed = {"input_text", "output_text", "text"}
            if not all(isinstance(part, dict) and part.get("type") in allowed and
                       isinstance(part.get("text"), str) for part in content):
                raise RequestError(f"input[{index}].content must contain text blocks only.")
            content = "\n".join(part["text"] for part in content)
        if not isinstance(content, str):
            raise RequestError(f"input[{index}].content must be text.")
        messages.append({"role": item["role"], "content": content})
    if messages[-1]["role"] != "user":
        raise RequestError("The last input message must have role user.")
    return messages


def complete(path: str, body: object, providers: dict[str, Provider], timeout: int) -> dict:
    if not isinstance(body, dict):
        raise RequestError("Request body must be a JSON object.")
    anthropic = path == "/v1/messages"
    responses = path == "/v1/responses"
    if body.get("stream") not in (None, False):
        raise RequestError("Streaming is not supported yet.")
    if anthropic:
        _reject(body, "tools", "tool_choice", "thinking", "output_config",
                "temperature", "top_p", "top_k", "stop_sequences")
    elif responses:
        _reject(body, "tools", "tool_choice", "text", "previous_response_id",
                "parallel_tool_calls", "include", "temperature", "top_p", "metadata")
    else:
        _reject(body, "tools", "tool_choice", "response_format", "functions",
                "function_call", "logprobs", "modalities", "audio", "stop",
                "temperature", "top_p", "seed", "presence_penalty", "frequency_penalty",
                "logit_bias", "max_completion_tokens", "parallel_tool_calls")
        if "n" in body and (type(body["n"]) is not int or body["n"] != 1):
            raise RequestError("Only n=1 is supported.")
    max_tokens = body.get("max_output_tokens") if responses else body.get("max_tokens")
    if max_tokens is not None and (type(max_tokens) is not int or not 1 <= max_tokens <= 100000):
        raise RequestError("Output token limit must be an integer from 1 to 100000.")
    selection = select(body, providers)
    messages = _response_messages(body) if responses else _messages(body, anthropic)
    result = generate(selection.provider, _prompt(messages, max_tokens), selection.model,
                      selection.effort, timeout)
    created = int(time.time())
    if anthropic:
        return {"id": f"msg_{uuid.uuid4().hex}", "type": "message", "role": "assistant",
                "model": selection.label, "content": [{"type": "text", "text": result.text}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": result.input_tokens or 0,
                          "output_tokens": result.output_tokens or 0}}
    if responses:
        return {"id": f"resp_{uuid.uuid4().hex}", "object": "response",
                "created_at": created, "status": "completed", "model": selection.label,
                "output": [{"id": f"msg_{uuid.uuid4().hex}", "type": "message",
                            "status": "completed", "role": "assistant",
                            "content": [{"type": "output_text", "text": result.text,
                                         "annotations": []}]}],
                "output_text": result.text,
                "usage": {"input_tokens": result.input_tokens or 0,
                          "output_tokens": result.output_tokens or 0,
                          "total_tokens": (result.input_tokens or 0) + (result.output_tokens or 0)}}
    return {"id": f"chatcmpl_{uuid.uuid4().hex}", "object": "chat.completion",
            "created": created, "model": selection.label,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": result.input_tokens or 0,
                      "completion_tokens": result.output_tokens or 0,
                      "total_tokens": (result.input_tokens or 0) + (result.output_tokens or 0)}}
