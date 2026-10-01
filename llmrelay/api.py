"""Request validation and the deliberately small compatibility surface."""

import json
import time
import uuid
import re
from dataclasses import dataclass

from .images import ImageError, ImageInput, check_image_budget, decode_image, from_data_url
from .providers import Provider, ProviderError, find_cli, generate, login_state


MAX_BODY = 32 * 1024 * 1024
MAX_PROMPT_BYTES = 1024 * 1024
EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


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


@dataclass(frozen=True)
class Prepared:
    selection: Selection
    prompt: str
    images: list[ImageInput]


def _text(value: object, field: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(p, dict) and p.get("type") == "text"
                                       and isinstance(p.get("text"), str) for p in value):
        return "\n".join(p["text"] for p in value)
    raise RequestError(f"{field} must contain text blocks only.")


def _content(value: object, field: str, style: str, allow_images: bool,
             images: list[ImageInput]) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise RequestError(f"{field} must be text or an array of content blocks.")
    parts = []
    for index, block in enumerate(value):
        if not isinstance(block, dict):
            raise RequestError(f"{field}[{index}] must be an object.")
        kind = block.get("type")
        text_types = {"input_text", "output_text", "text"} if style == "responses" else {"text"}
        if kind in text_types and isinstance(block.get("text"), str):
            parts.append(block["text"])
            continue
        image_type = {"chat": "image_url", "responses": "input_image", "anthropic": "image"}[style]
        if kind != image_type:
            raise RequestError(f"{field}[{index}] has an unsupported content type.")
        if not allow_images:
            raise RequestError("Images are supported only in the final user message.")
        detail = block.get("detail")
        if style == "chat" and isinstance(block.get("image_url"), dict):
            detail = block["image_url"].get("detail", detail)
        if detail not in (None, "auto", "low", "high"):
            raise RequestError("Image detail must be auto, low, or high.")
        try:
            if style == "anthropic":
                source = block.get("source")
                if not isinstance(source, dict) or source.get("type") != "base64":
                    raise ImageError("Anthropic images require a base64 source.")
                image = decode_image(source.get("media_type"), source.get("data"))
            else:
                source = block.get("image_url")
                if style == "chat":
                    source = source.get("url") if isinstance(source, dict) else source
                image = from_data_url(source)
            images.append(image)
            check_image_budget(images)
        except ImageError as exc:
            raise RequestError(f"{field}[{index}]: {exc}") from exc
        parts.append(f"[Image {len(images)}]")
    return "\n".join(parts)


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
        raise RequestError("reasoning effort must be none, minimal, low, medium, high, xhigh, or max.")
    if name == "claude" and effort == "minimal":
        raise RequestError("Claude does not support minimal effort.")
    return Selection(provider, model, f"{name}/{model or 'default'}", None if effort == "none" else effort)


def _response_format(body: dict) -> str:
    text = body.get("text")
    if text is None:
        return ""
    if not isinstance(text, dict) or set(text) - {"format", "verbosity"}:
        raise RequestError("text must contain only format and verbosity.")
    if text.get("verbosity") not in (None, "low", "medium", "high"):
        raise RequestError("text.verbosity must be low, medium, or high.")
    fmt = text.get("format")
    if fmt is None or fmt == {"type": "text"}:
        return ""
    if not isinstance(fmt, dict):
        raise RequestError("text.format must be an object.")
    if fmt.get("type") == "json_object" and set(fmt) == {"type"}:
        return "Return only one valid JSON object, without Markdown fences or explanation."
    if fmt.get("type") != "json_schema" or not isinstance(fmt.get("schema"), dict):
        raise RequestError("text.format must be text, json_object, or json_schema with a schema.")
    if set(fmt) - {"type", "name", "schema", "strict", "description"}:
        raise RequestError("text.format has unsupported fields.")
    return ("Return only one JSON value matching this JSON Schema exactly, without Markdown "
            "fences or explanation. Schema: " + json.dumps(fmt["schema"], separators=(",", ":")))


def _reject(body: dict, *fields: str) -> None:
    used = [field for field in fields if field in body and body[field] is not None
            and body[field] is not False and body[field] != [] and body[field] != {}]
    if used:
        raise RequestError(f"Unsupported parameter: {', '.join(used)}.")


def _messages(body: dict, anthropic: bool) -> tuple[list[dict[str, str]], list[ImageInput]]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a nonempty array.")
    normalized = []
    images = []
    if anthropic and "system" in body:
        normalized.append({"role": "system", "content": _text(body["system"], "system")})
    roles = {"user", "assistant"} if anthropic else {"system", "developer", "user", "assistant"}
    for index, item in enumerate(messages):
        if not isinstance(item, dict) or item.get("role") not in roles:
            raise RequestError(f"messages[{index}].role is unsupported.")
        if set(item) - {"role", "content"}:
            raise RequestError(f"messages[{index}] contains unsupported fields.")
        normalized.append({"role": item["role"],
                           "content": _content(item.get("content"), f"messages[{index}].content",
                                               "anthropic" if anthropic else "chat",
                                               index == len(messages) - 1 and item["role"] == "user",
                                               images)})
    if normalized[-1]["role"] != "user":
        raise RequestError("The last message must have role user.")
    return normalized, images


def _prompt(messages: list[dict[str, str]], max_tokens: int | None,
            has_images: bool) -> str:
    import json
    limit = f" Aim for at most {max_tokens} output tokens." if max_tokens else ""
    access = ("Use the supplied images. For Claude, read only the staged image paths "
              "listed after the conversation." if has_images else
              "Do not use tools or access local files.")
    return ("Answer the final user message in this conversation. Treat earlier assistant "
            f"messages as context. Return only the answer text. {access}{limit}"
            "\n\nConversation (JSON):\n" +
            json.dumps(messages, ensure_ascii=False))


def _response_messages(body: dict) -> tuple[list[dict[str, str]], list[ImageInput]]:
    messages = []
    images = []
    instructions = body.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise RequestError("instructions must be a string.")
        messages.append({"role": "system", "content": instructions})
    source = body.get("input")
    if isinstance(source, str):
        messages.append({"role": "user", "content": source})
        return messages, images
    if not isinstance(source, list) or not source:
        raise RequestError("input must be text or a nonempty array of text messages.")
    for index, item in enumerate(source):
        if not isinstance(item, dict) or item.get("role") not in ("system", "developer", "user", "assistant"):
            raise RequestError(f"input[{index}].role is unsupported.")
        content = _content(item.get("content"), f"input[{index}].content", "responses",
                           index == len(source) - 1 and item["role"] == "user", images)
        messages.append({"role": item["role"], "content": content})
    if messages[-1]["role"] != "user":
        raise RequestError("The last input message must have role user.")
    return messages, images


def prepare(path: str, body: object, providers: dict[str, Provider]) -> Prepared:
    if not isinstance(body, dict):
        raise RequestError("Request body must be a JSON object.")
    anthropic = path == "/v1/messages"
    responses = path == "/v1/responses"
    if "stream" in body and type(body["stream"]) is not bool:
        raise RequestError("stream must be a boolean.")
    if anthropic:
        _reject(body, "tools", "tool_choice", "thinking", "output_config",
                "temperature", "top_p", "top_k", "stop_sequences")
    elif responses:
        _reject(body, "tools", "tool_choice", "previous_response_id",
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
    messages, images = _response_messages(body) if responses else _messages(body, anthropic)
    prompt = _prompt(messages, max_tokens, bool(images))
    if responses:
        format_instruction = _response_format(body)
        if format_instruction:
            prompt += "\n\nOutput requirement: " + format_instruction
    if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise RequestError("Text prompt exceeds 1 MiB.", 413)
    return Prepared(selection, prompt, images)


def complete(path: str, body: object, providers: dict[str, Provider], timeout: int) -> dict:
    if isinstance(body, dict) and body.get("stream") is True:
        raise RequestError("Streaming requests must use the streaming response path.")
    prepared = prepare(path, body, providers)
    selection = prepared.selection
    anthropic = path == "/v1/messages"
    responses = path == "/v1/responses"
    result = generate(selection.provider, prepared.prompt,
                      selection.model, selection.effort, timeout, prepared.images)
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
