"""Translate Codex text deltas into OpenAI server-sent events."""

import json
import time
import uuid


def event(data: dict, name: str | None = None) -> bytes:
    prefix = f"event: {name}\n" if name else ""
    return (prefix + "data: " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) +
            "\n\n").encode("utf-8")


def chat_events(deltas, model: str):
    request_id = f"chatcmpl_{uuid.uuid4().hex}"
    created = int(time.time())

    def chunk(delta: dict, finish_reason=None):
        return {"id": request_id, "object": "chat.completion.chunk", "created": created,
                "model": model, "choices": [{"index": 0, "delta": delta,
                                             "finish_reason": finish_reason}]}

    yield event(chunk({"role": "assistant", "content": ""}))
    for text in deltas:
        yield event(chunk({"content": text}))
    yield event(chunk({}, "stop"))
    yield b"data: [DONE]\n\n"


def response_events(deltas, model: str):
    request_id = f"resp_{uuid.uuid4().hex}"
    message_id = f"msg_{uuid.uuid4().hex}"
    created = int(time.time())
    response = {"id": request_id, "object": "response", "created_at": created,
                "status": "in_progress", "model": model, "output": [],
                "usage": None}
    item = {"id": message_id, "type": "message", "status": "in_progress",
            "role": "assistant", "content": []}
    part = {"type": "output_text", "text": "", "annotations": []}
    sequence = 0

    def emit(name: str, **fields):
        nonlocal sequence
        sequence += 1
        return event({"type": name, "sequence_number": sequence, **fields}, name)

    yield emit("response.created", response=response.copy())
    yield emit("response.in_progress", response=response.copy())
    yield emit("response.output_item.added", response_id=request_id, output_index=0,
               item=item.copy())
    yield emit("response.content_part.added", response_id=request_id, item_id=message_id,
               output_index=0, content_index=0, part=part.copy())
    text_parts = []
    for text in deltas:
        text_parts.append(text)
        yield emit("response.output_text.delta", response_id=request_id, item_id=message_id,
                   output_index=0, content_index=0, delta=text)
    full_text = "".join(text_parts)
    part["text"] = full_text
    yield emit("response.output_text.done", response_id=request_id, item_id=message_id,
               output_index=0, content_index=0, text=full_text)
    yield emit("response.content_part.done", response_id=request_id, item_id=message_id,
               output_index=0, content_index=0, part=part)
    item["content"] = [part]
    item["status"] = "completed"
    yield emit("response.output_item.done", response_id=request_id, output_index=0, item=item)
    response["output"] = [item]
    response["status"] = "completed"
    response["output_text"] = full_text
    response["usage"] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    yield emit("response.completed", response=response)


def message_events(deltas, model: str):
    message = {"id": f"msg_{uuid.uuid4().hex}", "type": "message", "role": "assistant",
               "model": model, "content": [], "stop_reason": None, "stop_sequence": None,
               "usage": {"input_tokens": 0, "output_tokens": 0}}
    yield event({"type": "message_start", "message": message}, "message_start")
    yield event({"type": "content_block_start", "index": 0,
                 "content_block": {"type": "text", "text": ""}}, "content_block_start")
    for text in deltas:
        yield event({"type": "content_block_delta", "index": 0,
                     "delta": {"type": "text_delta", "text": text}}, "content_block_delta")
    yield event({"type": "content_block_stop", "index": 0}, "content_block_stop")
    yield event({"type": "message_delta", "delta": {"stop_reason": "end_turn",
                  "stop_sequence": None}, "usage": {"output_tokens": 0}}, "message_delta")
    yield event({"type": "message_stop"}, "message_stop")
