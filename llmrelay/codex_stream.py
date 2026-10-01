"""Incremental Codex text through the local app-server stdio protocol."""

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .images import ImageInput
from .providers import Provider, ProviderError, _clean_env, classify_error


def _lines(pipe, output: queue.Queue):
    try:
        for line in pipe:
            output.put(line)
    finally:
        output.put(None)


def _send(process, method: str, request_id: int | None, params: dict):
    message = {"method": method, "params": params}
    if request_id is not None:
        message["id"] = request_id
    try:
        process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        process.stdin.flush()
    except BrokenPipeError as exc:
        raise ProviderError("Codex app server closed while receiving the request.") from exc


def _stop_process(process):
    if process.poll() is None:
        if os.name == "nt":
            try:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=5, check=False)
            except (OSError, subprocess.TimeoutExpired):
                process.terminate()
        else:
            process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)
    if not process.stdin.closed:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
    process.stdout.close()


def stream_codex(provider: Provider, prompt: str, model: str | None,
                 effort: str | None, timeout: int,
                 images: list[ImageInput]):
    """Yield final-answer text deltas; stop the CLI when the iterator closes."""
    deadline = time.monotonic() + timeout
    with tempfile.TemporaryDirectory(prefix="llmrelay-") as directory:
        inputs = [{"type": "text", "text": prompt}]
        for index, image in enumerate(images, 1):
            path = Path(directory) / f"image-{index}{image.suffix}"
            path.write_bytes(image.data)
            inputs.append({"type": "localImage", "path": str(path)})
        try:
            command = (shutil.which("codex.exe") or provider.command) if os.name == "nt" else provider.command
            process = subprocess.Popen(
                [command, "app-server", "--stdio"], cwd=directory,
                env=_clean_env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                bufsize=1, creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        except OSError as exc:
            raise ProviderError(f"Could not start Codex app server: {exc}", 503,
                                "provider_unavailable") from exc
        output = queue.Queue()
        pending = []
        reader = threading.Thread(target=_lines, args=(process.stdout, output), daemon=True)
        reader.start()

        def receive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderError(f"Provider timed out after {timeout} seconds.", 504, "timeout")
            try:
                line = output.get(timeout=remaining)
            except queue.Empty as exc:
                raise ProviderError(f"Provider timed out after {timeout} seconds.", 504,
                                    "timeout") from exc
            if line is None:
                raise ProviderError("Codex app server closed before completing the response.")
            try:
                return json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProviderError("Codex app server sent invalid JSON.") from exc

        def reply(request_id):
            while True:
                event = receive()
                if event.get("id") == request_id:
                    if "error" in event:
                        message = str(event["error"].get("message", "Codex request failed."))
                        raise classify_error("codex", message)
                    return event.get("result") or {}
                if "id" in event and "method" in event:
                    process.stdin.write(json.dumps({"id": event["id"], "error": {
                        "code": -32601, "message": "LLMRelay does not handle server requests."}}) + "\n")
                    process.stdin.flush()
                else:
                    pending.append(event)

        try:
            _send(process, "initialize", 1, {"clientInfo": {
                "name": "llmrelay", "title": "LLMRelay", "version": "0.1.0"}})
            reply(1)
            _send(process, "initialized", None, {})
            if model is None:
                _send(process, "model/list", 4, {"limit": 50})
                listed = reply(4).get("data") or []
                default = next((entry for entry in listed if entry.get("isDefault")),
                               listed[0] if listed else None)
                if default:
                    model = default.get("model") or default.get("id")
            start = {"cwd": directory, "approvalPolicy": "never", "sandbox": "read-only",
                     "ephemeral": True, "serviceName": "llmrelay"}
            if model:
                start["model"] = model
            _send(process, "thread/start", 2, start)
            thread_id = reply(2).get("thread", {}).get("id")
            if not thread_id:
                raise ProviderError("Codex app server did not start a thread.")
            turn = {"threadId": thread_id, "input": inputs}
            if effort:
                turn["effort"] = effort
            _send(process, "turn/start", 3, turn)
            reply(3)
            seen = {}
            phases = {}
            final_text = None
            while True:
                event = pending.pop(0) if pending else receive()
                method = event.get("method")
                params = event.get("params") or {}
                if params.get("threadId") not in (None, thread_id):
                    continue
                if method == "item/started":
                    item = params.get("item") or {}
                    if item.get("type") == "agentMessage":
                        phases[item.get("id")] = item.get("phase")
                elif method == "item/agentMessage/delta":
                    delta = params.get("delta")
                    if (isinstance(delta, str) and delta and
                            phases.get(params.get("itemId")) != "commentary"):
                        item_id = params.get("itemId")
                        seen[item_id] = seen.get(item_id, "") + delta
                        yield delta
                elif method == "item/completed":
                    item = params.get("item") or {}
                    if (item.get("type") == "agentMessage" and
                            item.get("phase") != "commentary" and
                            isinstance(item.get("text"), str)):
                        final_text = item["text"]
                        item_id = item.get("id")
                        previous = seen.get(item_id, "")
                        if final_text.startswith(previous) and len(final_text) > len(previous):
                            yield final_text[len(previous):]
                        elif not previous and final_text:
                            yield final_text
                elif method == "turn/completed":
                    turn_result = params.get("turn") or {}
                    if turn_result.get("status") != "completed":
                        detail = str((turn_result.get("error") or {}).get("message", "Codex turn failed."))
                        raise classify_error("codex", detail)
                    if final_text is None:
                        raise ProviderError("Codex returned no final message.")
                    return
                elif method == "error":
                    detail = str((params.get("error") or {}).get("message", "Codex turn failed."))
                    raise classify_error("codex", detail)
                elif "id" in event and "method" in event:
                    process.stdin.write(json.dumps({"id": event["id"], "error": {
                        "code": -32601, "message": "LLMRelay does not handle server requests."}}) + "\n")
                    process.stdin.flush()
        finally:
            _stop_process(process)
            reader.join(timeout=1)
