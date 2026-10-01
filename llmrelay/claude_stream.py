"""Incremental Claude Code text from its documented stream-json mode."""

import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .codex_stream import _lines, _stop_process
from .images import ImageInput
from .providers import Provider, ProviderError, _clean_env, classify_error


def stream_claude(provider: Provider, prompt: str, model: str | None,
                  effort: str | None, timeout: int, images: list[ImageInput]):
    deadline = time.monotonic() + timeout
    with tempfile.TemporaryDirectory(prefix="llmrelay-") as directory:
        args = [provider.command, "-p", "--output-format", "stream-json", "--verbose",
                "--include-partial-messages"]
        if images:
            args += ["--restricted", "--tools", "Read", "--allowedTools", "Read",
                     "--permission-mode", "dontAsk"]
            paths = []
            for index, image in enumerate(images, 1):
                path = Path(directory) / f"image-{index}{image.suffix}"
                path.write_bytes(image.data)
                paths.append(path)
            prompt += "\n\nStaged images (read these files only):\n" + "\n".join(
                f"[Image {index}] {path}" for index, path in enumerate(paths, 1))
        else:
            args += ["--tools", ""]
        args += ["--disallowedTools", "mcp__*", "--no-session-persistence"]
        if model:
            args += ["--model", model]
        if effort:
            args += ["--effort", effort]
        args += ["Answer the conversation supplied on stdin."]
        try:
            process = subprocess.Popen(
                args, cwd=directory, env=_clean_env(), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                encoding="utf-8", errors="replace", bufsize=1,
                creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        except OSError as exc:
            raise ProviderError(f"Could not start Claude Code: {exc}", 503,
                                "provider_unavailable") from exc
        output = queue.Queue()
        reader = threading.Thread(target=_lines, args=(process.stdout, output), daemon=True)
        reader.start()
        try:
            try:
                process.stdin.write(prompt)
                process.stdin.close()
            except BrokenPipeError as exc:
                raise ProviderError("Claude Code closed while receiving the request.") from exc
            seen = ""
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProviderError(f"Provider timed out after {timeout} seconds.", 504, "timeout")
                try:
                    line = output.get(timeout=remaining)
                except queue.Empty as exc:
                    raise ProviderError(f"Provider timed out after {timeout} seconds.", 504,
                                        "timeout") from exc
                if line is None:
                    raise ProviderError("Claude Code closed before completing the response.")
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ProviderError("Claude Code sent invalid stream JSON.") from exc
                if item.get("type") == "stream_event":
                    delta = (item.get("event") or {}).get("delta") or {}
                    if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                        text = delta["text"]
                        seen += text
                        if text:
                            yield text
                elif item.get("type") == "result":
                    if item.get("is_error"):
                        raise classify_error("claude", str(item.get("result", "")) + " " +
                                             str(item.get("api_error_status", "")))
                    result = item.get("result")
                    if not isinstance(result, str):
                        raise ProviderError("Claude returned no final message.")
                    if result.startswith(seen) and len(result) > len(seen):
                        yield result[len(seen):]
                    elif not seen and result:
                        yield result
                    return
        finally:
            _stop_process(process)
            reader.join(timeout=1)
