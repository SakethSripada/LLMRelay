"""Small subprocess adapters. Credentials always stay with the vendor CLIs."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .images import ImageInput


class ProviderError(Exception):
    def __init__(self, message: str, status: int = 502, code: str = "provider_error"):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class Provider:
    name: str
    command: str


@dataclass(frozen=True)
class Result:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


def find_cli(name: str) -> str | None:
    # Python's which() can select an extensionless npm shim on Windows.
    names = (f"{name}.cmd", f"{name}.exe") if os.name == "nt" else (name,)
    return next((path for candidate in names if (path := shutil.which(candidate))), None)


def available() -> dict[str, Provider]:
    return {name: Provider(name, command) for name in ("codex", "claude")
            if (command := find_cli(name)) and login_state(name, command) == "subscription"}


def login_state(name: str, command: str) -> str:
    """Return subscription, api_key, signed_out, or unknown without exposing credentials."""
    args = [command, "login", "status"] if name == "codex" else [command, "auth", "status"]
    try:
        status = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=10, env=_clean_env(), check=False)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if status.returncode:
        return "signed_out"
    if name == "codex":
        output = status.stdout + status.stderr
        if "Logged in using ChatGPT" in output:
            return "subscription"
        if "API key" in output or "api key" in output:
            return "api_key"
        return "unknown"
    try:
        data = json.loads(status.stdout)
    except json.JSONDecodeError:
        return "unknown"
    if data.get("loggedIn") is not True:
        return "signed_out"
    if data.get("authMethod") == "claude.ai":
        return "subscription"
    if data.get("authMethod") in ("apiKey", "console"):
        return "api_key"
    return "unknown"


def _subscription_login(name: str, command: str) -> bool:
    return login_state(name, command) == "subscription"


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    # Prevent an ambient API key or cloud provider setting from silently billing
    # an API account instead of the account signed in through the CLI.
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY",
                "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
                "ANTHROPIC_BEDROCK_BASE_URL", "ANTHROPIC_VERTEX_BASE_URL",
                "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
                "CLAUDE_CODE_USE_FOUNDRY"):
        env.pop(key, None)
    return env


def _run(args: list[str], prompt: str, timeout: int,
         images: tuple[ImageInput, ...] | list[ImageInput] = (),
         provider: str | None = None) -> str:
    try:
        with tempfile.TemporaryDirectory(prefix="llmrelay-") as working_dir:
            paths = []
            for index, image in enumerate(images, 1):
                path = Path(working_dir) / f"image-{index}{image.suffix}"
                path.write_bytes(image.data)
                paths.append(path)
            if paths and provider == "codex":
                args = args[:-1] + [part for path in paths for part in ("--image", str(path))] + args[-1:]
            elif paths and provider == "claude":
                prompt += "\n\nStaged images (read these files only):\n" + "\n".join(
                    f"[Image {index}] {path}" for index, path in enumerate(paths, 1))
            result = subprocess.run(args, input=prompt, text=True, encoding="utf-8",
                                    errors="replace", capture_output=True, timeout=timeout,
                                    cwd=working_dir, env=_clean_env(), shell=False, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ProviderError(f"Provider timed out after {timeout} seconds.", 504, "timeout") from exc
    except OSError as exc:
        raise ProviderError(f"Could not start provider CLI: {exc}", 503, "provider_unavailable") from exc
    if result.returncode:
        name = Path(args[0]).stem
        if login_state(name, args[0]) in ("signed_out", "api_key"):
            raise ProviderError(f"{name} subscription sign-in is required. Run: "
                                f"python -m llmrelay login {name}", 401, "authentication_error")
        raise classify_error(name, result.stderr + "\n" + result.stdout)
    return result.stdout


def classify_error(name: str, detail: str) -> ProviderError:
    """Map known CLI failures to stable HTTP errors; do not return raw CLI logs."""
    lowered = detail.lower()
    login = f"python -m llmrelay login {name}"
    if name == "claude" and "--restricted" in lowered and re.search(
            r"unknown|unrecognized|unexpected|unsupported", lowered):
        return ProviderError("Claude Code is too old for restricted image reads. "
                             "Update Claude Code and retry.", 503, "provider_unavailable")
    if re.search(r"\b(401|unauthorized|unauthenticated|not logged in|login required)\b|auth(?:entication)? failed|authentication_error|oauth.*expir|session.*expir|token.*expir|invalid api key|no credentials", lowered):
        return ProviderError(f"{name} sign-in expired or was rejected. Run: {login}",
                             401, "authentication_error")
    if re.search(r"\b(429|rate limit|usage limit|quota|capacity limit|too many requests|monthly spend limit|daily limit)\b|you.ve hit.*limit", lowered):
        return ProviderError(f"{name} usage limit reached. Retry after the provider limit resets.",
                             429, "rate_limit_error")
    if re.search(r"(invalid|unknown|unsupported|unavailable|not found) model|model.{0,120}(invalid|unknown|unsupported|not supported|unavailable|not found)", lowered):
        return ProviderError(f"{name} rejected the requested model. Choose a model available to your account.",
                             400, "invalid_model")
    if re.search(r"(invalid|unsupported|corrupt|unreadable) image|image (.* )(invalid|unsupported|corrupt|unreadable)|failed to (decode|read|open) image", lowered):
        return ProviderError(f"{name} could not read the supplied image.", 400, "invalid_image")
    if re.search(r"econnreset|enotfound|connection refused|connection timed out|network error", lowered):
        return ProviderError(f"{name} could not reach its provider service. Check your network and retry.",
                             503, "provider_unavailable")
    return ProviderError(f"{name} CLI failed. Run the CLI directly to inspect its error.")


def generate(provider: Provider, prompt: str, model: str | None,
             effort: str | None, timeout: int,
             images: tuple[ImageInput, ...] | list[ImageInput] = ()) -> Result:
    if provider.name == "codex":
        args = [provider.command, "exec", "--json", "--sandbox", "read-only",
                "--skip-git-repo-check", "--ephemeral", "--ignore-user-config"]
        if model:
            args += ["--model", model]
        if effort:
            args += ["-c", f'model_reasoning_effort="{effort}"']
        args += ["-"]
        output = _run(args, prompt, timeout, images, "codex")
        answer = None
        usage = {}
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = event.get("item", {})
            if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                answer = item.get("text")
            if event.get("type") == "turn.failed":
                raise classify_error("codex", str(event.get("error", {})))
            if event.get("type") == "error":
                raise classify_error("codex", str(event.get("message", "")))
            if event.get("type") == "turn.completed":
                usage = event.get("usage", {})
        if not isinstance(answer, str):
            raise ProviderError("Codex returned no final message.")
        return Result(answer, usage.get("input_tokens"), usage.get("output_tokens"))

    args = [provider.command, "-p", "--output-format", "json"]
    if images:
        args += ["--restricted", "--tools", "Read", "--allowedTools", "Read",
                 "--permission-mode", "dontAsk"]
    else:
        args += ["--tools", ""]
    args += ["--disallowedTools", "mcp__*", "--no-session-persistence"]
    if model:
        args += ["--model", model]
    if effort:
        args += ["--effort", effort]
    args += ["Answer the conversation supplied on stdin."]
    output = _run(args, prompt, timeout, images, "claude")
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ProviderError("Claude returned invalid JSON.") from exc
    if data.get("is_error"):
        raise classify_error("claude", str(data.get("result", "")) + " " +
                             str(data.get("api_error_status", "")))
    if not isinstance(data.get("result"), str):
        raise ProviderError("Claude returned no final message.")
    usage = data.get("usage") or {}
    return Result(data["result"], usage.get("input_tokens"), usage.get("output_tokens"))
