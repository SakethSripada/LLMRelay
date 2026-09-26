"""Small subprocess adapters. Credentials always stay with the vendor CLIs."""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


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
            if (command := find_cli(name))}


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    # Prevent an ambient API key or cloud provider setting from silently billing
    # an API account instead of the account signed in through the CLI.
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY",
                "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK",
                "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY"):
        env.pop(key, None)
    return env


def _run(args: list[str], prompt: str, timeout: int) -> str:
    try:
        result = subprocess.run(args, input=prompt, text=True, encoding="utf-8",
                                errors="replace", capture_output=True, timeout=timeout,
                                cwd=Path(__file__).resolve().parent.parent, env=_clean_env(),
                                shell=False, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ProviderError(f"Provider timed out after {timeout} seconds.", 504, "timeout") from exc
    except OSError as exc:
        raise ProviderError(f"Could not start provider CLI: {exc}", 503, "provider_unavailable") from exc
    if result.returncode:
        # CLI stderr may contain prompt fragments or local paths. Keep the API
        # error short and avoid echoing provider logs to unrelated callers.
        detail = (result.stderr or result.stdout).strip().splitlines()
        last = detail[-1][:300] if detail else f"exit code {result.returncode}"
        raise ProviderError(f"{Path(args[0]).stem} failed: {last}")
    return result.stdout


def generate(provider: Provider, prompt: str, model: str | None,
             effort: str | None, timeout: int) -> Result:
    if provider.name == "codex":
        args = [provider.command, "exec", "--json", "--sandbox", "read-only",
                "--skip-git-repo-check", "--ephemeral", "--ignore-user-config"]
        if model:
            args += ["--model", model]
        if effort:
            args += ["-c", f'model_reasoning_effort="{effort}"']
        args += ["-"]
        output = _run(args, prompt, timeout)
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
                raise ProviderError("Codex reported a failed turn.")
            if event.get("type") == "turn.completed":
                usage = event.get("usage", {})
        if not isinstance(answer, str):
            raise ProviderError("Codex returned no final message.")
        return Result(answer, usage.get("input_tokens"), usage.get("output_tokens"))

    args = [provider.command, "-p", "--output-format", "json", "--tools", "",
            "--disallowedTools", "mcp__*", "--no-session-persistence"]
    if model:
        args += ["--model", model]
    if effort:
        args += ["--effort", effort]
    output = _run(args, prompt, timeout)
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ProviderError("Claude returned invalid JSON.") from exc
    if data.get("is_error") or not isinstance(data.get("result"), str):
        raise ProviderError("Claude returned an error or no final message.")
    usage = data.get("usage") or {}
    return Result(data["result"], usage.get("input_tokens"), usage.get("output_tokens"))
