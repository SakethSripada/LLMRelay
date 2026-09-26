"""Interactive sign-in through the installed vendor CLIs."""

import subprocess
import sys

from .providers import _clean_env, find_cli, login_state


INSTALL_URLS = {
    "codex": "https://developers.openai.com/codex/cli",
    "claude": "https://code.claude.com/docs/en/quickstart",
}


def status_lines() -> list[str]:
    lines = []
    for name in ("codex", "claude"):
        command = find_cli(name)
        state = login_state(name, command) if command else "not installed"
        lines.append(f"{name}: {state}")
    return lines


def choose_provider(target: str | None) -> str | None:
    if target:
        return target
    installed = [name for name in ("codex", "claude") if find_cli(name)]
    if len(installed) == 1:
        return installed[0]
    if not installed:
        return None
    if not sys.stdin.isatty():
        return None
    print("Choose a subscription to sign in:")
    for index, name in enumerate(installed, 1):
        print(f"  {index}. {name}")
    try:
        choice = input("Provider [1]: ").strip() or "1"
    except EOFError:
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(installed):
        return installed[int(choice) - 1]
    if choice in installed:
        return choice
    print("Choose a listed provider.", file=sys.stderr)
    return None


def sign_in(target: str | None = None) -> bool:
    name = choose_provider(target)
    if not name:
        print("Specify a provider: python -m llmrelay login codex|claude", file=sys.stderr)
        return False
    command = find_cli(name)
    if not command:
        print(f"{name} CLI is not installed. Install it from {INSTALL_URLS[name]}",
              file=sys.stderr)
        return False
    if not sys.stdin.isatty():
        print("Sign-in needs an interactive terminal. Run this command in a terminal:",
              file=sys.stderr)
        print(f"  python -m llmrelay login {name}", file=sys.stderr)
        return False
    args = [command, "login"] if name == "codex" else [command, "auth", "login"]
    print(f"Opening {name} sign-in. Complete the vendor's browser or terminal flow.", flush=True)
    try:
        result = subprocess.run(args, env=_clean_env(), check=False)
    except OSError as exc:
        print(f"Could not start {name} CLI: {exc}", file=sys.stderr)
        return False
    if result.returncode:
        print(f"{name} sign-in did not complete (exit code {result.returncode}).",
              file=sys.stderr)
        return False
    state = login_state(name, command)
    if state != "subscription":
        print(f"{name} reported {state}. Sign in with a subscription account, "
              "not an API key.", file=sys.stderr)
        return False
    print(f"{name} subscription is ready.", flush=True)
    return True


def first_run_sign_in() -> bool:
    installed = [name for name in ("codex", "claude") if find_cli(name)]
    if not installed:
        print("No supported CLI found. Install Codex or Claude Code:", file=sys.stderr)
        for name, url in INSTALL_URLS.items():
            print(f"  {name}: {url}", file=sys.stderr)
        return False
    print("No signed-in subscription found.", flush=True)
    for line in status_lines():
        print(f"  {line}", flush=True)
    if not sys.stdin.isatty():
        print("Run python -m llmrelay login codex|claude in an interactive terminal.",
              file=sys.stderr)
        return False
    return sign_in()
