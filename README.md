# LLMRelay

LLMRelay runs a small HTTP server on your computer and sends text requests through the Codex or Claude CLI you already signed into. It never reads or stores their tokens. It has no Python package dependencies.

## Start

Install Python 3.10+ and at least one vendor CLI, then sign in with a **subscription account**: [Codex CLI](https://learn.chatgpt.com/docs/developer-commands?surface=cli) (`codex login`) or [Claude Code](https://code.claude.com/docs/en/cli-reference) (`claude auth login`). API-key CLI logins are deliberately ignored.

From a clone of this repository, run:

```sh
python -m llmrelay
```

On some Windows systems, use `py -m llmrelay`. The command needs no install step. It prints the base URL and available providers, then stays running until you press Ctrl+C. Default base URL: `http://127.0.0.1:8765/v1`. Pass `--port 9000` to change the port.

## Use

OpenAI chat completions:

```sh
curl http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello"}]}'
```

Anthropic messages:

```sh
curl http://127.0.0.1:8765/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello"}]}'
```

OpenAI Responses also accepts a text `input` at `POST /v1/responses`.

Point an OpenAI SDK's `base_url` at `http://127.0.0.1:8765/v1` and use any placeholder API key if the SDK requires one. Point an Anthropic SDK's `base_url` at `http://127.0.0.1:8765`. SDKs that require a model can use `auto`. LLMRelay does not authenticate HTTP requests because it only listens on loopback; do not expose the port through a tunnel or reverse proxy.

`model` is optional. The default is the signed-in Codex CLI's default model, or Claude's if Codex is unavailable. Use `"model":"claude/sonnet"`, `"model":"codex/<model-name>"`, or `"provider":"claude"` to route explicitly. Unprefixed `claude-*` and `gpt-*` names also route to the matching CLI. Set `"reasoning_effort":"low"` (or `"reasoning":{"effort":"low"}`) to choose effort when that model supports it. `GET /health` and `GET /v1/models` show basic status.

## Scope

This is a text adapter, not a full implementation of either API. Each request starts a fresh CLI process. It accepts system, user, and assistant text messages, including text blocks. It does not support streaming, images, tools, structured output, or conversation IDs. Parameters that require those features return a JSON error. `max_tokens` and `max_output_tokens` are best-effort instructions to the CLI, not hard limits. Usage counts are taken from CLI output when present; a zero means that CLI did not report a count. Provider errors, timeouts, and busy limits return JSON errors with non-200 status codes.

The CLIs own authentication and usage accounting. LLMRelay clears common API-key environment variables before launching them and only enables CLIs that report a subscription login. Check your provider's current plan terms and limits. [OpenAI documents `codex exec` for scripted runs](https://learn.chatgpt.com/docs/developer-commands?surface=cli); [Anthropic currently says `claude -p` and third-party app usage draw from subscription limits](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan). This can change.

Run tests with `python -m unittest discover -s tests`.
