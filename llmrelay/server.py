"""Loopback HTTP server with bounded requests and predictable JSON errors."""

import argparse
import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .api import MAX_BODY, RequestError, complete
from .providers import ProviderError, available


class RelayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, providers, timeout: int, concurrency: int):
        super().__init__(address, RelayHandler)
        self.providers = providers
        self.timeout = timeout
        self.slots = threading.BoundedSemaphore(concurrency)


class RelayHandler(BaseHTTPRequestHandler):
    server: RelayServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # Avoid logging prompts, auth headers, and provider output.
        print(f"{self.client_address[0]} {format % args}", flush=True)

    def _send(self, status: int, data: dict):
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def _error(self, error):
        self._send(error.status, {"error": {"message": str(error), "type": error.code,
                                            "code": error.code}})

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "providers": list(self.server.providers)})
        elif self.path == "/v1/models":
            models = [{"id": "auto", "object": "model", "owned_by": "llmrelay"}]
            models += [{"id": f"{name}/default", "object": "model", "owned_by": name}
                       for name in self.server.providers]
            self._send(200, {"object": "list", "data": models})
        else:
            self._error(RequestError("Endpoint not found.", 404, "not_found"))

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/v1/messages"):
            self._error(RequestError("Endpoint not found.", 404, "not_found"))
            return
        if self.headers.get("Transfer-Encoding"):
            self._error(RequestError("Chunked request bodies are unsupported.", 411))
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._error(RequestError("Content-Length is required.", 411))
            return
        if length < 0 or length > MAX_BODY:
            self._error(RequestError("Request body exceeds 1 MiB.", 413))
            self.close_connection = True
            return
        if not self.server.slots.acquire(blocking=False):
            self._error(RequestError("Relay is busy; retry shortly.", 429, "busy"))
            self.close_connection = True
            return
        try:
            try:
                body = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeDecodeError):
                raise RequestError("Invalid JSON request body.")
            data = complete(self.path, body, self.server.providers, self.server.timeout)
            self._send(200, data)
        except (RequestError, ProviderError) as exc:
            self._error(exc)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            traceback.print_exc()
            try:
                self._error(RequestError("Internal relay error.", 500, "internal_error"))
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            self.server.slots.release()


def main():
    parser = argparse.ArgumentParser(description="Local API backed by Codex or Claude CLI")
    parser.add_argument("--port", type=int, default=int(os.getenv("LLMRELAY_PORT", "8765")))
    parser.add_argument("--timeout", type=int, default=120, help="CLI timeout in seconds")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.timeout < 1 or args.concurrency < 1:
        parser.error("port, timeout, and concurrency must be positive and valid")
    providers = available()
    if not providers:
        parser.error("Install and sign in to Codex CLI and/or Claude Code CLI first.")
    try:
        server = RelayServer(("127.0.0.1", args.port), providers,
                             args.timeout, args.concurrency)
    except OSError as exc:
        parser.error(f"Could not bind to 127.0.0.1:{args.port}: {exc}")
    print(f"LLMRelay: http://127.0.0.1:{args.port}/v1", flush=True)
    print(f"Available: {', '.join(providers)} | default: "
          f"{'codex' if 'codex' in providers else 'claude'}", flush=True)
    print("OpenAI chat: /v1/chat/completions | Anthropic messages: /v1/messages", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping LLMRelay.")
    finally:
        server.server_close()
