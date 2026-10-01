import http.client
import json
import threading
import unittest
from unittest.mock import patch

from llmrelay.providers import Provider, ProviderError
from llmrelay.server import RelayServer


PROVIDERS = {name: Provider(name, name) for name in ("codex", "claude")}


def chunks(*parts):
    yield from parts


def failure(after_text=False):
    if after_text:
        yield "partial"
    raise ProviderError("Provider limit reached.", 429, "rate_limit_error")


class StreamingHTTPTests(unittest.TestCase):
    def setUp(self):
        self.server = RelayServer(("127.0.0.1", 0), PROVIDERS.copy(), 10, 1)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, path, body):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        connection.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        status, content_type, raw = response.status, response.getheader("Content-Type"), response.read()
        connection.close()
        return status, content_type, raw.decode("utf-8")

    def test_chat_streams_for_both_providers(self):
        with patch("llmrelay.server.stream_codex", return_value=chunks("hel", "lo")), \
             patch("llmrelay.server.stream_claude", return_value=chunks("hel", "lo")):
            for provider in PROVIDERS:
                status, content_type, raw = self.request("/v1/chat/completions", {
                    "provider": provider, "stream": True,
                    "messages": [{"role": "user", "content": "Hi"}]})
                self.assertEqual(status, 200)
                self.assertIn("text/event-stream", content_type)
                self.assertIn('"content":"hel"', raw)
                self.assertIn('"content":"lo"', raw)
                self.assertIn('"finish_reason":"stop"', raw)
                self.assertTrue(raw.endswith("data: [DONE]\n\n"))

    def test_responses_and_anthropic_events(self):
        with patch("llmrelay.server.stream_codex", return_value=chunks("yes")):
            status, _, raw = self.request("/v1/responses", {"stream": True, "input": "Hi"})
            self.assertEqual(status, 200)
            self.assertIn("event: response.output_text.delta", raw)
            self.assertIn("event: response.completed", raw)
        with patch("llmrelay.server.stream_claude", return_value=chunks("yes")):
            status, _, raw = self.request("/v1/messages", {"model": "claude/sonnet",
                "stream": True, "messages": [{"role": "user", "content": "Hi"}]})
            self.assertEqual(status, 200)
            self.assertIn("event: content_block_delta", raw)
            self.assertIn("event: message_stop", raw)

    def test_errors_before_and_after_stream_starts(self):
        with patch("llmrelay.server.stream_codex", side_effect=lambda *args: failure()):
            status, content_type, raw = self.request("/v1/responses", {
                "stream": True, "input": "Hi"})
            self.assertEqual(status, 429)
            self.assertIn("application/json", content_type)
            self.assertEqual(json.loads(raw)["error"]["code"], "rate_limit_error")
        with patch("llmrelay.server.stream_codex", side_effect=lambda *args: failure(True)):
            status, content_type, raw = self.request("/v1/responses", {
                "stream": True, "input": "Hi"})
            self.assertEqual(status, 200)
            self.assertIn("event: error", raw)
            self.assertNotIn("event: response.completed", raw)


if __name__ == "__main__":
    unittest.main()
