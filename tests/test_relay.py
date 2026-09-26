import http.client
import json
import threading
import unittest
from unittest.mock import patch

from llmrelay.api import RequestError, complete, select
from llmrelay.providers import (Provider, ProviderError, Result, _subscription_login,
                                classify_error, generate, login_state)
from llmrelay.server import RelayServer


PROVIDERS = {"codex": Provider("codex", "codex"),
             "claude": Provider("claude", "claude")}


class RoutingTests(unittest.TestCase):
    def test_default_and_explicit_routing(self):
        self.assertEqual(select({}, PROVIDERS).provider.name, "codex")
        self.assertEqual(select({"model": "claude/sonnet"}, PROVIDERS).model, "sonnet")
        self.assertEqual(select({"model": "claude-sonnet-5"}, PROVIDERS).provider.name, "claude")
        self.assertEqual(select({"provider": "claude"}, PROVIDERS).model, None)

    def test_conflicts_and_missing_cli(self):
        with self.assertRaises(RequestError):
            select({"provider": "codex", "model": "claude/sonnet"}, PROVIDERS)
        with self.assertRaises(RequestError):
            select({"model": "codex/gpt&calc"}, PROVIDERS)
        with self.assertRaises(RequestError):
            select({"reasoning_effort": {}}, PROVIDERS)
        with patch("llmrelay.api.find_cli", return_value=None), self.assertRaises(RequestError) as context:
            select({"provider": "claude"}, {"codex": PROVIDERS["codex"]})
        self.assertEqual(context.exception.status, 503)

    @patch("llmrelay.api.find_cli", return_value="claude")
    @patch("llmrelay.api.login_state", return_value="signed_out")
    def test_unauthed_provider_gives_sign_in_command(self, *_):
        with self.assertRaises(RequestError) as context:
            select({"provider": "claude"}, {"codex": PROVIDERS["codex"]})
        self.assertEqual(context.exception.status, 401)
        self.assertIn("login claude", str(context.exception))

    @patch("llmrelay.api.generate", return_value=Result("Hello", 12, 3))
    def test_openai_and_anthropic_shapes(self, mock_generate):
        chat = complete("/v1/chat/completions", {"messages": [
            {"role": "user", "content": "Hi"}]}, PROVIDERS, 10)
        self.assertEqual(chat["choices"][0]["message"]["content"], "Hello")
        self.assertEqual(chat["usage"]["total_tokens"], 15)
        msg = complete("/v1/messages", {"system": "Be brief", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]}]}, PROVIDERS, 10)
        self.assertEqual(msg["content"][0]["text"], "Hello")
        response = complete("/v1/responses", {"input": "Hi"}, PROVIDERS, 10)
        self.assertEqual(response["output"][0]["content"][0]["text"], "Hello")
        self.assertEqual(mock_generate.call_count, 3)

    def test_unsupported_input_fails_before_cli_call(self):
        for body in ({"stream": True, "messages": [{"role": "user", "content": "Hi"}]},
                     {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
                     {"tools": [{"type": "function"}], "messages": [
                         {"role": "user", "content": "Hi"}]}):
            with self.subTest(body=body), self.assertRaises(RequestError):
                complete("/v1/chat/completions", body, PROVIDERS, 10)

    @patch("llmrelay.providers._run")
    def test_cli_parsing(self, run):
        run.return_value = '\n'.join([
            '{"type":"item.completed","item":{"type":"agent_message","text":"yes"}}',
            '{"type":"turn.completed","usage":{"input_tokens":2,"output_tokens":1}}'])
        self.assertEqual(generate(PROVIDERS["codex"], "Hi", None, "low", 10), Result("yes", 2, 1))
        self.assertIn("--sandbox", run.call_args.args[0])
        run.return_value = '{"result":"okay","usage":{"input_tokens":3,"output_tokens":2}}'
        self.assertEqual(generate(PROVIDERS["claude"], "Hi", "sonnet", None, 10),
                         Result("okay", 3, 2))

    @patch("llmrelay.providers.subprocess.run")
    def test_subscription_detection_rejects_api_login(self, run):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        run.return_value.stdout = "Logged in using ChatGPT"
        self.assertTrue(_subscription_login("codex", "codex"))
        run.return_value.stdout = "Logged in using API key"
        self.assertFalse(_subscription_login("codex", "codex"))
        run.return_value.stdout = '{"loggedIn":true,"authMethod":"apiKey"}'
        self.assertFalse(_subscription_login("claude", "claude"))
        run.return_value.stdout = '{"loggedIn":true,"authMethod":"claude.ai"}'
        self.assertTrue(_subscription_login("claude", "claude"))

    def test_cli_failure_categories(self):
        cases = [("OAuth session expired", 401, "authentication_error"),
                 ("429 rate limit exceeded", 429, "rate_limit_error"),
                 ("invalid model", 400, "invalid_model"),
                 ("connection refused", 503, "provider_unavailable"),
                 ("unexpected failure", 502, "provider_error")]
        for message, status, code in cases:
            with self.subTest(message=message):
                error = classify_error("claude", message)
                self.assertEqual((error.status, error.code), (status, code))

    @patch("llmrelay.providers.subprocess.run")
    def test_auth_status_from_codex_stderr(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = ""
        run.return_value.stderr = "Logged in using ChatGPT"
        self.assertEqual(login_state("codex", "codex"), "subscription")


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.server = RelayServer(("127.0.0.1", 0), PROVIDERS, 10, 1)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method, path, data=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        headers = {"Content-Type": "application/json"}
        connection.request(method, path, body=json.dumps(data) if data is not None else None,
                           headers=headers)
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    @patch("llmrelay.api.generate", return_value=Result("pong"))
    def test_http_round_trip_and_errors(self, _):
        status, health = self.request("GET", "/health")
        self.assertEqual((status, health["status"]), (200, "ok"))
        status, models = self.request("GET", "/v1/models")
        self.assertIn("auto", [item["id"] for item in models["data"]])
        status, data = self.request("POST", "/v1/chat/completions", {"messages": [
            {"role": "user", "content": "ping"}]})
        self.assertEqual((status, data["choices"][0]["message"]["content"]), (200, "pong"))
        status, data = self.request("POST", "/v1/responses", {"input": "ping"})
        self.assertEqual((status, data["output_text"]), (200, "pong"))
        status, error = self.request("POST", "/v1/messages", {"messages": []})
        self.assertEqual(status, 400)
        self.assertIn("messages", error["error"]["message"])

    def test_rejects_browser_origin(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        connection.request("POST", "/v1/responses", body='{"input":"hello"}', headers={
            "Content-Type": "application/json", "Origin": "https://example.com"})
        response = connection.getresponse()
        self.assertEqual(response.status, 403)
        response.read()
        connection.close()

    @patch("llmrelay.api.generate", side_effect=ProviderError("provider unavailable"))
    def test_provider_failure_is_json(self, _):
        status, data = self.request("POST", "/v1/chat/completions", {"messages": [
            {"role": "user", "content": "ping"}]})
        self.assertEqual(status, 502)
        self.assertEqual(data["error"]["code"], "provider_error")


if __name__ == "__main__":
    unittest.main()
