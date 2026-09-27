import base64
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llmrelay.api import RequestError, complete
from llmrelay.images import ImageInput
from llmrelay.providers import Provider, Result, _run


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)
ENCODED = base64.b64encode(PNG).decode()
URL = f"data:image/png;base64,{ENCODED}"
PROVIDERS = {"codex": Provider("codex", "codex"),
             "claude": Provider("claude", "claude")}


class VisionTests(unittest.TestCase):
    @patch("llmrelay.api.generate", return_value=Result("A tiny image"))
    def test_all_three_request_formats_pass_image_bytes(self, generate):
        bodies = [
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": URL, "detail": "auto"}}]}]}),
            ("/v1/responses", {"input": [{"role": "user", "content": [
                {"type": "input_text", "text": "What is this?"},
                {"type": "input_image", "image_url": URL}]}]}),
            ("/v1/messages", {"provider": "claude", "messages": [{"role": "user",
                "content": [{"type": "text", "text": "What is this?"},
                            {"type": "image", "source": {"type": "base64",
                              "media_type": "image/png", "data": ENCODED}}]}]})
        ]
        for path, body in bodies:
            with self.subTest(path=path):
                complete(path, body, PROVIDERS, 10)
                args = generate.call_args.args
                self.assertIn("[Image 1]", args[1])
                self.assertEqual(args[5], [ImageInput(PNG, "image/png")])

    @patch("llmrelay.api.generate")
    def test_rejects_remote_urls_and_invalid_bytes(self, generate):
        urls = ["https://example.com/image.png", "data:image/png;base64,%%%%",
                f"data:image/jpeg;base64,{ENCODED}"]
        for url in urls:
            with self.subTest(url=url), self.assertRaises(RequestError):
                complete("/v1/chat/completions", {"messages": [{"role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": url}}]}]},
                    PROVIDERS, 10)
        generate.assert_not_called()

    def test_prior_message_images_are_rejected(self):
        with self.assertRaises(RequestError) as context:
            complete("/v1/responses", {"input": [
                {"role": "user", "content": [{"type": "input_image", "image_url": URL}]},
                {"role": "assistant", "content": "I see it"},
                {"role": "user", "content": "And now?"}]}, PROVIDERS, 10)
        self.assertIn("final user message", str(context.exception))

    def test_rejects_more_than_twenty_images(self):
        with self.assertRaises(RequestError) as context:
            complete("/v1/responses", {"input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": URL} for _ in range(21)]}]},
                PROVIDERS, 10)
        self.assertIn("At most 20 images", str(context.exception))

    @patch("llmrelay.providers.subprocess.run")
    def test_staged_file_is_only_present_during_cli_call(self, run):
        visited = []

        def inspect(args, **kwargs):
            folder = Path(kwargs["cwd"])
            files = list(folder.iterdir())
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].read_bytes(), PNG)
            visited.append(folder)
            if "codex" in args[0]:
                self.assertEqual(args[-3], "--image")
                self.assertEqual(args[-2], str(files[0]))
            else:
                self.assertIn(str(files[0]), kwargs["input"])
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        run.side_effect = inspect
        image = ImageInput(PNG, "image/png")
        _run(["codex", "exec", "-"], "prompt", 10, [image], "codex")
        _run(["claude", "-p", "prompt"], "prompt", 10, [image], "claude")
        self.assertTrue(all(not folder.exists() for folder in visited))


if __name__ == "__main__":
    unittest.main()
