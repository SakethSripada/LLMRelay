import io
import unittest
from unittest.mock import patch

from llmrelay.auth import first_run_sign_in, sign_in


class AuthenticationTests(unittest.TestCase):
    @patch("llmrelay.auth.login_state", return_value="subscription")
    @patch("llmrelay.auth.subprocess.run")
    @patch("llmrelay.auth.sys.stdin.isatty", return_value=True)
    @patch("llmrelay.auth.find_cli", return_value="codex.cmd")
    def test_login_delegates_to_vendor_cli(self, _, tty, run, status):
        run.return_value.returncode = 0
        self.assertTrue(sign_in("codex"))
        self.assertEqual(run.call_args.args[0], ["codex.cmd", "login"])
        status.assert_called_once()

    @patch("llmrelay.auth.subprocess.run")
    @patch("llmrelay.auth.sys.stdin.isatty", return_value=False)
    @patch("llmrelay.auth.find_cli", return_value="claude")
    def test_noninteractive_login_explains_next_step(self, _, tty, run):
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertFalse(sign_in("claude"))
        self.assertIn("interactive terminal", stderr.getvalue())
        run.assert_not_called()

    @patch("llmrelay.auth.sign_in", return_value=True)
    @patch("llmrelay.auth.sys.stdin.isatty", return_value=True)
    @patch("llmrelay.auth.status_lines", return_value=["codex: signed_out"])
    @patch("llmrelay.auth.find_cli", side_effect=lambda name: "codex" if name == "codex" else None)
    def test_first_run_starts_sign_in(self, _, status, tty, sign_in_mock):
        self.assertTrue(first_run_sign_in())
        sign_in_mock.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
