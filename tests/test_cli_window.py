import io
import sys
import types
import unittest
from unittest.mock import MagicMock, Mock, patch

from REQreate import cli


class CliWindowTests(unittest.TestCase):
    def test_window_with_pywebview_available(self):
        mock_proc = Mock()
        mock_proc.poll.return_value = None

        mock_webview = MagicMock()
        mock_webview.settings = {}

        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch("REQreate.cli._wait_for_health") as mock_health, \
             patch.dict(sys.modules, {"webview": mock_webview}):
            result = cli.main(["app", "--window"])

        self.assertEqual(result, 0)
        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        self.assertEqual(cmd[:4], [sys.executable, "-m", "streamlit", "run"])
        self.assertEqual(cmd[4], cli.APP_PATH)
        self.assertIn("--server.headless", cmd)
        headless_idx = cmd.index("--server.headless")
        self.assertEqual(cmd[headless_idx + 1], "true")
        self.assertIn("--server.port", cmd)
        port_idx = cmd.index("--server.port")
        port = cmd[port_idx + 1]

        expected_url = f"http://localhost:{port}"
        mock_health.assert_called_once_with(expected_url, mock_proc)
        self.assertTrue(mock_webview.settings.get("ALLOW_DOWNLOADS"))
        mock_webview.create_window.assert_called_once_with(
            "REQreate", expected_url, width=1280, height=900
        )
        mock_webview.start.assert_called_once()
        mock_proc.terminate.assert_called_once()

    def test_window_with_custom_port(self):
        mock_proc = Mock()
        mock_proc.poll.return_value = None

        mock_webview = MagicMock()
        mock_webview.settings = {}

        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch("REQreate.cli._wait_for_health") as mock_health, \
             patch.dict(sys.modules, {"webview": mock_webview}):
            result = cli.main(["app", "--window", "--port", "9999"])

        self.assertEqual(result, 0)
        cmd = mock_popen.call_args[0][0]
        port_idx = cmd.index("--server.port")
        self.assertEqual(cmd[port_idx + 1], "9999")
        mock_health.assert_called_once_with("http://localhost:9999", mock_proc)
        mock_webview.create_window.assert_called_once_with(
            "REQreate", "http://localhost:9999", width=1280, height=900
        )

    def test_window_with_pywebview_missing_falls_back_to_browser(self):
        stdout = io.StringIO()
        mock_stcli = Mock()
        mock_stcli.main.return_value = 0

        with patch.dict(sys.modules, {"webview": None, "streamlit.web.cli": mock_stcli}), \
             patch("sys.stdout", stdout), \
             patch("sys.exit") as mock_exit:
            cli.main(["app", "--window"])

        self.assertIn('pip install "reqreate[app,desktop]"', stdout.getvalue())
        mock_stcli.main.assert_called_once()
        mock_exit.assert_called_once_with(0)

    def test_window_start_raising_terminates_proc_and_falls_back_to_browser(self):
        mock_proc = Mock()
        mock_proc.poll.return_value = None

        mock_webview = MagicMock()
        mock_webview.settings = {}
        mock_webview.start.side_effect = RuntimeError("Could not initialize GUI")

        stdout = io.StringIO()
        mock_stcli = Mock()
        mock_stcli.main.return_value = 0

        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch("REQreate.cli._wait_for_health"), \
             patch.dict(sys.modules, {"webview": mock_webview, "streamlit.web.cli": mock_stcli}), \
             patch("sys.stdout", stdout), \
             patch("sys.exit") as mock_exit:
            cli.main(["app", "--window"])

        mock_proc.terminate.assert_called_once()
        self.assertIn("Could not open desktop window", stdout.getvalue())
        mock_stcli.main.assert_called_once()
        mock_exit.assert_called_once_with(0)

    def test_without_window_behaves_as_before(self):
        mock_stcli = Mock()
        mock_stcli.main.return_value = 0

        with patch.dict(sys.modules, {"streamlit.web.cli": mock_stcli}), \
             patch("subprocess.Popen") as mock_popen, \
             patch("sys.exit") as mock_exit:
            cli.main(["app"])

        mock_popen.assert_not_called()
        mock_stcli.main.assert_called_once()
        self.assertEqual(
            sys.argv,
            [
                "streamlit", "run", cli.APP_PATH,
                "--server.port", "8501",
                "--server.address", "localhost",
            ]
        )
        mock_exit.assert_called_once_with(0)

    def test_without_window_with_custom_port_and_no_browser(self):
        mock_stcli = Mock()
        mock_stcli.main.return_value = 0

        with patch.dict(sys.modules, {"streamlit.web.cli": mock_stcli}), \
             patch("subprocess.Popen") as mock_popen, \
             patch("sys.exit") as mock_exit:
            cli.main(["app", "--port", "9000", "--no-browser"])

        mock_popen.assert_not_called()
        mock_stcli.main.assert_called_once()
        self.assertEqual(
            sys.argv,
            [
                "streamlit", "run", cli.APP_PATH,
                "--server.port", "9000",
                "--server.address", "localhost",
                "--server.headless", "true",
            ]
        )
        mock_exit.assert_called_once_with(0)

    def test_run_app_without_window_direct(self):
        mock_stcli = Mock()
        mock_stcli.main.return_value = 42

        args = types.SimpleNamespace(
            port=8501,
            address="localhost",
            no_browser=False,
            window=False,
        )
        with patch.dict(sys.modules, {"streamlit.web.cli": mock_stcli}), \
             patch("sys.exit") as mock_exit:
            cli._run_app(args)

        mock_stcli.main.assert_called_once()
        mock_exit.assert_called_once_with(42)

    def test_wait_for_health_success(self):
        mock_resp = Mock()
        mock_resp.status = 200
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=None)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            self.assertTrue(cli._wait_for_health("http://localhost:8501"))
            mock_urlopen.assert_called_once_with("http://localhost:8501/_stcore/health", timeout=1)

    def test_wait_for_health_premature_exit(self):
        mock_proc = Mock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        with self.assertRaises(RuntimeError):
            cli._wait_for_health("http://localhost:8501", proc=mock_proc, timeout=1)

    def test_wait_for_health_timeout(self):
        mock_proc = Mock()
        mock_proc.poll.return_value = None

        with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
            with self.assertRaises(TimeoutError):
                cli._wait_for_health("http://localhost:8501", proc=mock_proc, timeout=0.2)


if __name__ == "__main__":
    unittest.main()
