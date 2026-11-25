# Dashboard listener replacement validation
#
# mikael.mieskolainen@cern.ch (2026)

from io import BytesIO
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from twistboard.processes import is_dashboard
from twistboard.server import Handler


class DashboardServeTests(unittest.TestCase):
    """Check port conflicts and controlled dashboard replacement"""

    def setUp(self):
        """
        Create isolated dashboard files
        """
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.command = [sys.executable, "-m", "twistboard", "serve", "--root", str(self.root)]

    def stop_process(self, process):
        """
        Release a test subprocess

        Args:
            process:  Dashboard subprocess
        """
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)

    def start_dashboard(self, *arguments, ignore_term=False):
        """
        Start a dashboard and wait for its listening URL

        Args:
            arguments:  Additional server arguments
            ignore_term:  Keep SIGTERM blocked to test forced termination
        Returns:
            result: Process and assigned port
        """
        log = tempfile.TemporaryFile(mode="w+")
        self.addCleanup(log.close)
        previous_mask = None
        if ignore_term:
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        try:
            process = subprocess.Popen(
                [*self.command, *arguments], stdout=log, stderr=log,
                cwd=Path(__file__).resolve().parents[1],
            )
        finally:
            if previous_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        self.addCleanup(self.stop_process, process)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            log.seek(0)
            output = log.read()
            for line in output.splitlines():
                if line.startswith("Twistboard: http://"):
                    return process, int(line.split()[1].rsplit(":", 1)[1])
            if process.poll() is not None:
                self.fail(output)
            time.sleep(0.05)
        self.fail(f"Dashboard did not start: {output}")

    def run_dashboard(self, *arguments):
        """
        Capture a dashboard startup error

        Args:
            arguments:  Additional server arguments
        Returns:
            result: Command status and output
        """
        return subprocess.run(
            [*self.command, *arguments], capture_output=True, text=True,
            timeout=10, cwd=Path(__file__).resolve().parents[1],
        )

    def test_conflict_shows_force_without_traceback(self):
        """
        Report the force option while retaining the previous dashboard
        """
        previous, port = self.start_dashboard("--port", "0")
        result = self.run_dashboard("--port", str(port))
        self.assertEqual(result.returncode, 1)
        self.assertIn("--force", result.stderr)
        self.assertIn("--port 0", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertIsNone(previous.poll())

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux process control")
    def test_force_replaces_dashboard(self):
        """
        Replace a dashboard on the same port
        """
        previous, port = self.start_dashboard("--port", "0")
        replacement, new_port = self.start_dashboard("--port", str(port), "--force")
        self.assertEqual(new_port, port)
        self.assertEqual(previous.wait(timeout=5), -signal.SIGTERM)
        self.assertIsNone(replacement.poll())

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux process control")
    def test_force_kills_unresponsive_dashboard(self):
        """
        Escalate to SIGKILL when SIGTERM cannot stop the dashboard
        """
        previous, port = self.start_dashboard("--port", "0", ignore_term=True)
        _, new_port = self.start_dashboard("--port", str(port), "--force")
        self.assertEqual(new_port, port)
        self.assertEqual(previous.wait(timeout=5), -signal.SIGKILL)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux process control")
    def test_force_replaces_legacy_dashboard(self):
        """
        Replace a Quisboard process left running across the package rename
        """
        (self.root / "quisboard.py").write_text(
            "# Legacy dashboard entry point\n#\n"
            "# mikael.mieskolainen@cern.ch (2026)\n\n"
            "from twistboard.cli import main\nmain()\n"
        )
        command = [*self.command]
        command[2] = "quisboard"
        search_path = os.pathsep.join((str(self.root), os.environ.get("PYTHONPATH", "")))
        with patch.object(self, "command", command), patch.dict(
            os.environ, {"PYTHONPATH": search_path}
        ):
            previous, port = self.start_dashboard("--port", "0")
        replacement, new_port = self.start_dashboard("--port", str(port), "--force")
        self.assertEqual(new_port, port)
        self.assertEqual(previous.wait(timeout=5), -signal.SIGTERM)
        self.assertIsNone(replacement.poll())

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux process control")
    def test_force_preserves_unrelated_listener(self):
        """
        Leave a non-dashboard listener active
        """
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            result = self.run_dashboard("--port", str(port), "--force")
            self.assertEqual(result.returncode, 1)
            self.assertIn("no same-user Twistboard", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(listener.getsockname()[1], port)

    def test_force_on_free_port(self):
        """
        Start normally when no listener needs replacement
        """
        process, port = self.start_dashboard("--port", "0", "--force")
        self.assertGreater(port, 0)
        self.assertIsNone(process.poll())

    def test_command_identity(self):
        """
        Accept dashboard entry points and reject unrelated Python commands
        """
        for command in (
            ["python3", "-m", "twistboard", "serve"],
            ["python", "-u", "-m", "twistboard", "serve", "--root", "runs"],
            ["/env/bin/twistboard", "serve"],
            ["/env/bin/python", "/env/bin/twistboard", "serve"],
            ["python", "-m", "quisboard", "serve", "--root", "runs", "--open"],
            ["/env/bin/quisboard", "serve"],
            ["/env/bin/python", "/env/bin/quisboard", "serve"],
        ):
            self.assertTrue(is_dashboard(command), command)
        for command in (
            [], ["python", "-m", "twistboard", "run"],
            ["python", "app.py", "-m", "twistboard", "serve"],
            ["bash", "-c", "python -m twistboard serve"],
            ["python", "-m", "quisboard", "run"],
            ["python", "app.py", "-m", "quisboard", "serve"],
            ["python", "-m"],
        ):
            self.assertFalse(is_dashboard(command), command)

    def test_client_disconnect_stops_response(self):
        """
        Stop writes after a browser disconnects during headers or body output
        """
        for error in (BrokenPipeError, ConnectionResetError):
            for method, route in (
                ("GET", "/api/runs"),
                ("GET", "/style.css"),
                ("GET", "/missing"),
                ("POST", "/api/comparisons"),
                ("DELETE", "/api/runs/demo"),
            ):
                for write in (1, 2):
                    with self.subTest(error=error, method=method, route=route, write=write):
                        connection = Mock()
                        connection.makefile.return_value = BytesIO(
                            f"{method} {route} HTTP/1.0\r\n\r\n".encode()
                        )
                        connection.sendall.side_effect = [None] * (write - 1) + [error()]
                        handler = Handler(
                            connection, ("127.0.0.1", 12345), Mock(), root=self.root
                        )
                        self.assertTrue(handler.close_connection)
                        self.assertEqual(connection.sendall.call_count, write)

    def test_unexpected_response_error_is_not_hidden(self):
        """
        Retain unexpected response failures for diagnosis
        """
        connection = Mock()
        connection.makefile.return_value = BytesIO(b"GET /api/runs HTTP/1.0\r\n\r\n")
        connection.sendall.side_effect = RuntimeError("Response failure")
        with self.assertRaisesRegex(RuntimeError, "Response failure"):
            Handler(connection, ("127.0.0.1", 12345), Mock(), root=self.root)


if __name__ == "__main__":
    unittest.main()
