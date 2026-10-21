import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import psutil

from REQreate.webapp.job_queue import JobQueue, _child_env


class JobQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.queue = JobQueue(self.directory)
        self.commands = {}
        self.command_patch = patch.object(
            self.queue, "_command", side_effect=lambda job: self.commands[job["parameters"]["place"]]
        )
        self.command_patch.start()

    def tearDown(self):
        self.queue.close()
        self.command_patch.stop()
        self.temporary.cleanup()

    def submit(self, name, script):
        self.commands[name] = [sys.executable, "-u", "-c", script]
        return self.queue.submit_network(name)

    def wait_for(self, predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        self.fail(f"Timed out waiting for queue: {self.queue.jobs()}")

    def status(self, job):
        return next(item["status"] for item in self.queue.jobs() if item["id"] == job["id"])

    def test_fifo_and_only_one_process_at_a_time(self):
        first, position = self.submit(
            "first",
            "from pathlib import Path; import time; "
            "Path('first.started').touch(); "
            "\nwhile not Path('release').exists(): time.sleep(0.02)\n"
            "Path('first.finished').touch()",
        )
        second, second_position = self.submit(
            "second",
            "from pathlib import Path; "
            "assert Path('first.finished').exists(); Path('second.finished').touch()",
        )
        self.assertEqual(position, 1)
        self.assertEqual(second_position, 2)
        self.wait_for(lambda: (self.directory / "first.started").exists())
        self.assertEqual(self.status(first), "running")
        self.assertEqual(self.status(second), "queued")
        self.assertFalse((self.directory / "second.finished").exists())
        (self.directory / "release").touch()
        self.wait_for(lambda: self.status(second) == "done")
        self.assertEqual(self.status(first), "done")
        jobs = self.queue.jobs()
        self.assertLessEqual(jobs[0]["finished"], jobs[1]["started"])
        persisted = json.loads((self.directory / "queue" / f'{first["id"]}.json').read_text())
        self.assertEqual(persisted["exit_code"], 0)
        self.assertIsNotNone(persisted["started"])

    def test_cancel_queued(self):
        first, _ = self.submit("first", "import time; time.sleep(30)")
        self.wait_for(lambda: self.status(first) == "running")
        second, _ = self.submit("second", "raise AssertionError('must not run')")
        self.queue.cancel(second["id"])
        self.queue.cancel(first["id"])
        self.assertEqual(self.status(second), "cancelled")
        self.assertIsNone(self.queue.jobs()[1]["started"])
        self.assertFalse(Path(second["log_path"]).exists())

    def test_cancel_running_and_continue(self):
        first, _ = self.submit("first", "import time; print('started'); time.sleep(30)")
        self.wait_for(lambda: "started" in self.queue.tail(first["id"]))
        second, _ = self.submit("second", "print('next job')")
        self.queue.cancel(first["id"])
        self.wait_for(lambda: self.status(second) == "done")
        self.assertEqual(self.status(first), "cancelled")
        self.assertNotEqual(self.queue.jobs()[0]["exit_code"], 0)

    @unittest.skipIf(os.name == "nt", "POSIX process-group integration test")
    def test_cancel_stops_children(self):
        child = (
            "from pathlib import Path; import time; "
            "\nwhile True:\n"
            " with Path('heartbeat').open('a') as f: f.write('tick\\n')\n"
            " time.sleep(0.02)"
        )
        job, _ = self.submit(
            "parent",
            f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {child!r}]); "
            "time.sleep(30)",
        )
        heartbeat = self.directory / "heartbeat"
        self.wait_for(lambda: heartbeat.exists() and heartbeat.stat().st_size > 0)
        self.queue.cancel(job["id"])
        size = heartbeat.stat().st_size
        time.sleep(0.2)
        self.assertEqual(heartbeat.stat().st_size, size)

    def test_windows_cancel_uses_taskkill_tree(self):
        process = Mock()
        process.create_time.return_value = 10
        with patch("REQreate.webapp.job_queue.os.name", "nt"), patch(
            "REQreate.webapp.job_queue.psutil.Process", return_value=process
        ), patch("REQreate.webapp.job_queue.subprocess.run") as run:
            run.return_value.returncode = 0
            self.queue._stop_tree(1234, 10)
        run.assert_called_once_with(
            ["taskkill", "/PID", "1234", "/T", "/F"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def test_stop_tree_does_not_kill_reused_pid(self):
        process = Mock()
        process.create_time.return_value = 20
        with patch(
            "REQreate.webapp.job_queue.psutil.Process", return_value=process
        ), patch("REQreate.webapp.job_queue.subprocess.run") as run, patch(
            "REQreate.webapp.job_queue.os.killpg", create=True
        ) as kill:
            self.queue._stop_tree(1234, 10)
        run.assert_not_called()
        kill.assert_not_called()

    def test_windows_cancel_tolerates_already_exited_process(self):
        process = Mock()
        process.create_time.return_value = 10
        process.is_running.return_value = False
        with patch("REQreate.webapp.job_queue.os.name", "nt"), patch(
            "REQreate.webapp.job_queue.psutil.Process", return_value=process
        ), patch("REQreate.webapp.job_queue.subprocess.run") as run:
            run.return_value.returncode = 128
            self.queue._stop_tree(1234, 10)

    def test_failed_job_logs_exit_code_and_next_job_runs(self):
        first, _ = self.submit(
            "failure", "import sys; print('stdout'); print('stderr', file=sys.stderr); sys.exit(7)"
        )
        second, _ = self.submit("success", "print('ok')")
        self.wait_for(lambda: self.status(second) == "done")
        self.assertEqual(self.status(first), "failed")
        self.assertEqual(self.queue.jobs()[0]["exit_code"], 7)
        self.assertIn("stdout", self.queue.tail(first["id"]))
        self.assertIn("stderr", self.queue.tail(first["id"]))
        self.assertLessEqual(len(self.queue.tail(first["id"], max_bytes=4)), 4)

    def test_spawn_failure_does_not_stop_worker(self):
        self.commands["missing"] = [str(self.directory / "missing-executable")]
        job, _ = self.queue.submit_network("missing")
        second, _ = self.submit("second", "pass")
        self.wait_for(lambda: self.status(second) == "done")
        self.assertEqual(self.status(job), "failed")
        self.assertIn("Could not start job", self.queue.tail(job["id"]))

    def test_restart_interrupts_running_and_requires_resume_for_pending(self):
        first, _ = self.submit("first", "import time; time.sleep(30)")
        self.wait_for(lambda: self.status(first) == "running")
        second, _ = self.submit("second", "print('resumed')")
        self.queue.close()
        # Simulate the on-disk state left by an abrupt app exit.
        path = self.directory / "queue" / f'{first["id"]}.json'
        saved = json.loads(path.read_text())
        saved.update(status="running", finished=None, exit_code=None)
        path.write_text(json.dumps(saved))
        self.queue = JobQueue(self.directory)
        with patch.object(self.queue, "_command", return_value=[sys.executable, "-c", "pass"]):
            self.assertEqual(self.status(first), "interrupted")
            self.assertIsNotNone(self.queue.jobs()[0]["finished"])
            self.assertTrue(self.queue.paused)
            third, position = self.queue.submit_network("third")
            self.assertEqual(position, 2)
            time.sleep(0.1)
            self.assertEqual(self.status(second), "queued")
            self.assertEqual(self.status(third), "queued")
            self.queue.resume()
            self.wait_for(lambda: self.status(third) == "done")
            self.assertEqual(self.status(second), "done")
            jobs = self.queue.jobs()
            self.assertLessEqual(jobs[1]["finished"], jobs[2]["started"])

    def test_restart_stops_orphan_from_crashed_app(self):
        self.queue.close()
        script = (
            "import os, sys, time\n"
            "from pathlib import Path\n"
            "from REQreate.webapp.job_queue import JobQueue\n"
            "queue = JobQueue(Path.cwd())\n"
            "queue._command = lambda job: [sys.executable, '-c', 'import time; time.sleep(30)']\n"
            "job, _ = queue.submit_network('orphan')\n"
            "while not queue.jobs()[0].get('pid'): time.sleep(0.02)\n"
            "os._exit(0)\n"
        )
        subprocess.run(
            [sys.executable, "-c", script], cwd=self.directory, env=_child_env(), check=True, timeout=10,
        )
        record = json.loads(next((self.directory / "queue").glob("*.json")).read_text())
        orphan = psutil.Process(record["pid"])

        def stopped():
            # The orphan can exit between is_running() and status(), so a
            # vanished process counts as stopped rather than an error.
            try:
                return not orphan.is_running() or orphan.status() == psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                return True

        try:
            self.queue = JobQueue(self.directory)
            self.assertEqual(self.queue.jobs()[0]["status"], "interrupted")
            self.wait_for(stopped)
        finally:
            try:
                orphan.kill()
            except psutil.NoSuchProcess:
                pass

    def test_jobs_import_the_same_reqreate_as_the_app(self):
        with patch.dict(os.environ, {"PYTHONPATH": "elsewhere"}):
            paths = _child_env()["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(Path(paths[0]), Path(__file__).resolve().parents[1])
        self.assertEqual(paths[1:], ["elsewhere"])

    def test_crash_before_identity_is_saved_does_not_execute_cli_job(self):
        self.queue.close()
        script = (
            "import os, sys, time\n"
            "from pathlib import Path\n"
            "from REQreate.webapp.job_queue import JobQueue\n"
            "queue = JobQueue(Path.cwd())\n"
            "queue._command = lambda job: [sys.executable, '-u', '-m', 'REQreate.cli', '--version']\n"
            "save = queue._save\n"
            "def crash_before_save(job):\n"
            " if job.get('pid'):\n"
            "  Path('uncommitted_pid').write_text(str(job['pid']))\n"
            "  os._exit(0)\n"
            " save(job)\n"
            "queue._save = crash_before_save\n"
            "queue.submit_network('uncommitted')\n"
            "while True: time.sleep(0.02)\n"
        )
        subprocess.run(
            [sys.executable, "-c", script], cwd=self.directory, env=_child_env(), check=True, timeout=10,
        )
        pid = int((self.directory / "uncommitted_pid").read_text())

        def exited():
            try:
                return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                return True

        self.wait_for(exited)
        record = json.loads(next((self.directory / "queue").glob("*.json")).read_text())
        self.assertIsNone(record["pid"])
        self.assertEqual(Path(record["log_path"]).read_text(), "")
        self.queue = JobQueue(self.directory)
        self.assertEqual(self.queue.jobs()[0]["status"], "interrupted")

    def test_cli_gate_released_after_identity_is_saved(self):
        self.commands["version"] = [sys.executable, "-u", "-m", "REQreate.cli", "--version"]
        job, _ = self.queue.submit_network("version")
        self.wait_for(lambda: self.status(job) == "done")
        self.assertIn("reqreate", self.queue.tail(job["id"]).lower())
        record = json.loads((self.directory / "queue" / f'{job["id"]}.json').read_text())
        self.assertIsNotNone(record["pid"])
        self.assertIsNotNone(record["process_created"])

    def test_generate_snapshots_config_without_changing_output_basename(self):
        config = self.directory / "example.json"
        config.write_text('{"network": "First city"}')
        with self.queue._condition:
            first, _ = self.queue.submit_generate(config)
            config.write_text('{"network": "Second city"}')
            second, _ = self.queue.submit_generate(config)
            first_path = Path(first["parameters"]["config"])
            second_path = Path(second["parameters"]["config"])
            self.assertNotEqual(first_path, second_path)
            self.assertEqual(first_path.name, config.name)
            self.assertEqual(json.loads(first_path.read_text())["network"], "First city")
            self.assertEqual(json.loads(second_path.read_text())["network"], "Second city")
            self.assertEqual(
                JobQueue._command(self.queue, first),
                [sys.executable, "-u", "-m", "REQreate.cli", "generate", str(first_path)],
            )
            self.queue.cancel(first["id"])
            self.queue.cancel(second["id"])
        self.queue.clear_finished()
        self.assertEqual(self.queue.jobs(), [])
        self.assertFalse(first_path.exists())
        self.assertTrue(config.exists())
        self.assertEqual(list((self.directory / "queue").glob("*.json")), [])

    def test_network_command_arguments(self):
        with self.queue._condition:
            job, _ = self.queue.submit_network(
                "City with spaces", point=(4.5, 50.5), dist=1000, fixed_lines="osm"
            )
            self.assertEqual(
                JobQueue._command(self.queue, job),
                [sys.executable, "-u", "-m", "REQreate.cli", "network", "City with spaces",
                 "--point", "4.5", "50.5", "--dist", "1000", "--fixed-lines", "osm"],
            )
            self.queue.cancel(job["id"])

    def test_generate_rejects_config_outside_working_directory(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            config = Path(elsewhere) / "example.json"
            config.write_text('{"network": "Elsewhere"}')
            with self.assertRaises(ValueError):
                self.queue.submit_generate(config)
            with self.assertRaises(ValueError):
                self.queue.submit_generate(self.directory / ".." / "example.json")
        self.assertEqual(self.queue.jobs(), [])

    def test_submission_uses_form_config_even_if_shared_file_was_overwritten(self):
        config = self.directory / "example.json"
        config.write_text('{"network": "Other browser"}')
        with self.queue._condition:
            job, _ = self.queue.submit_generate(config, configuration={"network": "This browser"})
            snapshot = Path(job["parameters"]["config"])
            self.assertEqual(json.loads(snapshot.read_text()), {"network": "This browser"})
            self.queue.cancel(job["id"])


if __name__ == "__main__":
    unittest.main()
