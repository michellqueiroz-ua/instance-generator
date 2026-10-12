import contextlib
import io
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, call, patch

import REQreate
from REQreate import cli


class GenerateCliTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.previous_dir = os.getcwd()
        os.chdir(self.temp_dir.name)
        self.addCleanup(os.chdir, self.previous_dir)

    def _write_config(self, filename, network="Test City"):
        with open(filename, "w", encoding="utf-8") as config_file:
            config_file.write('{"network": "' + network + '"}')

    @contextlib.contextmanager
    def _mock_input_json(self):
        module = types.ModuleType("REQreate.input_json")
        input_json = Mock()
        module.input_json = input_json
        with patch.dict(sys.modules, {"REQreate.input_json": module}):
            with patch.object(REQreate, "input_json", module, create=True):
                yield input_json

    def test_folder_configs_are_sorted_and_macos_files_are_ignored(self):
        os.mkdir("configs")
        self._write_config("configs/b.json")
        self._write_config("configs/a.json")
        self._write_config("configs/._hidden.json")

        with self._mock_input_json() as input_json:
            result = cli.main(["generate", "configs", "--output", "batch"])

        self.assertEqual(result, 0)
        self.assertEqual(
            input_json.call_args_list,
            [
                call(os.path.abspath("configs") + os.sep, "a.json", "batch"),
                call(os.path.abspath("configs") + os.sep, "b.json", "batch"),
            ],
        )

    def test_skip_existing_skips_configs_with_an_existing_output(self):
        os.mkdir("configs")
        self._write_config("configs/a.json")
        self._write_config("configs/b.json")
        output_dir = os.path.join("Test City", "csv_format", "batch")
        os.makedirs(output_dir)
        with open(os.path.join(output_dir, "a_1.csv"), "w", encoding="utf-8"):
            pass

        with self._mock_input_json() as input_json:
            result = cli.main(
                ["generate", "configs", "--output", "batch", "--skip-existing"]
            )

        self.assertEqual(result, 0)
        input_json.assert_called_once_with(
            os.path.abspath("configs") + os.sep, "b.json", "batch"
        )

    def test_continue_on_error_reports_failed_configs_and_processes_remaining(self):
        os.mkdir("configs")
        self._write_config("configs/a.json")
        self._write_config("configs/b.json")

        with self._mock_input_json() as input_json:
            input_json.side_effect = [RuntimeError("generation failed"), None]
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = cli.main(
                    ["generate", "configs", "--continue-on-error"]
                )

        self.assertEqual(result, 1)
        self.assertEqual(
            input_json.call_args_list,
            [
                call(os.path.abspath("configs") + os.sep, "a.json", ""),
                call(os.path.abspath("configs") + os.sep, "b.json", ""),
            ],
        )
        self.assertIn("1 succeeded, 1 failed, 0 skipped", stdout.getvalue())
        self.assertIn(os.path.abspath("configs/a.json"), stderr.getvalue())
        self.assertIn(os.path.abspath("configs/a.json"), stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
