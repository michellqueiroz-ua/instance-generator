import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import lsf_summary

# The real example logs live in local_only/ of the main checkout (never pushed); LSF_REAL_LOGS overrides the folder.
REAL_LOGS = Path(os.environ.get("LSF_REAL_LOGS", r"C:\Users\micqu\Desktop\instance-generator\local_only\results_isa"))


def log(status="Successfully completed.", cpu="500.00 sec.", mem="1400 MB", run="520 sec.", swap="-", term=""):
    lines = ["Your job looked like:", "", "# LSBATCH: User input", "bash job.sh", "", status]
    if term:
        lines.append(term)
    lines += ["", "Resource usage summary:", ""]
    if cpu:
        lines.append(f"    CPU time :                                   {cpu}")
    if mem:
        lines.append(f"    Max Memory :                                 {mem}")
    lines.append("    Average Memory :                             1000.00 MB")
    if swap:
        lines.append(f"    Max Swap :                                   {swap}")
    if run:
        lines.append(f"    Run time :                                   {run}")
    lines += ["    Turnaround time :                            530 sec.", "", "The output (if any) is above this job summary.", ""]
    return "\n".join(lines)


def run_tool(*args):
    out = io.StringIO()
    with redirect_stdout(out):
        code = lsf_summary.main(list(args))
    return code, out.getvalue()


class ParseTests(unittest.TestCase):
    def test_a_completed_job_is_parsed(self):
        (job,) = lsf_summary.parse_log(log())
        self.assertTrue(job["ok"])
        self.assertEqual((job["cpu"], job["run"]), (500.0, 520.0))
        self.assertAlmostEqual(job["mem_gb"], 1400 / 1024)
        self.assertEqual(job["swap_gb"], "-")

    def test_an_exit_code_and_a_term_line_are_found(self):
        (job,) = lsf_summary.parse_log(log(status="Exited with exit code 137.", term="TERM_MEMLIMIT: job killed after reaching LSF memory usage limit."))
        self.assertFalse(job["ok"])
        self.assertEqual(job["status"], "exit 137")
        self.assertEqual(job["term"], ["TERM_MEMLIMIT"])

    def test_gb_units_and_a_numeric_swap(self):
        (job,) = lsf_summary.parse_log(log(mem="1.5 GB", swap="200 MB"))
        self.assertAlmostEqual(job["mem_gb"], 1.5)
        self.assertAlmostEqual(job["swap_gb"], 200 / 1024)

    def test_missing_fields_stay_missing(self):
        (job,) = lsf_summary.parse_log(log(cpu=None, mem=None, swap=None))
        self.assertIsNone(job["cpu"])
        self.assertIsNone(job["mem_gb"])
        self.assertIsNone(job["swap_gb"])
        self.assertIsNone(lsf_summary.utilisation(job))

    def test_a_file_with_several_jobs_gives_one_job_each(self):
        jobs = lsf_summary.parse_log(log(run="100 sec.") + "\n" + log(status="Exited with exit code 1.", run="200 sec."))
        self.assertEqual([j["run"] for j in jobs], [100.0, 200.0])
        self.assertEqual([j["ok"] for j in jobs], [True, False])

    def test_a_text_that_is_not_an_lsf_log_gives_no_job(self):
        self.assertEqual(lsf_summary.parse_log("hello\nworld\n"), [])


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def write(self, name, text):
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_good_solver_block_passes(self):
        self.write("rows_1_10.out", log(cpu="2000 sec.", run="2050 sec.", mem="1500 MB"))
        code, out = run_tool(str(self.dir))
        self.assertEqual(code, 0, out)
        self.assertIn("solver: 1 jobs, 0 failed", out)
        self.assertIn("PASS against the pass table", out)
        self.assertIn("0.98", out)

    def test_solver_fails_on_memory_utilisation_and_run_time(self):
        self.write("rows_1_10.out", log(cpu="2000 sec.", run="2050 sec.", mem="1800 MB"))
        self.write("rows_11_20.out", log(cpu="1000 sec.", run="2050 sec.", mem="1000 MB"))
        self.write("rows_21_30.out", log(cpu="6000 sec.", run="6000 sec.", mem="1000 MB"))
        code, out = run_tool(str(self.dir))
        self.assertEqual(code, 1)
        self.assertIn("Max Memory 1.76 GB > 1.7 GB", out)
        self.assertIn("utilisation 0.49 < 0.90", out)
        self.assertIn("Run time 6000 s > 5400 s", out)

    def test_a_killed_job_fails_any_kind(self):
        self.write("gen_1.out", log(status="Exited with exit code 140.", term="TERM_RUNLIMIT: job killed"))
        code, out = run_tool(str(self.dir))
        self.assertEqual(code, 1)
        self.assertIn("TERM_RUNLIMIT", out)

    def test_features_limits(self):
        self.write("features_batch_04_1.out", log(cpu="3000 sec.", run="7300 sec.", mem="1000 MB"))
        code, out = run_tool(str(self.dir))
        self.assertEqual(code, 1)
        self.assertIn("Run time 7300 s > 7200 s", out)

    def test_features_pass_has_no_utilisation_rule(self):
        self.write("features_batch_04_1.out", log(cpu="3000 sec.", run="5000 sec.", mem="1000 MB"))
        code, out = run_tool(str(self.dir))
        self.assertEqual(code, 0, out)

    def test_kind_option_overrides_the_file_name(self):
        self.write("whatever.out", log(cpu="500 sec.", run="1000 sec.", mem="1000 MB"))
        self.assertEqual(run_tool(str(self.dir / "whatever.out"), "--kind", "other")[0], 0)
        self.assertEqual(run_tool(str(self.dir / "whatever.out"), "--kind", "solver")[0], 1)

    def test_rows_per_block_sets_the_mean_row_time(self):
        self.write("rows_1_10.out", log(cpu="5000 sec.", run="5000 sec.", mem="1000 MB"))
        self.assertEqual(run_tool(str(self.dir))[0], 0)  # 500 s per row of 10
        code, out = run_tool(str(self.dir), "--rows-per-block", "4")
        self.assertEqual(code, 1)
        self.assertIn("mean row time 1250 s > 1100 s", out)

    def test_a_missing_field_is_printed_as_missing_and_not_invented(self):
        self.write("other.out", log(cpu=None, run="100 sec."))
        code, out = run_tool(str(self.dir))
        self.assertIn("CPU missing", out)
        self.assertIn("util missing", out)

    def test_a_folder_search_ignores_logs_that_are_not_lsf_logs(self):
        self.write("build.log", "just some text\n")
        self.write("rows_1_10.out", log(cpu="2000 sec.", run="2050 sec."))
        code, out = run_tool(str(self.dir))
        self.assertNotIn("build.log", out)

    def test_nothing_found_is_an_error(self):
        self.assertEqual(run_tool(str(self.dir))[0], 1)


@unittest.skipUnless((REAL_LOGS / "fixnet" / "logs" / "fixnet.out").is_file(), "real example logs not on this machine")
class RealLogTests(unittest.TestCase):
    def test_the_two_real_logs_match_their_text(self):
        code, out = run_tool(str(REAL_LOGS / "fixnet"), str(REAL_LOGS / "edgecheck"), "--kind", "other")
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"fixnet\.out\s+other\s+completed\s+-\s+CPU\s+134 s\s+Run\s+143 s\s+util\s+0\.94\s+Mem\s+1\.61 GB")
        self.assertRegex(out, r"edgecheck\.out\s+other\s+completed\s+-\s+CPU\s+55 s\s+Run\s+60 s\s+util\s+0\.92\s+Mem\s+1\.39 GB")


if __name__ == "__main__":
    unittest.main()
