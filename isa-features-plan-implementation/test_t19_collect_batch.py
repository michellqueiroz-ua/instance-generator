import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
for path in (THIS_DIR, THIS_DIR / "isa_set"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import collect_batch
from features_export import append_features_rows

PATH = "/work3/u/isa/Capital Region of Denmark/csv_format/ISA_b01_i001_1.csv"


def result_line(best, served=95, third=5, seconds=9.5):
    # filename served third total ppk extra ratio deadheading shared total_urt best_urt avgurt: ... elapsed duration
    return (f"{PATH} {served} {third} {served + third} 0.5 1.5 1.2 40 50 {best + 100} {best} avgurt: 60.5 7.1 "
            f"1001 10 0.3 0.3 0.3 0.0 4 4 0 0 20 30 {seconds} 9\n=== VEHICLE ROUTES ===\n")


class CollectBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        (self.work / "results").mkdir()
        keys = ["ISA_b01_i001_1_fleet1", "ISA_b01_i001_1_fleet2"]
        append_features_rows([(k, {}) for k in keys], csv_path=self.work / "results" / "features_batch_01.csv")
        runs = self.work / "solver" / "batch_01" / "results"
        runs.mkdir(parents=True)
        for key, sa, gr in ((keys[0], (800, 820, 780), 1000), (keys[1], (500, 520, 540), 600)):
            for seed, value in enumerate(sa, start=1):
                (runs / f"{key}_SA_s{seed}.txt").write_text(result_line(value))
            (runs / f"{key}_GR_s1.txt").write_text(result_line(gr))

    def test_parses_numbers_despite_spaces_in_the_path(self):
        row = collect_batch.parse_first_line(result_line(1234, served=90, third=10, seconds=42.5).splitlines()[0])
        self.assertEqual(row, {"served": 90.0, "third_party": 10.0, "best_total_urt": 1234.0, "elapsed_s": 42.5})

    def test_writes_runs_raw_results_and_perf_matrix(self):
        collect_batch.main(["--batch", "1", "--work", str(self.work)])
        raw = pd.read_csv(self.work / "results" / "raw_results_batch_01.csv", index_col="instance")
        self.assertAlmostEqual(raw.loc["ISA_b01_i001_1_fleet1", "SA"], 800.0)  # mean of the three seeds
        self.assertEqual(raw.loc["ISA_b01_i001_1_fleet2", "GR"], 600)
        perf = pd.read_csv(self.work / "results" / "perf_matrix_batch_01.csv", index_col="instance")
        self.assertEqual(perf.loc["ISA_b01_i001_1_fleet1", "SA"], 1.0)  # the best scores 1.0
        self.assertAlmostEqual(perf.loc["ISA_b01_i001_1_fleet1", "GR"], 0.8)
        self.assertEqual(len(pd.read_csv(self.work / "results" / "solver_runs_batch_01.csv")), 8)

    def test_a_missing_run_is_reported(self):
        (self.work / "solver" / "batch_01" / "results" / "ISA_b01_i001_1_fleet2_GR_s1.txt").unlink()
        with self.assertRaises(SystemExit) as context:
            collect_batch.main(["--batch", "1", "--work", str(self.work)])
        self.assertIn("lack SA or GR", str(context.exception))


if __name__ == "__main__":
    unittest.main()
