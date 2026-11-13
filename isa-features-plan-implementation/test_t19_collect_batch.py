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


def result_line(best, served=95, third=5, seconds=9.5, ratio=1.2):
    # filename served third total ppk extra ratio deadheading shared total_urt best_urt avgurt: ... elapsed duration
    return (f"{PATH} {served} {third} {served + third} 0.5 1.5 {ratio} 40 50 {best + 100} {best} avgurt: 60.5 7.1 "
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
        self.assertEqual(row, {"served": 90.0, "third_party": 10.0, "avg_extra_s": 1.5, "avg_ratio": 1.2,
                               "best_total_urt": 1234.0, "elapsed_s": 42.5})

    def test_writes_runs_and_both_scorings_of_sa(self):
        collect_batch.main(["--batch", "1", "--work", str(self.work)])
        results = self.work / "results"
        mean = pd.read_csv(results / "raw_results_mean_batch_01.csv", index_col="instance")
        best = pd.read_csv(results / "raw_results_best_batch_01.csv", index_col="instance")
        self.assertAlmostEqual(mean.loc["ISA_b01_i001_1_fleet1", "SA"], 800.0)  # mean of 800, 820, 780
        self.assertEqual(best.loc["ISA_b01_i001_1_fleet1", "SA"], 780)          # best seed
        self.assertEqual(best.loc["ISA_b01_i001_1_fleet2", "GR"], 600)
        perf_mean = pd.read_csv(results / "perf_matrix_mean_batch_01.csv", index_col="instance")
        perf_best = pd.read_csv(results / "perf_matrix_best_batch_01.csv", index_col="instance")
        self.assertEqual(perf_mean.loc["ISA_b01_i001_1_fleet1", "SA"], 1.0)  # the best scores 1.0
        self.assertAlmostEqual(perf_mean.loc["ISA_b01_i001_1_fleet1", "GR"], 0.8)
        self.assertAlmostEqual(perf_best.loc["ISA_b01_i001_1_fleet1", "GR"], 0.78)
        self.assertEqual(len(pd.read_csv(results / "solver_runs_batch_01.csv")), 8)

    def test_served_share_and_accepted_quality_are_scored_too(self):
        # fleet2: SA serves 80%, GR 95%; GR rides are slower (ratio 1.5) than SA's (1.2)
        runs = self.work / "solver" / "batch_01" / "results"
        for seed in (1, 2, 3):
            (runs / f"ISA_b01_i001_1_fleet2_SA_s{seed}.txt").write_text(result_line(500, served=80, third=20, ratio=1.2))
        (runs / "ISA_b01_i001_1_fleet2_GR_s1.txt").write_text(result_line(600, served=95, third=5, ratio=1.5))
        collect_batch.main(["--batch", "1", "--work", str(self.work)])
        results = self.work / "results"
        served = pd.read_csv(results / "perf_matrix_served_mean_batch_01.csv", index_col="instance")
        quality = pd.read_csv(results / "perf_matrix_quality_mean_batch_01.csv", index_col="instance")
        self.assertEqual(served.loc["ISA_b01_i001_1_fleet2", "GR"], 1.0)             # higher served share is better
        self.assertAlmostEqual(served.loc["ISA_b01_i001_1_fleet2", "SA"], 0.8 / 0.95)
        self.assertEqual(quality.loc["ISA_b01_i001_1_fleet2", "SA"], 1.0)            # lower ratio is better
        self.assertAlmostEqual(quality.loc["ISA_b01_i001_1_fleet2", "GR"], 1.2 / 1.5)

    def test_sa_variants_become_columns_and_are_compared_with_gr(self):
        runs = self.work / "solver" / "batch_01" / "results"
        for key, value in (("ISA_b01_i001_1_fleet1", 700), ("ISA_b01_i001_1_fleet2", 450)):
            for seed in (1, 2, 3):
                (runs / f"{key}_SA2000_s{seed}.txt").write_text(result_line(value))
        collect_batch.main(["--batch", "1", "--work", str(self.work)])
        perf = pd.read_csv(self.work / "results" / "perf_matrix_mean_batch_01.csv", index_col="instance")
        self.assertEqual(list(perf.columns), ["SA", "SA2000", "GR"])
        self.assertEqual(perf.loc["ISA_b01_i001_1_fleet1", "SA2000"], 1.0)  # 700 beats SA 800 and GR 1000
        self.assertAlmostEqual(perf.loc["ISA_b01_i001_1_fleet1", "SA"], 700 / 800)

    def test_partial_keeps_the_keys_that_have_every_algorithm(self):
        runs = self.work / "solver" / "batch_01" / "results"
        for seed in (1, 2, 3):  # a variant for fleet1 only
            (runs / f"ISA_b01_i001_1_fleet1_SA2000_s{seed}.txt").write_text(result_line(700))
        with self.assertRaises(SystemExit):
            collect_batch.main(["--batch", "1", "--work", str(self.work)])
        collect_batch.main(["--batch", "1", "--work", str(self.work), "--partial"])
        results = self.work / "results"
        perf = pd.read_csv(results / "perf_matrix_mean_batch_01_partial.csv", index_col="instance")
        self.assertEqual(list(perf.index), ["ISA_b01_i001_1_fleet1"])
        self.assertEqual(list(perf.columns), ["SA", "SA2000", "GR"])
        features = pd.read_csv(results / "features_batch_01_partial.csv")
        self.assertEqual(list(features["instance"]), ["ISA_b01_i001_1_fleet1"])

    def test_a_missing_run_is_reported(self):
        (self.work / "solver" / "batch_01" / "results" / "ISA_b01_i001_1_fleet2_GR_s1.txt").unlink()
        with self.assertRaises(SystemExit) as context:
            collect_batch.main(["--batch", "1", "--work", str(self.work)])
        self.assertIn("lack results of some of", str(context.exception))


if __name__ == "__main__":
    unittest.main()
