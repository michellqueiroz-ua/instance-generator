import contextlib
import csv
import io
import json
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
import gr_replication
import make_solver_jobs as jobs
from features_export import append_features_rows
from test_t19_collect_batch import result_line

N_INSTANCES, N_FLEETS = 100, 10


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


class SolverExtrasJobsTests(unittest.TestCase):
    """A fake batch of 100 instances x 10 fleets (batch 4 in size)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.work = Path(cls.tmp.name)
        net = cls.work / "Net"
        write_csv(net / "csv" / "Net.stations.csv", ["lat", "lon"], [[55.0, 12.0 + i * 0.01] for i in range(5)])
        write_csv(net / "travel_time_matrix" / "station_travel_time_matrix.csv", ["", "0"], [[0, 0]])
        fleets = {}
        for i in range(1, N_INSTANCES + 1):
            instance = f"ISA_b04_i{i:03d}_1"
            write_csv(net / "csv_format" / f"{instance}.csv", ["stops_orgn", "stops_dest"], [["[2]", "[3]"]] * 20)
            for f in range(1, N_FLEETS + 1):
                fleets[f"{instance}_fleet{f}"] = {
                    "instance": instance, "fleet_percent_of_requests": 25.0,
                    "composition": [{"capacity": 4, "percent": 60.0}, {"capacity": 8, "percent": 40.0}],
                }
        (cls.work / "design").mkdir()
        (cls.work / "design" / "batch_04_fleets.json").write_text(json.dumps(fleets))
        cls.out = cls.work / "solver" / "batch_04"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_jobs(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            jobs.main(["--batch", "4", "--work", str(self.work), "--network", "Net"] + list(extra))
        lines = (self.out / "jobs.csv").read_text().strip().splitlines()[1:]
        return [[c.strip() for c in line.split(";")] for line in lines]

    def test_default_rows_are_the_base_rows_only(self):
        rows = self.run_jobs()
        self.assertEqual(len(rows), 4000)
        # unchanged layout: per sorted fleet key SA s1, s2, s3, GR s1
        self.assertEqual([(r[0], r[2], r[3]) for r in rows[:4]],
                         [("ISA_b04_i001_1_fleet1", "SA", "1"), ("ISA_b04_i001_1_fleet1", "SA", "2"),
                          ("ISA_b04_i001_1_fleet1", "SA", "3"), ("ISA_b04_i001_1_fleet1", "GR", "1")])
        self.assertEqual(sum(1 for r in rows if r[2] == "GR"), 1000)

    def test_options_off_equal_the_options_never_given(self):
        plain = self.run_jobs()
        text_plain = (self.out / "jobs.csv").read_bytes()
        again = self.run_jobs("--gr-extra-seeds", "1", "--variants-instances", "", "--gr-extra-instances", "")
        self.assertEqual(plain, again)
        self.assertEqual(text_plain, (self.out / "jobs.csv").read_bytes())

    def test_planned_batch_4_extras_make_5200_rows_in_append_order(self):
        base = self.run_jobs()
        rows = self.run_jobs("--sa-variants", "1000", "--variants-instances", "1-10",
                             "--gr-extra-seeds", "10", "--gr-extra-instances", "1-10")
        self.assertEqual(len(rows), 4000 + 300 + 900)
        self.assertEqual(rows[:4000], base)  # row numbers of the base rows do not move
        variants, gr = rows[4000:4300], rows[4300:]
        self.assertTrue(all(r[2] == "SA1000" for r in variants))
        self.assertTrue(all(r[2] == "GR" and int(r[3]) >= 2 for r in gr))
        self.assertEqual({r[3] for r in gr}, {str(s) for s in range(2, 11)})
        self.assertEqual(len({r[0] for r in variants}), 100)
        self.assertEqual({r[0].split("_")[2] for r in gr}, {f"i{i:03d}" for i in range(1, 11)})
        # variant rows: sorted fleet key, then seed; GR extras: sorted fleet key, then seed
        self.assertEqual([(r[0], int(r[3])) for r in variants], sorted((r[0], int(r[3])) for r in variants))
        self.assertEqual([(r[0], int(r[3])) for r in gr], sorted((r[0], int(r[3])) for r in gr))

    def test_variant_rows_keep_their_numbers_when_gr_extras_are_added(self):
        without = self.run_jobs("--sa-variants", "1000", "--variants-instances", "1-10")
        with_gr = self.run_jobs("--sa-variants", "1000", "--variants-instances", "1-10",
                                "--gr-extra-seeds", "3", "--gr-extra-instances", "1-2")
        self.assertEqual(with_gr[:len(without)], without)
        self.assertEqual(len(with_gr) - len(without), 2 * 10 * 2)

    def test_variants_instances_intersects_with_variants_keys(self):
        rows = self.run_jobs("--sa-variants", "1000", "--variants-instances", "1-10",
                             "--variants-keys", "ISA_b04_i005_1_fleet3,ISA_b04_i050_1_fleet3")
        extra = rows[4000:]
        self.assertEqual({r[0] for r in extra}, {"ISA_b04_i005_1_fleet3"})  # i050 is outside 1-10: no union
        self.assertEqual(len(extra), 3)

    def test_variants_instances_intersects_with_max_fleet_percent(self):
        # every fleet has 5 vehicles for 20 requests = 25 per 100, so a limit of 24 leaves nothing
        with self.assertRaises(SystemExit):
            self.run_jobs("--sa-variants", "1000", "--variants-instances", "1-10",
                          "--variants-max-fleet-percent", "24")

    def test_bad_ranges_and_missing_instance_range_are_refused(self):
        for extra in (["--sa-variants", "1000", "--variants-instances", "10-1"],
                      ["--sa-variants", "1000", "--variants-instances", "a-b"],
                      ["--gr-extra-seeds", "5"],
                      ["--gr-extra-seeds", "5", "--gr-extra-instances", "200-300"]):
            with self.assertRaises(SystemExit, msg=extra):
                self.run_jobs(*extra)

    def test_single_number_range_is_one_instance(self):
        rows = self.run_jobs("--sa-variants", "1000", "--variants-instances", "7")
        self.assertEqual(len(rows), 4000 + 30)
        self.assertEqual({r[0].split("_")[2] for r in rows[4000:]}, {"i007"})


class GrSeedsInCollectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        (self.work / "results").mkdir()
        self.keys = ["ISA_b01_i001_1_fleet1", "ISA_b01_i001_1_fleet2"]
        append_features_rows([(k, {}) for k in self.keys], csv_path=self.work / "results" / "features_batch_01.csv")
        self.runs = self.work / "solver" / "batch_01" / "results"
        self.runs.mkdir(parents=True)
        for key, sa, gr in ((self.keys[0], 800, (1000, 1100, 900)), (self.keys[1], 500, (600, 640, 560))):
            for seed in (1, 2, 3):
                (self.runs / f"{key}_SA_s{seed}.txt").write_text(result_line(sa))
            for seed, value in enumerate(gr, start=1):
                (self.runs / f"{key}_GR_s{seed}.txt").write_text(result_line(value, served=90 + seed, third=10 - seed))

    def collect(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            collect_batch.main(["--batch", "1", "--work", str(self.work)] + list(extra))
        return pd.read_csv(self.work / "results" / "raw_results_mean_batch_01.csv", index_col=0)

    def test_extra_gr_seeds_do_not_change_the_base_matrices(self):
        with_extras = self.collect()
        perf_with = (self.work / "results" / "perf_matrix_mean_batch_01.csv").read_text()
        for path in self.runs.glob("*_GR_s[23].txt"):
            path.unlink()
        base_only = self.collect()
        pd.testing.assert_frame_equal(with_extras, base_only)
        self.assertEqual((self.work / "results" / "perf_matrix_mean_batch_01.csv").read_text(), perf_with)
        self.assertEqual(with_extras.loc[self.keys[0], "GR"], 1000)  # seed 1 only
        self.assertEqual(with_extras.loc[self.keys[1], "GR"], 600)

    def test_gr_seeds_max_averages_over_more_seeds(self):
        raw = self.collect("--gr-seeds-max", "3")
        self.assertAlmostEqual(raw.loc[self.keys[1], "GR"], 600.0)  # mean of 600, 640, 560
        self.assertAlmostEqual(self.collect("--gr-seeds-max", "2").loc[self.keys[0], "GR"], 1050.0)  # 1000, 1100
        self.assertAlmostEqual(self.collect("--gr-seeds-max", "2").loc[self.keys[1], "GR"], 620.0)

    def test_solver_runs_csv_keeps_every_seed(self):
        self.collect()
        table = pd.read_csv(self.work / "results" / "solver_runs_batch_01.csv")
        self.assertEqual(len(table), 2 * 3 + 2 * 3)


class GrReplicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        runs = self.work / "solver" / "batch_04" / "results"
        runs.mkdir(parents=True)
        values = {"ISA_b04_i001_1_fleet1": (1000, 1100, 900, 1000), "ISA_b04_i001_1_fleet2": (600, 800, 700, 750),
                  "ISA_b04_i001_1_fleet3": (500,)}  # one seed only: left out
        for key, vs in values.items():
            for seed, v in enumerate(vs, start=1):
                (runs / f"{key}_GR_s{seed}.txt").write_text(result_line(v))
            (runs / f"{key}_SA_s1.txt").write_text(result_line(1))

    def test_summary_file_and_printout(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            gr_replication.main(["--batch", "4", "--work", str(self.work)])
        table = pd.read_csv(self.work / "results" / "gr_replication_batch_04.csv", index_col="key")
        self.assertEqual(list(table.index), ["ISA_b04_i001_1_fleet1", "ISA_b04_i001_1_fleet2"])
        row = table.loc["ISA_b04_i001_1_fleet1"]
        self.assertEqual(row["n_seeds"], 4)
        self.assertAlmostEqual(row["urt_mean"], 1000.0)
        self.assertAlmostEqual(row["urt_sd"], (20000 / 3) ** 0.5)  # sample sd of 1000, 1100, 900, 1000
        self.assertEqual((row["urt_min"], row["urt_max"]), (900, 1100))
        self.assertAlmostEqual(row["served_mean"], 0.95)
        self.assertAlmostEqual(row["urt_cv"], row["urt_sd"] / 1000.0)
        out = buffer.getvalue()
        self.assertIn("2 keys with 2 or more GR seeds", out)
        # fleet1: |1000 - 1000| <= sd; fleet2: mean 712.5, sd about 85, |600 - 712.5| = 112.5 > sd
        self.assertIn("within one sd of the mean over the seeds: 1/2 keys", out)
        self.assertIn("median coefficient of variation", out)


if __name__ == "__main__":
    unittest.main()
