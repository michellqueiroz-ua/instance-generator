import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
for path in (THIS_DIR, THIS_DIR / "isa_set"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import make_solver_jobs as jobs


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


class SolverJobsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        net = self.work / "Net"
        # 5 stations on a line; station 2 is the centre
        write_csv(net / "csv" / "Net.stations.csv", ["lat", "lon"], [[55.0, 12.0 + i * 0.01] for i in range(5)])
        write_csv(net / "travel_time_matrix" / "station_travel_time_matrix.csv", ["", "0"], [[0, 0]])
        # 20 requests; they use stations 2 and 3, so the depot cannot be station 2
        write_csv(net / "csv_format" / "ISA_b01_i001_1.csv", ["stops_orgn", "stops_dest"], [["[2]", "[3]"]] * 20)
        fleets = {
            "ISA_b01_i001_1_fleet1": {
                "instance": "ISA_b01_i001_1", "fleet_percent_of_requests": 25.0,
                "composition": [{"capacity": 4, "percent": 60.0}, {"capacity": 8, "percent": 40.0}],
            },
            "ISA_b01_i002_1_fleet1": {  # instance not generated yet
                "instance": "ISA_b01_i002_1", "fleet_percent_of_requests": 25.0,
                "composition": [{"capacity": 4, "percent": 100.0}, {"capacity": 8, "percent": 0.0}],
            },
        }
        (self.work / "design").mkdir()
        (self.work / "design" / "batch_01_fleets.json").write_text(json.dumps(fleets))
        jobs.main(["--batch", "1", "--work", str(self.work), "--network", "Net", "--sa-seeds", "2"])
        self.out = self.work / "solver" / "batch_01"

    def rows(self):
        lines = (self.out / "jobs.csv").read_text().strip().splitlines()[1:]
        return [[c.strip() for c in line.split(";")] for line in lines]

    def test_runs_per_fleet_and_skipped_missing_instance(self):
        rows = self.rows()
        self.assertEqual([(r[2], r[3]) for r in rows], [("SA", "1"), ("SA", "2"), ("GR", "1")])
        self.assertTrue(all(r[0] == "ISA_b01_i001_1_fleet1" for r in rows))

    def test_sa_variants_are_appended_after_the_standard_rows(self):
        standard = self.rows()
        jobs.main(["--batch", "1", "--work", str(self.work), "--network", "Net", "--sa-seeds", "2",
                   "--sa-variants", "1000,2000"])
        rows = self.rows()
        self.assertEqual(rows[:len(standard)], standard)  # row numbers of finished runs do not move
        self.assertEqual([(r[2], r[3]) for r in rows[len(standard):]],
                         [("SA1000", "1"), ("SA1000", "2"), ("SA2000", "1"), ("SA2000", "2")])
        run_rows = (self.out / "run_rows.sh").read_text()
        self.assertIn("--max_static_iterations ${algo#SA}", run_rows)

    def test_variants_can_be_limited_to_small_fleets(self):
        # the fleet of i001 has 5 vehicles for 20 requests: 25 vehicles per 100 requests
        jobs.main(["--batch", "1", "--work", str(self.work), "--network", "Net", "--sa-seeds", "1",
                   "--sa-variants", "1000", "--variants-max-fleet-percent", "24"])
        self.assertEqual([r[2] for r in self.rows()], ["SA", "GR"])
        jobs.main(["--batch", "1", "--work", str(self.work), "--network", "Net", "--sa-seeds", "1",
                   "--sa-variants", "1000", "--variants-max-fleet-percent", "25"])
        self.assertEqual([r[2] for r in self.rows()], ["SA", "GR", "SA1000"])

    def test_vehicle_counts_round_trip_through_the_solver_percentages(self):
        row = self.rows()[0]
        n_requests = int(row[4])
        counts = [math.ceil(float(row[5]) / 100 * n_requests), math.ceil(float(row[6]) / 100 * n_requests)]
        self.assertEqual(sum(counts), 5)  # 25% of 20 requests
        self.assertEqual(counts, [3, 2])

    def test_depot_is_a_station_no_request_uses(self):
        depot = int(self.rows()[0][7])
        self.assertNotIn(depot, (2, 3))
        self.assertEqual(depot, 1)  # closest to the centre among 0, 1, 4 (tie broken by order)

    def test_submit_script_has_resources_and_matrix_path(self):
        submit = (self.out / "submit.sh").read_text()
        self.assertIn("-n 1", submit)
        self.assertIn("rusage[mem=${MEM}GB]", submit)
        self.assertIn("run_rows.sh", submit)
        self.assertNotIn("\r", submit)
        run_rows = (self.out / "run_rows.sh").read_text()
        self.assertIn("station_travel_time_matrix.csv", run_rows)
        self.assertNotIn("__TRAVEL_TIME__", run_rows)
        self.assertNotIn("__SA_ARGS__", run_rows)
        self.assertIn("--init_temperature 1.3", run_rows)
        self.assertIn("--greedy_only", run_rows)


if __name__ == "__main__":
    unittest.main()
