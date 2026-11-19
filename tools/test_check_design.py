import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
ISA = REPO / "isa-features-plan-implementation" / "isa_set"
DESIGN = ISA / "design.json"
sys.path.insert(0, str(TOOLS))

import check_design

ZONES = [("zone0", 100, 0), ("zone1", 50, 50), ("zone2", 0, 100), ("zone3", 50, 50)]


def make_real_batch(work, batch=4, n=100):
    """Runs the real make_batch.py with a fake zone-weights file (as test_t20_zone_weights.py does)."""
    weights = Path(work) / "zone_weights.csv"
    with open(weights, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "n_stations", "n_residential", "n_attraction", "central_share"])
        for name, home, place in ZONES:
            writer.writerow([name, 3, home, place, 0.9 if name == "zone1" else 0.0])
    subprocess.run([sys.executable, str(ISA / "make_batch.py"), "--batch", str(batch), "--n", str(n),
                    "--work", str(Path(work) / "w"), "--zone-weights", str(weights)],
                   check=True, capture_output=True, text=True)
    return Path(work) / "w"


class CheckDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.good = make_real_batch(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.copy_tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.copy_tmp.cleanup)
        self.work = Path(self.copy_tmp.name) / "w"
        shutil.copytree(self.good, self.work)

    def run_check(self):
        return check_design.run(DESIGN, self.work, 4, 100)

    def failed_names(self):
        return [name for name, ok, _ in self.run_check().lines if not ok]

    def edit_csv(self, change):
        path = self.work / "design" / "batch_04.csv"
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        change(rows)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def edit_fleets(self, change):
        path = self.work / "design" / "batch_04_fleets.json"
        fleets = json.loads(path.read_text(encoding="utf-8"))
        change(fleets)
        path.write_text(json.dumps(fleets), encoding="utf-8")

    def test_a_real_batch_passes_every_check(self):
        report = self.run_check()
        self.assertEqual(report.failed, [], report.text())
        self.assertGreater(len(report.lines), 15)

    def test_exit_code_is_0_on_a_good_batch(self):
        self.assertEqual(check_design.main(["--design-json", str(DESIGN), "--batch-dir", str(self.work),
                                            "--batch", "4"]), 0)

    def test_a_setting_out_of_range_fails_range_and_latin_hypercube(self):
        def change(rows):
            rows[0]["urgency_mean_min"] = "99"
        self.edit_csv(change)
        failed = self.failed_names()
        self.assertIn("instances: urgency_mean_min in [5, 40]", failed)
        self.assertIn("instances: urgency_mean_min Latin hypercube", failed)

    def test_two_points_in_one_slice_fail_the_hypercube(self):
        def change(rows):
            rows[1]["concentration"] = rows[0]["concentration"]
        self.edit_csv(change)
        self.assertEqual(self.failed_names(), ["instances: concentration Latin hypercube"])

    def test_a_wrong_fixed_start_hour_fails(self):
        def change(rows):
            rows[3]["start_hour"] = "12.0"
        self.edit_csv(change)
        self.assertEqual(self.failed_names(), ["instances: start_hour fixed at 10.0"])

    def test_a_fleet_moved_out_of_its_stratum_fails_the_strata_check(self):
        def change(fleets):
            key = "ISA_b04_i001_1_fleet1"
            others = sorted(f["fleet_percent_of_requests"] for k, f in fleets.items() if f["instance"] == "ISA_b04_i001_1")
            fleets[key]["fleet_percent_of_requests"] = 24.9 if fleets[key]["fleet_percent_of_requests"] < 24 else 10.1
            self.assertEqual(len(others), 10)
        self.edit_fleets(change)
        failed = self.failed_names()
        self.assertEqual(failed, ["fleets: one fleet per stratum (10 strata of 10-25%)"])

    def test_a_missing_fleet_fails_count_and_per_instance(self):
        def change(fleets):
            del fleets["ISA_b04_i002_1_fleet3"]
        self.edit_fleets(change)
        failed = self.failed_names()
        self.assertIn("fleets: count", failed)
        self.assertIn("fleets: exactly 10 per instance", failed)

    def test_a_fleet_percent_out_of_range_and_a_bad_share_fail(self):
        def change(fleets):
            fleets["ISA_b04_i005_1_fleet2"]["fleet_percent_of_requests"] = 40
            fleets["ISA_b04_i006_1_fleet2"]["composition"] = [{"capacity": 4, "percent": 0}, {"capacity": 8, "percent": 130}]
        self.edit_fleets(change)
        failed = self.failed_names()
        self.assertIn("fleets: percent in [10, 25]", failed)
        self.assertIn("fleets: 8-seat share in [0, 100]%", failed)
        self.assertIn("fleets: composition sums to 100%", failed)

    def test_a_wrong_walking_time_in_the_first_config_fails(self):
        path = sorted((self.work / "configs" / "batch_04").glob("*.json"))[0]
        config = json.loads(path.read_text(encoding="utf-8"))
        for attr in config["attributes"]:
            if attr["name"] == "max_walking":
                attr["pdf"] = [{"type": "uniform", "loc": 300, "scale": 600}]
        path.write_text(json.dumps(config), encoding="utf-8")
        failed = self.failed_names()
        self.assertEqual(failed, [f"configs: {path.name} max_walking loc = 600, scale = 0"])

    def test_exit_code_is_1_on_a_bad_batch(self):
        self.edit_csv(lambda rows: rows[0].update(inbound_share="7"))
        self.assertEqual(check_design.main(["--design-json", str(DESIGN), "--batch-dir", str(self.work),
                                            "--batch", "4"]), 1)

    def test_missing_files_fail_instead_of_crashing(self):
        report = check_design.run(DESIGN, self.work / "nowhere", 4, 100)
        self.assertTrue(report.failed)

    def test_stratum_matching_allows_rounding_at_a_boundary(self):
        # strata of 10-25% with 10 draws are 1.5 wide; stratum 1 starts at 11.5, and 11.496 only counts with the tolerance
        values = [10.5 + 1.5 * s for s in range(10)]
        values[1] = 11.496
        self.assertTrue(check_design.stratum_matches(values, 10, 25, 10, 0.006))
        self.assertFalse(check_design.stratum_matches(values, 10, 25, 10, 0.0))


if __name__ == "__main__":
    unittest.main()
