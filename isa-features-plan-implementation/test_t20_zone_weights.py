import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
for path in (THIS_DIR, THIS_DIR / "isa_set"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import make_batch
import zone_weights

ZONES = [("zone0", 100.0, 0.0), ("zone1", 50.0, 50.0), ("zone2", 0.0, 100.0), ("zone3", 50.0, 50.0)]


class OdWeightsTests(unittest.TestCase):
    def test_inbound_origins_follow_homes_and_destinations_follow_attractions(self):
        origin, destination = make_batch.od_weights(ZONES, inbound_share=1.0, concentration=1.0)
        self.assertAlmostEqual(sum(origin), 1.0)
        self.assertAlmostEqual(sum(destination), 1.0)
        self.assertAlmostEqual(origin[0], 0.5)       # 100 of 200 homes
        self.assertEqual(origin[2], 0.0)             # no homes in zone2
        self.assertAlmostEqual(destination[2], 0.5)  # 100 of 200 attractions
        self.assertEqual(destination[0], 0.0)

    def test_outbound_is_the_mirror_image(self):
        inbound = make_batch.od_weights(ZONES, 1.0, 1.0)
        outbound = make_batch.od_weights(ZONES, 0.0, 1.0)
        self.assertEqual(outbound[0], inbound[1])
        self.assertEqual(outbound[1], inbound[0])

    def test_zero_concentration_is_uniform(self):
        origin, destination = make_batch.od_weights(ZONES, 0.3, 0.0)
        self.assertEqual(origin, [0.25] * 4)
        self.assertEqual(destination, [0.25] * 4)

    def test_concentration_packs_the_demand_into_dense_zones(self):
        _, flat = make_batch.od_weights(ZONES, 1.0, 1.0)
        _, packed = make_batch.od_weights(ZONES, 1.0, 2.0)
        self.assertGreater(packed[2], flat[2])
        self.assertLess(packed[1], flat[1])

    def test_scenario_labels(self):
        self.assertEqual([make_batch.scenario_label(x) for x in (0.0, 0.5, 1.0)],
                         ["c_outbound", "b_both_ways", "a_inbound"])


class BatchWithZonesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        folder = self.work / "Capital Region of Denmark" / "csv"
        folder.mkdir(parents=True)
        with open(folder / "Capital Region of Denmark.zone_weights.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "n_stations", "n_residential", "n_attraction"])
            for name, home, place in ZONES:
                writer.writerow([name, 3, home, place])

    def run_batch(self, n=8):
        make_batch.main(["--batch", "2", "--n", str(n), "--work", str(self.work)])
        configs = sorted((self.work / "configs" / "batch_02").glob("*.json"))
        return [json.loads(p.read_text()) for p in configs]

    def test_configs_carry_zone_weights_and_fixed_start(self):
        for config in self.run_batch():
            params = {p["name"]: p for p in config["parameters"]}
            self.assertEqual(params["zones_all"]["size"], 4)
            self.assertEqual(params["zones_all"]["value"], ["zone0", "zone1", "zone2", "zone3"])
            attrs = {a["name"]: a for a in config["attributes"]}
            for end in ("origin", "destination"):
                self.assertEqual(attrs[end]["subset_zones"], "zones_all")
                self.assertEqual(len(attrs[end]["weights"]), 4)
                self.assertAlmostEqual(sum(attrs[end]["weights"]), 1.0, places=5)
            self.assertEqual(params["min_early_departure"]["value"], 14.0)
            self.assertEqual(params["max_dtt"]["value"], 5400)

    def test_batch_csv_has_factors_and_scenario_and_start_hour_is_not_drawn(self):
        self.run_batch(n=20)
        with open(self.work / "design" / "batch_02.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(all(0 <= float(r["inbound_share"]) <= 1 and 0 <= float(r["concentration"]) <= 2 for r in rows))
        self.assertTrue(all(float(r["start_hour"]) == 14.0 for r in rows))
        self.assertEqual({r["scenario"] for r in rows}, {"a_inbound", "b_both_ways", "c_outbound"})

    def test_missing_zone_weights_stop_the_batch(self):
        (self.work / "Capital Region of Denmark" / "csv" / "Capital Region of Denmark.zone_weights.csv").unlink()
        with self.assertRaises(SystemExit) as context:
            self.run_batch()
        self.assertIn("zone_weights.sh", str(context.exception))


class ZoneCountTests(unittest.TestCase):
    def test_points_are_counted_in_the_rectangle_that_holds_them(self):
        zones = [{"name": "a", "bounds": (0, 0, 1, 1)}, {"name": "b", "bounds": (1, 0, 2, 1)}]
        self.assertEqual(zone_weights.count_points(zones, [0.5, 0.5, 0.2, 5.0], [0.5, 1.5, 1.9, 0.5]), [1, 2])

    def test_strips_group_zones_by_their_southern_edge(self):
        zones = [{"name": "a", "bounds": (0, 0, 1, 1)}, {"name": "b", "bounds": (1, 0, 2, 1)},
                 {"name": "c", "bounds": (0, 1, 1, 2)}]
        self.assertEqual([[z["name"] for z in s] for s in zone_weights.strips(zones)], [["a", "b"], ["c"]])

    def test_a_restart_uses_the_cached_strips(self):
        zones = [{"name": "a", "bounds": (0, 0, 1, 1), "n_stations": 1},
                 {"name": "c", "bounds": (0, 1, 1, 2), "n_stations": 1}]
        calls = []

        def fake_download(bounds, tags):
            calls.append(bounds)
            return [bounds[1] + 0.5], [0.5]

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            first = zone_weights.build(zones, Path(tmp), download=fake_download, log=lambda m: None)
            self.assertEqual(len(calls), 4)  # 2 strips x (residential, attraction)
            second = zone_weights.build(zones, Path(tmp), download=fake_download, log=lambda m: None)
            self.assertEqual(len(calls), 4)
            self.assertEqual(first, second)
            self.assertEqual(first, {"a": [1, 1], "c": [1, 1]})


if __name__ == "__main__":
    unittest.main()
