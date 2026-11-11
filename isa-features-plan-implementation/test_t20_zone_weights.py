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

    def test_central_zones_are_removed_with_factor_zero_and_kept_with_one(self):
        zones = [(n, h, a, 0.8 if n == "zone1" else 0.1) for n, h, a in ZONES]  # only zone1 is mostly central
        origin, destination = make_batch.od_weights(zones, 0.5, 1.0, central_factor=0.0)
        self.assertEqual(origin[1], 0.0)
        self.assertEqual(destination[1], 0.0)
        self.assertAlmostEqual(sum(origin), 1.0)
        self.assertAlmostEqual(sum(destination), 1.0)
        kept = make_batch.od_weights(zones, 0.5, 1.0, central_factor=1.0)
        self.assertGreater(kept[0][1], 0.0)

    def test_the_threshold_decides_which_boundary_zones_are_removed(self):
        zones = [(n, h, a, {"zone0": 0.0, "zone1": 0.3, "zone2": 0.6, "zone3": 0.0}[n]) for n, h, a in ZONES]
        strict = make_batch.od_weights(zones, 0.5, 1.0, central_factor=0.0, central_threshold=0.5)[0]
        loose = make_batch.od_weights(zones, 0.5, 1.0, central_factor=0.0, central_threshold=0.2)[0]
        self.assertGreater(strict[1], 0.0)   # 30% central: kept at threshold 0.5
        self.assertEqual(loose[1], 0.0)      # removed at threshold 0.2
        self.assertEqual(strict[2], 0.0)     # 60% central: removed at both
        self.assertEqual(loose[2], 0.0)

    def test_zero_factor_removes_a_zone_even_at_the_lowest_concentration(self):
        zones = [(n, h, a, 1.0 if n == "zone3" else 0.0) for n, h, a in ZONES]
        origin, _ = make_batch.od_weights(zones, 0.5, 0.1, central_factor=0.0)
        self.assertEqual(origin[3], 0.0)

    def test_the_decay_scale_goes_to_the_destination_only(self):
        design = json.loads((THIS_DIR / "isa_set" / "design.json").read_text(encoding="utf-8"))
        settings = {"start_hour": 14.0, "requests": 100, "peak_sd_min": 30, "urgency_mean_min": 10, "urgency_sd_min": 3,
                    "arrival_slack_min": 20, "inbound_share": 0.5, "concentration": 1.0}
        design["spatial"]["trip_decay_km"] = 5
        config = make_batch.make_config(design, settings, 1, False, ZONES)
        attrs = {a["name"]: a for a in config["attributes"]}
        self.assertEqual(attrs["destination"]["distance_decay_m"], 5000)
        self.assertNotIn("distance_decay_m", attrs["origin"])
        design["spatial"]["trip_decay_km"] = None
        config = make_batch.make_config(design, settings, 1, False, ZONES)
        self.assertNotIn("distance_decay_m", {a["name"]: a for a in config["attributes"]}["destination"])

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
            writer.writerow(["name", "n_stations", "n_residential", "n_attraction", "central_share"])
            for name, home, place in ZONES:
                writer.writerow([name, 3, home, place, 0.9 if name == "zone1" else 0.0])

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
            self.assertEqual(params["max_dtt"]["value"], 3600)
            self.assertEqual(attrs["origin"]["weights"][1], 0.0)       # zone1 is central: removed from the service area
            self.assertEqual(attrs["destination"]["weights"][1], 0.0)
            self.assertEqual(params["max_early_departure"]["value"], 18.0)  # 4 h period from 14:00

    def test_volume_is_fixed_at_5000_requests_and_the_set_option_holds_a_setting(self):
        for config in self.run_batch():
            self.assertEqual(config["requests"], 5000)
        make_batch.main(["--batch", "3", "--n", "2", "--work", str(self.work),
                         "--set", "requests_per_hour=250", "--set", "inbound_share=1"])
        with open(self.work / "design" / "batch_03.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([(float(r["requests_per_hour"]), float(r["inbound_share"]), r["scenario"]) for r in rows],
                         [(250.0, 1.0, "a_inbound")] * 2)
        with self.assertRaises(SystemExit):
            make_batch.main(["--batch", "4", "--n", "1", "--work", str(self.work), "--set", "nonsense=1"])

    def test_batch_csv_has_factors_and_scenario_and_start_hour_is_not_drawn(self):
        self.run_batch(n=20)
        with open(self.work / "design" / "batch_02.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(all(0 <= float(r["inbound_share"]) <= 1 and 0.1 <= float(r["concentration"]) <= 1.5 for r in rows))
        self.assertTrue(all(float(r["start_hour"]) == 14.0 for r in rows))
        self.assertEqual({r["scenario"] for r in rows}, {"a_inbound", "b_both_ways", "c_outbound"})

    def test_a_weights_file_without_central_share_stops_the_batch(self):
        path = self.work / "Capital Region of Denmark" / "csv" / "Capital Region of Denmark.zone_weights.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "n_stations", "n_residential", "n_attraction"])
            writer.writerow(["zone0", 3, 10, 10])
        with self.assertRaises(SystemExit) as context:
            self.run_batch()
        self.assertIn("central_share", str(context.exception))

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
