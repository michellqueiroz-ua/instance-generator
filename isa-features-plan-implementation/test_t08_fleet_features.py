import math
import sys
import unittest
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from compute_features_scaffold import compute_features


class TestT08FleetFeatures(unittest.TestCase):
    def test_fleet_features_formula_sanity(self) -> None:
        instance = {
            "requests": [{"id": i} for i in range(12)],
            "vehicles": [
                {"capacity": 10},
                {"capacity": 20},
                {"capacity": 30},
            ],
        }

        feats = compute_features(instance)

        expected_mean = 20.0
        expected_std = math.sqrt(((10.0 - expected_mean) ** 2 + (20.0 - expected_mean) ** 2 + (30.0 - expected_mean) ** 2) / 3.0)
        expected_cv = expected_std / expected_mean

        self.assertAlmostEqual(feats["fleet_size"], 3.0, places=9)
        self.assertAlmostEqual(feats["capacity_cv"], expected_cv, places=9)
        self.assertAlmostEqual(feats["fleet_utilization_ratio"], 60.0 / 12.0, places=9)
        self.assertAlmostEqual(feats["request_vehicle_ratio"], 12.0 / 3.0, places=9)

    def test_zero_division_and_empty_inputs(self) -> None:
        instance = {
            "requests": [{"id": i} for i in range(5)],
            "vehicles": [],
        }

        feats = compute_features(instance)

        self.assertAlmostEqual(feats["fleet_size"], 0.0, places=9)
        self.assertTrue(math.isnan(feats["capacity_cv"]))
        self.assertAlmostEqual(feats["fleet_utilization_ratio"], 0.0, places=9)
        self.assertTrue(math.isnan(feats["request_vehicle_ratio"]))

    def test_empty_requests_and_malformed_entries(self) -> None:
        instance = {
            "requests": [],
            "vehicles": [
                {"capacity": 10},
                {"capacity": 0},
                {"capacity": "bad"},
                {},
            ],
        }

        feats = compute_features(instance)

        self.assertAlmostEqual(feats["fleet_size"], 4.0, places=9)
        self.assertAlmostEqual(feats["capacity_cv"], 0.0, places=9)
        self.assertTrue(math.isnan(feats["fleet_utilization_ratio"]))
        self.assertAlmostEqual(feats["request_vehicle_ratio"], 0.0, places=9)

    def test_non_list_vehicles_returns_nan(self) -> None:
        feats = compute_features({"requests": [{"id": 1}], "vehicles": {"capacity": 10}})

        self.assertTrue(math.isnan(feats["fleet_size"]))
        self.assertTrue(math.isnan(feats["capacity_cv"]))
        self.assertTrue(math.isnan(feats["fleet_utilization_ratio"]))
        self.assertTrue(math.isnan(feats["request_vehicle_ratio"]))


if __name__ == "__main__":
    unittest.main()
