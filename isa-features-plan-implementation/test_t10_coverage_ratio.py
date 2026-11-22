import logging
import math
import sys
import unittest
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from compute_features_scaffold import compute_features


class TestT10CoverageRatio(unittest.TestCase):
    def test_coverage_ratio_deterministic_and_configurable_grid(self) -> None:
        instance = {
            "coverage_grid_n": 31,
            "walk_speed": 0.5,
            "stops": [
                {"id": "s0", "x": 0.005, "y": 0.005},
            ],
            "requests": [
                {
                    "origin": {"x": 0.0, "y": 0.0},
                    "destination": {"x": 0.01, "y": 0.01},
                }
            ],
            "vehicles": [],
        }

        feats_a = compute_features(instance)
        feats_b = compute_features(instance)

        self.assertAlmostEqual(feats_a["coverage_ratio"], feats_b["coverage_ratio"], places=12)
        self.assertTrue(0.0 <= feats_a["coverage_ratio"] <= 1.0)
        self.assertLess(feats_a["coverage_ratio"], 1.0)

        instance_finer = dict(instance)
        instance_finer["coverage_grid_n"] = 61
        feats_finer = compute_features(instance_finer)
        self.assertTrue(0.0 <= feats_finer["coverage_ratio"] <= 1.0)

    def test_runtime_guard_caps_grid_size(self) -> None:
        stops = []
        for i in range(120):
            for j in range(5):
                stops.append({"id": f"s{i}_{j}", "x": i * 0.0001, "y": j * 0.0001})

        instance = {
            "coverage_grid_n": 2000,
            "walk_speed": 1.2,
            "stops": stops,
            "requests": [
                {
                    "origin": {"x": 0.0, "y": 0.0},
                    "destination": {"x": 0.02, "y": 0.01},
                }
            ],
            "vehicles": [],
        }

        with self.assertLogs("compute_features_scaffold", level=logging.INFO) as captured:
            feats = compute_features(instance)

        self.assertTrue(0.0 <= feats["coverage_ratio"] <= 1.0)
        self.assertIn("coverage_ratio grid_n capped", "\n".join(captured.output))

    def test_coverage_ratio_returns_nan_without_stops(self) -> None:
        instance = {
            "requests": [
                {
                    "origin": {"x": 5.68, "y": 50.85},
                    "destination": {"x": 5.69, "y": 50.86},
                }
            ],
            "vehicles": [],
        }

        feats = compute_features(instance)
        self.assertTrue(math.isnan(feats["coverage_ratio"]))

    def test_walk_speed_default_fallback_is_logged(self) -> None:
        instance = {
            "coverage_grid_n": 25,
            "stops": [
                {"id": "s0", "x": 0.005, "y": 0.005},
            ],
            "requests": [
                {
                    "origin": {"x": 0.0, "y": 0.0},
                    "destination": {"x": 0.01, "y": 0.01},
                }
            ],
            "vehicles": [],
        }

        with self.assertLogs("compute_features_scaffold", level=logging.INFO) as captured:
            feats = compute_features(instance)

        self.assertTrue(0.0 <= feats["coverage_ratio"] <= 1.0)
        self.assertIn("coverage_ratio walk-speed fallback used", "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()
