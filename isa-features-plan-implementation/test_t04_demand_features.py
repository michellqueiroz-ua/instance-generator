import ast
import csv
import logging
import math
import sys
import unittest
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from compute_features_scaffold import FEATURE_KEYS, compute_features


def _build_real_sample_instance(max_rows: int = 80) -> dict:
    """Build an instance-like payload from repository sample CSV request rows."""
    csv_path = THIS_DIR / "fixtures" / "test_instance.csv"
    rows = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            if len(rows) >= max_rows:
                break

            try:
                origin_lat_lon = ast.literal_eval(rec["origin"])
                dest_lat_lon = ast.literal_eval(rec["destination"])
                tw_start_min = float(rec["earliest_departure"]) / 60.0
                tw_end_min = float(rec["latest_departure"]) / 60.0
            except (ValueError, SyntaxError):
                continue

            if not isinstance(origin_lat_lon, list) or not isinstance(dest_lat_lon, list):
                continue
            if len(origin_lat_lon) < 2 or len(dest_lat_lon) < 2:
                continue

            rows.append(
                {
                    "origin": {"x": float(origin_lat_lon[1]), "y": float(origin_lat_lon[0])},
                    "destination": {"x": float(dest_lat_lon[1]), "y": float(dest_lat_lon[0])},
                    "tw_start": tw_start_min,
                    "tw_end": tw_end_min,
                    "demand": 1,
                }
            )

    return {
        # Intentionally missing service_area_km2 to test fallback path.
        "requests": rows,
        "vehicles": [],
    }


class TestT04DemandFeatures(unittest.TestCase):
    def test_toy_instance_demand_metrics(self) -> None:
        toy_instance = {
            "service_area_km2": 100.0,
            "requests": [
                {
                    "origin": {"x": 4.40, "y": 51.21},
                    "destination": {"x": 4.42, "y": 51.22},
                    "tw_start": 480,
                    "tw_end": 510,
                    "demand": 2,
                },
                {
                    "origin": {"x": 4.41, "y": 51.20},
                    "destination": {"x": 4.43, "y": 51.23},
                    "tw_start": 490,
                    "tw_end": 540,
                    "demand": 1,
                },
            ],
            "vehicles": [
                {"capacity": 8, "depot": {"x": 4.39, "y": 51.20}},
                {"capacity": 12, "depot": {"x": 4.45, "y": 51.24}},
            ],
        }

        feats = compute_features(toy_instance)

        self.assertEqual(len(feats), len(FEATURE_KEYS))
        self.assertAlmostEqual(feats["n_requests"], 2.0)
        self.assertAlmostEqual(feats["demand_density"], 0.02)
        self.assertTrue(0.0 <= feats["spatial_cluster_coeff"] <= 1.0)

    def test_real_sample_fallback_logs_once(self) -> None:
        instance = _build_real_sample_instance(max_rows=80)
        self.assertGreaterEqual(len(instance["requests"]), 2)

        with self.assertLogs("compute_features_scaffold", level=logging.INFO) as captured:
            feats = compute_features(instance)

        self.assertEqual(feats["n_requests"], float(len(instance["requests"])))
        self.assertTrue(math.isfinite(feats["demand_density"]))
        self.assertGreater(feats["demand_density"], 0.0)
        self.assertTrue(math.isfinite(feats["spatial_cluster_coeff"]))
        self.assertTrue(0.0 <= feats["spatial_cluster_coeff"] <= 1.0)

        fallback_logs = [
            line for line in captured.output if "service_area_km2 fallback used" in line
        ]
        self.assertEqual(len(fallback_logs), 1)


if __name__ == "__main__":
    unittest.main()
