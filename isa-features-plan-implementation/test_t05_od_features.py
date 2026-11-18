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

from compute_features_scaffold import compute_features


def _build_instance_from_csv(max_rows: int = 80) -> tuple:
    """Build instance-like data and expected driving distances (km) from CSV sample."""
    csv_path = THIS_DIR / "fixtures" / "test_instance.csv"
    requests = []
    expected_km = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            if len(requests) >= max_rows:
                break

            try:
                origin_lat_lon = ast.literal_eval(rec["origin"])
                destination_lat_lon = ast.literal_eval(rec["destination"])
                direct_distance_m = float(rec["direct_distance"])
            except (ValueError, SyntaxError):
                continue

            if (
                not isinstance(origin_lat_lon, list)
                or not isinstance(destination_lat_lon, list)
                or len(origin_lat_lon) < 2
                or len(destination_lat_lon) < 2
            ):
                continue

            requests.append(
                {
                    "origin": {"x": float(origin_lat_lon[1]), "y": float(origin_lat_lon[0])},
                    "destination": {
                        "x": float(destination_lat_lon[1]),
                        "y": float(destination_lat_lon[0]),
                    },
                    "direct_distance": direct_distance_m,
                }
            )
            expected_km.append(direct_distance_m / 1000.0)

    return ({"requests": requests, "vehicles": []}, expected_km)


class TestT05ODDistanceFeatures(unittest.TestCase):
    def test_uses_drive_distance_matrix_by_default(self) -> None:
        instance = {
            "requests": [
                {
                    "origin": {"x": 5.0, "y": 50.0},
                    "destination": {"x": 5.1, "y": 50.1},
                    "originid": 0,
                    "destinationid": 1,
                },
                {
                    "origin": {"x": 5.2, "y": 50.2},
                    "destination": {"x": 5.3, "y": 50.3},
                    "originid": 2,
                    "destinationid": 1,
                },
                {
                    "origin": {"x": 5.4, "y": 50.4},
                    "destination": {"x": 5.5, "y": 50.5},
                    "originid": 1,
                    "destinationid": 2,
                },
            ],
            # Meters, explicitly marked as such.
            "drive_distance_matrix_m": [
                [0, 5000, 12000],
                [5200, 0, 10000],
                [12500, 8000, 0],
            ],
            "vehicles": [],
        }

        feats = compute_features(instance)

        expected = [5.0, 8.0, 10.0]
        expected_mean = sum(expected) / len(expected)
        expected_var = sum((x - expected_mean) ** 2 for x in expected) / len(expected)
        expected_std = expected_var ** 0.5

        self.assertAlmostEqual(feats["od_distance_mean"], expected_mean, places=9)
        self.assertAlmostEqual(feats["od_distance_std"], expected_std, places=9)

    def test_sample_instance_direct_distance_sanity(self) -> None:
        instance, expected_km = _build_instance_from_csv(max_rows=80)
        self.assertGreaterEqual(len(instance["requests"]), 2)

        with self.assertLogs("compute_features_scaffold", level=logging.INFO) as captured:
            feats = compute_features(instance)

        expected_mean = sum(expected_km) / len(expected_km)
        expected_var = sum((x - expected_mean) ** 2 for x in expected_km) / len(expected_km)
        expected_std = expected_var ** 0.5

        self.assertAlmostEqual(feats["od_distance_mean"], expected_mean, places=6)
        self.assertAlmostEqual(feats["od_distance_std"], expected_std, places=6)

        fallback_logs = [
            line for line in captured.output if "od_distance fallback used" in line
        ]
        self.assertEqual(len(fallback_logs), 0)

    def test_haversine_fallback_only_when_driving_missing(self) -> None:
        instance = {
            "requests": [
                {
                    "origin": {"x": 5.68, "y": 50.85},
                    "destination": {"x": 5.69, "y": 50.86},
                },
                {
                    "origin": {"x": 5.66, "y": 50.84},
                    "destination": {"x": 5.67, "y": 50.85},
                },
            ],
            "vehicles": [],
        }

        with self.assertLogs("compute_features_scaffold", level=logging.INFO) as captured:
            feats = compute_features(instance)

        self.assertTrue(math.isfinite(feats["od_distance_mean"]))
        self.assertTrue(math.isfinite(feats["od_distance_std"]))
        self.assertGreater(feats["od_distance_mean"], 0.0)

        fallback_logs = [
            line for line in captured.output if "od_distance fallback used" in line
        ]
        self.assertEqual(len(fallback_logs), 1)


if __name__ == "__main__":
    unittest.main()
