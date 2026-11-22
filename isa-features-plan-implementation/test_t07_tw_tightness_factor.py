import logging
import math
import sys
import unittest
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from compute_features_scaffold import compute_features


class TestT07TwTightnessFactor(unittest.TestCase):
    def test_formula_with_primary_driving_matrix(self) -> None:
        instance = {
            "requests": [
                {"tw_start": 0, "tw_end": 30, "originid": 0, "destinationid": 1},
                {"tw_start": 0, "tw_end": 20, "originid": 1, "destinationid": 2},
                {"tw_start": 10, "tw_end": 60, "originid": 2, "destinationid": 0},
            ],
            # Seconds; 600s=10min, 900s=15min, 1500s=25min.
            "travel_time_matrix_sec": [
                [1, 600, 700],
                [620, 1, 900],
                [1500, 880, 1],
            ],
            "vehicles": [],
        }

        feats = compute_features(instance)

        expected = [30.0 / 10.0, 20.0 / 15.0, 50.0 / 25.0]
        expected_mean = sum(expected) / len(expected)
        self.assertAlmostEqual(feats["tw_tightness_factor"], expected_mean, places=9)

        # Plausible range sanity check for this toy case.
        self.assertTrue(1.0 <= feats["tw_tightness_factor"] <= 3.5)

    def test_skip_invalid_rows_when_primary_exists(self) -> None:
        instance = {
            "requests": [
                {
                    "tw_start": 0,
                    "tw_end": 30,
                    "originid": 0,
                    "destinationid": 1,
                    "direct_travel_time": 1200,
                },
                {
                    "tw_start": 5,
                    "tw_end": 35,
                    "originid": 1,
                    "destinationid": 2,
                    "direct_travel_time": 600,
                },
                {
                    "tw_start": 10,
                    "tw_end": 30,
                    "originid": 2,
                    "destinationid": 1,
                    "direct_travel_time": 300,
                },
            ],
            # Matrix present -> primary exists. Row (1,2) has 0 and (2,1) has -10,
            # so both must be skipped rather than using direct_travel_time fallback.
            "travel_time_matrix_sec": [
                [1, 600, 500],
                [700, 1, 0],
                [800, -10, 1],
            ],
            "vehicles": [],
        }

        feats = compute_features(instance)

        # Only first request contributes: 30 / 10 = 3.0
        self.assertAlmostEqual(feats["tw_tightness_factor"], 3.0, places=9)

    def test_fallback_used_only_when_primary_missing(self) -> None:
        instance = {
            "requests": [
                {"tw_start": 0, "tw_end": 30, "direct_travel_time": 600},
                {"tw_start": 10, "tw_end": 35, "direct_travel_time": 1500},
                {"tw_start": 10, "tw_end": 20, "direct_travel_time": 0},
            ],
            "vehicles": [],
        }

        with self.assertLogs("compute_features_scaffold", level=logging.INFO) as captured:
            feats = compute_features(instance)

        # Factors from valid fallback rows: 30/10 and 25/25 -> mean 2.0
        self.assertAlmostEqual(feats["tw_tightness_factor"], 2.0, places=9)

        fallback_logs = [
            line for line in captured.output if "tw_tightness fallback used" in line
        ]
        self.assertEqual(len(fallback_logs), 1)

    def test_nan_for_malformed_or_empty_requests(self) -> None:
        feats_bad = compute_features({"requests": {"tw_start": 0, "tw_end": 1}, "vehicles": []})
        self.assertTrue(math.isnan(feats_bad["tw_tightness_factor"]))

        feats_empty = compute_features({"requests": [], "vehicles": []})
        self.assertTrue(math.isnan(feats_empty["tw_tightness_factor"]))


if __name__ == "__main__":
    unittest.main()
