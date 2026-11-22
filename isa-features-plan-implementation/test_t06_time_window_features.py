import math
import sys
import unittest
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from compute_features_scaffold import compute_features


class TestT06TimeWindowFeatures(unittest.TestCase):
    def test_time_window_formulas_on_toy_data(self) -> None:
        instance = {
            "requests": [
                {"tw_start": 0, "tw_end": 30},
                {"tw_start": 10, "tw_end": 25},
                {"tw_start": 40, "tw_end": 70},
                {"tw_start": 50, "tw_end": 55},
            ],
            "vehicles": [],
        }

        feats = compute_features(instance)

        expected_widths = [30.0, 15.0, 30.0, 5.0]
        expected_mean = sum(expected_widths) / len(expected_widths)
        expected_var = sum((x - expected_mean) ** 2 for x in expected_widths) / len(expected_widths)
        expected_std = expected_var ** 0.5

        # Overlaps: (0,1) and (2,3) over 6 total pairs.
        expected_overlap_ratio = 2.0 / 6.0

        self.assertAlmostEqual(feats["tw_width_mean"], expected_mean, places=9)
        self.assertAlmostEqual(feats["tw_width_std"], expected_std, places=9)
        self.assertAlmostEqual(feats["tw_overlap_ratio"], expected_overlap_ratio, places=9)
        self.assertAlmostEqual(feats["tight_tw_fraction"], 0.5, places=9)

    def test_empty_or_non_list_requests_returns_nan(self) -> None:
        instance_empty = {"requests": [], "vehicles": []}
        feats_empty = compute_features(instance_empty)

        self.assertTrue(math.isnan(feats_empty["tw_width_mean"]))
        self.assertTrue(math.isnan(feats_empty["tw_width_std"]))
        self.assertTrue(math.isnan(feats_empty["tw_overlap_ratio"]))
        self.assertTrue(math.isnan(feats_empty["tight_tw_fraction"]))

        instance_bad = {"requests": {"tw_start": 0, "tw_end": 10}, "vehicles": []}
        feats_bad = compute_features(instance_bad)

        self.assertTrue(math.isnan(feats_bad["tw_width_mean"]))
        self.assertTrue(math.isnan(feats_bad["tw_width_std"]))
        self.assertTrue(math.isnan(feats_bad["tw_overlap_ratio"]))
        self.assertTrue(math.isnan(feats_bad["tight_tw_fraction"]))

    def test_malformed_entries_are_skipped_without_crashing(self) -> None:
        instance = {
            "requests": [
                {"tw_start": 100, "tw_end": 130},
                {"tw_start": "bad", "tw_end": 120},
                {"tw_start": 140, "tw_end": 130},
                {},
                "not-a-dict",
            ],
            "vehicles": [],
        }

        feats = compute_features(instance)

        self.assertAlmostEqual(feats["tw_width_mean"], 30.0, places=9)
        self.assertAlmostEqual(feats["tw_width_std"], 0.0, places=9)
        self.assertTrue(math.isnan(feats["tw_overlap_ratio"]))
        self.assertAlmostEqual(feats["tight_tw_fraction"], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
