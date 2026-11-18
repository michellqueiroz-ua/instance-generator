import math
import sys
import unittest
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from compute_features_scaffold import compute_features


class TestT11DynamicPlaceholders(unittest.TestCase):
    def test_static_placeholders_are_stable_and_numeric(self) -> None:
        instance = {
            "requests": [
                {"tw_start": 480, "tw_end": 510},
                {"tw_start": 540, "tw_end": 600},
            ],
            "vehicles": [],
        }

        feats = compute_features(instance)

        self.assertAlmostEqual(feats["request_flux_dynamism"], 0.0, places=9)
        self.assertAlmostEqual(feats["static_to_dynamic_ratio"], 100.0, places=9)
        # issue_time defaults to 0 for static placeholder behavior.
        self.assertAlmostEqual(feats["avg_request_urgency"], (480.0 + 540.0) / 2.0, places=9)

    def test_dynamic_inputs_remain_future_compatible(self) -> None:
        instance = {
            "requests": [
                {"tw_start": 600, "tw_end": 660, "issue_time": 120},
                {"tw_start": 720, "tw_end": 780, "issue_time": 300},
                {"tw_start": 840, "tw_end": 900, "issue_time": 540},
            ],
            "vehicles": [],
        }

        feats = compute_features(instance)

        self.assertTrue(0.0 <= feats["request_flux_dynamism"] <= 1.0)
        self.assertTrue(0.0 <= feats["static_to_dynamic_ratio"] <= 100.0)
        self.assertGreater(feats["avg_request_urgency"], 0.0)

    def test_empty_or_malformed_requests_are_numeric(self) -> None:
        feats_empty = compute_features({"requests": [], "vehicles": []})
        self.assertAlmostEqual(feats_empty["request_flux_dynamism"], 0.0, places=9)
        self.assertAlmostEqual(feats_empty["static_to_dynamic_ratio"], 100.0, places=9)
        self.assertAlmostEqual(feats_empty["avg_request_urgency"], 0.0, places=9)

        feats_bad = compute_features({"requests": {"tw_start": 1}, "vehicles": []})
        self.assertAlmostEqual(feats_bad["request_flux_dynamism"], 0.0, places=9)
        self.assertAlmostEqual(feats_bad["static_to_dynamic_ratio"], 100.0, places=9)
        self.assertAlmostEqual(feats_bad["avg_request_urgency"], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
