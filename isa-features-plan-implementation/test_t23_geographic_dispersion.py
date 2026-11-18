"""
T23 - geographic_dispersion (Queiroz, Lucas & Sorensen 2024, EJOR 317, section 5.4, p. 703).

gd = mu + omega, in minutes. The hand-computed toy instance below follows the formula with ths = 10 min
and n = 5; a brute-force port of the loops in REQreate/geographic_dispersion.py (ODBRP branch) checks
the vectorised implementation on a random instance.
"""
import math
import sys
import time
import unittest
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from compute_features_scaffold import FEATURE_KEYS, compute_features


def _request(ed, la, orgn, dest):
    return {"tw_start": ed, "tw_end": ed + 5.0, "latest_arrival": la, "stops_orgn": orgn, "stops_dest": dest}


def _brute_force(requests, matrix_sec, ths=10.0, n_nearest=5):
    """Straight port of the tool's ODBRP loops (quadratic), in minutes."""
    n = len(requests)
    mu_sum = 0.0
    for r in requests:
        pairs = [matrix_sec[o][d] for o in r["stops_orgn"] for d in r["stops_dest"] if o != d]
        mu_sum += sum(pairs) / len(pairs)
    mu = mu_sum / n
    total = 0.0
    for i, ri in enumerate(requests):
        lo_, ld_ = [], []
        for j, rj in enumerate(requests):
            if i == j:
                continue
            if abs(rj["tw_start"] - ri["tw_start"]) <= ths:
                lo_.extend(rj["stops_orgn"])
            if abs(rj["latest_arrival"] - ri["tw_start"]) <= ths:
                lo_.extend(rj["stops_dest"])
            if abs(rj["latest_arrival"] - ri["latest_arrival"]) <= ths:
                ld_.extend(rj["stops_dest"])
            if abs(rj["tw_start"] - ri["latest_arrival"]) <= ths:
                ld_.extend(rj["stops_orgn"])
        for own, cand in ((ri["stops_orgn"], lo_), (ri["stops_dest"], ld_)):
            cand = [c for c in dict.fromkeys(cand) if c not in own]
            times = sorted(matrix_sec[s][c] for s in own for c in cand)[:n_nearest]
            total += (sum(times) / len(times)) if times else 0.0
    return mu / 60.0 + total / (2 * n) / 60.0


class TestT23GeographicDispersion(unittest.TestCase):
    def test_feature_is_listed(self) -> None:
        self.assertIn("geographic_dispersion", FEATURE_KEYS)

    def test_hand_computed_toy_instance(self) -> None:
        # Stations 0..5. Direct times (s): 0-1 600, 2-3 1200, 4-5 1800; 0-2 300; every other pair 5000.
        m = [[5000.0] * 6 for _ in range(6)]
        for a, b, t in ((0, 1, 600.0), (2, 3, 1200.0), (4, 5, 1800.0), (0, 2, 300.0)):
            m[a][b] = m[b][a] = t
        requests = [
            _request(0.0, 30.0, [0], [1]),     # A
            _request(5.0, 100.0, [2], [3]),    # B: ed within 10 min of A's ed
            _request(200.0, 260.0, [4], [5]),  # C: alone in time
        ]
        # mu = (10 + 20 + 30) / 3 = 20 min.
        # omega: A: L^o = {2} -> 300 s = 5 min, L^d = {} -> 0; B: L^o = {0} -> 5 min, L^d = {} -> 0; C: 0.
        # omega = (5 + 5 + 0) / (2 * 3) = 10 / 6; gd = 20 + 10 / 6.
        feats = compute_features({"requests": requests, "travel_time_matrix_sec": m, "vehicles": []})
        self.assertAlmostEqual(feats["geographic_dispersion"], 20.0 + 10.0 / 6.0, places=9)
        self.assertAlmostEqual(_brute_force(requests, m), 20.0 + 10.0 / 6.0, places=9)

    def test_matches_brute_force_on_random_instance(self) -> None:
        rng = np.random.default_rng(7)
        n_stations, n_requests = 40, 70
        pts = rng.uniform(0, 10, size=(n_stations, 2))
        matrix = np.linalg.norm(pts[:, None] - pts[None, :], axis=2) * 120.0  # seconds, symmetric
        requests = []
        for _ in range(n_requests):
            ed = float(rng.uniform(0, 90))
            orgn = [int(x) for x in rng.choice(n_stations, size=int(rng.integers(1, 4)), replace=False)]
            dest = [int(x) for x in rng.choice(n_stations, size=int(rng.integers(1, 4)), replace=False)]
            requests.append(_request(ed, ed + float(rng.uniform(10, 40)), orgn, dest))
        feats = compute_features(
            {"requests": requests, "travel_time_matrix_sec": matrix, "vehicles": []}
        )
        self.assertAlmostEqual(feats["geographic_dispersion"], _brute_force(requests, matrix.tolist()), places=9)

    def test_missing_inputs_give_nan_and_time_is_small(self) -> None:
        feats = compute_features({"requests": [_request(0.0, 30.0, [0], [1])], "vehicles": []})
        self.assertTrue(math.isnan(feats["geographic_dispersion"]))
        rng = np.random.default_rng(1)
        matrix = rng.uniform(60, 2400, size=(300, 300))
        requests = []
        for _ in range(2000):
            ed = float(rng.uniform(0, 240))
            requests.append(_request(ed, ed + 30.0, [int(x) for x in rng.integers(0, 300, 3)],
                                     [int(x) for x in rng.integers(0, 300, 3)]))
        instance = {"requests": requests, "travel_time_matrix_sec": matrix, "vehicles": []}
        started = time.time()
        first = compute_features(instance)["geographic_dispersion"]
        elapsed = time.time() - started
        self.assertTrue(math.isfinite(first))
        self.assertLess(elapsed, 20.0)  # about 0.5 s on a laptop; loose bound for slow CI
        started = time.time()
        self.assertEqual(compute_features(instance)["geographic_dispersion"], first)  # cached per instance
        self.assertLess(time.time() - started, 5.0)


if __name__ == "__main__":
    unittest.main()
