import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
for path in (THIS_DIR, THIS_DIR / "isa_set"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import features_batch

MATRIX = np.array([[0, 600, 700], [620, 0, 900], [1500, 880, 0]], dtype=np.int32)


def request(origin, destination, start_min, width_min):
    return {
        "originx": 12.5, "originy": 55.7, "destinationx": 12.6, "destinationy": 55.8,
        "stops_orgn": [origin], "stops_dest": [destination], "time_stamp": start_min * 60,
        "earliest_departure": start_min * 60, "latest_departure": (start_min + width_min) * 60,
        "direct_travel_time": 60.0,
    }


class FeaturesBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        net = self.work / "Net"
        (net / "json_format").mkdir(parents=True)
        (net / "travel_time_matrix").mkdir()
        np.save(net / "travel_time_matrix" / "station_travel_time_matrix.npy", MATRIX)
        requests = {str(i): request(i % 3, (i + 1) % 3, 10 * i, 30) for i in range(6)}
        fleets = {}
        for n in (1, 2):
            (net / "json_format" / f"ISA_b01_i00{n}_1.json").write_text(json.dumps({"requests": requests}))
            for k, share in ((1, 0.0), (2, 50.0)):
                fleets[f"ISA_b01_i00{n}_1_fleet{k}"] = {
                    "instance": f"ISA_b01_i00{n}_1", "fleet_percent_of_requests": 50.0 * k,
                    "composition": [{"capacity": 4, "percent": 100 - share}, {"capacity": 8, "percent": share}],
                }
        fleets["ISA_b01_i003_1_fleet1"] = dict(fleets["ISA_b01_i001_1_fleet1"], instance="ISA_b01_i003_1")  # no json
        (self.work / "design").mkdir()
        (self.work / "design" / "batch_01_fleets.json").write_text(json.dumps(fleets))

    def test_one_row_per_fleet_with_fleet_features_and_matrix(self):
        status = features_batch.main(["--batch", "1", "--work", str(self.work), "--network", "Net"])
        self.assertEqual(status, 0)
        with open(self.work / "results" / "features_batch_01.csv", newline="") as f:
            rows = {r["instance"]: r for r in csv.DictReader(f)}
        self.assertEqual(sorted(rows), [f"ISA_b01_i00{n}_1_fleet{k}" for n in (1, 2) for k in (1, 2)])
        self.assertEqual(float(rows["ISA_b01_i001_1_fleet1"]["fleet_size"]), 3.0)   # 50% of 6 requests
        self.assertEqual(float(rows["ISA_b01_i001_1_fleet2"]["fleet_size"]), 6.0)
        # width 30 min over the matrix times (10, 15, 25 min), not the 1 min direct time
        self.assertLess(float(rows["ISA_b01_i001_1_fleet1"]["tw_tightness_factor"]), 10)

    def test_range_selects_instances(self):
        features_batch.main(["--batch", "1", "--work", str(self.work), "--network", "Net", "--first", "2", "--last", "2"])
        with open(self.work / "results" / "features_batch_01.csv", newline="") as f:
            self.assertEqual({r["instance"].split("_fleet")[0] for r in csv.DictReader(f)}, {"ISA_b01_i002_1"})


if __name__ == "__main__":
    unittest.main()
