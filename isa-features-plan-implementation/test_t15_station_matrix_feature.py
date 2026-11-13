import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from compute_features_scaffold import compute_features
from reqreate_adapter import find_station_matrix, load_reqreate_instance

# Seconds between three stations; direct_travel_time on the requests is deliberately different,
# so the test shows which source tw_tightness used.
MATRIX = np.array([[0, 600, 700], [620, 0, 900], [1500, 880, 0]], dtype=np.int32)


def request(origin, destination, tw_start, tw_end):
    return {
        "originx": 12.5, "originy": 55.7, "destinationx": 12.6, "destinationy": 55.8,
        "stops_orgn": [origin], "stops_dest": [destination],
        "earliest_departure": tw_start * 60, "latest_departure": tw_end * 60, "direct_travel_time": 60.0,
    }


class StationMatrixFeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.location = Path(self.tmp.name) / "Place"
        (self.location / "json_format").mkdir(parents=True)
        (self.location / "travel_time_matrix").mkdir()
        np.save(self.location / "travel_time_matrix" / "station_travel_time_matrix.npy", MATRIX)
        self.json_path = self.location / "json_format" / "inst.json"
        requests = {
            "0": request(0, 1, 0, 30),   # 600 s = 10 min
            "1": request(1, 2, 0, 20),   # 900 s = 15 min
            "2": request(2, 0, 10, 60),  # 1500 s = 25 min
        }
        self.json_path.write_text(json.dumps({"requests": requests}))

    def test_matrix_is_found_from_the_instance_path(self):
        self.assertTrue(find_station_matrix(self.json_path).endswith("station_travel_time_matrix.npy"))

    def test_tw_tightness_uses_the_shared_matrix(self):
        instance = load_reqreate_instance(self.json_path)
        self.assertIn("travel_time_matrix_sec", instance)
        result = compute_features(instance)["tw_tightness_factor"]
        self.assertAlmostEqual(result, np.mean([30 / 10, 20 / 15, 50 / 25]))

    def test_without_a_matrix_it_falls_back_to_direct_time(self):
        (self.location / "travel_time_matrix" / "station_travel_time_matrix.npy").unlink()
        instance = load_reqreate_instance(self.json_path)
        self.assertNotIn("travel_time_matrix_sec", instance)

    def test_matrix_of_another_network_is_rejected(self):
        stations = self.location / "csv"
        stations.mkdir()
        (stations / "Place.stations.csv").write_text(
            "lat,lon,type,osmid_drive,osmid_walk\n55.7,12.5,0,1,1\n55.8,12.6,0,2,2\n"
        )
        with self.assertRaises(ValueError):
            load_reqreate_instance(self.json_path)


if __name__ == "__main__":
    unittest.main()
