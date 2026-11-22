import logging
import math
import sys
import unittest
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from compute_features_scaffold import compute_features


class TestT09SpatialNetworkFeatures(unittest.TestCase):
    def test_csv_style_inputs_compute_all_three_features(self) -> None:
        instance = {
            "service_area_km2": 0.1,
            "stops": [
                {"station_index": 0, "lon": 5.6800, "lat": 50.8500, "type": 0},
                {"station_index": 1, "lon": 5.6810, "lat": 50.8510, "type": 0},
                {"station_index": 2, "lon": 5.6820, "lat": 50.8520, "type": 0},
                {"station_index": 3, "lon": 5.6830, "lat": 50.8530, "type": 0},
            ],
            "requests": [
                {
                    "origin": {"x": 5.6805, "y": 50.8505},
                    "destination": {"x": 5.6815, "y": 50.8515},
                    "walk_speed": 1.0,
                    "stops_orgn": "[0, 1]",
                    "stops_orgn_walking_distance": "[100, 700]",
                    "stops_dest": "[1]",
                    "stops_dest_walking_distance": "[300]",
                },
                {
                    "origin": {"x": 5.6825, "y": 50.8525},
                    "destination": {"x": 5.6835, "y": 50.8535},
                    # Missing walk_speed -> fallback speed should be used.
                    "stops_orgn": "[2, 3]",
                    "stops_orgn_walking_distance": "[820, 900]",
                    "stops_dest": "[0, 3]",
                    "stops_dest_walking_distance": "[500, 1200]",
                },
            ],
            # A complete stop matrix must not be read as road degree.
            "travel_time_matrix_sec": [
                [1, 100, 110, 120],
                [100, 1, 130, 140],
                [110, 130, 1, 150],
                [120, 140, 150, 1],
            ],
            "vehicles": [],
        }

        with self.assertLogs("compute_features_scaffold", level=logging.INFO) as captured:
            feats = compute_features(instance)

        # station_density = 4 / 0.1 = 40
        self.assertAlmostEqual(feats["station_density"], 40.0, places=9)

        # Endpoint candidate counts = [1,1,1,1] -> mean 1.0
        self.assertAlmostEqual(feats["bsa_flexibility"], 1.0, places=9)

        # No road degree and no road_network: NaN, not n_stops - 1.
        self.assertTrue(math.isnan(feats["avg_station_degree"]))

        logs = "\n".join(captured.output)
        self.assertIn("bsa_flexibility walk-speed fallback used", logs)
        self.assertIn("bsa_flexibility=1.0 indicates one accessible stop", logs)
        self.assertIn("avg_station_degree: no road degree", logs)

    def test_avg_station_degree_from_road_network_edges(self) -> None:
        instance = {
            "service_area_km2": 5.0,
            "stops": [
                {"id": "a", "x": 5.68, "y": 50.85},
                {"id": "b", "x": 5.69, "y": 50.86},
                {"id": "c", "x": 5.70, "y": 50.87},
            ],
            "road_network": {
                "edges": [
                    {"u": "a", "v": "b", "travel_time_min": 2.0},
                    {"u": "b", "v": "c", "travel_time_min": 2.0},
                    {"u": "c", "v": "a", "travel_time_min": 2.0},
                ]
            },
            "requests": [],
            "vehicles": [],
        }

        feats = compute_features(instance)

        # Undirected neighbor degrees are [2,2,2].
        self.assertAlmostEqual(feats["avg_station_degree"], 2.0, places=9)

    def test_degenerate_inputs_return_nan_or_safe_numeric(self) -> None:
        instance = {
            "requests": ["bad", {"stops_orgn": "not-a-list"}],
            "vehicles": [],
        }

        feats = compute_features(instance)

        self.assertTrue(math.isnan(feats["station_density"]))
        self.assertTrue(math.isnan(feats["bsa_flexibility"]))
        self.assertTrue(math.isnan(feats["avg_station_degree"]))


class TestStopRoadDegrees(unittest.TestCase):
    def test_degree_counts_distinct_road_neighbours_both_directions(self) -> None:
        import networkx as nx
        from reqreate_adapter import add_stop_road_degrees

        graph = nx.MultiDiGraph()
        # Stop node 1: two-way street to 2 (parallel edges), one-way in from 3,
        # one-way out to 4 -> neighbours {2, 3, 4}.
        graph.add_edge(1, 2)
        graph.add_edge(1, 2)
        graph.add_edge(2, 1)
        graph.add_edge(3, 1)
        graph.add_edge(1, 4)
        stops = [{"osmid_drive": 1}, {"osmid_drive": 99}]

        self.assertEqual(add_stop_road_degrees(stops, graph), 1)
        self.assertEqual(stops[0]["degree"], 3)
        self.assertNotIn("degree", stops[1])

        feats = compute_features({"stops": stops, "requests": [], "vehicles": []})
        self.assertAlmostEqual(feats["avg_station_degree"], 3.0, places=9)


if __name__ == "__main__":
    unittest.main()
