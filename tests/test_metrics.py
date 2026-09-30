import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import pandas as pd

from REQreate.dynamism import dynamism, dynamism2
from REQreate.geographic_dispersion import geographic_dispersion
from REQreate.urgency import urgency


class MetricTests(unittest.TestCase):
    def test_dynamism_is_zero_for_requests_at_the_same_time(self):
        self.assertEqual(dynamism([0, 0, 0], 0, 12), 0)

        requests = pd.DataFrame({"time_stamp": [0, 0, 0]})
        self.assertEqual(dynamism2(requests, 0, 12), 0)

    def test_dynamism_is_one_for_evenly_spaced_requests(self):
        timestamps = [8, 0, 4]
        self.assertEqual(dynamism(timestamps, 0, 12), 1)

        requests = pd.DataFrame({"time_stamp": timestamps})
        self.assertEqual(dynamism2(requests, 0, 12), 1)

    def test_urgency_uses_positive_timestamps_and_population_statistics(self):
        requests = pd.DataFrame(
            {
                "time_stamp": [10, 20, 0],
                "latest_departure": [10, 24, 1000],
            }
        )

        self.assertEqual(urgency(requests), (2, 2))

    def test_geographic_dispersion_on_a_synthetic_graph(self):
        graph = nx.MultiDiGraph(crs="epsg:4326")
        graph.add_node(0, x=0.0, y=0.0)
        graph.add_node(1, x=0.01, y=0.0)
        graph.add_node(2, x=0.02, y=0.0)
        graph.add_edge(0, 1, length=100, travel_time=10)
        graph.add_edge(1, 0, length=100, travel_time=10)
        graph.add_edge(1, 2, length=200, travel_time=20)
        graph.add_edge(2, 1, length=200, travel_time=20)

        class SyntheticNetwork:
            place_name = "synthetic"
            G_drive = graph

            def _return_estimated_distance_drive(self, origin, destination):
                return nx.dijkstra_path_length(
                    self.G_drive, origin, destination, weight="length"
                )

        requests = pd.DataFrame(
            {
                "earliest_departure": [1000, 1100],
                "direct_travel_time": [10, 20],
                "originnode_drive": [0, 1],
                "destinationnode_drive": [1, 2],
            }
        )
        instance = SimpleNamespace(network=SyntheticNetwork())

        with tempfile.TemporaryDirectory() as directory:
            csv_directory = os.path.join(directory, "synthetic", "csv_format")
            os.makedirs(csv_directory)
            requests.to_csv(os.path.join(csv_directory, "requests.csv"), index=False)

            with patch("os.getcwd", return_value=directory):
                result = geographic_dispersion(instance, "DARP", "requests.csv")

        self.assertEqual(result, 77.5)


if __name__ == "__main__":
    unittest.main()
