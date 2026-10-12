import unittest

import networkx as nx
import osmnx as ox
import pandas as pd

from REQreate.network_class import Network


class GreatCircleCompatibilityTests(unittest.TestCase):
    def test_osmnx_exposes_renamed_distance_function(self):
        self.assertTrue(hasattr(ox.distance, "great_circle"))
        self.assertFalse(hasattr(ox.distance, "great_circle_vec"))

    def test_adding_school_and_stop_selects_nodes(self):
        graph = nx.MultiDiGraph(crs="epsg:4326")
        graph.add_node(1, x=6.0, y=50.0)
        graph.add_node(2, x=6.01, y=50.01)
        graph.add_node(3, x=6.02, y=50.0)
        graph.add_edge(1, 2, length=100)
        graph.add_edge(2, 3, length=100)

        network = Network(
            "synthetic",
            graph.copy(),
            graph.copy(),
            None,
            pd.DataFrame(),
        )
        network.schools = pd.DataFrame(columns=["school_name"])

        network.add_new_school("Synthetic School", 6.004, 50.001)
        network.add_new_stop("bus", 6.004, 50.001)

        for frame, columns in (
            (network.schools, ("osmid_walk", "osmid_drive")),
            (network.bus_stations, ("osmid_walk", "osmid_drive")),
        ):
            for column in columns:
                self.assertIn(frame.iloc[0][column], graph.nodes)


if __name__ == "__main__":
    unittest.main()
