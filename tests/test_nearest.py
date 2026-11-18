import unittest

import networkx as nx
import numpy as np
import osmnx as ox

from REQreate.nearest import nearest_node


def grid_graph(n=20):
    rng = np.random.default_rng(1)
    G = nx.MultiDiGraph(crs="epsg:4326")
    for i in range(n * n):
        G.add_node(i, x=12.3 + rng.uniform(0, 0.3), y=55.5 + rng.uniform(0, 0.2))
    return G


class NearestNodeTests(unittest.TestCase):
    def test_same_answer_as_osmnx(self):
        G = grid_graph()
        rng = np.random.default_rng(2)
        for _ in range(30):
            lon, lat = 12.2 + rng.uniform(0, 0.5), 55.4 + rng.uniform(0, 0.4)
            self.assertEqual(nearest_node(G, lon, lat), ox.nearest_nodes(G, lon, lat))

    def test_index_is_per_graph(self):
        a, b = grid_graph(5), grid_graph(6)
        self.assertIn(nearest_node(a, 12.4, 55.6), a.nodes)
        self.assertIn(nearest_node(b, 12.4, 55.6), b.nodes)


if __name__ == "__main__":
    unittest.main()
