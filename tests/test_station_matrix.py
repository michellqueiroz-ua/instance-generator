import unittest

import networkx as nx
import numpy as np

from REQreate.compute_distance_matrix import LazyDistanceMatrix, travel_times_between


def road_graph(n=60, seed=3):
    rng = np.random.default_rng(seed)
    G = nx.MultiDiGraph()
    for i in range(n):
        G.add_edge(i, (i + 1) % n, travel_time=float(rng.uniform(5, 90)))  # a ring keeps it strongly connected
    for _ in range(3 * n):
        u, v = rng.integers(0, n, 2)
        if u != v:
            G.add_edge(int(u), int(v), travel_time=float(rng.uniform(5, 90)))
    return G


class TravelTimesBetweenTests(unittest.TestCase):
    def test_same_values_as_lazy_matrix(self):
        G = road_graph()
        nodes = [3, 17, 17, 42, 0, 59]  # a repeated node (two stations on one node)
        lazy = LazyDistanceMatrix(G, "travel_time", int)
        expected = np.array([[lazy.get(u, v) for v in nodes] for u in nodes])
        for chunk in (1, 2, 64):
            np.testing.assert_array_equal(travel_times_between(G, nodes, chunk=chunk), expected)

    def test_unreachable_is_minus_one(self):
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, travel_time=10.0)
        G.add_node(2)
        result = travel_times_between(G, [0, 1, 2])
        self.assertEqual(result[0, 1], 10)
        self.assertEqual(result[1, 0], -1)
        self.assertEqual(result[0, 2], -1)

    def test_progress_is_reported(self):
        calls = []
        travel_times_between(road_graph(10), list(range(10)), chunk=4, progress=lambda a, b: calls.append((a, b)))
        self.assertEqual(calls[-1], (10, 10))


if __name__ == "__main__":
    unittest.main()
