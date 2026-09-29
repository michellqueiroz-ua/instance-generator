import os
import random
import tempfile
import unittest

import networkx as nx
import pandas as pd

from REQreate import compute_distance_matrix as cdm


def synthetic_graph(n=60, seed=5):
    random.seed(seed)
    graph = nx.MultiDiGraph(crs='epsg:4326')
    base = nx.connected_watts_strogatz_graph(n, 4, 0.2, seed=seed)
    for node in base.nodes():
        graph.add_node(node, x=6.0 + node * 1e-4, y=50.0 + node * 1e-4)
    for u, v in base.edges():
        length = float(random.randint(50, 900))
        for a, b in ((u, v), (v, u)):
            graph.add_edge(a, b, length=length, travel_time=length / 13.9)
    return graph


class RayShimTests(unittest.TestCase):
    def test_dummy_is_only_installed_when_the_import_failed(self):
        # The shim used to replace the real module unconditionally, which
        # silently disabled the `parallel` extra.
        if cdm.RAY_AVAILABLE:
            self.assertEqual(getattr(cdm.ray, '__name__', None), 'ray')
            self.assertNotIsInstance(cdm.ray, cdm.DummyRay)
        else:
            self.assertIsInstance(cdm.ray, cdm.DummyRay)

    def test_remote_helpers_exist_exactly_when_ray_does(self):
        self.assertEqual(cdm.RAY_AVAILABLE, hasattr(cdm, 'shortest_path_nx_ss'))


class DistanceMatrixTests(unittest.TestCase):
    def test_matrices_match_networkx_on_a_small_graph(self):
        graph = synthetic_graph()
        stops = pd.DataFrame({'osmid_walk': list(graph.nodes())[:5]})

        with tempfile.TemporaryDirectory() as save_dir:
            walk, travel_time, distance, unreachable = cdm._get_distance_matrix(
                graph.copy(), graph.copy(), stops, save_dir, 'synthetic')

            self.assertTrue(os.path.isfile(
                os.path.join(save_dir, 'csv', 'synthetic.dist.walk.csv')))

        expected = dict(nx.single_source_dijkstra_path_length(
            graph, 0, weight='length'))
        for node, length in expected.items():
            self.assertEqual(walk.loc[0, str(node)], float(length))
            self.assertEqual(distance.loc[0, str(node)], float(length))

        expected_tt = dict(nx.single_source_dijkstra_path_length(
            graph, 0, weight='travel_time'))
        for node, seconds in expected_tt.items():
            self.assertEqual(travel_time.loc[0, str(node)], int(seconds))

        self.assertEqual(unreachable, [])
        self.assertEqual(set(map(str, distance.dtypes)), {'float64'})
        self.assertEqual(set(map(str, travel_time.dtypes)), {'int64'})

    def test_update_walk_matrix_adds_the_new_stops(self):
        graph = synthetic_graph()
        stops = pd.DataFrame({'osmid_walk': list(graph.nodes())[:5]})

        with tempfile.TemporaryDirectory() as save_dir:
            cdm._get_distance_matrix(
                graph.copy(), graph.copy(), stops, save_dir, 'synthetic')

            extra = list(graph.nodes())[5:8]
            updated = cdm._update_distance_matrix_walk(
                graph.copy(), extra, save_dir, 'synthetic')

        for node in extra:
            self.assertIn(node, updated.index)
            expected = dict(nx.single_source_dijkstra_path_length(
                graph, node, weight='length'))
            for target, length in expected.items():
                self.assertEqual(updated.loc[node, str(target)], float(length))


if __name__ == '__main__':
    unittest.main()
