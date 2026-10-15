import math
import os
import pickle
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


class LazyDistanceMatrixTests(unittest.TestCase):
    def multigraph(self):
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(range(6))
        graph.add_edge(0, 1, length=5.5, travel_time=3.2)
        graph.add_edge(0, 1, length=2.5, travel_time=1.2)
        graph.add_edge(1, 2, length=0.0, travel_time=1.1)
        graph.add_edge(0, 2, length=2.5, travel_time=2.0)
        graph.add_edge(2, 0, length=4.0, travel_time=0.5)
        graph.add_edge(3, 4, length=7.0, travel_time=4.0)
        return graph

    def test_values_match_networkx_on_multigraph(self):
        graph = self.multigraph()
        for weight, value_type in (("length", float), ("travel_time", int)):
            matrix = cdm.LazyDistanceMatrix(graph, weight, value_type)
            for origin in graph.nodes:
                expected = nx.single_source_dijkstra_path_length(
                    graph, origin, weight=weight)
                for target in graph.nodes:
                    value = matrix.loc[origin, str(target)]
                    if target in expected:
                        self.assertAlmostEqual(
                            value, value_type(expected[target]), places=3)
                        self.assertIsInstance(value, value_type)
                    else:
                        self.assertTrue(math.isnan(value))
                        # Callers test unreachability with this string check.
                        self.assertEqual(str(value), 'nan')

    def test_unknown_nodes_raise_key_error(self):
        matrix = cdm.LazyDistanceMatrix(self.multigraph(), "length")
        with self.assertRaises(KeyError):
            matrix.loc[99, "0"]
        with self.assertRaises(KeyError):
            matrix.loc[0, "99"]
        self.assertIn(0, matrix)
        self.assertNotIn(99, matrix)

    def test_accepts_string_and_numpy_node_ids(self):
        import numpy as np
        matrix = cdm.LazyDistanceMatrix(self.multigraph(), "length")
        self.assertEqual(matrix.loc[np.int64(0), "2"], matrix.loc["0", 2])
        self.assertEqual(matrix.loc[0, "2.0"], matrix.loc[0, 2])

    def test_cache_is_capped(self):
        graph = synthetic_graph()
        matrix = cdm.LazyDistanceMatrix(graph, "length", max_rows=5)
        for origin in list(graph.nodes)[:20]:
            matrix.loc[origin, "0"]
        self.assertEqual(len(matrix._rows), 5)

    def test_pickles_without_the_cache(self):
        graph = synthetic_graph()
        matrix = cdm.LazyDistanceMatrix(graph, "travel_time", int)
        before = matrix.loc[0, "10"]
        restored = pickle.loads(pickle.dumps(matrix))
        self.assertEqual(len(restored._rows), 0)
        self.assertEqual(restored.loc[0, "10"], before)

    def test_whole_matrix_access_fails_loudly(self):
        matrix = cdm.LazyDistanceMatrix(self.multigraph(), "length")
        with self.assertRaisesRegex(AttributeError, "no longer materialised"):
            matrix.index


class GetDistanceMatrixTests(unittest.TestCase):
    def test_matrices_match_networkx_and_write_no_csv(self):
        graph = synthetic_graph()
        stops = pd.DataFrame({'osmid_walk': list(graph.nodes())[:5]})

        with tempfile.TemporaryDirectory() as save_dir:
            walk, travel_time, distance, unreachable = cdm._get_distance_matrix(
                graph.copy(), graph.copy(), stops, save_dir, 'synthetic')
            self.assertEqual(os.listdir(save_dir), [])

        expected = dict(nx.single_source_dijkstra_path_length(
            graph, 0, weight='length'))
        for node, length in expected.items():
            self.assertAlmostEqual(walk.loc[0, str(node)], float(length), places=2)
            self.assertAlmostEqual(distance.loc[0, str(node)], float(length), places=2)

        expected_tt = dict(nx.single_source_dijkstra_path_length(
            graph, 0, weight='travel_time'))
        for node, seconds in expected_tt.items():
            self.assertEqual(travel_time.loc[0, str(node)], int(seconds))

        self.assertEqual(unreachable, [])

    def test_update_walk_matrix_answers_the_new_stops(self):
        graph = synthetic_graph()
        extra = list(graph.nodes())[5:8]
        with tempfile.TemporaryDirectory() as save_dir:
            updated = cdm._update_distance_matrix_walk(
                graph.copy(), extra, save_dir, 'synthetic')

        for node in extra:
            self.assertIn(node, updated)
            expected = dict(nx.single_source_dijkstra_path_length(
                graph, node, weight='length'))
            for target, length in expected.items():
                self.assertAlmostEqual(
                    updated.loc[node, str(target)], float(length), places=2)


if __name__ == '__main__':
    unittest.main()
