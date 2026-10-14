import os
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd

from REQreate import rank_model
from REQreate.retrieve_POIs import zone_probability_matrix, zone_rank_matrix


def reference_ranks(distances, pois):
    """The original triple loop from retrieve_POIs.calc_rank_between_zones."""
    n = len(pois)
    ranks = np.zeros((n, n), dtype=int)
    for u in range(n):
        for v in range(n):
            duv = distances[u][v]
            for w in range(n):
                if w != u and w != v:
                    if distances[u][w] < duv:
                        ranks[u][v] += pois[w]
    return ranks


def reference_probabilities(ranks, alpha):
    """The original triple loop from calc_probability_travel_between_zones."""
    alpha = alpha * -1
    n = len(ranks)
    probabilities = np.zeros((n, n))
    for u in range(n):
        for v in range(n):
            if u != v:
                p1 = ranks[u][v]
                r2 = 0
                for w in range(n):
                    p2 = ranks[w][v]
                    if p2 != 0:
                        r2 += p2 ** alpha
                probabilities[u][v] = p1 ** alpha / r2 if p1 != 0 else 0
    return probabilities


class VectorisedMatchesOriginalTests(unittest.TestCase):
    def random_case(self, seed, n=12):
        rng = np.random.default_rng(seed)
        distances = rng.uniform(100, 5000, (n, n))
        np.fill_diagonal(distances, 0)
        distances[rng.random((n, n)) < 0.1] = np.nan  # unreachable pairs
        pois = rng.integers(0, 30, n)
        pois[rng.random(n) < 0.2] = 0  # zones without POIs
        return distances, pois

    def test_ranks_match_the_original_loop(self):
        for seed in range(5):
            distances, pois = self.random_case(seed)
            np.testing.assert_array_equal(
                zone_rank_matrix(distances, pois), reference_ranks(distances, pois))

    def test_probabilities_match_the_original_loop(self):
        for seed in range(5):
            distances, pois = self.random_case(seed)
            ranks = reference_ranks(distances, pois)
            np.testing.assert_allclose(
                zone_probability_matrix(ranks, 0.86),
                reference_probabilities(ranks, 0.86), rtol=1e-12, atol=0)


class FakeMatrix:
    def __init__(self, graph):
        self.lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight='length'))
        self.loc = self

    def __getitem__(self, key):
        origin, destination = key
        return float(self.lengths[int(origin)].get(int(destination), np.nan))


class FakeNetwork:
    def __init__(self):
        graph = nx.MultiDiGraph(crs='epsg:4326')
        for i in range(9):
            graph.add_node(i, x=6.0 + (i % 3) * 0.01, y=50.0 + (i // 3) * 0.01)
        for i in range(9):
            for j in (i + 1, i + 3):
                if j < 9 and not (j == i + 1 and i % 3 == 2):
                    graph.add_edge(i, j, length=700.0)
                    graph.add_edge(j, i, length=700.0)
        self.G_drive = graph
        self.shortest_dist_drive = FakeMatrix(graph)
        self.zones = pd.DataFrame({
            'center_x': [graph.nodes[i]['x'] for i in range(9)],
            'center_y': [graph.nodes[i]['y'] for i in range(9)],
        })


def fake_density(network, pois):
    network.zones['number_pois'] = [3, 0, 5, 1, 2, 0, 4, 1, 6]
    total = network.zones['number_pois'].sum()
    network.zones['density_pois'] = network.zones['number_pois'] / total * 100


class OnDemandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    @patch.object(rank_model, 'attribute_density_zones', side_effect=fake_density)
    @patch.object(rank_model, 'get_POIs_matrix_csv', return_value=None)
    def test_poi_density_is_downloaded_once(self, download, _density):
        network = FakeNetwork()
        self.assertTrue(rank_model.add_poi_density(network, 'Testville'))
        self.assertFalse(rank_model.add_poi_density(network, 'Testville'))
        self.assertEqual(download.call_count, 1)

    @patch.object(rank_model, 'attribute_density_zones', side_effect=fake_density)
    @patch.object(rank_model, 'get_POIs_matrix_csv', return_value=None)
    def test_zone_probabilities_fetch_pois_store_and_write_csv(self, download, _density):
        network = FakeNetwork()
        self.assertTrue(rank_model.add_zone_probabilities(network, 'Testville'))
        self.assertEqual(download.call_count, 1)

        probabilities = network.zone_probabilities
        self.assertEqual(probabilities.shape, (9, 9))
        self.assertTrue((np.diag(probabilities.to_numpy()) == 0).all())
        for name in ('zone_ranks', 'zone_probabilities'):
            self.assertTrue(os.path.isfile(
                os.path.join('Testville', 'csv', f'Testville.{name}.csv')))

        # Second call reuses what is stored on the network.
        self.assertFalse(rank_model.add_zone_probabilities(network, 'Testville'))
        self.assertEqual(download.call_count, 1)


if __name__ == '__main__':
    unittest.main()
