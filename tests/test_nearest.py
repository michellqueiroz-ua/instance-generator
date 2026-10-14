import os
import random
import unittest
from unittest.mock import patch

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from osmnx.distance import great_circle
from shapely.geometry import Point, Polygon

from REQreate import overpass_config
from REQreate.nearest import centroid_coordinates, nearest_edge_endpoints, nearest_nodes
from REQreate.retrieve_bus_stations import bus_station_rows, drop_repeated_stops


def grid_graph(side=15, seed=3):
    rng = random.Random(seed)
    base = nx.grid_2d_graph(side, side)
    graph = nx.MultiDiGraph(crs='epsg:4326')
    ids = {}
    for (i, j) in base.nodes:
        ids[(i, j)] = 1000 + i * side + j
        graph.add_node(ids[(i, j)], x=6.0 + j * 0.002 + rng.uniform(-3e-4, 3e-4),
                       y=50.0 + i * 0.002 + rng.uniform(-3e-4, 3e-4))
    for a, b in base.edges:
        for u, v in ((a, b), (b, a)):
            graph.add_edge(ids[u], ids[v], length=200.0)
    return graph


def random_points(n, seed=7):
    rng = random.Random(seed)
    lats = [50.0 + rng.uniform(0, 0.028) for _ in range(n)]
    lons = [6.0 + rng.uniform(0, 0.028) for _ in range(n)]
    return lats, lons


class BatchedLookupTests(unittest.TestCase):
    def test_edge_endpoints_match_one_by_one(self):
        graph = grid_graph()
        lats, lons = random_points(40)
        expected = []
        for lat, lon in zip(lats, lons):
            u, v, _ = ox.nearest_edges(graph, lon, lat)
            expected.append(min((u, v), key=lambda n: great_circle(
                lat, lon, graph.nodes[n]['y'], graph.nodes[n]['x'])))
        self.assertEqual(nearest_edge_endpoints(graph, lats, lons), expected)

    def test_nodes_match_one_by_one(self):
        graph = grid_graph()
        lats, lons = random_points(40)
        expected = [ox.nearest_nodes(graph, lon, lat) for lat, lon in zip(lats, lons)]
        self.assertEqual(nearest_nodes(graph, lats, lons), expected)

    def test_empty_input(self):
        graph = grid_graph()
        self.assertEqual(nearest_edge_endpoints(graph, [], []), [])
        self.assertEqual(nearest_nodes(graph, [], []), [])

    def test_centroids_of_points_and_polygons(self):
        features = gpd.GeoDataFrame(geometry=[
            Point(6.01, 50.01),
            Polygon([(6.0, 50.0), (6.002, 50.0), (6.002, 50.002), (6.0, 50.002)]),
        ], crs='epsg:4326')
        lats, lons = centroid_coordinates(features)
        self.assertAlmostEqual(lats[0], 50.01)
        self.assertAlmostEqual(lons[1], 6.001)


class BusStationTests(unittest.TestCase):
    def test_rows_skip_non_bus_stops_and_keep_order(self):
        graph = grid_graph()
        lats, lons = random_points(5)
        features = gpd.GeoDataFrame(
            {'highway': ['bus_stop', 'bus_stop', 'platform', 'bus_stop', 'bus_stop']},
            geometry=[Point(lon, lat) for lat, lon in zip(lats, lons)], crs='epsg:4326')
        rows = bus_station_rows(graph, graph, features)
        self.assertEqual(len(rows), 4)
        self.assertEqual([r['lat'] for r in rows], [lats[i] for i in (0, 1, 3, 4)])
        self.assertTrue(all(r['type'] == 0 for r in rows))

    def test_dedupe_matches_the_original_pair_loop(self):
        rng = random.Random(1)
        stops = pd.DataFrame({
            'osmid_walk': [rng.randint(1, 50) for _ in range(60)],
            'osmid_drive': [rng.randint(1, 20) for _ in range(60)],
        })
        expected = stops.copy()
        drop_index_list = []
        for index1, stop1 in expected.iterrows():
            if index1 not in drop_index_list:
                for index2, stop2 in expected.iterrows():
                    if index2 not in drop_index_list and index1 != index2:
                        if stop1['osmid_drive'] == stop2['osmid_drive']:
                            drop_index_list.append(index2)
        for index_to_drop in drop_index_list:
            expected = expected.drop(index_to_drop)

        pd.testing.assert_frame_equal(drop_repeated_stops(stops), expected)


class CacheFolderTests(unittest.TestCase):
    def test_override(self):
        with patch.dict(os.environ, {'REQREATE_CACHE_DIR': r'D:\osm'}):
            self.assertEqual(overpass_config.get_cache_folder(), r'D:\osm')

    def test_default_is_per_user_not_cwd(self):
        with patch.dict(os.environ, {'REQREATE_CACHE_DIR': ''}):
            folder = overpass_config.get_cache_folder()
        self.assertTrue(os.path.isabs(folder))
        self.assertTrue(folder.endswith(os.path.join('reqreate', 'osm-cache')))

    def test_configure_sets_osmnx_cache_folder(self):
        previous = ox.settings.cache_folder
        self.addCleanup(setattr, ox.settings, 'cache_folder', previous)
        with patch.dict(os.environ, {'REQREATE_CACHE_DIR': r'D:\osm'}):
            overpass_config.configure_overpass()
        self.assertEqual(ox.settings.cache_folder, r'D:\osm')
        self.assertTrue(ox.settings.use_cache)


if __name__ == '__main__':
    unittest.main()
