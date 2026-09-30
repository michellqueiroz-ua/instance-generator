import random
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
from geopy.distance import distance
from shapely.geometry import Polygon

from REQreate.network_class import Network
from REQreate.request_distribution_class import RequestDistributionTime
from REQreate.spatial_distribution_class import SpatialDistribution


class RequestDistributionTimeTests(unittest.TestCase):
    def test_uniform_samples_match_configured_distribution(self):
        np.random.seed(2718)
        distribution = RequestDistributionTime(2, 14, 6000, "uniform")

        distribution.sample_times()

        self.assertEqual(len(distribution.demand), 6000)
        self.assertGreaterEqual(distribution.demand.min(), 2)
        self.assertLess(distribution.demand.max(), 14)
        self.assertAlmostEqual(distribution.demand.mean(), 8, delta=0.12)
        self.assertAlmostEqual(distribution.demand.std(), 12 / np.sqrt(12), delta=0.12)

    def test_normal_samples_match_configured_distribution(self):
        np.random.seed(3141)
        distribution = RequestDistributionTime(25, 4, 6000, "normal")

        distribution.sample_times()

        self.assertEqual(len(distribution.demand), 6000)
        self.assertAlmostEqual(distribution.demand.mean(), 25, delta=0.12)
        self.assertAlmostEqual(distribution.demand.std(), 4, delta=0.12)

    def test_fixed_seed_reproduces_sampled_times(self):
        distribution = RequestDistributionTime(0, 1, 100, "normal")

        np.random.seed(42)
        distribution.sample_times()
        first_sample = distribution.demand.copy()
        np.random.seed(42)
        distribution.sample_times()

        np.testing.assert_array_equal(distribution.demand, first_sample)


class SpatialDistributionTests(unittest.TestCase):
    def test_randomly_sampled_zone_ids_are_in_range_and_reproducible(self):
        distribution = SpatialDistribution(4000, 3000, None)

        np.random.seed(123)
        distribution.randomly_sample_origin_zones(7)
        distribution.randomly_sample_destination_zones(5)
        origins = distribution.origin_zones.copy()
        destinations = distribution.destination_zones.copy()

        self.assertEqual(len(origins), 4000)
        self.assertEqual(len(destinations), 3000)
        self.assertTrue(np.all((origins >= 0) & (origins < 7)))
        self.assertTrue(np.all((destinations >= 0) & (destinations < 5)))

        np.random.seed(123)
        distribution.randomly_sample_origin_zones(7)
        distribution.randomly_sample_destination_zones(5)
        np.testing.assert_array_equal(distribution.origin_zones, origins)
        np.testing.assert_array_equal(distribution.destination_zones, destinations)

    def test_default_zone_lists_are_not_shared_between_instances(self):
        first = SpatialDistribution(0, 0, None)
        second = SpatialDistribution(0, 0, None)

        first.origin_zones.append(3)
        first.destination_zones.append(4)

        self.assertEqual(second.origin_zones, [])
        self.assertEqual(second.destination_zones, [])

    def test_default_zone_coordinates_stay_inside_the_zone(self):
        network = Network.__new__(Network)
        network.G_drive = nx.MultiDiGraph()
        network.G_drive.add_node(0, x=0, y=0)
        zone = Polygon([(-0.001, -0.001), (0.001, -0.001),
                        (0.001, 0.001), (-0.001, 0.001)])

        with patch("REQreate.network_class.ox.nearest_nodes", return_value=0):
            point = network._get_random_coord(zone, 91)

        self.assertTrue(zone.contains(point))

    def test_circle_and_radius_samples_stay_within_their_boundaries(self):
        network = Network.__new__(Network)
        center = (50.0, 6.0)
        radius = 750

        np.random.seed(19)
        circle_point = network._get_random_coord_circle(
            radius, center[0], center[1], 19)
        circle_distance = distance(center, (circle_point.y, circle_point.x)).m
        self.assertLessEqual(circle_distance, radius)

        enclosing_zone = Polygon([
            (5.95, 49.95), (6.05, 49.95), (6.05, 50.05), (5.95, 50.05)
        ])
        random.seed(19)
        radius_point = network._get_random_coord_radius(
            center[0], center[1], radius, enclosing_zone, 19)

        self.assertTrue(enclosing_zone.contains(radius_point))
        self.assertLessEqual(
            distance(center, (radius_point.y, radius_point.x)).m, radius + 1)


if __name__ == "__main__":
    unittest.main()
