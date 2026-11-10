import unittest

try:
    from REQreate.passenger_requests import zones_near_point
except ImportError:  # the generator's dependencies (osmnx, shapely, ...) are not installed
    zones_near_point = None


def grid(columns=25, rows=3, size=0.04):
    """Row-major grid of rectangles like the network's zone grid: label = row * columns + column."""
    return [(row * columns + col, (col * size, row * size, (col + 1) * size, (row + 1) * size))
            for row in range(rows) for col in range(columns)]


@unittest.skipIf(zones_near_point is None, "generator dependencies missing")
class ZonesNearPointTests(unittest.TestCase):
    def test_a_point_deep_inside_a_zone_only_finds_that_zone(self):
        self.assertEqual(zones_near_point(grid(), 0.06, 0.06, 100.0), [26])

    def test_a_point_near_the_north_border_also_finds_the_zone_to_the_north(self):
        # zone 26 is column 1 of row 1; the zone to the north is zone 51 (+25), which the old index offsets skipped
        found = zones_near_point(grid(), 0.06, 0.079, 800.0)
        self.assertIn(26, found)
        self.assertIn(51, found)

    def test_a_point_near_a_corner_finds_the_diagonal_zones(self):
        found = zones_near_point(grid(), 0.08, 0.04, 800.0)
        self.assertEqual(sorted(found), [1, 2, 26, 27])

    def test_zones_beyond_walking_range_are_not_returned(self):
        found = zones_near_point(grid(), 0.06, 0.06, 800.0)
        self.assertNotIn(51, found)
        self.assertNotIn(1, found)


if __name__ == "__main__":
    unittest.main()
