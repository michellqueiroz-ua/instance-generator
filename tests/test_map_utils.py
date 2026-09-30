import unittest

import folium
from folium.plugins import HeatMap
import pandas as pd

from REQreate.webapp.attribute_library import (
    build_attributes_list,
    get_attributes_for_problem,
    get_required_attributes,
)
from REQreate.webapp.map_utils import create_heatmap, create_request_map


class MapUtilsTests(unittest.TestCase):
    def setUp(self):
        self.requests = pd.DataFrame([
            {"origin_lat": 50.0, "origin_lon": 5.0, "dest_lat": 51.0, "dest_lon": 6.0},
            {"origin_lat": 52.0, "origin_lon": 7.0, "dest_lat": 53.0, "dest_lon": 8.0},
        ])

    def test_request_map_layers_and_optional_markers(self):
        stations = pd.DataFrame([{"lat": 50.0, "lon": 5.0, "station_id": 1}])
        hospitals = pd.DataFrame([{"lat": 51.0, "lon": 6.0, "hospital_name": "Clinic"}])
        request_map = create_request_map(
            self.requests, bus_stations_df=stations, hospitals_df=hospitals
        )

        self.assertIsInstance(request_map, folium.Map)
        self.assertEqual(request_map.location, [51.5, 6.5])
        layers = {
            child.layer_name: child
            for child in request_map._children.values()
            if isinstance(child, folium.FeatureGroup)
        }
        self.assertEqual(
            set(layers),
            {"Origins (Green)", "Destinations (Red)", "Request Paths", "Bus Stations", "Hospitals"},
        )
        self.assertEqual(
            sum(isinstance(child, folium.PolyLine) for child in layers["Request Paths"]._children.values()),
            2,
        )

        no_lines = create_request_map(self.requests, show_lines=False)
        self.assertFalse(any(
            isinstance(child, folium.FeatureGroup) and child.layer_name == "Request Paths"
            for child in no_lines._children.values()
        ))

    def test_heatmap_layers_use_expected_coordinates(self):
        for layer, expected in (
            ("origins", [[50.0, 5.0], [52.0, 7.0]]),
            ("destinations", [[51.0, 6.0], [53.0, 8.0]]),
            ("both", [[50.0, 5.0], [52.0, 7.0], [51.0, 6.0], [53.0, 8.0]]),
        ):
            with self.subTest(layer=layer):
                heatmap = create_heatmap(self.requests, layer)
                self.assertEqual(heatmap.location, [51.0, 6.0])
                heat = next(child for child in heatmap._children.values() if isinstance(child, HeatMap))
                self.assertEqual(heat.data, expected)


class AttributeLibraryTests(unittest.TestCase):
    def test_required_attributes_and_unknown_problem(self):
        for problem in ("DARP", "ODBRP", "Patient Transport"):
            with self.subTest(problem=problem):
                available = get_attributes_for_problem(problem)
                self.assertTrue(available)
                self.assertEqual(
                    get_required_attributes(problem),
                    [name for name, info in available.items() if info["required"]],
                )
        self.assertEqual(get_attributes_for_problem("unknown"), {})
        self.assertEqual(get_required_attributes("unknown"), [])

    def test_build_attributes_list_selects_and_overrides_templates(self):
        default = get_attributes_for_problem("DARP")["time_stamp"]["template"]
        attrs = build_attributes_list(
            "DARP", ["time_stamp", "not_an_attribute"],
            {"time_stamp": {"dynamism": 40}},
        )
        self.assertEqual(len(attrs), 1)
        self.assertEqual(attrs[0]["name"], "time_stamp")
        self.assertEqual(attrs[0]["dynamism"], 40)
        self.assertEqual(default["dynamism"], 0)
        self.assertEqual(build_attributes_list("unknown", ["time_stamp"]), [])


if __name__ == "__main__":
    unittest.main()
