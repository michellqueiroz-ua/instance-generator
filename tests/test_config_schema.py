import json
import tempfile
import unittest
from pathlib import Path

from REQreate.config_schema import validate_config


class ConfigSchemaTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "network": "Example City",
            "problem": "DARP",
            "parameters": [
                {"name": "start_time", "type": "integer", "value": 7, "time_unit": "h"}
            ],
            "places": [
                {"name": "depot", "type": "location", "lon": 4.3, "lat": 51.2},
                {"name": "service_area", "type": "zone", "centroid": True, "radius": 1000},
            ],
            "attributes": [
                {
                    "name": "departure",
                    "type": "integer",
                    "time_unit": "s",
                    "pdf": [{"type": "uniform", "loc": 0, "scale": 3600}],
                }
            ],
        }

    def test_valid_configuration(self):
        self.assertEqual(validate_config(self.config), [])

    def test_missing_network_is_invalid(self):
        config = {**self.config}
        del config["network"]

        errors = validate_config(config)

        self.assertTrue(any("network" in error for error in errors))

    def test_invalid_units_are_reported(self):
        config = {**self.config, "parameters": [{"name": "start", "time_unit": "days"}]}

        errors = validate_config(config)

        self.assertTrue(any("time_unit" in error for error in errors))

    def test_pdf_type_and_parameters_are_checked(self):
        config = {
            **self.config,
            "attributes": [
                {"name": "departure", "type": "integer", "pdf": [{"type": "gaussian"}]}
            ],
        }

        errors = validate_config(config)

        self.assertTrue(any("type" in error for error in errors))

    def test_pdf_distribution_parameters_are_required(self):
        config = {
            **self.config,
            "attributes": [
                {"name": "departure", "type": "integer", "pdf": [{"type": "gamma"}]}
            ],
        }

        errors = validate_config(config)

        self.assertTrue(any("'loc' is a required property" in error for error in errors))
        self.assertTrue(any("'aux' is a required property" in error for error in errors))

    def test_location_requires_coordinates_or_centroid(self):
        config = {**self.config, "places": [{"name": "depot", "type": "location"}]}

        errors = validate_config(config)

        self.assertTrue(any("valid under any of the given schemas" in error for error in errors))

    def test_path_input_and_unreadable_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(self.config), encoding="utf-8")
            self.assertEqual(validate_config(path), [])
            path.write_text("{", encoding="utf-8")
            self.assertTrue(validate_config(path))


if __name__ == "__main__":
    unittest.main()
