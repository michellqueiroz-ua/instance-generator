import json
import os
import sys
import tempfile
import unittest
from datetime import time
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from REQreate.webapp.attribute_library import get_required_attributes


APP_PATH = Path(__file__).resolve().parents[1] / "REQreate" / "webapp" / "app.py"


class WebAppTests(unittest.TestCase):
    @staticmethod
    def widget(app, kind, label):
        return next(widget for widget in getattr(app, kind) if widget.label == label)

    @classmethod
    def setUpClass(cls):
        # The app imports its sibling modules as top-level modules when launched by Streamlit.
        sys.path.insert(0, str(APP_PATH.parent))

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(APP_PATH.parent))

    def test_renders_and_switches_problem_types(self):
        app = AppTest.from_file(str(APP_PATH)).run()
        self.assertEqual(len(app.exception), 0)

        for problem in ("DARP", "ODBRP", "Patient Transport"):
            with self.subTest(problem=problem):
                self.widget(app, "selectbox", "Select Problem Type").set_value(problem).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(self.widget(app, "selectbox", "Select Problem Type").value, problem)
                self.assertTrue(
                    set(get_required_attributes(problem)).issubset(
                        self.widget(app, "multiselect", "Select attributes to include:").value
                    )
                )

    def test_generated_configuration_for_each_problem(self):
        for problem in ("DARP", "ODBRP", "Patient Transport"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                with patch("REQreate.input_json.input_json") as generate:
                    previous_dir = os.getcwd()
                    try:
                        os.chdir(directory)
                        app = AppTest.from_file(str(APP_PATH)).run()
                        self.widget(app, "selectbox", "Select Problem Type").set_value(problem).run()
                        if problem == "Patient Transport":
                            selected = self.widget(app, "multiselect", "Select attributes to include:")
                            selected.set_value(selected.value + ["max_ride_time"]).run()
                        self.widget(app, "text_input", "Location (City, Country)").set_value("Testville, Country").run()
                        self.assertEqual(len(app.exception), 0)
                        self.widget(app, "text_input", "Output Folder Name").set_value("offline-test")
                        self.widget(app, "number_input", "Number of Requests").set_value(23)
                        self.widget(app, "number_input", "Replicate Number").set_value(4)
                        self.widget(app, "number_input", "Vehicle Speed (km/h)").set_value(35.0)
                        self.widget(app, "time_input", "Start Time").set_value(time(9, 15))
                        self.widget(app, "time_input", "End Time").set_value(time(10, 45))
                        self.widget(app, "slider", "Dynamism").set_value(0.5)
                        self.widget(app, "number_input", "Reaction Time / Urgency (seconds)").set_value(180)
                        self.widget(app, "number_input", "Max Delay (seconds)").set_value(900)
                        self.widget(app, "button", "🚀 Generate Instance").click().run()

                        self.assertEqual(len(app.exception), 0)
                        config_name = f"offline-test_{problem}_23req.json"
                        config_path = Path("examples/webapp_instances") / config_name
                        with config_path.open() as config_file:
                            config = json.load(config_file)
                        generate.assert_called_once_with(
                            "examples/webapp_instances/", config_name, ""
                        )
                        self.assertEqual(config["network"], "Testville, Country")
                        self.assertEqual(config["problem"], "ODBRP" if problem == "ODBRP" else "DARP")
                        self.assertEqual(config["seed"], 4)
                        self.assertEqual(config["requests"], 23)
                        self.assertEqual(config["set_fixed_speed"], {
                            "vehicle_speed_data": 35.0, "vehicle_speed_data_unit": "kmh"
                        })
                        parameters = {entry["name"]: entry for entry in config["parameters"]}
                        self.assertEqual(parameters["min_early_departure"]["value"], 9.25)
                        self.assertEqual(parameters["max_early_departure"]["value"], 10.75)
                        attributes = {entry["name"]: entry for entry in config["attributes"]}
                        self.assertEqual(attributes["time_stamp"]["pdf"][0], {
                            "type": "uniform", "loc": 33300, "scale": 5400
                        })
                        self.assertEqual(attributes["time_stamp"]["dynamism"], 50)
                        if problem == "ODBRP":
                            self.assertEqual(attributes["reaction_time"]["pdf"][0]["loc"], 180)
                            self.assertEqual(config["travel_time_matrix"], ["bus_stations"])
                            self.assertIn("900", attributes["latest_arrival"]["expression"])
                        else:
                            self.assertEqual(attributes["pickup_to"]["expression"], "pickup_from + 180")
                            self.assertEqual(attributes["dropoff_to"]["expression"], "dropoff_from + 900")
                            if problem == "Patient Transport":
                                self.assertEqual(parameters["hospitals"]["locs"], "hospitals")
                                self.assertEqual(attributes["max_ride_time"]["expression"], "drivingDuration * 1.5")
                    finally:
                        os.chdir(previous_dir)


if __name__ == "__main__":
    unittest.main()
