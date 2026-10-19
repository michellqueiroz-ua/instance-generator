import json
import os
import sys
import tempfile
import unittest
from datetime import time
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from REQreate.webapp.attribute_library import get_required_attributes


APP_PATH = Path(__file__).resolve().parents[1] / "REQreate" / "webapp" / "app.py"

# Patient Transport is hidden by default while its hospital lookup is fixed;
# the tests below still exercise it by switching it on.
ENABLE_PATIENT_TRANSPORT = {"REQREATE_ENABLE_PATIENT_TRANSPORT": "1"}


class WebAppTests(unittest.TestCase):
    @staticmethod
    def widget(app, kind, label):
        return next(widget for widget in getattr(app, kind) if widget.label == label)

    @staticmethod
    def make_network(directory, name):
        pickle_dir = Path(directory) / name / "pickle"
        pickle_dir.mkdir(parents=True)
        (pickle_dir / f"{name}.network.class.pkl").touch()

    @classmethod
    def setUpClass(cls):
        # The app imports its sibling modules as top-level modules when launched by Streamlit.
        sys.path.insert(0, str(APP_PATH.parent))

    def setUp(self):
        self.queue_patch = patch("REQreate.webapp.job_queue.JobQueue")
        self.queue_factory = self.queue_patch.start()
        self.queue = self.queue_factory.return_value
        self.queue._closed = False
        self.queue.submit_generate.return_value = ({"id": "test-job"}, 1)
        self.queue.submit_network.return_value = ({"id": "test-network"}, 2)
        self.queue.jobs.return_value = []
        self.queue.paused = False
        self.addCleanup(self.queue_patch.stop)
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(APP_PATH.parent))

    def test_instance_form_disables_enter_submission(self):
        # AppTest cannot simulate Enter, so check the form's submission setting.
        with tempfile.TemporaryDirectory() as directory:
            self.make_network(directory, "Test City")
            previous_dir = os.getcwd()
            try:
                os.chdir(directory)
                with patch("streamlit.form", wraps=st.form) as form:
                    app = AppTest.from_file(str(APP_PATH)).run()
            finally:
                os.chdir(previous_dir)
        self.assertEqual(len(app.exception), 0)
        form.assert_called_once_with("instance_form", enter_to_submit=False)

    def test_instance_generation_requires_a_built_network(self):
        with tempfile.TemporaryDirectory() as directory:
            previous_dir = os.getcwd()
            try:
                os.chdir(directory)
                app = AppTest.from_file(str(APP_PATH)).run()
            finally:
                os.chdir(previous_dir)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("No networks yet" in item.value for item in app.info))
        self.assertFalse(any(widget.label == "🚀 Generate Instance" for widget in app.button))

    def test_network_form_disables_enter_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            previous_dir = os.getcwd()
            try:
                os.chdir(directory)
                with patch("streamlit.form", wraps=st.form) as form:
                    app = AppTest.from_file(str(APP_PATH)).run()
                    self.widget(app, "radio", "Navigation").set_value("Networks").run()
            finally:
                os.chdir(previous_dir)

        self.assertEqual(len(app.exception), 0)
        form.assert_called_once_with("network_form", enter_to_submit=False)

    def test_patient_transport_hidden_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_network(directory, "Test City")
            previous_dir = os.getcwd()
            try:
                os.chdir(directory)
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("REQREATE_ENABLE_PATIENT_TRANSPORT", None)
                    app = AppTest.from_file(str(APP_PATH)).run()
            finally:
                os.chdir(previous_dir)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(
            self.widget(app, "selectbox", "Select Problem Type").options, ["DARP", "ODBRP"]
        )

    @patch.dict(os.environ, ENABLE_PATIENT_TRANSPORT)
    def test_renders_and_switches_problem_types(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_network(directory, "Test City")
            previous_dir = os.getcwd()
            try:
                os.chdir(directory)
                app = AppTest.from_file(str(APP_PATH)).run()
                self.assertEqual(len(app.exception), 0)

                for problem in ("DARP", "ODBRP", "Patient Transport"):
                    with self.subTest(problem=problem):
                        self.widget(app, "selectbox", "Select Problem Type").set_value(problem).run()
                        self.assertEqual(len(app.exception), 0)
                        self.assertEqual(
                            self.widget(app, "selectbox", "Select Problem Type").value,
                            problem,
                        )
                        self.assertTrue(
                            set(get_required_attributes(problem)).issubset(
                                self.widget(app, "multiselect", "Select attributes to include:").value
                            )
                        )
            finally:
                os.chdir(previous_dir)

    def test_output_folder_defaults_to_submitted_location(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("REQreate.input_json.input_json") as generate:
                previous_dir = os.getcwd()
                try:
                    os.chdir(directory)
                    self.make_network(directory, "Antwerp, Belgium")
                    app = AppTest.from_file(str(APP_PATH)).run()
                    self.widget(app, "selectbox", "Built Network").set_value("Antwerp, Belgium")
                    self.widget(app, "button", "🚀 Generate Instance").click().run()

                    self.assertEqual(len(app.exception), 0)
                    config_name = "Antwerp_Belgium_DARP_10req.json"
                    config_path = Path(directory) / "examples/webapp_instances" / config_name
                    with config_path.open() as config_file:
                        config = json.load(config_file)
                    generate.assert_not_called()
                    self.queue.submit_generate.assert_called_once_with(
                        str(Path("examples/webapp_instances") / config_name), configuration=config
                    )
                    self.assertTrue(any(
                        "Added to the queue (position 1)" in item.value for item in app.success
                    ))
                    self.assertEqual(config["network"], "Antwerp, Belgium")
                finally:
                    os.chdir(previous_dir)

    @patch.dict(os.environ, ENABLE_PATIENT_TRANSPORT)
    def test_generated_configuration_for_each_problem(self):
        for problem in ("DARP", "ODBRP", "Patient Transport"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                self.queue.reset_mock()
                with patch("REQreate.input_json.input_json") as generate:
                    previous_dir = os.getcwd()
                    try:
                        os.chdir(directory)
                        self.make_network(directory, "Testville, Country")
                        app = AppTest.from_file(str(APP_PATH)).run()
                        self.widget(app, "selectbox", "Select Problem Type").set_value(problem).run()
                        if problem == "Patient Transport":
                            selected = self.widget(app, "multiselect", "Select attributes to include:")
                            selected.set_value(selected.value + ["max_ride_time"]).run()
                        self.widget(app, "selectbox", "Built Network").set_value("Testville, Country")
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
                        generate.assert_not_called()
                        self.queue.submit_generate.assert_called_once_with(
                            str(config_path), configuration=config
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

    def test_network_download_adds_a_job(self):
        with tempfile.TemporaryDirectory() as directory:
            previous_dir = os.getcwd()
            try:
                os.chdir(directory)
                app = AppTest.from_file(str(APP_PATH)).run()
                self.widget(app, "radio", "Navigation").set_value("Networks").run()
                self.widget(app, "text_input", "Place or network name").set_value("Test City")
                self.widget(app, "button", "Download network").click().run()
                self.assertEqual(len(app.exception), 0)
                self.queue.submit_network.assert_called_once_with(
                    "Test City", point=None, dist=None, fixed_lines=None
                )
                self.assertTrue(any(
                    "Added to the queue (position 2)" in item.value for item in app.success
                ))
            finally:
                os.chdir(previous_dir)

    def test_queue_page_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            previous_dir = os.getcwd()
            try:
                os.chdir(directory)
                self.queue.paused = True
                app = AppTest.from_file(str(APP_PATH)).run()
                self.widget(app, "radio", "Navigation").set_value("Queue").run()
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any("paused" in item.value for item in app.info))
                self.widget(app, "button", "Resume").click().run()
                self.queue.resume.assert_called_once()
                self.widget(app, "button", "Clear finished").click().run()
                self.queue.clear_finished.assert_called_once()
                self.assertEqual(len(app.exception), 0)
            finally:
                os.chdir(previous_dir)

    def test_worker_is_shared_across_browser_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            previous_dir = os.getcwd()
            try:
                os.chdir(directory)
                first = AppTest.from_file(str(APP_PATH)).run()
                st.cache_resource.clear()
                second = AppTest.from_file(str(APP_PATH)).run()
                self.assertEqual(len(first.exception), 0)
                self.assertEqual(len(second.exception), 0)
                self.queue_factory.assert_called_once_with(str(Path(directory).resolve()))
            finally:
                os.chdir(previous_dir)

    def test_queue_shows_running_log_and_can_cancel(self):
        self.queue.jobs.return_value = [{
            "id": "running12345", "type": "network",
            "parameters": {"place": "Test City"}, "status": "running",
            "created": "2026-01-01T00:00:00+00:00",
            "started": "2026-01-01T00:00:01+00:00", "finished": None, "exit_code": None,
        }]
        self.queue.tail.return_value = "Network progress"
        with tempfile.TemporaryDirectory() as directory:
            previous_dir = os.getcwd()
            try:
                os.chdir(directory)
                app = AppTest.from_file(str(APP_PATH)).run()
                self.widget(app, "radio", "Navigation").set_value("Queue").run()
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any("Network progress" in item.value for item in app.code))
                self.widget(app, "button", "Cancel network running1").click().run()
                self.queue.cancel.assert_called_once_with("running12345")
                self.assertEqual(len(app.exception), 0)
            finally:
                os.chdir(previous_dir)


if __name__ == "__main__":
    unittest.main()
