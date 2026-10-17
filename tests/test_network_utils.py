import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from REQreate.retrieve_network import build_network
from REQreate.webapp.network_utils import list_networks


class NetworkListingTests(unittest.TestCase):
    def test_lists_only_networks_with_expected_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("zeta", "Alpha"):
                pickle_dir = root / name / "pickle"
                pickle_dir.mkdir(parents=True)
                network_file = pickle_dir / f"{name}.network.class.pkl"
                network_file.write_bytes(b"network")
            (root / "not-built" / "pickle").mkdir(parents=True)
            (root / "not-built" / "pickle" / "other.pkl").write_bytes(b"other")

            networks = list_networks(root)

        self.assertEqual([network["name"] for network in networks], ["Alpha", "zeta"])
        self.assertEqual(networks[0]["size_bytes"], len(b"network"))
        self.assertRegex(networks[0]["modified"], r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


class BuildNetworkTests(unittest.TestCase):
    def test_build_network_rejects_path_traversal(self):
        with patch("REQreate.retrieve_network.download_network_information") as download:
            with self.assertRaisesRegex(ValueError, "without path separators"):
                build_network("../outside")
        download.assert_not_called()

    @patch("REQreate.retrieve_network.download_network_information")
    def test_build_network_forwards_network_settings_without_generating_instance(
        self, download
    ):
        network = object()
        download.return_value = network
        with tempfile.TemporaryDirectory() as directory:
            previous_dir = os.getcwd()
            try:
                os.chdir(directory)
                with patch("REQreate.retrieve_network.output_fixed_route_network"):
                    result = build_network(
                        "Test City",
                        point=(4.4, 51.2),
                        dist=1200,
                        get_fixed_lines="osm",
                        max_speed_factor=0.6,
                    )
            finally:
                os.chdir(previous_dir)

        self.assertIs(result, network)
        download.assert_called_once_with(
            place_name="Test City",
            vehicle_speed_data="max",
            vehicle_speed=None,
            max_speed_factor=0.6,
            get_fixed_lines="osm",
            graph_from_point=True,
            lon=4.4,
            lat=51.2,
            dist=1200,
        )
