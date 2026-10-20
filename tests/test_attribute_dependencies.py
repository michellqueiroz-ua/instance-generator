import sys
import unittest
from pathlib import Path

import networkx as nx

from REQreate.input_json import _add_attribute_dependencies

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "REQreate" / "webapp"))
import attribute_library  # noqa: E402  (imported as a top-level module, as the app does)


def _graph(attributes):
    """Dependency graph nodes as input_json builds them from config attributes."""
    GA = nx.DiGraph()
    for attribute in attributes:
        GA.add_node(attribute["name"])
        for key in ("expression", "constraints"):
            if key in attribute:
                GA.nodes[attribute["name"]][key] = attribute[key]
    return GA


class AttributeDependencyTests(unittest.TestCase):
    def test_stops_without_walking_attributes_is_a_clear_error(self):
        GA = _graph([
            {"name": "origin", "type": "location"},
            {"name": "stops_orgn", "expression": "stops(origin)"},
        ])
        with self.assertRaises(ValueError) as raised:
            _add_attribute_dependencies(GA)
        self.assertIn("max_walking", str(raised.exception))
        self.assertIn("walk_speed", str(raised.exception))

    def test_stops_links_walking_attributes(self):
        GA = _graph([
            {"name": "origin", "type": "location"},
            {"name": "max_walking"},
            {"name": "walk_speed"},
            {"name": "stops_orgn", "expression": "stops(origin)", "constraints": ["len(stops_orgn) > 0"]},
        ])
        _add_attribute_dependencies(GA)
        self.assertTrue(GA.has_edge("origin", "stops_orgn"))
        self.assertTrue(GA.has_edge("max_walking", "stops_orgn"))
        self.assertTrue(GA.has_edge("walk_speed", "stops_orgn"))
        self.assertFalse(GA.has_edge("stops_orgn", "stops_orgn"))

    def test_web_app_default_selection_builds_for_every_problem(self):
        # The web app pre-selects exactly the required attributes. An ODBRP
        # instance made that way used to crash with "dictionary changed size
        # during iteration" because stops() needs two optional attributes.
        for problem in ("DARP", "ODBRP", "Patient Transport"):
            with self.subTest(problem=problem):
                required = attribute_library.get_required_attributes(problem)
                attributes = attribute_library.build_attributes_list(problem, required)
                _add_attribute_dependencies(_graph(attributes))


if __name__ == "__main__":
    unittest.main()
