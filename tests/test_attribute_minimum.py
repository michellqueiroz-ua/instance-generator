import unittest

from REQreate.passenger_requests import attribute_minimum


class AttributeMinimumTests(unittest.TestCase):
    def test_reads_the_lower_bound_of_the_attribute(self):
        node = {"constraints": ["reaction_time >= 60", "reaction_time <= 5000"]}
        self.assertEqual(attribute_minimum(node, "reaction_time"), 60)

    def test_strict_and_spaceless_forms_and_floats(self):
        self.assertEqual(attribute_minimum({"constraints": ["reaction_time>90.5"]}, "reaction_time"), 90.5)

    def test_other_attributes_and_no_constraints_give_the_default(self):
        self.assertEqual(attribute_minimum({"constraints": ["time_stamp >= 0"]}, "reaction_time"), 0)
        self.assertEqual(attribute_minimum({}, "reaction_time"), 0)

    def test_the_largest_bound_wins_and_negative_bounds_keep_zero(self):
        node = {"constraints": ["reaction_time >= -5", "reaction_time >= 30", "reaction_time >= 20"]}
        self.assertEqual(attribute_minimum(node, "reaction_time"), 30)


if __name__ == "__main__":
    unittest.main()
