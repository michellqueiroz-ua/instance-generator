import unittest

from REQreate.passenger_requests import eval_expression


class AllowedExpressionTests(unittest.TestCase):
    # Attribute names are substituted by their values before evaluation, so
    # these mirror what eval_expression sees for the example configs.

    def test_arithmetic(self):
        self.assertEqual(eval_expression("100 + 20"), 120)
        self.assertEqual(eval_expression("100 - 20 * 2"), 60)
        self.assertEqual(eval_expression("7 / 2"), 3.5)
        self.assertEqual(eval_expression("7 // 2"), 3)
        self.assertEqual(eval_expression("7 % 2"), 1)
        self.assertEqual(eval_expression("2 ** 3"), 8)
        self.assertEqual(eval_expression("300 + 120 + (60) + 3600"), 4080)
        self.assertEqual(eval_expression("600 * 1.5"), 900.0)

    def test_comparisons(self):
        self.assertTrue(eval_expression("5 >= 0"))
        self.assertFalse(eval_expression("5 <= 0"))
        self.assertTrue(eval_expression("5 > 4"))
        self.assertTrue(eval_expression("5 == 5.0"))
        self.assertTrue(eval_expression("5 != 4"))
        self.assertTrue(eval_expression("0 <= 5 <= 10"))

    def test_len_and_set(self):
        self.assertTrue(eval_expression("len([1, 2, 3]) > 0"))
        self.assertFalse(eval_expression("len([]) > 0"))
        self.assertTrue(eval_expression("not (set([1, 2]) & set([3, 4]))"))
        self.assertFalse(eval_expression("not (set([1, 2]) & set([2, 4]))"))
        self.assertEqual(eval_expression("set([1, 1, 2])"), {1, 2})

    def test_min_max_abs_round(self):
        self.assertEqual(eval_expression("min(3, 1, 2)"), 1)
        self.assertEqual(eval_expression("max([3, 1, 2])"), 3)
        self.assertEqual(eval_expression("abs(-4)"), 4)
        self.assertEqual(eval_expression("round(2.567, 2)"), 2.57)
        self.assertEqual(eval_expression("round(2.4)"), 2)

    def test_boolean_logic(self):
        self.assertTrue(eval_expression("True and not False"))
        self.assertTrue(eval_expression("False or 1 > 0"))
        self.assertFalse(eval_expression("1 > 0 and 2 < 1"))
        self.assertEqual(eval_expression("5 if 1 > 0 else 6"), 5)


class RejectedExpressionTests(unittest.TestCase):

    def assertRejected(self, expression):
        with self.assertRaises(NameError):
            eval_expression(expression)

    def test_import_is_rejected(self):
        self.assertRejected("__import__('os').system('echo pwned')")

    def test_open_is_rejected(self):
        self.assertRejected("open('/etc/passwd').read()")

    def test_eval_and_exec_are_rejected(self):
        self.assertRejected("eval('1 + 1')")
        self.assertRejected("exec('x = 1')")

    def test_attribute_escapes_are_rejected(self):
        self.assertRejected("().__class__.__bases__")
        self.assertRejected("().__class__.__bases__[0].__subclasses__()")
        self.assertRejected("'{0.__class__}'.format(1)")

    def test_escapes_inside_lambda_are_rejected(self):
        self.assertRejected("(lambda: ().__class__.__bases__)()")
        self.assertRejected("(lambda: __import__('os'))()")

    def test_escapes_inside_comprehensions_are_rejected(self):
        self.assertRejected("[c for c in ().__class__.__bases__]")
        self.assertRejected("[x.__class__ for x in [1]]")
        self.assertRejected("{x: x.__class__ for x in [1]}")
        self.assertRejected("list(x.__class__ for x in [1])")


class EdgeCaseTests(unittest.TestCase):

    def test_empty_string_raises_syntax_error(self):
        with self.assertRaises(SyntaxError):
            eval_expression("")

    def test_syntax_error(self):
        with self.assertRaises(SyntaxError):
            eval_expression("1 +")
        with self.assertRaises(SyntaxError):
            eval_expression("x = 1")

    def test_unknown_variable_is_rejected(self):
        with self.assertRaises(NameError) as ctx:
            eval_expression("time_stamp + reaction_time")
        self.assertIn("time_stamp", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
