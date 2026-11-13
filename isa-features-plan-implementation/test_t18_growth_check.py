import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
for path in (THIS_DIR, THIS_DIR / "isa_set"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import growth_check

COLUMNS = ["f1", "f2", "f3", "constant"]


def write_batch(path, data):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", *COLUMNS])
        for i, row in enumerate(data):
            w.writerow([f"inst{i}", *row, 7.0])


def draw(rng, n, shift=0.0):
    return rng.normal(size=(n, 3)) + shift


class GrowthCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.rng = np.random.default_rng(0)
        write_batch(self.dir / "features_batch_01.csv", draw(self.rng, 400))

    def run_check(self, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = growth_check.main(["--new", str(self.dir / "features_batch_02.csv"), *extra])
        return status, out.getvalue()

    def test_a_batch_like_the_existing_one_stops(self):
        write_batch(self.dir / "features_batch_02.csv", draw(self.rng, 200))
        status, text = self.run_check()
        self.assertEqual(status, 1, text)
        self.assertIn("STOP: far share", text)

    def test_a_batch_in_new_territory_continues(self):
        write_batch(self.dir / "features_batch_02.csv", draw(self.rng, 200, shift=6.0))
        status, text = self.run_check()
        self.assertEqual(status, 0, text)
        self.assertIn("CONTINUE", text)

    def test_cap_stops_even_in_new_territory(self):
        write_batch(self.dir / "features_batch_02.csv", draw(self.rng, 200, shift=6.0))
        status, text = self.run_check("--cap", "500")
        self.assertEqual(status, 1, text)
        self.assertIn("cap", text)

    def test_constant_column_is_left_out_and_first_batch_has_nothing_to_compare(self):
        write_batch(self.dir / "features_batch_02.csv", draw(self.rng, 50))
        _, text = self.run_check()
        self.assertIn("features used 3 of 4", text)
        (self.dir / "features_batch_01.csv").unlink()
        status, _ = self.run_check()
        self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()
