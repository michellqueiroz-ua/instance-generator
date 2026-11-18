import contextlib
import csv
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
for path in (THIS_DIR, THIS_DIR / "isa_set"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fix_unservable

NETWORK = "Capital Region of Denmark"
NAME = "ISA_b04_i001_1"


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


class FixUnservableTests(unittest.TestCase):
    """A tiny fake network: station matrix M[i, j] = 100 + |i - j| s, the 15 requests of fixtures/test_instance.csv."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        net = self.work / NETWORK
        (net / "csv_format").mkdir(parents=True)
        (net / "travel_time_matrix").mkdir()
        idx = np.arange(1000)
        np.save(net / "travel_time_matrix" / "station_travel_time_matrix.npy",
                (100 + np.abs(idx[:, None] - idx[None, :])).astype(np.int32))
        rows = read_rows(THIS_DIR / "fixtures" / "test_instance.csv")
        self.header = rows[0]
        self.col = {name: i for i, name in enumerate(self.header)}
        self.bad = 3      # request 3: station 761 -> 26, best = 100 + 735 = 835 s
        rows[1 + self.bad][self.col["latest_arrival"]] = str(int(rows[1 + self.bad][self.col["earliest_departure"]]) + 500)
        self.csv = net / "csv_format" / f"{NAME}.csv"
        with open(self.csv, "w", encoding="utf-8", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows(rows)
        self.rows = rows
        self.log = self.work / "design" / "batch_04_unservable_fix.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def run_fix(self, *extra):
        return fix_unservable.main(["--batch", "4", "--work", str(self.work), *extra])

    def test_dry_run_writes_nothing(self):
        before = self.csv.read_bytes()
        self.run_fix()
        self.assertEqual(self.csv.read_bytes(), before)
        self.assertFalse(self.log.exists())
        self.assertFalse((self.csv.parent / "orig").exists())

    def test_apply_fixes_only_the_infeasible_request(self):
        before = self.csv.read_bytes()
        self.run_fix("--apply")
        after = read_rows(self.csv)
        self.assertEqual(len(after), len(self.rows))                       # row count unchanged
        la, ed = self.col["latest_arrival"], self.col["earliest_departure"]
        for k, (old, new) in enumerate(zip(self.rows, after)):
            if k == 1 + self.bad:
                self.assertEqual(int(new[la]), int(old[ed]) + 835 + 300)   # earliest departure + best + 300 s margin
                self.assertEqual([c for i, c in enumerate(new) if i != la], [c for i, c in enumerate(old) if i != la])
            else:
                self.assertEqual(new, old)                                 # feasible requests untouched
        self.assertEqual((self.csv.parent / "orig" / self.csv.name).read_bytes(), before)
        log = read_rows(self.log)
        self.assertEqual(log[0], fix_unservable.LOG_HEADER)
        self.assertEqual(log[1][:5], [NAME, str(self.bad), str(int(self.rows[1 + self.bad][ed]) + 500),
                                      str(int(self.rows[1 + self.bad][ed]) + 835 + 300), "835"])
        self.assertEqual(len(log), 2)
        # byte level: every line except the changed one is identical (same quoting, same line endings)
        old_lines, new_lines = before.splitlines(keepends=True), self.csv.read_bytes().splitlines(keepends=True)
        self.assertEqual(len(old_lines), len(new_lines))
        self.assertEqual([i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b], [1 + self.bad])

    def test_apply_is_idempotent(self):
        self.run_fix("--apply")
        once, log_once = self.csv.read_bytes(), self.log.read_bytes()
        self.run_fix("--apply")
        self.assertEqual(self.csv.read_bytes(), once)
        self.assertEqual(self.log.read_bytes(), log_once)                  # the first log is kept

    def test_clean_batch_still_writes_an_empty_log(self):
        self.run_fix("--apply")
        (self.csv.parent / "orig" / self.csv.name).replace(self.csv)       # restore, then make a clean instance
        rows = read_rows(self.csv)
        rows[1 + self.bad][self.col["latest_arrival"]] = str(int(rows[1 + self.bad][self.col["earliest_departure"]]) + 5000)
        with open(self.csv, "w", encoding="utf-8", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows(rows)
        self.log.unlink()
        self.run_fix("--apply")
        self.assertEqual(read_rows(self.log), [fix_unservable.LOG_HEADER])

    def test_nothing_to_change_refreshes_the_log_modification_time(self):
        self.run_fix("--apply")
        old = self.log.stat().st_mtime - 1000
        os.utime(self.log, (old, old))
        content = self.log.read_bytes()
        self.run_fix("--apply")
        self.assertEqual(self.log.read_bytes(), content)
        self.assertGreater(self.log.stat().st_mtime, old + 500)
        self.assertGreaterEqual(self.log.stat().st_mtime, self.csv.stat().st_mtime)   # what the gate in hpc_submit_next_batch.sh compares

    def test_second_apply_with_changes_appends_to_a_second_log(self):
        self.run_fix("--apply")
        first = self.log.read_bytes()
        rows = read_rows(self.csv)                       # a later change: request 5 becomes unservable too
        rows[1 + 5][self.col["latest_arrival"]] = str(int(rows[1 + 5][self.col["earliest_departure"]]) + 1)
        with open(self.csv, "w", encoding="utf-8", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows(rows)
        old = self.log.stat().st_mtime - 1000
        os.utime(self.log, (old, old))
        self.run_fix("--apply")
        self.assertEqual(self.log.read_bytes(), first)                     # the primary log is not overwritten
        log2 = read_rows(self.work / "design" / "batch_04_unservable_fix_2.csv")
        self.assertEqual(log2[0], fix_unservable.LOG_HEADER)
        self.assertEqual([r[1] for r in log2[1:]], ["5"])
        self.assertEqual(len(log2[1]), len(fix_unservable.LOG_HEADER))
        self.assertTrue(log2[1][-1])                                       # applied_at is filled
        self.assertGreaterEqual(self.log.stat().st_mtime, self.csv.stat().st_mtime)
        self.run_fix("--apply")                                            # and now nothing is left to change
        self.assertEqual(len(read_rows(self.work / "design" / "batch_04_unservable_fix_2.csv")), 2)

    def test_apply_prints_the_json_note(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.run_fix("--apply")
        self.assertIn("json_format/*.json was NOT rewritten", out.getvalue())
        self.assertIn("geographic_dispersion", out.getvalue())

    def test_walking_option_only_enlarges_the_new_window(self):
        self.run_fix("--apply", "--include-walking")
        row = read_rows(self.csv)[1 + self.bad]
        ed = int(row[self.col["earliest_departure"]])
        self.assertGreater(int(row[self.col["latest_arrival"]]), ed + 835 + 300)


if __name__ == "__main__":
    unittest.main()
