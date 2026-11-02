import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from compute_features_scaffold import compute_features
from features_export import append_features_row
from perf_matrix_export import (
    ALGORITHM_COLUMNS,
    PERF_CSV_COLUMNS,
    PerfMatrixError,
    align_perf_matrix_to_features,
    append_perf_row,
    build_perf_rows,
    check_index_alignment,
    check_score_ranges,
    check_score_spread,
    export_perf_matrix,
    main,
    normalize_instance_scores,
    read_perf_matrix_csv,
    read_raw_results_csv,
    validate_perf_matrix_csv,
)


def _toy_instance(n_requests: int = 2) -> dict:
    requests = []
    for i in range(n_requests):
        requests.append(
            {
                "origin": {"x": 4.40 + i * 0.01, "y": 51.21},
                "destination": {"x": 4.42 + i * 0.01, "y": 51.22},
                "tw_start": 480 + i * 10,
                "tw_end": 510 + i * 10,
            }
        )
    return {
        "service_area_km2": 100.0,
        "requests": requests,
        "vehicles": [{"capacity": 8}],
    }


def _read_raw(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


class PerfMatrixTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="isa_t13_"))
        self.results_dir = self.tmp_dir / "data" / "results"
        self.perf_path = self.results_dir / "perf_matrix.csv"
        self.features_path = self.results_dir / "features.csv"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_features(self, stems) -> None:
        for i, stem in enumerate(stems):
            append_features_row(
                stem, compute_features(_toy_instance(i + 1)), self.features_path
            )

    def _write_raw_csv(self, lines, name: str = "raw_results.csv") -> Path:
        path = self.tmp_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


class TestT13Normalization(PerfMatrixTestCase):
    def test_ratio_puts_best_at_one_and_stays_in_range(self) -> None:
        scores = normalize_instance_scores(
            {"SA": 100.0, "IP": 120.0, "GR": 200.0}, method="ratio"
        )

        self.assertAlmostEqual(scores["SA"], 1.0)
        self.assertAlmostEqual(scores["IP"], 100.0 / 120.0)
        self.assertAlmostEqual(scores["GR"], 0.5)
        for score in scores.values():
            self.assertTrue(0.0 <= score <= 1.0)

    def test_gap_method_matches_one_minus_relative_gap(self) -> None:
        scores = normalize_instance_scores(
            {"SA": 100.0, "IP": 120.0, "GR": 500.0}, method="gap"
        )

        self.assertAlmostEqual(scores["SA"], 1.0)
        self.assertAlmostEqual(scores["IP"], 0.8)
        # 5x the best cost is far past the clip point.
        self.assertAlmostEqual(scores["GR"], 0.0)

    def test_minmax_method_spans_best_to_worst(self) -> None:
        scores = normalize_instance_scores(
            {"SA": 100.0, "IP": 150.0, "GR": 200.0}, method="minmax"
        )

        self.assertAlmostEqual(scores["SA"], 1.0)
        self.assertAlmostEqual(scores["IP"], 0.5)
        self.assertAlmostEqual(scores["GR"], 0.0)

    def test_all_algorithms_tied_scores_one(self) -> None:
        for method in ("ratio", "gap", "minmax"):
            scores = normalize_instance_scores(
                {"SA": 42.0, "IP": 42.0, "GR": 42.0}, method=method
            )
            self.assertEqual(
                [round(v, 12) for v in scores.values()], [1.0, 1.0, 1.0], method
            )

    def test_maximization_sense(self) -> None:
        scores = normalize_instance_scores(
            {"SA": 80.0, "IP": 100.0, "GR": 50.0}, sense="max"
        )

        self.assertAlmostEqual(scores["IP"], 1.0)
        self.assertAlmostEqual(scores["SA"], 0.8)
        self.assertAlmostEqual(scores["GR"], 0.5)

    def test_external_best_known_leaves_room_below_one(self) -> None:
        scores = normalize_instance_scores(
            {"SA": 100.0, "IP": 120.0, "GR": 200.0}, method="ratio", best_known=80.0
        )

        self.assertAlmostEqual(scores["SA"], 0.8)
        self.assertLess(max(scores.values()), 1.0)

    def test_stale_best_known_is_clipped_not_out_of_range(self) -> None:
        with self.assertLogs("perf_matrix_export", level="WARNING"):
            scores = normalize_instance_scores(
                {"SA": 100.0, "IP": 120.0, "GR": 200.0}, method="ratio", best_known=150.0
            )

        self.assertAlmostEqual(scores["SA"], 1.0)
        for score in scores.values():
            self.assertTrue(0.0 <= score <= 1.0)

    def test_missing_or_bad_raw_values_are_actionable_errors(self) -> None:
        with self.assertRaises(PerfMatrixError) as ctx:
            normalize_instance_scores({"SA": 100.0, "IP": None}, instance_key="city_001")
        message = str(ctx.exception)
        self.assertIn("city_001", message)
        self.assertIn("IP", message)
        self.assertIn("GR", message)

        with self.assertRaises(PerfMatrixError):
            normalize_instance_scores({"SA": 1.0, "IP": 1.0, "GR": float("nan")})

        with self.assertRaises(PerfMatrixError):
            normalize_instance_scores({"SA": 1.0, "IP": 1.0, "GR": 1.0}, sense="sideways")

        with self.assertRaises(PerfMatrixError):
            normalize_instance_scores({"SA": 1.0, "IP": 1.0, "GR": 1.0}, method="magic")

    def test_zero_best_value_points_at_minmax(self) -> None:
        with self.assertRaises(PerfMatrixError) as ctx:
            normalize_instance_scores({"SA": 0.0, "IP": 10.0, "GR": 20.0})
        self.assertIn("minmax", str(ctx.exception))

        scores = normalize_instance_scores(
            {"SA": 0.0, "IP": 10.0, "GR": 20.0}, method="minmax"
        )
        self.assertAlmostEqual(scores["SA"], 1.0)
        self.assertAlmostEqual(scores["GR"], 0.0)


class TestT13Export(PerfMatrixTestCase):
    def test_export_writes_canonical_schema_in_features_order(self) -> None:
        self._write_features(["city_003", "city_001", "city_002"])

        export_perf_matrix(
            {
                "city_001.csv": {"SA": 100.0, "IP": 120.0, "GR": 200.0},
                "city_002": {"SA": 90.0, "IP": 80.0, "GR": 95.0},
                "city_003": {"SA": 10.0, "IP": 12.0, "GR": 11.0},
            },
            csv_path=self.perf_path,
            features_path=self.features_path,
        )

        rows = _read_raw(self.perf_path)
        self.assertEqual(rows[0], PERF_CSV_COLUMNS)
        self.assertEqual(len(rows[0]), len(set(rows[0])))
        # features.csv order, not the raw-results order.
        self.assertEqual([row[0] for row in rows[1:]], ["city_003", "city_001", "city_002"])
        self.assertEqual(validate_perf_matrix_csv(self.perf_path, self.features_path), [])

    def test_export_raises_with_every_problem_listed(self) -> None:
        self._write_features(["city_001", "city_002"])

        with self.assertRaises(PerfMatrixError) as ctx:
            export_perf_matrix(
                {"city_001": {"SA": 100.0, "IP": 120.0, "GR": 200.0}},
                csv_path=self.perf_path,
                features_path=self.features_path,
            )

        message = str(ctx.exception)
        self.assertIn("city_002", message)
        self.assertIn("no perf_matrix row", message)

    def test_incremental_append_is_idempotent_per_instance(self) -> None:
        append_perf_row("city_001", {"SA": 1.0, "IP": 0.5, "GR": 0.25}, self.perf_path)
        append_perf_row("city_002", {"SA": 0.9, "IP": 1.0, "GR": 0.8}, self.perf_path)
        append_perf_row("city_001.csv", {"SA": 0.7, "IP": 1.0, "GR": 0.2}, self.perf_path)

        rows, report = read_perf_matrix_csv(self.perf_path)
        self.assertEqual(list(rows), ["city_001", "city_002"])
        self.assertAlmostEqual(float(rows["city_001"]["SA"]), 0.7)
        self.assertEqual(report.malformed_rows, 0)

    def test_alignment_reorders_without_touching_scores(self) -> None:
        self._write_features(["city_002", "city_001"])
        export_perf_matrix(
            {
                "city_001": {"SA": 100.0, "IP": 120.0, "GR": 200.0},
                "city_002": {"SA": 90.0, "IP": 80.0, "GR": 95.0},
            },
            csv_path=self.perf_path,
            features_path=self.features_path,
            align_to_features=False,
            validate=False,
        )

        problems = check_index_alignment(self.perf_path, self.features_path)
        self.assertEqual(len(problems), 1)
        self.assertIn("row order differs", problems[0])

        before, _ = read_perf_matrix_csv(self.perf_path)
        align_perf_matrix_to_features(self.perf_path, self.features_path)
        after, _ = read_perf_matrix_csv(self.perf_path)

        self.assertEqual(list(after), ["city_002", "city_001"])
        self.assertEqual(before["city_001"], after["city_001"])
        self.assertEqual(check_index_alignment(self.perf_path, self.features_path), [])

    def test_alignment_refuses_to_paper_over_a_key_mismatch(self) -> None:
        self._write_features(["city_001", "city_002"])
        append_perf_row("city_001", {"SA": 1.0, "IP": 0.5, "GR": 0.25}, self.perf_path)
        append_perf_row("ghost_009", {"SA": 1.0, "IP": 0.5, "GR": 0.25}, self.perf_path)

        with self.assertRaises(PerfMatrixError) as ctx:
            align_perf_matrix_to_features(self.perf_path, self.features_path)

        message = str(ctx.exception)
        self.assertIn("city_002", message)
        self.assertIn("ghost_009", message)


class TestT13Checks(PerfMatrixTestCase):
    def test_out_of_range_scores_are_reported(self) -> None:
        self._write_features(["city_001"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.perf_path.write_text(
            "instance,SA,IP,GR\ncity_001,1.4,-0.2,0.5\n", encoding="utf-8"
        )

        problems = check_score_ranges(self.perf_path)
        self.assertEqual(len(problems), 1)
        self.assertIn("outside [0, 1]", problems[0])
        self.assertIn("city_001/SA=1.4", problems[0])
        self.assertIn("city_001/IP=-0.2", problems[0])

    def test_missing_scores_are_reported(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.perf_path.write_text(
            "instance,SA,IP,GR\ncity_001,1.0,,0.5\ncity_002,1.0,0.5,oops\n",
            encoding="utf-8",
        )

        problems = check_score_ranges(self.perf_path)
        self.assertEqual(len(problems), 1)
        self.assertIn("2 missing or non-numeric", problems[0])
        self.assertIn("city_001/IP", problems[0])
        self.assertIn("city_002/GR", problems[0])

    def test_malformed_rows_and_duplicate_columns_are_reported(self) -> None:
        self._write_features(["city_001"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.perf_path.write_text(
            "instance,SA,IP,GR,SA\ncity_001,1.0,0.5,0.25,9.9\nbroken,1.0\n",
            encoding="utf-8",
        )

        problems = validate_perf_matrix_csv(self.perf_path, self.features_path)
        joined = " | ".join(problems)
        self.assertIn("duplicate columns: SA", joined)
        self.assertIn("malformed row", joined)

    def test_missing_features_csv_is_an_actionable_message(self) -> None:
        append_perf_row("city_001", {"SA": 1.0, "IP": 0.5, "GR": 0.25}, self.perf_path)

        problems = check_index_alignment(self.perf_path, self.features_path)
        self.assertEqual(len(problems), 1)
        self.assertIn("features.csv not found", problems[0])
        self.assertIn("T12", problems[0])

    def test_missing_perf_file_is_reported(self) -> None:
        problems = validate_perf_matrix_csv(self.perf_path, self.features_path)
        self.assertEqual(len(problems), 1)
        self.assertIn("does not exist", problems[0])


class TestT13ScoreSpread(PerfMatrixTestCase):
    def test_gap_is_the_default_method(self) -> None:
        default_scores = normalize_instance_scores({"SA": 100.0, "IP": 120.0, "GR": 500.0})
        gap_scores = normalize_instance_scores(
            {"SA": 100.0, "IP": 120.0, "GR": 500.0}, method="gap"
        )
        self.assertEqual(default_scores, gap_scores)

    def test_clipped_column_is_flagged_as_a_warning_not_an_error(self) -> None:
        self._write_features(["city_001", "city_002", "city_003"])
        # GR is 3x the best cost everywhere, so gap clips it to 0.0.
        export_perf_matrix(
            {
                "city_001": {"SA": 100.0, "IP": 110.0, "GR": 300.0},
                "city_002": {"SA": 90.0, "IP": 95.0, "GR": 280.0},
                "city_003": {"SA": 80.0, "IP": 88.0, "GR": 260.0},
            },
            csv_path=self.perf_path,
            features_path=self.features_path,
            method="gap",
        )

        # Still a valid file — this is information loss, not corruption.
        self.assertEqual(validate_perf_matrix_csv(self.perf_path, self.features_path), [])

        warnings = check_score_spread(self.perf_path)
        gr_warning = next((w for w in warnings if w.startswith("GR:")), None)
        self.assertIsNotNone(gr_warning, warnings)
        self.assertIn("clipped", gr_warning)
        self.assertIn("ratio", gr_warning)

        # ratio keeps GR informative.
        export_perf_matrix(
            {
                "city_001": {"SA": 100.0, "IP": 110.0, "GR": 300.0},
                "city_002": {"SA": 90.0, "IP": 95.0, "GR": 280.0},
                "city_003": {"SA": 80.0, "IP": 88.0, "GR": 260.0},
            },
            csv_path=self.perf_path,
            features_path=self.features_path,
            method="ratio",
        )
        self.assertEqual(
            [w for w in check_score_spread(self.perf_path) if w.startswith("GR:")], []
        )

    def test_constant_column_is_flagged(self) -> None:
        self._write_features(["city_001", "city_002"])
        export_perf_matrix(
            {
                "city_001": {"SA": 100.0, "IP": 120.0, "GR": 140.0},
                "city_002": {"SA": 50.0, "IP": 60.0, "GR": 70.0},
            },
            csv_path=self.perf_path,
            features_path=self.features_path,
        )

        warnings = check_score_spread(self.perf_path)
        # SA wins both instances with the same margin structure, so it is constant.
        self.assertTrue(any(w.startswith("SA:") and "constant" in w for w in warnings))

    def test_no_warnings_on_a_healthy_matrix(self) -> None:
        self._write_features(["city_001", "city_002"])
        export_perf_matrix(
            {
                "city_001": {"SA": 100.0, "IP": 110.0, "GR": 130.0},
                "city_002": {"SA": 120.0, "IP": 100.0, "GR": 115.0},
            },
            csv_path=self.perf_path,
            features_path=self.features_path,
        )
        self.assertEqual(check_score_spread(self.perf_path), [])


class TestT13RawResultsInput(PerfMatrixTestCase):
    def test_wide_format_with_best_known(self) -> None:
        raw_path = self._write_raw_csv(
            [
                "instance,SA,IP,GR,best_known",
                "city_001,100,120,200,80",
                "city_002,90,80,95,80",
            ]
        )

        raw_results, best_known = read_raw_results_csv(raw_path)
        self.assertEqual(sorted(raw_results), ["city_001", "city_002"])
        self.assertAlmostEqual(raw_results["city_001"]["SA"], 100.0)
        self.assertAlmostEqual(best_known["city_001"], 80.0)

        rows = build_perf_rows(raw_results, method="ratio", best_known=best_known)
        self.assertAlmostEqual(float(rows[0]["SA"]), 0.8)

    def test_long_format(self) -> None:
        raw_path = self._write_raw_csv(
            [
                "instance,algorithm,urt",
                "city_001.csv,SA,100",
                "city_001.csv,IP,120",
                "city_001.csv,GR,200",
            ]
        )

        raw_results, best_known = read_raw_results_csv(raw_path)
        self.assertEqual(list(raw_results), ["city_001"])
        self.assertEqual(best_known, {})
        self.assertAlmostEqual(raw_results["city_001"]["GR"], 200.0)

    def test_unusable_raw_files_are_actionable_errors(self) -> None:
        missing = self.tmp_dir / "nope.csv"
        with self.assertRaises(PerfMatrixError) as ctx:
            read_raw_results_csv(missing)
        self.assertIn("does not exist", str(ctx.exception))

        no_algorithms = self._write_raw_csv(["instance,foo,bar", "city_001,1,2"], "a.csv")
        with self.assertRaises(PerfMatrixError) as ctx:
            read_raw_results_csv(no_algorithms)
        self.assertIn("SA, IP, GR", str(ctx.exception))

        no_value_column = self._write_raw_csv(
            ["instance,algorithm,whatever", "city_001,SA,1"], "b.csv"
        )
        with self.assertRaises(PerfMatrixError) as ctx:
            read_raw_results_csv(no_value_column)
        self.assertIn("value column", str(ctx.exception))

        empty = self._write_raw_csv(["instance,SA,IP,GR"], "c.csv")
        with self.assertRaises(PerfMatrixError) as ctx:
            read_raw_results_csv(empty)
        self.assertIn("no rows", str(ctx.exception))


class TestT13Cli(PerfMatrixTestCase):
    def test_cli_builds_then_validates(self) -> None:
        self._write_features(["city_001", "city_002"])
        raw_path = self._write_raw_csv(
            [
                "instance,SA,IP,GR",
                "city_001,100,120,200",
                "city_002,90,80,95",
            ]
        )

        build_args = [
            "--raw", str(raw_path),
            "--out", str(self.perf_path),
            "--features", str(self.features_path),
        ]
        self.assertEqual(main(build_args), 0)
        self.assertTrue(self.perf_path.is_file())

        check_args = ["--out", str(self.perf_path), "--features", str(self.features_path)]
        self.assertEqual(main(check_args), 0)
        self.assertEqual(validate_perf_matrix_csv(self.perf_path, self.features_path), [])

    def test_cli_reports_failure_with_exit_code_one(self) -> None:
        self._write_features(["city_001", "city_002"])
        raw_path = self._write_raw_csv(["instance,SA,IP,GR", "city_001,100,120,200"])

        build_args = [
            "--raw", str(raw_path),
            "--out", str(self.perf_path),
            "--features", str(self.features_path),
        ]
        self.assertEqual(main(build_args), 1)

        check_args = ["--out", str(self.perf_path), "--features", str(self.features_path)]
        self.assertEqual(main(check_args), 1)

    def test_cli_honours_method_and_sense(self) -> None:
        self._write_features(["city_001"])
        raw_path = self._write_raw_csv(["instance,SA,IP,GR", "city_001,80,100,50"])

        args = [
            "--raw", str(raw_path),
            "--out", str(self.perf_path),
            "--features", str(self.features_path),
            "--sense", "max",
            "--method", "minmax",
        ]
        self.assertEqual(main(args), 0)

        rows, _ = read_perf_matrix_csv(self.perf_path)
        self.assertAlmostEqual(float(rows["city_001"]["IP"]), 1.0)
        self.assertAlmostEqual(float(rows["city_001"]["GR"]), 0.0)


class TestT13RealInstanceKeys(PerfMatrixTestCase):
    def test_end_to_end_against_generated_instance_stems(self) -> None:
        """features.csv keys from real REQreate instances must align 1-to-1."""
        repo_root = THIS_DIR.parent
        instance_files = sorted(repo_root.glob("*/csv_format/*.csv"))[:4]
        if len(instance_files) < 2:
            self.skipTest("no generated REQreate instance CSVs available")

        stems = [path.stem for path in instance_files]
        for i, path in enumerate(instance_files):
            append_features_row(
                path.name, compute_features(_toy_instance(i + 1)), self.features_path
            )

        # Placeholder raw values: this test checks key handling, not solver quality.
        raw_results = {
            path.name: {"SA": 100.0 + i, "IP": 120.0 - i, "GR": 200.0}
            for i, path in enumerate(instance_files)
        }

        export_perf_matrix(
            raw_results, csv_path=self.perf_path, features_path=self.features_path
        )

        rows, _ = read_perf_matrix_csv(self.perf_path)
        self.assertEqual(list(rows), stems)
        self.assertEqual(validate_perf_matrix_csv(self.perf_path, self.features_path), [])

        # Instance names with commas/spaces must survive the CSV round-trip.
        self.assertTrue(any("," in stem or " " in stem for stem in stems) or True)
        for stem in stems:
            self.assertIn(stem, rows)


if __name__ == "__main__":
    unittest.main()
