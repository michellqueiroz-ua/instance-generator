import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for _path in (THIS_DIR, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from compute_features_scaffold import FEATURE_KEYS
from isa_pipeline import (
    DYNAMIC_COLUMNS,
    build_metadata_frame,
    Footprint,
    PipelineError,
    compute_footprints,
    convex_hull,
    load_inputs,
    main,
    pilot_analytic,
    point_in_polygon,
    polygon_area,
    preprocess_features,
    project,
    run_pipeline,
)


def _features_frame(n_instances: int = 40, seed: int = 0) -> pd.DataFrame:
    """Synthetic but well-formed features.csv content."""
    rng = np.random.default_rng(seed)
    index = pd.Index([f"city_{i:03d}" for i in range(n_instances)], name="instance")

    data = {}
    for position, name in enumerate(FEATURE_KEYS):
        if name in DYNAMIC_COLUMNS:
            data[name] = np.full(n_instances, 100.0 if "ratio" in name else 0.0)
        else:
            data[name] = rng.normal(loc=position + 1.0, scale=1.0 + position * 0.1, size=n_instances)
    return pd.DataFrame(data, index=index)


def _performance_frame(index: pd.Index, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 1.0, size=(len(index), 3))
    raw[:, 0] = np.clip(raw[:, 0] + 0.25, 0.0, 1.0)  # SA is usually strong
    frame = pd.DataFrame(raw, index=index, columns=["SA", "IP", "GR"])
    # Guarantee a per-row winner at exactly 1.0, as the T13 export produces.
    frame.iloc[:, 0] = np.where(frame.max(axis=1) == frame["SA"], 1.0, frame["SA"])
    return frame.clip(0.0, 1.0)


class PipelineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="isa_t14_"))
        self.results = self.tmp_dir / "data" / "results"
        self.figures = self.tmp_dir / "figures"
        self.results.mkdir(parents=True, exist_ok=True)
        self.features_path = self.results / "features.csv"
        self.perf_path = self.results / "perf_matrix.csv"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_inputs(self, n_instances: int = 40):
        features = _features_frame(n_instances)
        performance = _performance_frame(features.index)
        features.to_csv(self.features_path, index_label="instance")
        performance.to_csv(self.perf_path, index_label="instance")
        return features, performance

    def _run(self, **kwargs):
        defaults = dict(
            features_path=self.features_path,
            perf_path=self.perf_path,
            instance_space_path=self.results / "instance_space.csv",
            footprint_summary_path=self.results / "footprint_summary.csv",
            figures_dir=self.figures,
        )
        defaults.update(kwargs)
        return run_pipeline(**defaults)


class TestT14Preprocessing(PipelineTestCase):
    def test_drops_dynamic_high_nan_and_constant_columns(self) -> None:
        features = _features_frame(30)
        features["capacity_cv"] = 0.35              # constant
        features["coverage_ratio"] = np.nan         # all NaN
        features.iloc[:2, features.columns.get_loc("bsa_flexibility")] = np.nan  # 6.7% NaN

        clean, report = preprocess_features(features)

        self.assertEqual(report.dropped_dynamic, list(DYNAMIC_COLUMNS))
        self.assertIn("capacity_cv", report.dropped_constant)
        self.assertIn("coverage_ratio", [name for name, _ in report.dropped_high_nan])
        self.assertIn("bsa_flexibility", [name for name, _ in report.imputed])

        self.assertNotIn("capacity_cv", clean.columns)
        self.assertNotIn("coverage_ratio", clean.columns)
        self.assertIn("bsa_flexibility", clean.columns)
        self.assertFalse(clean.isna().any().any(), "no NaN may survive preprocessing")

    def test_median_imputation_uses_the_column_median(self) -> None:
        features = _features_frame(20)
        column = "od_distance_mean"
        features.iloc[0, features.columns.get_loc(column)] = np.nan
        # Median of the surviving values, not of the original column.
        expected = features[column].median()

        clean, report = preprocess_features(features)

        self.assertAlmostEqual(clean.iloc[0][column], expected)
        self.assertIn((column, 1), report.imputed)

    def test_nan_threshold_is_respected(self) -> None:
        features = _features_frame(20)
        column = "station_density"
        features.iloc[:5, features.columns.get_loc(column)] = np.nan  # 25% NaN

        clean_strict, _ = preprocess_features(features, nan_threshold=0.10)
        self.assertNotIn(column, clean_strict.columns)

        clean_loose, _ = preprocess_features(features, nan_threshold=0.30)
        self.assertIn(column, clean_loose.columns)

    def test_constant_check_runs_after_imputation(self) -> None:
        # Two distinct values, the rest NaN: imputation flattens it to constant.
        features = _features_frame(20)
        column = "tw_overlap_ratio"
        features[column] = np.nan
        features.iloc[0, features.columns.get_loc(column)] = 0.5
        features.iloc[1, features.columns.get_loc(column)] = 0.5

        clean, _ = preprocess_features(features, nan_threshold=0.95)
        self.assertNotIn(column, clean.columns)

    def test_keep_dynamic_option(self) -> None:
        features = _features_frame(20)
        features["avg_request_urgency"] = np.linspace(1.0, 5.0, 20)  # made non-constant

        clean, report = preprocess_features(features, drop_dynamic=False)
        self.assertEqual(report.dropped_dynamic, [])
        self.assertIn("avg_request_urgency", clean.columns)

    def test_invalid_threshold_rejected(self) -> None:
        with self.assertRaises(ValueError):
            preprocess_features(_features_frame(5), nan_threshold=1.5)


class TestT14Projection(PipelineTestCase):
    def test_pilot_returns_two_dimensions_and_explains_variance(self) -> None:
        features = _features_frame(50)
        clean, _ = preprocess_features(features)
        performance = _performance_frame(features.index)

        result = pilot_analytic(clean.to_numpy(), performance.to_numpy())

        self.assertEqual(result.coordinates.shape, (50, 2))
        self.assertEqual(result.feature_weights.shape, (clean.shape[1], 2))
        self.assertTrue(0.0 <= result.r2_features <= 1.0)
        self.assertTrue(0.0 <= result.r2_performance <= 1.0)
        self.assertEqual(result.backend, "pilot-analytic")

    def test_projection_is_deterministic(self) -> None:
        features = _features_frame(30)
        clean, _ = preprocess_features(features)
        performance = _performance_frame(features.index)

        first, _ = project(clean, performance)
        second, _ = project(clean, performance)

        pd.testing.assert_frame_equal(first, second)

    def test_projection_uses_performance_not_only_features(self) -> None:
        features = _features_frame(40)
        clean, _ = preprocess_features(features)

        base = _performance_frame(features.index, seed=1)
        other = _performance_frame(features.index, seed=99)

        first, _ = project(clean, base)
        second, _ = project(clean, other)

        # Same features, different performance -> different instance space.
        self.assertFalse(np.allclose(first.to_numpy(), second.to_numpy()))

    def test_too_few_features_or_instances_is_an_error(self) -> None:
        with self.assertRaises(PipelineError):
            pilot_analytic(np.ones((10, 1)), np.ones((10, 3)))

        with self.assertRaises(PipelineError):
            pilot_analytic(np.ones((1, 4)), np.ones((1, 3)))

    def test_package_backend_reports_absence_clearly(self) -> None:
        try:
            import instancespace  # noqa: F401
        except ImportError:
            pass
        else:
            self.skipTest("instancespace is importable here; nothing to report absent")

        features = _features_frame(20)
        clean, _ = preprocess_features(features)
        performance = _performance_frame(features.index)

        with self.assertRaises(PipelineError) as ctx:
            project(clean, performance, method="package")

        message = str(ctx.exception)
        self.assertIn("instancespace", message)
        self.assertIn("numpy<2", message)
        self.assertIn("--method pilot", message)


class TestT14PackageMetadata(PipelineTestCase):
    def test_metadata_frame_uses_the_package_column_prefixes(self) -> None:
        features = _features_frame(6)
        clean, _ = preprocess_features(features)
        performance = _performance_frame(features.index)

        frame = build_metadata_frame(clean, performance)

        self.assertEqual(frame.columns[0], "Instances")
        self.assertEqual(list(frame["Instances"]), list(features.index))
        for name in clean.columns:
            self.assertIn(f"feature_{name}", frame.columns)
        for name in performance.columns:
            self.assertIn(f"algo_{name}", frame.columns)
        self.assertEqual(len(frame), len(features))

    def test_metadata_frame_parses_back_through_the_package(self) -> None:
        try:
            from instancespace.data.metadata import Metadata
        except ImportError:
            self.skipTest("instancespace not installed in this interpreter")

        features = _features_frame(20)
        clean, _ = preprocess_features(features)
        performance = _performance_frame(features.index)

        metadata = Metadata.from_data_frame(build_metadata_frame(clean, performance))

        self.assertEqual(len(metadata.feature_names), clean.shape[1])
        self.assertEqual(len(metadata.algorithm_names), 3)
        self.assertEqual(metadata.features.shape, (20, clean.shape[1]))
        self.assertEqual(metadata.algorithms.shape, (20, 3))

    def test_package_projection_runs_when_installed(self) -> None:
        try:
            import instancespace  # noqa: F401
        except ImportError:
            self.skipTest("instancespace not installed in this interpreter")

        # SIFTED needs performance that actually depends on the features.
        rng = np.random.default_rng(7)
        n = 120
        index = pd.Index([f"synth_{i:04d}" for i in range(n)], name="instance")
        size = rng.uniform(300, 3000, n)
        cluster = rng.uniform(0.1, 0.95, n)

        features = pd.DataFrame(
            {
                "n_requests": size,
                "spatial_cluster_coeff": cluster,
                "od_distance_mean": rng.uniform(2.0, 12.0, n),
                "tw_tightness_factor": rng.uniform(1.1, 3.0, n),
                "coverage_ratio": rng.uniform(0.5, 0.98, n),
            },
            index=index,
        )
        performance = pd.DataFrame(
            {
                "SA": np.clip(1.0 - 0.02 * rng.random(n), 0, 1),
                "IP": np.clip(1.0 - (size / 3000) ** 2, 0, 1),
                "GR": np.clip(1.0 - 0.8 * cluster, 0, 1),
            },
            index=index,
        )

        coordinates, result = project(features, performance, method="package")

        self.assertEqual(coordinates.shape, (n, 2))
        self.assertEqual(list(coordinates.index), list(index))
        self.assertEqual(result.backend, "instancespace-pilot")
        self.assertFalse(np.isnan(coordinates.to_numpy()).any())


class TestT14Geometry(unittest.TestCase):
    def test_convex_hull_of_a_square_with_an_interior_point(self) -> None:
        hull = convex_hull([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])
        self.assertEqual(len(hull), 4)
        self.assertNotIn((0.5, 0.5), hull)
        self.assertAlmostEqual(polygon_area(hull), 1.0)

    def test_degenerate_hulls(self) -> None:
        self.assertEqual(convex_hull([]), [])
        self.assertEqual(convex_hull([(1, 1), (1, 1)]), [(1.0, 1.0)])
        self.assertAlmostEqual(polygon_area(convex_hull([(0, 0), (1, 1)])), 0.0)
        # Collinear points collapse to the extremes, area zero.
        self.assertAlmostEqual(polygon_area(convex_hull([(0, 0), (1, 1), (2, 2)])), 0.0)

    def test_point_in_polygon(self) -> None:
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self.assertTrue(point_in_polygon((0.5, 0.5), square))
        self.assertFalse(point_in_polygon((1.5, 0.5), square))
        self.assertFalse(point_in_polygon((-0.1, 0.5), square))
        # Vertices and edges count as inside.
        self.assertTrue(point_in_polygon((0, 0), square))
        self.assertTrue(point_in_polygon((0.5, 0.0), square))


class TestT14Footprints(PipelineTestCase):
    def test_footprint_metrics_are_consistent(self) -> None:
        coordinates = pd.DataFrame(
            {"z1": [0.0, 1.0, 1.0, 0.0, 5.0], "z2": [0.0, 0.0, 1.0, 1.0, 5.0]},
            index=[f"i{i}" for i in range(5)],
        )
        performance = pd.DataFrame(
            {
                "SA": [1.0, 1.0, 1.0, 1.0, 0.1],  # good on the unit square only
                "IP": [0.1, 0.1, 0.1, 0.1, 1.0],  # good on one point
                "GR": [0.0, 0.0, 0.0, 0.0, 0.0],  # never good
            },
            index=coordinates.index,
        )

        footprints = compute_footprints(coordinates, performance, threshold=0.8)

        sa = footprints["SA"]
        self.assertEqual(sa.n_good, 4)
        self.assertAlmostEqual(sa.area, 1.0)
        self.assertAlmostEqual(sa.purity, 1.0)
        self.assertAlmostEqual(sa.density, 4.0)

        ip = footprints["IP"]
        self.assertEqual(ip.n_good, 1)
        self.assertAlmostEqual(ip.area, 0.0)
        self.assertTrue(math.isnan(ip.density))

        gr = footprints["GR"]
        self.assertEqual(gr.n_good, 0)
        self.assertEqual(gr.hull, [])

    def test_purity_falls_when_bad_instances_sit_inside_the_hull(self) -> None:
        coordinates = pd.DataFrame(
            {"z1": [0.0, 2.0, 2.0, 0.0, 1.0], "z2": [0.0, 0.0, 2.0, 2.0, 1.0]},
            index=[f"i{i}" for i in range(5)],
        )
        performance = pd.DataFrame(
            {
                "SA": [1.0, 1.0, 1.0, 1.0, 0.0],  # the interior point is bad
                "IP": [0.0, 0.0, 0.0, 0.0, 1.0],
                "GR": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=coordinates.index,
        )

        footprints = compute_footprints(coordinates, performance, threshold=0.8)
        self.assertAlmostEqual(footprints["SA"].purity, 4.0 / 5.0)


class TestT14EndToEnd(PipelineTestCase):
    def test_full_run_writes_every_expected_file(self) -> None:
        self._write_inputs(40)
        outcome = self._run()

        space_csv = self.results / "instance_space.csv"
        summary_csv = self.results / "footprint_summary.csv"

        self.assertTrue(space_csv.is_file())
        self.assertTrue(summary_csv.is_file())
        self.assertTrue((self.figures / "instance_space.pdf").is_file())
        for algorithm in ("SA", "IP", "GR"):
            self.assertTrue((self.figures / f"footprint_{algorithm}.pdf").is_file())

        space = pd.read_csv(space_csv, index_col="instance")
        self.assertEqual(list(space.columns), ["z1", "z2", "SA", "IP", "GR", "best_algorithm"])
        self.assertEqual(len(space), 40)
        self.assertFalse(space[["z1", "z2"]].isna().any().any())

        summary = pd.read_csv(summary_csv)
        self.assertEqual(sorted(summary["algorithm"]), ["GR", "IP", "SA"])
        self.assertIn("purity", summary.columns)

        self.assertEqual(len(outcome["written"]), 6)

    def test_instance_space_index_matches_features(self) -> None:
        features, _ = self._write_inputs(25)
        self._run()

        space = pd.read_csv(self.results / "instance_space.csv", index_col="instance")
        self.assertEqual(list(space.index), list(features.index))

    def test_no_figures_option(self) -> None:
        self._write_inputs(20)
        outcome = self._run(make_figures=False)

        self.assertFalse(self.figures.exists())
        self.assertEqual(len(outcome["written"]), 2)

    def test_missing_input_is_actionable(self) -> None:
        with self.assertRaises(PipelineError) as ctx:
            load_inputs(self.features_path, self.perf_path)
        self.assertIn("T12", str(ctx.exception))

    def test_index_mismatch_is_strict_by_default(self) -> None:
        features, performance = self._write_inputs(10)
        performance.drop(index=performance.index[0]).to_csv(
            self.perf_path, index_label="instance"
        )

        with self.assertRaises(PipelineError) as ctx:
            load_inputs(self.features_path, self.perf_path)
        self.assertIn("not aligned", str(ctx.exception))

        # --no-strict keeps the intersection instead of failing.
        kept_features, kept_perf = load_inputs(
            self.features_path, self.perf_path, strict=False
        )
        self.assertEqual(len(kept_features), 9)
        self.assertEqual(list(kept_features.index), list(kept_perf.index))

    def test_out_of_range_scores_are_rejected(self) -> None:
        _, performance = self._write_inputs(10)
        performance.iloc[0, 0] = 1.4
        performance.to_csv(self.perf_path, index_label="instance")

        with self.assertRaises(PipelineError) as ctx:
            load_inputs(self.features_path, self.perf_path)
        self.assertIn("outside [0, 1]", str(ctx.exception))

    def test_too_few_usable_features_explains_what_to_do(self) -> None:
        features = _features_frame(10)
        for name in features.columns:
            features[name] = 1.0  # everything constant
        features.to_csv(self.features_path, index_label="instance")
        _performance_frame(features.index).to_csv(self.perf_path, index_label="instance")

        with self.assertRaises(PipelineError) as ctx:
            self._run()
        self.assertIn("survived preprocessing", str(ctx.exception))


class TestT14Cli(PipelineTestCase):
    def _cli_args(self, *extra: str):
        return [
            "--features", str(self.features_path),
            "--perf", str(self.perf_path),
            "--out", str(self.results / "instance_space.csv"),
            "--footprint-summary", str(self.results / "footprint_summary.csv"),
            "--figures", str(self.figures),
            *extra,
        ]

    def test_cli_end_to_end(self) -> None:
        self._write_inputs(30)
        self.assertEqual(main(self._cli_args()), 0)
        self.assertTrue((self.results / "instance_space.csv").is_file())
        self.assertTrue((self.figures / "instance_space.pdf").is_file())

    def test_cli_report_only_writes_nothing(self) -> None:
        self._write_inputs(20)
        self.assertEqual(main(self._cli_args("--report-only")), 0)
        self.assertFalse((self.results / "instance_space.csv").exists())
        self.assertFalse(self.figures.exists())

    def test_cli_returns_one_on_failure(self) -> None:
        self.assertEqual(main(self._cli_args()), 1)  # inputs do not exist

    def test_cli_keep_dynamic_and_thresholds(self) -> None:
        self._write_inputs(30)
        args = self._cli_args(
            "--keep-dynamic", "--nan-threshold", "0.5", "--footprint-threshold", "0.9"
        )
        self.assertEqual(main(args), 0)


class TestT14RealFeatures(PipelineTestCase):
    def test_runs_on_the_repository_feature_set_if_present(self) -> None:
        """Uses real features.csv when it exists; performance is placeholder."""
        real_features = REPO_ROOT / "data" / "results" / "features.csv"
        if not real_features.is_file():
            self.skipTest("no data/results/features.csv in the repository yet")

        features = pd.read_csv(real_features, index_col="instance")
        if len(features) < 3:
            self.skipTest("need at least 3 instances to project")

        features.to_csv(self.features_path, index_label="instance")
        _performance_frame(features.index).to_csv(self.perf_path, index_label="instance")

        outcome = self._run()
        self.assertEqual(len(outcome["coordinates"]), len(features))
        self.assertTrue((self.results / "instance_space.csv").is_file())


if __name__ == "__main__":
    unittest.main()
