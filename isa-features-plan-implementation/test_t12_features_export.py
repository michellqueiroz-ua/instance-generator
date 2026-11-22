import csv
import logging
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from compute_features_scaffold import FEATURE_KEYS, compute_features
from features_export import (
    FEATURE_CSV_COLUMNS,
    INSTANCE_COLUMN,
    append_features_row,
    append_features_rows,
    export_features_for_instance,
    instance_stem,
    normalize_feature_row,
    read_features_csv,
    validate_features_csv,
)
import reqreate_adapter


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
        "vehicles": [{"capacity": 8}, {"capacity": 12}],
    }


def _read_raw(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


class FeaturesExportTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="isa_t12_"))
        self.csv_path = self.tmp_dir / "data" / "results" / "features.csv"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


class TestT12SchemaAndKeys(FeaturesExportTestCase):
    def test_header_is_canonical_and_has_no_duplicate_columns(self) -> None:
        append_features_row("alpha_001.json", compute_features(_toy_instance()), self.csv_path)

        rows = _read_raw(self.csv_path)
        header = rows[0]

        self.assertEqual(header, FEATURE_CSV_COLUMNS)
        self.assertEqual(header[0], INSTANCE_COLUMN)
        self.assertEqual(len(header), len(set(header)), "duplicate column in header")
        self.assertEqual(len(header), len(FEATURE_KEYS) + 1)
        self.assertEqual(len(rows), 2)

    def test_instance_key_is_the_file_stem(self) -> None:
        self.assertEqual(instance_stem("antwerp_001.json"), "antwerp_001")
        self.assertEqual(instance_stem("antwerp_001.csv"), "antwerp_001")
        self.assertEqual(instance_stem("antwerp_001"), "antwerp_001")
        self.assertEqual(instance_stem("/a/b/antwerp_001.CSV"), "antwerp_001")
        self.assertEqual(instance_stem(r"C:\out\csv_format\city_ODBRP_13req_1.csv"), "city_ODBRP_13req_1")
        # Dots that are not a known extension are preserved.
        self.assertEqual(instance_stem("inst_v1.2"), "inst_v1.2")

        with self.assertRaises(ValueError):
            instance_stem("   ")

    def test_row_never_widens_or_narrows_the_schema(self) -> None:
        features = compute_features(_toy_instance())
        features["not_a_feature"] = 1.0
        del features["n_requests"]

        with self.assertLogs("features_export", level="WARNING"):
            row = normalize_feature_row("beta_001", features)

        self.assertEqual(sorted(row), sorted(FEATURE_CSV_COLUMNS))
        self.assertEqual(row["n_requests"], "")


class TestT12AppendBehaviour(FeaturesExportTestCase):
    def test_repeated_generation_appends_valid_rows(self) -> None:
        for i in range(5):
            export_features_for_instance(
                _toy_instance(n_requests=i + 1),
                f"city_{i:03d}.json",
                csv_path=self.csv_path,
            )

        rows = _read_raw(self.csv_path)
        self.assertEqual(len(rows), 6, "expected header + 5 rows")
        self.assertEqual(rows[0], FEATURE_CSV_COLUMNS)

        for row in rows[1:]:
            self.assertEqual(len(row), len(FEATURE_CSV_COLUMNS))
            self.assertTrue(row[0])
            for cell in row[1:]:
                if cell:
                    float(cell)  # every non-empty cell parses as a number

        self.assertEqual([row[0] for row in rows[1:]], [f"city_{i:03d}" for i in range(5)])
        self.assertEqual(validate_features_csv(self.csv_path), [])

    def test_regenerating_an_instance_replaces_its_row(self) -> None:
        append_features_row("city_001.json", compute_features(_toy_instance(2)), self.csv_path)
        append_features_row("city_002.json", compute_features(_toy_instance(2)), self.csv_path)
        append_features_row("city_001.csv", compute_features(_toy_instance(7)), self.csv_path)

        rows, report = read_features_csv(self.csv_path)

        self.assertEqual(list(rows), ["city_001", "city_002"])
        self.assertEqual(float(rows["city_001"]["n_requests"]), 7.0)
        self.assertEqual(report.malformed_rows, 0)
        self.assertEqual(validate_features_csv(self.csv_path), [])

    def test_on_duplicate_skip_and_error(self) -> None:
        append_features_row("city_001", compute_features(_toy_instance(2)), self.csv_path)

        append_features_row(
            "city_001",
            compute_features(_toy_instance(9)),
            self.csv_path,
            on_duplicate="skip",
        )
        rows, _ = read_features_csv(self.csv_path)
        self.assertEqual(float(rows["city_001"]["n_requests"]), 2.0)

        with self.assertRaises(ValueError):
            append_features_row(
                "city_001",
                compute_features(_toy_instance(9)),
                self.csv_path,
                on_duplicate="error",
            )

        with self.assertRaises(ValueError):
            append_features_row(
                "city_002",
                compute_features(_toy_instance(2)),
                self.csv_path,
                on_duplicate="nonsense",
            )

    def test_batch_append_matches_sequential_append(self) -> None:
        other_csv = self.tmp_dir / "batch" / "features.csv"

        items = [(f"city_{i:03d}", compute_features(_toy_instance(i + 1))) for i in range(3)]
        append_features_rows(items, csv_path=other_csv)
        for ref, features in items:
            append_features_row(ref, features, self.csv_path)

        self.assertEqual(_read_raw(other_csv), _read_raw(self.csv_path))

    def test_nan_is_written_as_empty_cell(self) -> None:
        # No stops → spatial features are NaN by design.
        instance = _toy_instance()
        features = compute_features(instance)
        self.assertTrue(math.isnan(features["station_density"]))

        append_features_row("city_042", features, self.csv_path)

        rows = _read_raw(self.csv_path)
        cell = rows[1][FEATURE_CSV_COLUMNS.index("station_density")]
        self.assertEqual(cell, "")
        # No literal "nan"/"inf" text anywhere in the row.
        self.assertNotIn("nan", ",".join(rows[1]).lower())
        self.assertNotIn("inf", ",".join(rows[1]).lower())

    def test_non_finite_and_non_numeric_values_become_empty(self) -> None:
        features = compute_features(_toy_instance())
        features["demand_density"] = float("inf")
        features["capacity_cv"] = "not a number"
        features["coverage_ratio"] = None

        append_features_row("city_weird", features, self.csv_path)

        rows, report = read_features_csv(self.csv_path)
        row = rows["city_weird"]
        self.assertEqual(row["demand_density"], "")
        self.assertEqual(row["capacity_cv"], "")
        self.assertEqual(row["coverage_ratio"], "")
        self.assertEqual(report.malformed_rows, 0)


class TestT12Protections(FeaturesExportTestCase):
    def _write_raw(self, lines) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def test_malformed_rows_are_dropped_on_next_append(self) -> None:
        good = ",".join(["good_001"] + ["1.0"] * len(FEATURE_KEYS))
        short = "short_row,1.0,2.0"
        long_row = ",".join(["long_001"] + ["1.0"] * (len(FEATURE_KEYS) + 3))
        keyless = ",".join([""] + ["1.0"] * len(FEATURE_KEYS))

        self._write_raw([",".join(FEATURE_CSV_COLUMNS), good, short, long_row, keyless])

        _, report = read_features_csv(self.csv_path)
        self.assertEqual(report.malformed_rows, 3)

        append_features_row("new_001", compute_features(_toy_instance()), self.csv_path)

        rows = _read_raw(self.csv_path)
        self.assertEqual([row[0] for row in rows[1:]], ["good_001", "new_001"])
        for row in rows:
            self.assertEqual(len(row), len(FEATURE_CSV_COLUMNS))
        self.assertEqual(validate_features_csv(self.csv_path), [])

    def test_duplicate_columns_are_collapsed(self) -> None:
        header = FEATURE_CSV_COLUMNS + ["n_requests", "coverage_ratio"]
        row = ["dup_001"] + ["1.0"] * len(FEATURE_KEYS) + ["999.0", "999.0"]
        self._write_raw([",".join(header), ",".join(row)])

        _, report = read_features_csv(self.csv_path)
        self.assertEqual(sorted(report.duplicate_columns), ["coverage_ratio", "n_requests"])

        append_features_row("dup_002", compute_features(_toy_instance()), self.csv_path)

        rows = _read_raw(self.csv_path)
        self.assertEqual(rows[0], FEATURE_CSV_COLUMNS)
        self.assertEqual(len(rows[0]), len(set(rows[0])))
        # First occurrence wins, so the shadowing duplicate value is discarded.
        self.assertEqual(rows[1][FEATURE_CSV_COLUMNS.index("n_requests")], "1.0")
        self.assertEqual(validate_features_csv(self.csv_path), [])

    def test_reordered_and_unknown_columns_are_migrated(self) -> None:
        header = [INSTANCE_COLUMN, "coverage_ratio", "n_requests", "legacy_score"]
        self._write_raw([",".join(header), "old_001,0.5,42,7.5"])

        append_features_row("new_001", compute_features(_toy_instance()), self.csv_path)

        rows, report = read_features_csv(self.csv_path)
        self.assertEqual(report.unknown_columns, [])
        self.assertEqual(list(rows), ["old_001", "new_001"])
        self.assertEqual(float(rows["old_001"]["n_requests"]), 42.0)
        self.assertEqual(float(rows["old_001"]["coverage_ratio"]), 0.5)
        self.assertEqual(rows["old_001"]["tw_width_mean"], "")
        self.assertEqual(_read_raw(self.csv_path)[0], FEATURE_CSV_COLUMNS)

    def test_duplicate_instance_rows_are_collapsed_to_one(self) -> None:
        first = ",".join(["dup_001"] + ["1.0"] * len(FEATURE_KEYS))
        second = ",".join(["dup_001"] + ["2.0"] * len(FEATURE_KEYS))
        self._write_raw([",".join(FEATURE_CSV_COLUMNS), first, second])

        rows, report = read_features_csv(self.csv_path)
        self.assertEqual(report.duplicate_keys, ["dup_001"])
        self.assertEqual(float(rows["dup_001"]["n_requests"]), 2.0, "last row wins")

        append_features_row("dup_002", compute_features(_toy_instance()), self.csv_path)
        self.assertEqual(validate_features_csv(self.csv_path), [])
        self.assertEqual(len(_read_raw(self.csv_path)), 3)

    def test_missing_file_reports_and_creates_cleanly(self) -> None:
        self.assertEqual(len(validate_features_csv(self.csv_path)), 1)
        self.assertFalse(self.csv_path.exists())

        append_features_row("first_001", compute_features(_toy_instance()), self.csv_path)

        self.assertTrue(self.csv_path.is_file())
        self.assertEqual(validate_features_csv(self.csv_path), [])

    def test_failed_write_leaves_no_temp_files(self) -> None:
        append_features_row("keep_001", compute_features(_toy_instance()), self.csv_path)

        with self.assertRaises(TypeError):
            append_features_row("bad_002", ["not", "a", "dict"], self.csv_path)

        leftovers = list(self.csv_path.parent.glob("*.tmp"))
        self.assertEqual(leftovers, [])
        self.assertEqual(validate_features_csv(self.csv_path), [])


class TestT12ReqreateAdapter(FeaturesExportTestCase):
    def test_converts_reqreate_units_and_keys(self) -> None:
        raw = {
            "requests": {
                "0": {
                    "originx": 5.6858,
                    "originy": 50.8357,
                    "destinationx": 5.6663,
                    "destinationy": 50.8480,
                    "originnode_drive": 2503257628,
                    "destinationnode_drive": 10781536345,
                    "time_stamp": 25200,
                    "earliest_departure": 25200,
                    "latest_departure": 25340,
                    "latest_arrival": 26350,
                    "direct_travel_time": 550,
                    "direct_distance": 3112.5,
                    "walk_speed": 1.3888888888888888,
                    "stops_orgn": [1],
                    "stops_dest": [2],
                    "stops_orgn_walking_distance": [66],
                    "stops_dest_walking_distance": [164],
                }
            },
            "travel_time_matrix": [
                [0.0, 300.0, 600.0],
                [300.0, 0.0, 420.0],
                [600.0, 420.0, 0.0],
            ],
        }

        instance = reqreate_adapter.build_instance_dict(raw)
        request = instance["requests"][0]

        self.assertAlmostEqual(request["origin"]["x"], 5.6858)
        self.assertAlmostEqual(request["origin"]["y"], 50.8357)
        self.assertAlmostEqual(request["tw_start"], 420.0)
        self.assertAlmostEqual(request["tw_end"], 25340 / 60.0)
        self.assertAlmostEqual(request["issue_time"], 420.0)
        self.assertEqual(request["origin_idx"], 1)
        self.assertEqual(request["destination_idx"], 2)
        # OSM node ids must not leak through: they would break matrix lookups.
        self.assertNotIn("originnode_drive", request)
        self.assertIn("travel_time_matrix_sec", instance)

        features = compute_features(instance)
        # tw_width 2.33 min over a 7 min stop-to-stop trip.
        self.assertAlmostEqual(features["tw_tightness_factor"], (140 / 60.0) / 7.0, places=9)
        self.assertAlmostEqual(features["od_distance_mean"], 3.1125, places=6)

    def test_darp_field_names_map_to_the_same_features(self) -> None:
        raw = {
            "requests": {
                "0": {
                    "originx": 5.6858,
                    "originy": 50.8357,
                    "destinationx": 5.6663,
                    "destinationy": 50.8480,
                    "time_stamp": 25200,
                    "pickup_from": 25200,
                    "pickup_to": 25560,
                    "dropoff_from": 26097,
                    "dropoff_to": 26697,
                    "drivingDistance": 2922.333248499852,
                    "drivingDuration": 537,
                }
            }
        }

        instance = reqreate_adapter.build_instance_dict(raw)
        request = instance["requests"][0]

        self.assertAlmostEqual(request["tw_start"], 420.0)
        self.assertAlmostEqual(request["tw_end"], 426.0)
        self.assertAlmostEqual(request["direct_distance"], 2922.333248499852)
        self.assertAlmostEqual(request["direct_travel_time"], 537.0)

        features = compute_features(instance)
        self.assertAlmostEqual(features["tw_width_mean"], 6.0, places=9)
        self.assertAlmostEqual(features["od_distance_mean"], 2.922333248499852, places=9)
        self.assertAlmostEqual(features["tw_tightness_factor"], 6.0 / (537 / 60.0), places=9)

    def test_endpoint_pair_list_fallback_is_lat_lon(self) -> None:
        raw = {
            "requests": [
                {
                    "origin": [50.8357, 5.6858],
                    "destination": [50.8480, 5.6663],
                    "earliest_departure": 25200,
                    "latest_departure": 25340,
                }
            ]
        }
        request = reqreate_adapter.build_instance_dict(raw)["requests"][0]
        self.assertAlmostEqual(request["origin"]["x"], 5.6858)
        self.assertAlmostEqual(request["origin"]["y"], 50.8357)

    def test_service_area_from_points_is_metric(self) -> None:
        # ~1 km x ~1 km box near Maastricht.
        points = [
            (5.6858, 50.8357),
            (5.6858 + 0.014222, 50.8357),
            (5.6858 + 0.014222, 50.8357 + 0.009043),
            (5.6858, 50.8357 + 0.009043),
        ]
        area = reqreate_adapter.service_area_km2_from_points(points)
        self.assertAlmostEqual(area, 1.0, places=1)

    def test_end_to_end_on_generated_reqreate_instance(self) -> None:
        repo_root = THIS_DIR.parent
        candidates = sorted(repo_root.glob("*/json_format/*.json"))
        if not candidates:
            self.skipTest("no generated REQreate instance JSON available")

        json_path = candidates[0]
        instance = reqreate_adapter.load_reqreate_instance(json_path)

        self.assertGreater(len(instance["requests"]), 0)

        features = compute_features(instance)
        self.assertEqual(list(features), FEATURE_KEYS)

        append_features_row(json_path.name, features, self.csv_path)
        append_features_row(json_path.name, features, self.csv_path)

        rows = _read_raw(self.csv_path)
        self.assertEqual(len(rows), 2, "re-export must not duplicate the row")
        self.assertEqual(rows[1][0], json_path.stem)
        self.assertEqual(validate_features_csv(self.csv_path), [])

        # n_requests must always be populated on a real instance.
        self.assertNotEqual(rows[1][FEATURE_CSV_COLUMNS.index("n_requests")], "")


class TestT12PipelineHook(FeaturesExportTestCase):
    def test_hook_writes_a_row_and_never_raises(self) -> None:
        repo_root = THIS_DIR.parent
        reqreate_dir = repo_root / "REQreate"
        if str(reqreate_dir) not in sys.path:
            sys.path.insert(0, str(reqreate_dir))

        from isa_features_hook import export_generated_instance_features

        candidates = sorted(repo_root.glob("*/json_format/*.json"))
        if not candidates:
            self.skipTest("no generated REQreate instance JSON available")

        ok = export_generated_instance_features(
            json_path=str(candidates[0]),
            instance_ref="hooked_instance_1.csv",
            csv_path=str(self.csv_path),
        )
        self.assertTrue(ok)

        rows, _ = read_features_csv(self.csv_path)
        self.assertEqual(list(rows), ["hooked_instance_1"])

        # A broken input is swallowed, not raised, so generation keeps running.
        logging.disable(logging.CRITICAL)
        try:
            self.assertFalse(
                export_generated_instance_features(
                    json_path=str(self.tmp_dir / "does_not_exist.json"),
                    instance_ref="missing_1.csv",
                    csv_path=str(self.csv_path),
                )
            )
        finally:
            logging.disable(logging.NOTSET)

        rows_after, _ = read_features_csv(self.csv_path)
        self.assertEqual(list(rows_after), ["hooked_instance_1"])


if __name__ == "__main__":
    unittest.main()
