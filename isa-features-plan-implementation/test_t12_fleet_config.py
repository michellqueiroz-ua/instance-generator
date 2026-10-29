import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from compute_features_scaffold import compute_features
from fleet_config import (
    DEFAULT_FLEET_CONFIG_PATH,
    detect_problem_type,
    fleet_from_solver_config,
    load_fleet_config,
    resolve_fleet,
    resolve_fleet_spec,
    vehicles_from_spec,
)


def _instance_with(n_requests: int, vehicles) -> dict:
    return {
        "service_area_km2": 100.0,
        "requests": [
            {
                "origin": {"x": 4.40, "y": 51.21},
                "destination": {"x": 4.42, "y": 51.22},
                "tw_start": 480,
                "tw_end": 510,
            }
            for _ in range(n_requests)
        ],
        "vehicles": vehicles,
    }


# Fleet FF from the Heterogeneous ODBRP paper, Table 13.
FF_SPEC = {
    "fleet_percent_of_requests": 20,
    "composition": [
        {"capacity": 4, "percent": 60},
        {"capacity": 8, "percent": 40},
    ],
}


class TestCompositionSpec(unittest.TestCase):
    def test_paper_fleet_ff_reproduces_table_13(self) -> None:
        # |R| = 300 -> |B| = 20% = 60 vehicles: 36 of capacity 4, 24 of capacity 8.
        fleet = vehicles_from_spec(FF_SPEC, n_requests=300)
        capacities = [v["capacity"] for v in fleet]

        self.assertEqual(len(fleet), 60)
        self.assertEqual(capacities.count(4.0), 36)
        self.assertEqual(capacities.count(8.0), 24)

    def test_fleet_scales_with_request_count(self) -> None:
        sizes = [
            len(vehicles_from_spec(FF_SPEC, n_requests=n)) for n in (300, 1500, 3000)
        ]
        self.assertEqual(sizes, [60, 300, 600])

    def test_three_type_composition_f2(self) -> None:
        f2 = {
            "fleet_percent_of_requests": 12,
            "composition": [
                {"capacity": 4, "percent": 40},
                {"capacity": 8, "percent": 45},
                {"capacity": 12, "percent": 15},
            ],
        }
        fleet = vehicles_from_spec(f2, n_requests=1200)
        capacities = [v["capacity"] for v in fleet]

        self.assertEqual(len(fleet), 144)  # 12% of 1200
        self.assertEqual(capacities.count(4.0), 58)   # 40% of 144 = 57.6
        self.assertEqual(capacities.count(8.0), 65)   # 45% of 144 = 64.8
        self.assertEqual(capacities.count(12.0), 21)  # 15% of 144 = 21.6

    def test_parts_always_sum_to_the_fleet_size(self) -> None:
        for n_requests in range(1, 400):
            fleet = vehicles_from_spec(FF_SPEC, n_requests=n_requests)
            expected = max(1, round(n_requests * 0.20))
            self.assertEqual(len(fleet), expected, n_requests)

    def test_small_instances_still_get_a_fleet(self) -> None:
        fleet = vehicles_from_spec(FF_SPEC, n_requests=13)
        self.assertEqual(len(fleet), 3)  # round(2.6)
        self.assertEqual(sorted(v["capacity"] for v in fleet), [4.0, 4.0, 8.0])

        # Never zero vehicles, even for a single request.
        self.assertEqual(len(vehicles_from_spec(FF_SPEC, n_requests=1)), 1)

    def test_composition_errors_are_actionable(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            vehicles_from_spec(FF_SPEC)
        self.assertIn("n_requests", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            vehicles_from_spec({"fleet_percent_of_requests": 20}, n_requests=100)
        self.assertIn("composition", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            vehicles_from_spec(
                {
                    "fleet_percent_of_requests": 20,
                    "composition": [
                        {"capacity": 4, "percent": 60},
                        {"capacity": 8, "percent": 60},
                    ],
                },
                n_requests=100,
            )
        self.assertIn("sum to 120", str(ctx.exception))

    def test_capacity_cv_is_non_zero_and_stable(self) -> None:
        features = compute_features(
            _instance_with(300, vehicles_from_spec(FF_SPEC, n_requests=300))
        )
        # 36 x cap 4 + 24 x cap 8 -> mean 5.6, population std 1.959...
        self.assertAlmostEqual(features["fleet_size"], 60.0)
        self.assertAlmostEqual(features["capacity_cv"], 1.9595917942265424 / 5.6, places=9)
        self.assertAlmostEqual(features["request_vehicle_ratio"], 5.0)


class TestFleetSpecShapes(unittest.TestCase):
    def test_flat_spec(self) -> None:
        fleet = vehicles_from_spec({"number_vehicles": 5, "capacity": 4})
        self.assertEqual(len(fleet), 5)
        self.assertEqual({v["capacity"] for v in fleet}, {4.0})

    def test_solver_spec_with_mixed_capacities(self) -> None:
        fleet = vehicles_from_spec(
            {
                "type_vehicles": 2,
                "number_vehicles1": 5,
                "number_vehicles2": 3,
                "number_vehicles3": 0,
                "capacity_vehicles": [4, 8],
            }
        )
        self.assertEqual(len(fleet), 8)
        capacities = sorted(v["capacity"] for v in fleet)
        self.assertEqual(capacities.count(4.0), 5)
        self.assertEqual(capacities.count(8.0), 3)

    def test_demand_scaled_spec(self) -> None:
        fleet = vehicles_from_spec(
            {"requests_per_vehicle": 10, "capacity": 4}, n_requests=45
        )
        self.assertEqual(len(fleet), 5)  # ceil(45 / 10)

        self.assertEqual(
            len(vehicles_from_spec({"requests_per_vehicle": 10, "capacity": 4}, 100)), 10
        )

        with self.assertRaises(ValueError) as ctx:
            vehicles_from_spec({"requests_per_vehicle": 10, "capacity": 4})
        self.assertIn("n_requests", str(ctx.exception))

    def test_empty_spec_means_no_fleet(self) -> None:
        self.assertEqual(vehicles_from_spec({}), [])
        self.assertEqual(vehicles_from_spec(None), [])

    def test_bad_specs_are_actionable_errors(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            vehicles_from_spec({"number_vehicles1": 5})
        self.assertIn("capacity_vehicles", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            vehicles_from_spec({"number_vehicles": 5, "capacity": 0})
        self.assertIn("capacity", str(ctx.exception))

        with self.assertRaises(ValueError):
            vehicles_from_spec({"number_vehicles": "many", "capacity": 4})

        with self.assertRaises(ValueError) as ctx:
            vehicles_from_spec({"number_vehicles": 10**9, "capacity": 4})
        self.assertIn("plausible", str(ctx.exception))

    def test_from_solver_config_object(self) -> None:
        class FakeSolverConfig:
            type_vehicles = 1
            number_vehicles1 = 5
            number_vehicles2 = 0
            number_vehicles3 = 0
            capacity_vehicles = [4]

        fleet = fleet_from_solver_config(FakeSolverConfig())
        self.assertEqual(len(fleet), 5)
        self.assertEqual(fleet[0]["capacity"], 4.0)


class TestFleetResolution(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="isa_fleet_"))
        self.config_path = self.tmp_dir / "fleet_config.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_config(self, config: dict) -> dict:
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        return load_fleet_config(self.config_path)

    def test_problem_type_detection_from_instance_name(self) -> None:
        self.assertEqual(
            detect_problem_type("Maastricht_Netherlands_DARP_17req_1"), "DARP"
        )
        self.assertEqual(
            detect_problem_type("Copenhagen_Municipality_Denmark_ODBRP_10req_1"), "ODBRP"
        )
        self.assertEqual(
            detect_problem_type("Maastricht_Netherlands_Patient Transport_8req_1"),
            "Patient Transport",
        )
        self.assertIsNone(detect_problem_type("some_other_instance_1"))

    def test_most_specific_match_wins(self) -> None:
        config = self._write_config(
            {
                "default": {"number_vehicles": 5, "capacity": 4},
                "by_problem_type": {"DARP": {"number_vehicles": 8, "capacity": 6}},
                "by_instance": {"city_special_1": {"number_vehicles": 2, "capacity": 20}},
            }
        )

        self.assertEqual(len(resolve_fleet("city_special_1", config=config)), 2)
        self.assertEqual(
            len(resolve_fleet("Maastricht_Netherlands_DARP_17req_1", config=config)), 8
        )
        self.assertEqual(len(resolve_fleet("anything_else", config=config)), 5)

    def test_missing_config_leaves_fleet_empty(self) -> None:
        config = load_fleet_config(self.tmp_dir / "does_not_exist.json")
        self.assertEqual(config, {})
        self.assertEqual(resolve_fleet("city_001", config=config), [])
        self.assertEqual(resolve_fleet_spec("city_001", config=config), {})

    def test_instance_ref_may_be_a_file_name(self) -> None:
        config = self._write_config({"default": {"number_vehicles": 3, "capacity": 4}})
        self.assertEqual(len(resolve_fleet("city_001.csv", config=config)), 3)


class TestFleetFeatureEffect(unittest.TestCase):
    def test_fleet_turns_nan_features_into_numbers(self) -> None:
        without = compute_features(_instance_with(20, []))
        self.assertTrue(math.isnan(without["capacity_cv"]))
        self.assertTrue(math.isnan(without["request_vehicle_ratio"]))

        fleet = vehicles_from_spec({"number_vehicles": 5, "capacity": 4})
        with_fleet = compute_features(_instance_with(20, fleet))

        self.assertEqual(with_fleet["fleet_size"], 5.0)
        self.assertAlmostEqual(with_fleet["capacity_cv"], 0.0)
        self.assertAlmostEqual(with_fleet["request_vehicle_ratio"], 4.0)
        self.assertAlmostEqual(with_fleet["fleet_utilization_ratio"], 1.0)

    def test_mixed_capacities_give_non_zero_capacity_cv(self) -> None:
        fleet = vehicles_from_spec(
            {
                "type_vehicles": 2,
                "number_vehicles1": 5,
                "number_vehicles2": 3,
                "capacity_vehicles": [4, 8],
            }
        )
        features = compute_features(_instance_with(20, fleet))
        self.assertGreater(features["capacity_cv"], 0.0)

    def test_demand_scaled_fleet_varies_across_instances(self) -> None:
        spec = {"requests_per_vehicle": 10, "capacity": 4}
        sizes = [
            compute_features(
                _instance_with(n, vehicles_from_spec(spec, n_requests=n))
            )["fleet_size"]
            for n in (10, 50, 100)
        ]
        self.assertEqual(sizes, [1.0, 5.0, 10.0])
        # A constant fleet would make this column zero-variance and MATILDA
        # would drop it.
        self.assertGreater(max(sizes) - min(sizes), 0.0)


class TestShippedFleetConfig(unittest.TestCase):
    def test_repo_config_is_the_paper_fleet_ff(self) -> None:
        self.assertTrue(
            DEFAULT_FLEET_CONFIG_PATH.is_file(), DEFAULT_FLEET_CONFIG_PATH
        )

        config = load_fleet_config(DEFAULT_FLEET_CONFIG_PATH)
        fleet = resolve_fleet(
            "Maastricht_Netherlands_ODBRP_300req_1", n_requests=300, config=config
        )

        # Table 13: |B| = 20% of |R|, 60% capacity 4 and 40% capacity 8.
        capacities = [v["capacity"] for v in fleet]
        self.assertEqual(len(fleet), 60)
        self.assertEqual(capacities.count(4.0), 36)
        self.assertEqual(capacities.count(8.0), 24)

    def test_every_shipped_preset_expands(self) -> None:
        import json

        raw = json.loads(DEFAULT_FLEET_CONFIG_PATH.read_text(encoding="utf-8"))
        presets = raw.get("_presets") or {}
        self.assertTrue(presets)

        for name, spec in presets.items():
            clean = {k: v for k, v in spec.items() if not k.startswith("_")}
            fleet = vehicles_from_spec(clean, n_requests=1200)
            self.assertGreater(len(fleet), 0, name)

    def test_hook_uses_the_config_when_no_vehicles_are_passed(self) -> None:
        repo_root = THIS_DIR.parent
        reqreate_dir = repo_root / "REQreate"
        if str(reqreate_dir) not in sys.path:
            sys.path.insert(0, str(reqreate_dir))

        from isa_features_hook import export_generated_instance_features
        from features_export import read_features_csv

        candidates = sorted(repo_root.glob("*/json_format/*.json"))
        if not candidates:
            self.skipTest("no generated REQreate instance JSON available")

        tmp_dir = Path(tempfile.mkdtemp(prefix="isa_fleet_hook_"))
        try:
            csv_path = tmp_dir / "features.csv"
            self.assertTrue(
                export_generated_instance_features(
                    json_path=str(candidates[0]),
                    instance_ref="fleet_hooked_1.csv",
                    csv_path=str(csv_path),
                )
            )

            rows, _ = read_features_csv(csv_path)
            row = rows["fleet_hooked_1"]

            n_requests = float(row["n_requests"])
            fleet_size = float(row["fleet_size"])

            # FF sizes the fleet at 20% of demand, so |B| tracks |R|.
            self.assertAlmostEqual(fleet_size, max(1, round(n_requests * 0.20)))
            # Mixed capacities -> capacity_cv is a real number, not NaN or 0.
            self.assertGreater(float(row["capacity_cv"]), 0.0)
            self.assertAlmostEqual(float(row["request_vehicle_ratio"]), 5.0, places=1)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
