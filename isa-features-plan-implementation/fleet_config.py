"""
fleet_config.py — fleet inputs for features 11-14 (ISA-HODBRP)

REQreate instances describe demand only: there is no fleet anywhere in the
generated JSON/CSV.  The fleet is a solver input (``solver_utils.SolverConfig``:
``type_vehicles``, ``number_vehicles1..3``, ``capacity_vehicles``), so without
it ``capacity_cv`` and ``request_vehicle_ratio`` are NaN on every instance and
MATILDA drops them.

This module resolves the fleet used for the experiment from a small JSON file
(``fleet_config.json`` next to this module, or ``$ISA_FLEET_CONFIG``) and
expands it into the ``vehicles`` list the feature code expects.

Config layout — every section optional, most specific match wins::

    {
      "default":          {"number_vehicles": 5, "capacity": 4},
      "by_problem_type":  {"DARP": {"number_vehicles": 8, "capacity": 6}},
      "by_instance":      {"Maastricht_Netherlands_ODBRP_13req_1": {...}}
    }

A fleet spec accepts four interchangeable shapes:

  ``{"fleet_percent_of_requests": 20,
     "composition": [{"capacity": 4, "percent": 60},
                     {"capacity": 8, "percent": 40}]}``
      Heterogeneous fleet sized as a percentage of demand — the shape used in
      Melis & Queiroz, *Heterogeneous ODBRP*, Tables 9 and 13.  ``|B|`` is
      ``round(percent/100 * n_requests)`` and the type split uses the
      largest-remainder method so the parts always sum to ``|B|`` exactly.
      This is the only shape that varies both ``fleet_size`` and
      ``capacity_cv`` meaningfully, so it is what the shipped config uses.

  ``{"number_vehicles": 5, "capacity": 4}``
      Flat fleet: 5 vehicles of capacity 4.

  ``{"type_vehicles": 2, "number_vehicles1": 5, "number_vehicles2": 3,
     "capacity_vehicles": [4, 8]}``
      The solver's own shape, copied straight from a SolverConfig.

  ``{"requests_per_vehicle": 10, "capacity": 4}``
      Homogeneous fleet that scales with demand: ``ceil(n_requests / 10)``
      vehicles.

A fleet that is constant across the whole instance set makes ``fleet_size``
and ``capacity_cv`` zero-variance columns, which MATILDA discards during
preprocessing — prefer a demand-scaled shape.
"""

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


LOGGER = logging.getLogger(__name__)

ENV_FLEET_CONFIG = "ISA_FLEET_CONFIG"
DEFAULT_FLEET_CONFIG_PATH = _THIS_DIR / "fleet_config.json"

# Problem types REQreate encodes in the instance file name.
PROBLEM_TYPES = ("ODBRP", "DARP", "Patient Transport", "ODBRPFL", "SBRP")

_MAX_VEHICLES = 100_000


def resolve_fleet_config_path(path: Optional[Any] = None) -> Path:
    """
    Resolve the fleet config file.

    Precedence: explicit argument > ``ISA_FLEET_CONFIG`` env var >
    ``fleet_config.json`` next to this module.
    """
    if path is not None:
        return Path(str(path)).expanduser()

    from_env = os.environ.get(ENV_FLEET_CONFIG)
    if from_env and from_env.strip():
        return Path(from_env.strip()).expanduser()

    return DEFAULT_FLEET_CONFIG_PATH


def load_fleet_config(path: Optional[Any] = None) -> Dict[str, Any]:
    """
    Load the fleet config, or return an empty config when the file is absent.

    A missing config is not an error: fleet features simply stay NaN, exactly
    as before this module existed.
    """
    config_path = resolve_fleet_config_path(path)
    if not config_path.is_file():
        LOGGER.info(
            "no fleet config at %s; fleet features (capacity_cv, "
            "request_vehicle_ratio) will be NaN",
            config_path,
        )
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"fleet config must be a JSON object: {config_path}")

    return config


def detect_problem_type(instance_ref: Any) -> Optional[str]:
    """
    Read the problem type out of a REQreate instance name.

    ``Maastricht_Netherlands_DARP_17req_1`` -> ``"DARP"``.  Returns None when
    no known problem type appears in the name.
    """
    text = str(instance_ref)
    # Longest first so ODBRPFL is not swallowed by ODBRP.
    for problem_type in sorted(PROBLEM_TYPES, key=len, reverse=True):
        needle = problem_type.replace(" ", "")
        if needle.lower() in text.replace(" ", "").replace("_", "").lower():
            return problem_type
    return None


def _positive_int(value: Any, field: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"fleet spec field {field!r} must be an integer, got {value!r}")
    if number < 0:
        raise ValueError(f"fleet spec field {field!r} must be >= 0, got {number}")
    return number


def _positive_float(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"fleet spec field {field!r} must be a number, got {value!r}")
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"fleet spec field {field!r} must be > 0, got {number}")
    return number


def _split_by_percent(total: int, percents: Sequence[float]) -> List[int]:
    """
    Split ``total`` vehicles across types by percentage.

    Uses the largest-remainder method so the parts sum to ``total`` exactly,
    instead of drifting by a vehicle or two through independent rounding.
    """
    exact = [total * (percent / 100.0) for percent in percents]
    counts = [int(math.floor(value)) for value in exact]

    remaining = total - sum(counts)
    if remaining > 0:
        order = sorted(
            range(len(exact)), key=lambda i: (exact[i] - counts[i], exact[i]), reverse=True
        )
        for position in range(remaining):
            counts[order[position % len(order)]] += 1

    return counts


def _vehicles_from_composition(
    spec: Dict[str, Any], n_requests: Optional[int]
) -> List[Dict[str, float]]:
    """Expand the percentage-composition shape (paper Tables 9 and 13)."""
    composition = spec.get("composition")
    if not isinstance(composition, (list, tuple)) or not composition:
        raise ValueError(
            "fleet spec with 'fleet_percent_of_requests' also needs a non-empty "
            "'composition' list of {capacity, percent} entries"
        )

    if n_requests is None:
        raise ValueError(
            "fleet spec is sized as a percentage of requests but the instance "
            "request count is unknown; pass n_requests"
        )

    fleet_percent = _positive_float(
        spec.get("fleet_percent_of_requests"), "fleet_percent_of_requests"
    )

    capacities: List[float] = []
    percents: List[float] = []
    for index, entry in enumerate(composition):
        if not isinstance(entry, dict):
            raise ValueError(
                f"composition[{index}] must be an object with 'capacity' and 'percent'"
            )
        percent = float(entry.get("percent", 0.0))
        if percent <= 0.0:
            continue
        capacities.append(_positive_float(entry.get("capacity"), f"composition[{index}].capacity"))
        percents.append(percent)

    if not capacities:
        raise ValueError("fleet composition has no vehicle type with percent > 0")

    total_percent = sum(percents)
    if abs(total_percent - 100.0) > 1.0:
        raise ValueError(
            f"fleet composition percentages sum to {total_percent:g}%, expected 100%"
        )

    total = max(1, int(round(float(n_requests) * fleet_percent / 100.0)))
    if total > _MAX_VEHICLES:
        raise ValueError(f"fleet spec expands to {total} vehicles; that is not plausible")

    vehicles: List[Dict[str, float]] = []
    for capacity, count in zip(capacities, _split_by_percent(total, percents)):
        vehicles.extend({"capacity": capacity} for _ in range(count))

    return vehicles


def vehicles_from_spec(
    spec: Dict[str, Any], n_requests: Optional[int] = None
) -> List[Dict[str, float]]:
    """
    Expand a fleet spec into ``[{"capacity": c}, ...]``.

    See the module docstring for the four accepted shapes.  Returns an empty
    list for an empty spec, which leaves fleet features NaN.
    """
    if not spec:
        return []
    if not isinstance(spec, dict):
        raise ValueError(f"fleet spec must be an object, got {type(spec).__name__}")

    if "fleet_percent_of_requests" in spec or "composition" in spec:
        return _vehicles_from_composition(spec, n_requests)

    # Solver shape: per-type counts with a capacity list.
    if any(key.startswith("number_vehicles") and key != "number_vehicles" for key in spec):
        capacities = spec.get("capacity_vehicles")
        if not isinstance(capacities, (list, tuple)) or not capacities:
            raise ValueError(
                "fleet spec with number_vehicles1..N also needs 'capacity_vehicles' "
                "(one capacity per vehicle type)"
            )

        n_types = _positive_int(spec.get("type_vehicles", len(capacities)), "type_vehicles")
        n_types = max(1, min(n_types, len(capacities)))

        vehicles: List[Dict[str, float]] = []
        for type_index in range(n_types):
            count = _positive_int(
                spec.get(f"number_vehicles{type_index + 1}", 0),
                f"number_vehicles{type_index + 1}",
            )
            capacity = _positive_float(
                capacities[type_index], f"capacity_vehicles[{type_index}]"
            )
            vehicles.extend({"capacity": capacity} for _ in range(count))

        if not vehicles:
            raise ValueError("fleet spec expands to zero vehicles; check number_vehicles*")
        return vehicles

    capacity = _positive_float(spec.get("capacity", spec.get("vehicle_capacity")), "capacity")

    if "requests_per_vehicle" in spec:
        per_vehicle = _positive_float(spec["requests_per_vehicle"], "requests_per_vehicle")
        if n_requests is None:
            raise ValueError(
                "fleet spec uses 'requests_per_vehicle' but the instance request "
                "count is unknown; pass n_requests"
            )
        count = max(1, int(math.ceil(float(n_requests) / per_vehicle)))
    else:
        count = _positive_int(
            spec.get("number_vehicles", spec.get("n_vehicles")), "number_vehicles"
        )

    if count <= 0:
        raise ValueError("fleet spec expands to zero vehicles; check number_vehicles")
    if count > _MAX_VEHICLES:
        raise ValueError(f"fleet spec expands to {count} vehicles; that is not plausible")

    return [{"capacity": capacity} for _ in range(count)]


def fleet_from_solver_config(config: Any) -> List[Dict[str, float]]:
    """
    Build the vehicles list from a ``solver_utils.SolverConfig`` (or a dict).

    Lets an experiment reuse the exact fleet handed to the solver instead of
    restating it in fleet_config.json.
    """
    if isinstance(config, dict):
        source = config
    else:
        source = {
            key: getattr(config, key)
            for key in (
                "type_vehicles",
                "number_vehicles1",
                "number_vehicles2",
                "number_vehicles3",
                "capacity_vehicles",
            )
            if hasattr(config, key)
        }

    if not source:
        return []
    return vehicles_from_spec(source)


def resolve_fleet_spec(
    instance_ref: Any,
    config: Optional[Dict[str, Any]] = None,
    problem_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pick the fleet spec for one instance: by_instance > by_problem_type > default.
    """
    if config is None:
        config = load_fleet_config()
    if not config:
        return {}

    from csv_table import instance_stem

    stem = instance_stem(instance_ref)

    by_instance = config.get("by_instance") or {}
    if stem in by_instance:
        return by_instance[stem]

    by_problem_type = config.get("by_problem_type") or {}
    detected = problem_type or detect_problem_type(stem)
    if detected and detected in by_problem_type:
        return by_problem_type[detected]

    return config.get("default") or {}


def resolve_fleet(
    instance_ref: Any,
    n_requests: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    problem_type: Optional[str] = None,
) -> List[Dict[str, float]]:
    """
    Resolve the vehicles list for one instance.

    Returns an empty list when no config applies, which keeps fleet features
    NaN rather than inventing a fleet.
    """
    spec = resolve_fleet_spec(instance_ref, config=config, problem_type=problem_type)
    if not spec:
        return []
    return vehicles_from_spec(spec, n_requests=n_requests)


def _smoke_call() -> None:
    config = load_fleet_config()
    print(f"config: {resolve_fleet_config_path()}")

    for name in (
        "Maastricht_Netherlands_ODBRP_13req_1",
        "Maastricht_Netherlands_DARP_17req_1",
        "Copenhagen_Municipality_Denmark_ODBRP_1000req_1",
    ):
        fleet = resolve_fleet(name, n_requests=100, config=config)
        capacities = sorted({vehicle["capacity"] for vehicle in fleet})
        print(
            f"  {name}: problem={detect_problem_type(name)}, "
            f"{len(fleet)} vehicle(s), capacities={capacities}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _smoke_call()
