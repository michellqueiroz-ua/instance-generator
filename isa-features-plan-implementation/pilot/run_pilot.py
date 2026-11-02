"""
run_pilot.py - T14 pilot (ISA-HODBRP)

Generate a small Copenhagen ODBRP instance set with a 7-8 am peak, so the
feature spread can be checked before the full experiment is sized.

    python isa-features-plan-implementation/pilot/run_pilot.py                  # configs + generate + features
    python isa-features-plan-implementation/pilot/run_pilot.py --configs-only
    python isa-features-plan-implementation/pilot/run_pilot.py --features-only  # recompute features.csv

Design: 12 configs x 2 replicas = 24 generated instances, each paired with
3 fleets = 72 ISA instances.
    requests        200, 500, 1000
    peak spread     time_stamp ~ normal(7:30, sd), kept within 6:30-8:30
                    sd 15 min (about 95% of requests between 7:00 and 8:00)
                    sd 30 min (about 68% between 7:00 and 8:00)
    time windows    tight: ~10 min reaction, 15 min arrival slack
                    loose: ~30 min reaction, 45 min arrival slack
    replicas        2
    seed            one per config (1..12), so every config draws its own
                    locations and times
    fleet           10%, 20%, 30% of requests, 60% 4-seat / 40% 8-seat

No 'dynamism' on time_stamp: it replaces the pdf with evenly spaced times
between loc and loc + scale, which flattened the peak in the first pilot.

Everything is written under pilot/work/ (git-ignored):
    work/configs/                 one JSON per config
    work/Copenhagen Municipality, Denmark/  network + generated instances (csv_format/)
    work/results/features.csv     one row per (instance, fleet), key <instance>_fleet<pct>
    work/results/fleets.json      the fleet behind each key, for the solver runs

The generator runs as a subprocess with this checkout first on PYTHONPATH, so
an older REQreate installed in the environment is never picked up.
"""

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PILOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PILOT_DIR.parents[1]
WORK_DIR = PILOT_DIR / "work"
CONFIG_DIR = WORK_DIR / "configs"
RESULTS_DIR = WORK_DIR / "results"
FEATURES_CSV = RESULTS_DIR / "features.csv"
FLEETS_JSON = RESULTS_DIR / "fleets.json"
# Rows the generation hook writes (default fleet only); features.csv is rebuilt from the instances.
GENERATION_FEATURES_CSV = RESULTS_DIR / "generation_features.csv"
ISA_DIR = PILOT_DIR.parent

NETWORK = "Copenhagen Municipality, Denmark"  # "Copenhagen, Denmark" geocodes to a village in New York
REQUESTS = [200, 500, 1000]
PEAK_SD_MIN = [15, 30]
TIME_WINDOWS = {
    # reaction time ~ normal(mean, sd) in seconds; slack added to latest arrival
    "tight": {"reaction_mean": 600, "reaction_sd": 120, "arrival_slack": 900},
    "loose": {"reaction_mean": 1800, "reaction_sd": 300, "arrival_slack": 2700},
}
REPLICAS = 2
FLEET_PERCENTS = [10, 20, 30]
FLEET_COMPOSITION = [{"capacity": 4, "percent": 60}, {"capacity": 8, "percent": 40}]

PEAK_CENTRE_S = 7.5 * 3600  # 7:30
WINDOW_START_H = 6.5        # requests are kept between 6:30 and 8:30
WINDOW_END_H = 8.5


def make_config(requests, peak_sd_min, tw_name, seed):
    tw = TIME_WINDOWS[tw_name]
    return {
        "seed": seed,
        "network": NETWORK,
        "problem": "ODBRP",
        "set_fixed_speed": {"vehicle_speed_data": 20, "vehicle_speed_data_unit": "kmh"},
        "replicas": REPLICAS,
        "requests": requests,
        "instance_filename": ["network", "problem", "requests"],
        "parameters": [
            {"name": "min_early_departure", "type": "float", "value": WINDOW_START_H, "time_unit": "h"},
            {"name": "max_early_departure", "type": "float", "value": WINDOW_END_H, "time_unit": "h"},
            {"name": "min_dtt", "type": "integer", "value": 300, "time_unit": "s"},
            {"name": "max_dtt", "type": "integer", "value": 2400, "time_unit": "s"},
            {"name": "graphml", "type": "graphml", "value": True},
        ],
        "attributes": [
            {
                "name": "time_stamp", "type": "integer", "time_unit": "s",
                "pdf": [{"type": "normal", "loc": PEAK_CENTRE_S, "scale": peak_sd_min * 60}],
                "constraints": [
                    "time_stamp >= 0",
                    "time_stamp >= min_early_departure",
                    "time_stamp <= max_early_departure",
                ],
            },
            {
                "name": "reaction_time", "type": "integer", "time_unit": "s",
                "pdf": [{"type": "normal", "loc": tw["reaction_mean"], "scale": tw["reaction_sd"]}],
                "constraints": ["reaction_time >= 60"],
            },
            {
                "name": "earliest_departure", "type": "integer", "time_unit": "s",
                "expression": "time_stamp",
                "constraints": ["earliest_departure >= 0", "earliest_departure >= min_early_departure"],
            },
            {
                "name": "latest_departure", "type": "integer", "time_unit": "s",
                "expression": "time_stamp + reaction_time",
            },
            {
                "name": "latest_arrival", "type": "integer", "time_unit": "s",
                "expression": f"earliest_departure + direct_travel_time + reaction_time + {tw['arrival_slack']}",
            },
            {"name": "origin", "type": "location"},
            {"name": "destination", "type": "location"},
            {
                "name": "stops_orgn", "type": "array_primitives", "expression": "stops(origin)",
                "constraints": ["len(stops_orgn) > 0"],
            },
            {
                "name": "stops_dest", "type": "array_primitives", "expression": "stops(destination)",
                "constraints": ["len(stops_dest) > 0", "not (set(stops_orgn) & set(stops_dest))"],
            },
            {
                "name": "max_walking", "type": "integer", "time_unit": "s",
                "pdf": [{"type": "uniform", "loc": 550, "scale": 50}],
            },
            {
                "name": "walk_speed", "type": "real", "speed_unit": "mps",
                "pdf": [{"type": "uniform", "loc": 1.38889, "scale": 0}],
            },
            {
                "name": "direct_travel_time", "type": "integer", "time_unit": "s",
                "expression": "dtt(origin,destination)",
                "constraints": ["direct_travel_time >= min_dtt", "direct_travel_time <= max_dtt"],
            },
            {
                "name": "direct_distance", "type": "integer", "length_unit": "m",
                "expression": "dist_drive(origin,destination)",
            },
        ],
        "travel_time_matrix": ["bus_stations"],
    }


def config_name(requests, peak_sd_min, tw_name):
    return f"Copenhagen_ODBRP_{requests:04d}req_peak{peak_sd_min}_{tw_name}.json"


def write_configs():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for stale in CONFIG_DIR.glob("*.json"):  # configs from an earlier design
        stale.unlink()
    names = []
    designs = itertools.product(REQUESTS, PEAK_SD_MIN, TIME_WINDOWS)
    for seed, (requests, peak, tw_name) in enumerate(designs, start=1):
        name = config_name(requests, peak, tw_name)
        (CONFIG_DIR / name).write_text(
            json.dumps(make_config(requests, peak, tw_name, seed), indent=2), encoding="utf-8"
        )
        names.append(name)
    return names


def generate():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env["ISA_FEATURES_CSV"] = str(GENERATION_FEATURES_CSV)
    command = [
        sys.executable, "-u", "-m", "REQreate.cli", "generate", str(CONFIG_DIR),
        "--skip-existing", "--continue-on-error",
    ]
    return subprocess.call(command, cwd=WORK_DIR, env=env)


def instance_jsons(config_names):
    """Generated instance JSONs (all replicas) for the current configs."""
    stems = {name[: -len(".json")] for name in config_names}
    found = sorted((WORK_DIR / NETWORK / "json_format").rglob("*.json"))
    return [path for path in found if re.sub(r"_\d+$", "", path.stem) in stems]


def write_features(config_names):
    """One features.csv row per (instance, fleet) and the matching fleets.json."""
    for path in (REPO_ROOT, ISA_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from compute_features_scaffold import compute_features
    from features_export import write_features_csv, normalize_feature_row
    from fleet_config import vehicles_from_spec
    from reqreate_adapter import load_reqreate_instance

    rows, fleets = [], {}
    paths = instance_jsons(config_names)
    for path in paths:
        instance = load_reqreate_instance(path)
        n_requests = len(instance.get("requests", []))
        for percent in FLEET_PERCENTS:
            spec = {"fleet_percent_of_requests": percent, "composition": FLEET_COMPOSITION}
            instance["vehicles"] = vehicles_from_spec(spec, n_requests=n_requests)
            key = f"{path.stem}_fleet{percent}"
            rows.append(normalize_feature_row(key, compute_features(instance)))
            fleets[key] = {"instance": path.stem, "n_vehicles": len(instance["vehicles"]), **spec}
        print(f"  {path.stem}: {n_requests} requests")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_features_csv(rows, FEATURES_CSV)
    FLEETS_JSON.write_text(json.dumps(fleets, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows ({len(paths)} instances x {len(FLEET_PERCENTS)} fleets) to {FEATURES_CSV}")
    return 0 if rows else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--configs-only", action="store_true", help="write the configs and stop")
    parser.add_argument("--features-only", action="store_true", help="only rebuild features.csv")
    args = parser.parse_args(argv)

    names = write_configs()
    print(f"Wrote {len(names)} configs ({REPLICAS} replicas each) to {CONFIG_DIR}")
    if args.configs_only:
        return 0
    if not args.features_only:
        status = generate()
        if status:
            print("Some configs failed; features are computed for the ones that succeeded.")
    return write_features(names)


if __name__ == "__main__":
    sys.exit(main())
