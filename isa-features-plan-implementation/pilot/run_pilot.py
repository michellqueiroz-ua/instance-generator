"""
run_pilot.py - T14 pilot (ISA-HODBRP)

Generate a small Copenhagen ODBRP instance set with a 7-8 am peak, so the
feature spread can be checked before the full experiment is sized.

    python isa-features-plan-implementation/pilot/run_pilot.py            # write configs + generate
    python isa-features-plan-implementation/pilot/run_pilot.py --configs-only

Design (2 x 3 x 2 x 2 = 24 instances):
    requests        200, 500, 1000
    peak spread     sd 15 min (about 95% of requests between 7:00 and 8:00)
                    sd 30 min (about 68% between 7:00 and 8:00)
    time windows    tight: ~10 min reaction, 15 min arrival slack
                    loose: ~30 min reaction, 45 min arrival slack
    seeds           1, 2

Everything is written under pilot/work/ (git-ignored):
    work/configs/                 one JSON per instance
    work/Copenhagen Municipality, Denmark/  network + generated instances (csv_format/)
    work/results/features.csv     one feature row per instance

The generator runs as a subprocess with this checkout first on PYTHONPATH, so
an older REQreate installed in the environment is never picked up.
"""

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

PILOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PILOT_DIR.parents[1]
WORK_DIR = PILOT_DIR / "work"
CONFIG_DIR = WORK_DIR / "configs"
FEATURES_CSV = WORK_DIR / "results" / "features.csv"

NETWORK = "Copenhagen Municipality, Denmark"  # "Copenhagen, Denmark" geocodes to a village in New York
REQUESTS = [200, 500, 1000]
PEAK_SD_MIN = [15, 30]
TIME_WINDOWS = {
    # reaction time ~ normal(mean, sd) in seconds; slack added to latest arrival
    "tight": {"reaction_mean": 600, "reaction_sd": 120, "arrival_slack": 900},
    "loose": {"reaction_mean": 1800, "reaction_sd": 300, "arrival_slack": 2700},
}
SEEDS = [1, 2]

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
        "replicas": 1,
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
                "dynamism": 100,
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


def config_name(requests, peak_sd_min, tw_name, seed):
    return f"Copenhagen_ODBRP_{requests:04d}req_peak{peak_sd_min}_{tw_name}_s{seed}.json"


def write_configs():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    names = []
    for requests, peak, tw_name, seed in itertools.product(REQUESTS, PEAK_SD_MIN, TIME_WINDOWS, SEEDS):
        name = config_name(requests, peak, tw_name, seed)
        (CONFIG_DIR / name).write_text(
            json.dumps(make_config(requests, peak, tw_name, seed), indent=2), encoding="utf-8"
        )
        names.append(name)
    return names


def generate():
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env["ISA_FEATURES_CSV"] = str(FEATURES_CSV)
    command = [
        sys.executable, "-u", "-m", "REQreate.cli", "generate", str(CONFIG_DIR),
        "--skip-existing", "--continue-on-error",
    ]
    return subprocess.call(command, cwd=WORK_DIR, env=env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--configs-only", action="store_true", help="write the configs and stop")
    args = parser.parse_args(argv)

    names = write_configs()
    print(f"Wrote {len(names)} configs to {CONFIG_DIR}")
    if args.configs_only:
        return 0
    status = generate()
    print(f"Features: {FEATURES_CSV}")
    return status


if __name__ == "__main__":
    sys.exit(main())
