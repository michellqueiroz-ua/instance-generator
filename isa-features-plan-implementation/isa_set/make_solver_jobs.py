"""
make_solver_jobs.py - ISA instance set: write the SA / GR solver jobs of one batch.

    python make_solver_jobs.py --batch 1 [--work /work3/$USER/isa] [--sa-seeds 3]

Reads   <work>/design/batch_XX_fleets.json                          the fleets per instance
        <work>/<network>/csv_format/<instance>.csv                  the generated instances
        <work>/<network>/csv/*.stations.csv                         station positions (for the depot)
        <work>/<network>/travel_time_matrix/station_travel_time_matrix.csv
Writes  <work>/solver/batch_XX/jobs.csv      one row per solver run
        <work>/solver/batch_XX/submit.sh     submits one single-core LSF job per row
Nothing is run here; submit.sh is run on a login node, after the solver is built (see its header).

Runs per (instance, fleet): SA with seeds 1..sa-seeds and GR (SA binary with --greedy_only).
The solver takes each vehicle type's count as a percentage of the requests, rounded up, so a count is
written as (count - 0.5) / requests * 100, which rounds back to exactly the count.

Depot: the solver crashes when a passenger's stop equals the depot, and the ISA ignores depots, so each
instance gets the station closest to the mean station position that none of its requests uses.

Only the standard library and numpy are used.
"""

import argparse
import ast
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from fleet_config import vehicles_from_spec  # noqa: E402

CAPACITIES = [4, 8]
# Same SA settings as the published MDHODBRP runs (see pilot/make_solver_jobs.py).
SA_ARGS = "--init_temperature 1.3 --maxnrep 350 --increase_rep 800"

TEMPLATES = HERE / "hpc"


def percent_for(count, n_requests):
    """Percentage string the solver turns back into ``count`` via ceil(float)."""
    if count == 0:
        return "0"
    text = f"{(count - 0.5) / n_requests * 100:.6f}"
    decoded = math.ceil(np.float32(np.float32(text) / np.float32(100)) * np.float32(n_requests))
    if decoded != count:
        raise ValueError(f"cannot encode {count} vehicles for {n_requests} requests ({text}% -> {decoded})")
    return text


def read_requests(path):
    """(number of requests, stations used as a stop by any request)."""
    used, n = set(), 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n += 1
            for column in ("stops_orgn", "stops_dest"):
                try:
                    used.update(ast.literal_eval(row[column]))
                except (SyntaxError, ValueError) as error:
                    raise SystemExit(f"{path}, request {n}: cannot read {column} = {row[column]!r} ({error})")
    return n, used


def read_stations(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [(float(r["lat"]), float(r["lon"])) for r in rows]


def central_station(stations, excluded):
    """Index of the station closest to the mean position among those not in ``excluded``."""
    lat = sum(s[0] for s in stations) / len(stations)
    lon = sum(s[1] for s in stations) / len(stations)
    cos = math.cos(math.radians(lat))
    best, best_distance = None, float("inf")
    for index, (y, x) in enumerate(stations):
        if index in excluded:
            continue
        distance = (y - lat) ** 2 + ((x - lon) * cos) ** 2
        if distance < best_distance:
            best, best_distance = index, distance
    if best is None:
        raise SystemExit("every station is a stop of some request; no safe depot")
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--work", default=os.environ.get("ISA_WORK", f"/work3/{os.environ.get('USER', 'user')}/isa"))
    parser.add_argument("--network", default="Capital Region of Denmark")
    parser.add_argument("--sa-seeds", type=int, default=3)
    args = parser.parse_args(argv)

    work = Path(args.work)
    network = work / args.network
    tag = f"{args.batch:02d}"
    fleets = json.loads((work / "design" / f"batch_{tag}_fleets.json").read_text(encoding="utf-8"))
    matrix = network / "travel_time_matrix" / "station_travel_time_matrix.csv"
    if not matrix.is_file():
        raise SystemExit(f"{matrix} is missing: run station_matrix.sh first")
    stations = read_stations(next((network / "csv").glob("*.stations.csv")))

    out = work / "solver" / f"batch_{tag}"
    out.mkdir(parents=True, exist_ok=True)

    rows = ["key; instance; algo; seed; total_requests; n1; n2; depot"]
    cache = {}
    missing = []
    for key, fleet in sorted(fleets.items()):
        path = network / "csv_format" / f"{fleet['instance']}.csv"
        if not path.is_file():
            missing.append(path.name)
            continue
        if path not in cache:
            n_requests, used = read_requests(path)
            cache[path] = (n_requests, central_station(stations, used))
        n_requests, depot = cache[path]
        vehicles = vehicles_from_spec(fleet, n_requests=n_requests)
        counts = [sum(1 for v in vehicles if v["capacity"] == c) for c in CAPACITIES]
        if sum(counts) != len(vehicles):
            raise SystemExit(f"{key}: fleet has capacities other than {CAPACITIES}")
        n1, n2 = (percent_for(c, n_requests) for c in counts)
        for algo, seeds in (("SA", range(1, args.sa_seeds + 1)), ("GR", [1])):
            for seed in seeds:
                rows.append(f"{key}; {path}; {algo}; {seed}; {n_requests}; {n1}; {n2}; {depot}")

    # open() instead of Path.write_text(newline=): the login node runs Python 3.9
    with open(out / "jobs.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rows) + "\n")
    # read_text turns any CRLF of a Windows checkout into LF; the files must run on Linux
    for name in ("submit.sh", "run_rows.sh"):
        text = (TEMPLATES / ("solver_" + name)).read_text(encoding="utf-8")
        text = text.replace("__TRAVEL_TIME__", str(matrix)).replace("__SA_ARGS__", SA_ARGS)
        with open(out / name, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    print(f"{len(rows) - 1} runs for {len(fleets) - len(missing)} fleets of {len(cache)} instances in {out}")
    if missing:
        print(f"skipped {len(missing)} fleets whose instance csv does not exist yet, e.g. {missing[0]}")


if __name__ == "__main__":
    main()
