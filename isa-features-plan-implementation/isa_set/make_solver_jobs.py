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
With --sa-variants 1000,2000,10000 there are also SA1000, SA2000 and SA10000 (same seeds), which stop each SA call
after that many counted moves instead of 200 (--max_static_iterations). Their rows come after ALL the standard rows
(rows 4001.. in a batch of 100 instances x 10 fleets x 4 rows; base rows keep their numbers), per fleet 3 rows
(SA seeds) for each cap, in the order sorted fleet key, cap, seed. --variants-max-fleet-percent X keeps only fleets
with at most X vehicles per 100 requests; --variants-keys KEY[,KEY] keeps only those fleets (probe of one
instance--fleet pair: 3 rows for one cap); --variants-instances A-B keeps only the fleets of instances A to B (the
instance number of ISA_b04_i007_1 is 7); with several of these filters a fleet must pass all of them (intersection).
--gr-extra-seeds N --gr-extra-instances A-B add GR seeds 2..N for the fleets of instances A to B (GR is randomised and
runs once per pair otherwise); these rows come last, after the SA variant rows, in the order sorted fleet key, seed.
collect_batch.py ignores GR seeds above 1 unless --gr-seeds-max says otherwise. Run them with PER_JOB=1..3 and a longer WALL_H (see solver_submit.sh).
jobs.csv, submit.sh and run_rows.sh are written to *.tmp and renamed, so a job that is running never reads a half-written file.
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
import re
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


def instance_number(fleet):
    """Number of the instance in its name: ISA_b04_i007_1 -> 7."""
    match = re.search(r"_i(\d+)_", fleet["instance"])
    if not match:
        raise SystemExit(f"cannot read the instance number from {fleet['instance']!r}")
    return int(match.group(1))


def parse_range(text, option):
    """'1-10' -> (1, 10); '5' -> (5, 5); '' -> None (no filter)."""
    text = text.strip()
    if not text:
        return None
    try:
        first, _, last = text.partition("-")
        low, high = int(first), int(last or first)
    except ValueError:
        raise SystemExit(f"{option}: expected A-B (for example 1-10), got {text!r}")
    if low > high:
        raise SystemExit(f"{option}: {text!r} is an empty range")
    return low, high


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
    parser.add_argument("--sa-variants", default="",
                        help="comma-separated move caps, e.g. 1000,2000,10000: extra SA runs named SA<cap> with "
                             "--max_static_iterations <cap>, written after the standard rows")
    parser.add_argument("--variants-max-fleet-percent", type=float, default=None,
                        help="only fleets with at most this many vehicles per 100 requests get the SA variants "
                             "(batch 1: with ample fleets SA already reaches the sum of the direct times and the "
                             "variants take 40 times as long)")
    parser.add_argument("--variants-keys", default="",
                        help="comma-separated fleet keys (e.g. ISA_b04_i001_1_fleet7): only these fleets get the SA "
                             "variants (probe of one instance--fleet pair); combines with --variants-max-fleet-percent")
    parser.add_argument("--variants-instances", default="",
                        help="A-B: only the fleets of instances A to B get the SA variants (instance number as in "
                             "ISA_b04_i007_1 = 7); with --variants-keys and --variants-max-fleet-percent a fleet "
                             "must pass every filter")
    parser.add_argument("--gr-extra-seeds", type=int, default=1,
                        help="N: add GR runs with seeds 2..N for the instances of --gr-extra-instances (default 1: none)")
    parser.add_argument("--gr-extra-instances", default="",
                        help="A-B: instances that get the extra GR seeds (required when --gr-extra-seeds > 1)")
    args = parser.parse_args(argv)

    work = Path(args.work)
    network = work / args.network
    tag = f"{args.batch:02d}"
    fleets = json.loads((work / "design" / f"batch_{tag}_fleets.json").read_text(encoding="utf-8"))
    matrix = network / "travel_time_matrix" / "station_travel_time_matrix.csv"
    if not matrix.is_file():
        raise SystemExit(f"{matrix} is missing: run station_matrix.sh first")
    stations = read_stations(next((network / "csv").glob("*.stations.csv")))

    caps = [int(c) for c in args.sa_variants.split(",") if c.strip()]
    only = {k.strip() for k in args.variants_keys.split(",") if k.strip()}
    variant_range = parse_range(args.variants_instances, "--variants-instances")
    gr_range = parse_range(args.gr_extra_instances, "--gr-extra-instances")
    if args.gr_extra_seeds < 1:
        raise SystemExit("--gr-extra-seeds must be at least 1")
    if args.gr_extra_seeds > 1 and gr_range is None:
        raise SystemExit("--gr-extra-seeds needs --gr-extra-instances A-B (no default: it would run for every instance)")
    if only - set(fleets):   # before anything is created or written
        raise SystemExit(f"--variants-keys: unknown fleet key(s) {sorted(only - set(fleets))[:3]}")

    out = work / "solver" / f"batch_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    rows = ["key; instance; algo; seed; total_requests; n1; n2; depot"]
    variant_rows = []
    gr_rows = []
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
        fleet_percent = 100.0 * len(vehicles) / n_requests
        for cap in caps if ((args.variants_max_fleet_percent is None
                             or fleet_percent <= args.variants_max_fleet_percent)
                            and (not only or key in only)
                            and (variant_range is None
                                 or variant_range[0] <= instance_number(fleet) <= variant_range[1])) else []:
            for seed in range(1, args.sa_seeds + 1):
                variant_rows.append(f"{key}; {path}; SA{cap}; {seed}; {n_requests}; {n1}; {n2}; {depot}")
        if gr_range is not None and gr_range[0] <= instance_number(fleet) <= gr_range[1]:
            for seed in range(2, args.gr_extra_seeds + 1):
                gr_rows.append(f"{key}; {path}; GR; {seed}; {n_requests}; {n1}; {n2}; {depot}")
    # appended, so the rows of the standard runs keep their numbers (finished runs are skipped by name)
    # then the extra GR seeds, so the variant rows keep their numbers whether or not GR extras are asked for
    n_base = len(rows) - 1
    rows.extend(variant_rows)
    rows.extend(gr_rows)
    if (args.variants_keys or args.variants_instances) and not variant_rows:
        raise SystemExit("--variants-keys/--variants-instances: no variant rows written (is --sa-variants given, is the "
                         "fleet's instance csv there, do the other variants filters exclude it?)")
    if args.gr_extra_seeds > 1 and not gr_rows:
        raise SystemExit("--gr-extra-seeds: no GR rows written (does --gr-extra-instances match an instance with a csv?)")

    # Each file is written to <name>.tmp and renamed (os.replace is atomic on the same folder), so a running job that
    # reads jobs.csv or run_rows.sh never sees a truncated file.
    # open() instead of Path.write_text(newline=): the login node runs Python 3.9
    def write_atomic(path, text):
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)

    write_atomic(out / "jobs.csv", "\n".join(rows) + "\n")
    # read_text turns any CRLF of a Windows checkout into LF; the files must run on Linux
    for name in ("submit.sh", "run_rows.sh"):
        text = (TEMPLATES / ("solver_" + name)).read_text(encoding="utf-8")
        text = text.replace("__TRAVEL_TIME__", str(matrix)).replace("__SA_ARGS__", SA_ARGS)
        write_atomic(out / name, text)
    if variant_rows or gr_rows:
        print(f"{len(rows) - 1} runs: {n_base} base, {len(variant_rows)} SA variant (rows {n_base + 1}-"
              f"{n_base + len(variant_rows)}), {len(gr_rows)} extra GR "
              f"(rows {n_base + len(variant_rows) + 1}-{len(rows) - 1})")
    print(f"{len(rows) - 1} runs for {len(fleets) - len(missing)} fleets of {len(cache)} instances in {out}")
    if missing:
        print(f"skipped {len(missing)} fleets whose instance csv does not exist yet, e.g. {missing[0]}")


if __name__ == "__main__":
    main()
