"""
make_solver_jobs.py - T14 pilot (ISA-HODBRP)

Build a self-contained bundle for running the ODBRP solver (SA and GR) on the
pilot instances on the HPC. Nothing is run here.

    python isa-features-plan-implementation/pilot/make_solver_jobs.py

Reads   work/results/fleets.json           (written by run_pilot.py)
        work/<network>/csv_format/*.csv    the generated instances
        work/<network>/travel_time_matrix/ the station travel-time matrix
Writes  work/hpc_bundle/
            instances/*.csv                one per generated instance
            travel_time.csv                station x station, seconds
            jobs.csv                       one row per solver run
            submit.sh                      submits one DTU LSF (bsub) job per row
            odbrp-parallel.cpp, Makefile   solver source (with --greedy_only)
            README.txt                     how to build the solver and submit

Solver runs per (instance, fleet) key in fleets.json:
    SA   seeds 1..SA_SEEDS (simulated annealing, the solver's default)
    GR   seed 1 with --greedy_only (the insertion solution, no SA)

GR needs a solver built with the --greedy_only flag; see README.txt.

Fleet: the solver reads --number_vehicles<k> as a percentage of the requests
and rounds up (ceil of a float), so each type count is encoded as
(count - 0.5) / requests * 100, which rounds back to exactly ``count``.
"""

import json
import os
import math
import shutil
import sys
from pathlib import Path

import numpy as np

PILOT_DIR = Path(__file__).resolve().parent
WORK_DIR = PILOT_DIR / "work"
sys.path.insert(0, str(PILOT_DIR))
from run_pilot import FLEETS_JSON, NETWORK  # noqa: E402

NETWORK_DIR = WORK_DIR / NETWORK
BUNDLE = WORK_DIR / "hpc_bundle"
# MDHODBRP-P solver source (odbrp-parallel.cpp + Makefile); override with ODBRP_SOLVER_DIR.
SOLVER_DIR = Path(os.environ.get("ODBRP_SOLVER_DIR", r"C:\Users\micqu\Desktop\ODBRP_P\MDHODBRP-P"))

SA_SEEDS = 3
# Same SA settings as the published MDHODBRP runs. They passed "--lamba"
# (sic), which the solver ignores, so lambda stays at its default; it is
# left out here for the same behaviour.
SA_ARGS = ["--init_temperature", "1.3", "--maxnrep", "350", "--increase_rep", "800"]
CAPACITIES = [4, 8]


def type_counts(n_vehicles, composition):
    """Vehicles per capacity, split exactly as fleet_config does."""
    sys.path.insert(0, str(PILOT_DIR.parent))
    from fleet_config import vehicles_from_spec

    # 100% of n_vehicles "requests" gives exactly n_vehicles, split by composition.
    vehicles = vehicles_from_spec(
        {"fleet_percent_of_requests": 100, "composition": composition}, n_requests=n_vehicles
    )
    return [sum(1 for v in vehicles if v["capacity"] == c) for c in CAPACITIES]


def percent_for(count, n_requests):
    """Percentage string the solver turns back into ``count`` via ceil(float)."""
    if count == 0:
        return "0"
    text = f"{(count - 0.5) / n_requests * 100:.6f}"
    decoded = math.ceil(np.float32(np.float32(text) / np.float32(100)) * np.float32(n_requests))
    if decoded != count:
        raise ValueError(f"cannot encode {count} vehicles for {n_requests} requests ({text}% -> {decoded})")
    return text


def used_stations(instance_csvs):
    """Stations that appear in any request's stops_orgn or stops_dest."""
    import ast
    import pandas as pd

    used = set()
    for path in instance_csvs:
        requests = pd.read_csv(path, usecols=["stops_orgn", "stops_dest"])
        for column in ("stops_orgn", "stops_dest"):
            for stops in requests[column]:
                used.update(ast.literal_eval(stops))
    return used


def central_station(excluded):
    """
    Depot: the station closest to the mean station position that no request
    uses as a stop.

    The solver crashes when a passenger's stop equals the depot: in
    cheapest_destination2 (and three similar places) a drop-off at the
    route's final stop, the depot, reads stops[v][i+2], one past the route.
    """
    import pandas as pd

    stations = pd.read_csv(next((NETWORK_DIR / "csv").glob("*.stations.csv")))
    lat, lon = stations["lat"].mean(), stations["lon"].mean()
    distance = (stations["lat"] - lat) ** 2 + ((stations["lon"] - lon) * math.cos(math.radians(lat))) ** 2
    candidates = distance.drop(index=[s for s in excluded if s in distance.index])
    if candidates.empty:
        raise SystemExit("every station is a stop of some request; no safe depot")
    return int(candidates.idxmin())


def main():
    fleets = json.loads(FLEETS_JSON.read_text(encoding="utf-8"))
    # Empty the bundle but keep the folder (a shell may have it open on Windows).
    BUNDLE.mkdir(parents=True, exist_ok=True)
    for entry in BUNDLE.iterdir():
        shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
    (BUNDLE / "instances").mkdir()

    matrix = sorted((NETWORK_DIR / "travel_time_matrix").glob("*.csv"))[0]
    shutil.copyfile(matrix, BUNDLE / "travel_time.csv")
    source_cpp = (SOLVER_DIR / "odbrp-parallel.cpp").read_text(encoding="utf-8")
    if "--greedy_only" not in source_cpp:
        raise SystemExit(f"{SOLVER_DIR / 'odbrp-parallel.cpp'} has no --greedy_only flag; GR runs would run SA")
    for name in ("odbrp-parallel.cpp", "Makefile"):
        # LF endings: with CRLF, make on Linux names the binary "a.out\r".
        text = (SOLVER_DIR / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        (BUNDLE / name).write_text(text, encoding="utf-8", newline="\n")
    instances = sorted({NETWORK_DIR / "csv_format" / f"{f['instance']}.csv" for f in fleets.values()})
    depot = central_station(used_stations(instances))

    rows = ["key; instance; algo; seed; total_requests; n1; n2"]
    copied = set()
    for key, fleet in sorted(fleets.items()):
        instance = fleet["instance"]
        source = NETWORK_DIR / "csv_format" / f"{instance}.csv"
        if instance not in copied:
            shutil.copyfile(source, BUNDLE / "instances" / source.name)
            copied.add(instance)
        n_requests = sum(1 for _ in source.open(encoding="utf-8")) - 1
        counts = type_counts(fleet["n_vehicles"], fleet["composition"])
        n1, n2 = (percent_for(c, n_requests) for c in counts)
        for algo, seeds in (("SA", range(1, SA_SEEDS + 1)), ("GR", [1])):
            for seed in seeds:
                rows.append(f"{key}; {source.name}; {algo}; {seed}; {n_requests}; {n1}; {n2}")
    (BUNDLE / "jobs.csv").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")

    solver_args = (
        f"--number_depots 1 --depot {depot} --type_vehicles 2 "
        f"--capacity_vehicles {' '.join(map(str, CAPACITIES))} {' '.join(SA_ARGS)}"
    )
    submit = f"""#!/bin/bash
# Submit one DTU LSF job per row of jobs.csv. Run from this folder on a login node.
# Usage: bash submit.sh            submit all rows
#        bash submit.sh 5          submit only the first 5 rows (a test)
# The solver reserves ~6 GB of fixed-size arrays at start; a lower memory limit
# makes it segfault before printing anything. One OpenMP thread is enough.
cd "$(dirname "${{BASH_SOURCE[0]}}")"
mkdir -p results logs
limit=${{1:-0}}
n=0
tail -n +2 jobs.csv | while IFS=';' read -r key instance algo seed total n1 n2; do
    key=$(echo $key); instance=$(echo $instance); algo=$(echo $algo); seed=$(echo $seed)
    total=$(echo $total); n1=$(echo $n1); n2=$(echo $n2)
    n=$((n+1)); [ "$limit" -gt 0 ] && [ "$n" -gt "$limit" ] && break
    greedy=""; [ "$algo" = "GR" ] && greedy="--greedy_only"
    name="${{key}}_${{algo}}_s${{seed}}"
    [ -s "results/$name.txt" ] && continue   # already done
    bsub -J "isa_$name" -q hpc -n 1 -R "span[hosts=1]" -R "rusage[mem=16GB]" -M 16GB -W 24:00 \
         -o "logs/$name.out" -e "logs/$name.err" \
         "OMP_NUM_THREADS=1 ./a.out --filename_requests 3 instances/$instance --seed $seed \
          --filename_travel_time travel_time.csv --output_file results/$name.txt \
          {solver_args} --number_vehicles1 $n1 --number_vehicles2 $n2 --number_vehicles3 0 \
          --total_requests $total $greedy"
done
"""
    (BUNDLE / "submit.sh").write_text(submit, encoding="utf-8", newline="\n")
    (BUNDLE / "README.txt").write_text(
        f"""ISA pilot solver bundle (DTU HPC, LSF): {len(rows) - 1} runs, {len(fleets)} (instance, fleet)
keys, {len(copied)} instances, depot = station {depot} (closest to the mean station position among stations no request uses).

1. Copy this folder to the HPC.
2. Build the solver in this folder (source copied from {SOLVER_DIR}, with --greedy_only):
       make
3. Test with a few jobs first:  bash submit.sh 4     (then check logs/ and results/)
4. Submit the rest:             bash submit.sh       (finished runs are skipped)
5. Copy results/ back to work/hpc_bundle/results/ for the performance matrix.
""",
        encoding="utf-8", newline="\n",
    )
    print(f"Wrote {len(rows) - 1} runs for {len(fleets)} keys ({len(copied)} instances) to {BUNDLE}")
    print(f"Depot: station {depot}")


if __name__ == "__main__":
    main()
