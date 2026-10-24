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
            jobs.csv                       one row per solver run (atools data file)
            run_solver.slurm               runs one row of jobs.csv
            submit.sh                      submits the whole array
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


def central_station():
    """Index of the station closest to the mean station position (the depot)."""
    import pandas as pd

    stations = pd.read_csv(next((NETWORK_DIR / "csv").glob("*.stations.csv")))
    lat, lon = stations["lat"].mean(), stations["lon"].mean()
    distance = (stations["lat"] - lat) ** 2 + ((stations["lon"] - lon) * math.cos(math.radians(lat))) ** 2
    return int(distance.idxmin())


def main():
    fleets = json.loads(FLEETS_JSON.read_text(encoding="utf-8"))
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    (BUNDLE / "instances").mkdir(parents=True)

    matrix = sorted((NETWORK_DIR / "travel_time_matrix").glob("*.csv"))[0]
    shutil.copyfile(matrix, BUNDLE / "travel_time.csv")
    depot = central_station()

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

    slurm = f"""#!/bin/bash
#SBATCH --job-name=isa-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=skylake

module load atools
module load calcua/2024a GCCcore/13.3.0

source <(aenv --data $1)
mkdir -p results
greedy=""
[ "$algo" = "GR" ] && greedy="--greedy_only"
srun -n 1 -c ${{SLURM_CPUS_PER_TASK}} ./a.out --filename_requests 3 "instances/$instance" \\
    --seed $seed --filename_travel_time travel_time.csv \\
    --output_file "results/${{key}}_${{algo}}_s${{seed}}.txt" \\
    --number_depots 1 --depot {depot} \\
    --type_vehicles 2 --number_vehicles1 $n1 --number_vehicles2 $n2 --number_vehicles3 0 \\
    --capacity_vehicles {' '.join(map(str, CAPACITIES))} \\
    {' '.join(SA_ARGS)} --total_requests $total_requests $greedy
"""
    (BUNDLE / "run_solver.slurm").write_text(slurm, encoding="utf-8", newline="\n")
    (BUNDLE / "submit.sh").write_text(
        "#!/bin/bash\nmodule load atools\n"
        "sbatch --array $(arange --data jobs.csv) run_solver.slurm jobs.csv\n",
        encoding="utf-8", newline="\n",
    )
    (BUNDLE / "README.txt").write_text(
        f"""ISA pilot solver bundle: {len(rows) - 1} runs, {len(fleets)} (instance, fleet) keys,
{len(copied)} instances, depot = station {depot} (closest to the mean station position).

1. Copy this folder to the HPC.
2. Build the solver here (needs the --greedy_only flag for the GR runs):
       cp /path/to/MDHODBRP-P/odbrp-parallel.cpp /path/to/MDHODBRP-P/Makefile .
       module load calcua/2024a GCCcore/13.3.0 && make
3. bash submit.sh
4. Copy results/ back to work/hpc_bundle/results/ for the performance matrix.
""",
        encoding="utf-8", newline="\n",
    )
    print(f"Wrote {len(rows) - 1} runs for {len(fleets)} keys ({len(copied)} instances) to {BUNDLE}")
    print(f"Depot: station {depot}")


if __name__ == "__main__":
    main()
