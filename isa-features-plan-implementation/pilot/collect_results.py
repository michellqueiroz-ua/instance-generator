"""
collect_results.py - T14 pilot (ISA-HODBRP)

Turn the solver result files from the HPC into the pilot performance matrix.

    python isa-features-plan-implementation/pilot/collect_results.py

Reads   work/hpc_bundle/results/<key>_<SA|GR>_s<seed>.txt   (copied back from the HPC)
        work/results/features.csv                          (instance order and keys)
Writes  work/results/solver_runs.csv     one row per run: key, algo, seed, served, URT, seconds
        work/results/raw_results.csv     per key: mean SA URT over seeds, GR URT
        work/results/perf_matrix.csv     normalized scores (ratio: best / value)

The objective is the solver's best_total_user_ride_time: total ride time of all
passengers, where a passenger handed to a third party counts as 4x their direct
travel time, so serving fewer passengers is not rewarded.

Only SA and GR exist for the pilot (IP needs Gurobi on the HPC), so the matrix
is built with ISA_ALGORITHMS=SA,GR. ``ratio`` normalization is used because SA
is often under half of GR's cost, which ``gap`` would clip to 0.
"""

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("ISA_ALGORITHMS", "SA,GR")

PILOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PILOT_DIR.parent))

import pandas as pd  # noqa: E402
from perf_matrix_export import export_perf_matrix  # noqa: E402

WORK_DIR = PILOT_DIR / "work"
RESULTS_DIR = WORK_DIR / "hpc_bundle" / "results"
OUT_DIR = WORK_DIR / "results"
FEATURES_CSV = OUT_DIR / "features.csv"

NAME = re.compile(r"^(?P<key>.+)_(?P<algo>SA|GR)_s(?P<seed>\d+)\.txt$")
# First line of a result file, as written at the end of odbrp-parallel.cpp main().
FIELDS = {"served": 1, "third_party": 2, "best_total_urt": 10, "elapsed_s": -2}


def read_runs():
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.txt")):
        match = NAME.match(path.name)
        if not match:
            continue
        tokens = path.read_text(encoding="utf-8").splitlines()[0].split()
        row = {"key": match["key"], "algo": match["algo"], "seed": int(match["seed"])}
        for field, index in FIELDS.items():
            row[field] = float(tokens[index])
        rows.append(row)
    if not rows:
        raise SystemExit(f"no result files in {RESULTS_DIR}")
    return pd.DataFrame(rows)


def main():
    runs = read_runs()
    runs.to_csv(OUT_DIR / "solver_runs.csv", index=False)

    raw = runs.groupby(["key", "algo"])["best_total_urt"].mean().unstack("algo")
    expected = pd.read_csv(FEATURES_CSV)["instance"]
    missing = sorted(set(expected) - set(raw.dropna().index))
    if missing:
        raise SystemExit(f"{len(missing)} keys lack SA or GR results, e.g. {missing[:3]}")
    raw = raw.loc[expected]
    raw.rename_axis("instance").to_csv(OUT_DIR / "raw_results.csv")

    export_perf_matrix(
        {key: values.to_dict() for key, values in raw.iterrows()},
        csv_path=OUT_DIR / "perf_matrix.csv",
        features_path=FEATURES_CSV,
        sense="min",
        method="ratio",
    )

    per_algo = runs.groupby("algo").agg(
        runs=("key", "size"),
        served_pct=("served", lambda s: 100 * s.sum() / (s + runs.loc[s.index, "third_party"]).sum()),
        mean_seconds=("elapsed_s", "mean"),
        max_seconds=("elapsed_s", "max"),
    )
    print(per_algo.round(1).to_string())
    sa_wins = (raw["SA"] < raw["GR"]).sum()
    print(f"SA better than GR on {sa_wins}/{len(raw)} keys; "
          f"mean SA/GR cost ratio {(raw['SA'] / raw['GR']).mean():.2f}")
    print(f"Wrote {OUT_DIR / 'perf_matrix.csv'}")


if __name__ == "__main__":
    main()
