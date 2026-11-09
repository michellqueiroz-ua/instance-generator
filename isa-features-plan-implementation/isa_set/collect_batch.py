"""
collect_batch.py - ISA instance set: turn the solver results of one batch into its performance matrix.

    python collect_batch.py --batch 1 [--work /work3/$USER/isa]

Reads   <work>/solver/batch_XX/results/<key>_<SA|GR>_s<seed>.txt   the solver output, one file per run
        <work>/results/features_batch_XX.csv                       the feature rows (instance order and keys)
Writes  <work>/results/solver_runs_batch_XX.csv    one row per run: key, algo, seed, served, third party, URT, seconds
        <work>/results/raw_results_batch_XX.csv    per key: mean SA URT over the seeds, GR URT
        <work>/results/perf_matrix_batch_XX.csv    normalized scores (ratio: best / value; best = 1.0)

The objective is the solver's best_total_user_ride_time: total ride time of all passengers, where a passenger
handed to a third party counts as 4 x the direct travel time. Only SA and GR exist for the batches.

The first line of a result file starts with the instance path, which contains spaces ("Capital Region of
Denmark"), so the numbers are located from the token that ends in ".csv", not from the start of the line.
"""

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("ISA_ALGORITHMS", "SA,GR")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd  # noqa: E402
from perf_matrix_export import export_perf_matrix  # noqa: E402

NAME = re.compile(r"^(?P<key>.+)_(?P<algo>SA|GR)_s(?P<seed>\d+)\.txt$")
# Position of each number after the instance path, as written at the end of odbrp-parallel.cpp main().
FIELDS = {"served": 1, "third_party": 2, "best_total_urt": 10}


def parse_first_line(line):
    """The numbers of a result file's first line, whatever spaces the instance path contains."""
    tokens = line.split()
    path_end = next((i for i, t in enumerate(tokens) if t.endswith(".csv")), None)
    if path_end is None:
        raise ValueError("no instance path ending in .csv on the first line")
    row = {name: float(tokens[path_end + offset]) for name, offset in FIELDS.items()}
    row["elapsed_s"] = float(tokens[-2])
    return row


def read_runs(results_dir):
    rows, unreadable = [], []
    for path in sorted(results_dir.glob("*.txt")):
        match = NAME.match(path.name)
        if not match:
            continue
        with open(path, encoding="utf-8") as f:
            first = f.readline()
        try:
            row = {"key": match["key"], "algo": match["algo"], "seed": int(match["seed"]), **parse_first_line(first)}
        except (ValueError, IndexError) as error:
            unreadable.append(f"{path.name} ({error})")
            continue
        rows.append(row)
    return pd.DataFrame(rows), unreadable


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--work", default=os.environ.get("ISA_WORK", f"/work3/{os.environ.get('USER', 'user')}/isa"))
    args = parser.parse_args(argv)

    tag = f"{args.batch:02d}"
    work = Path(args.work)
    results = work / "results"
    features_csv = results / f"features_batch_{tag}.csv"

    runs, unreadable = read_runs(work / "solver" / f"batch_{tag}" / "results")
    if unreadable:
        raise SystemExit(f"{len(unreadable)} result files cannot be read, e.g. {unreadable[:3]}")
    if runs.empty:
        raise SystemExit("no result files")
    runs.to_csv(results / f"solver_runs_batch_{tag}.csv", index=False)

    raw = runs.groupby(["key", "algo"])["best_total_urt"].mean().unstack("algo")
    expected = pd.read_csv(features_csv)["instance"]
    complete = set(raw.dropna().index)
    missing = [key for key in expected if key not in complete]
    if missing:
        raise SystemExit(f"{len(missing)} keys lack SA or GR results, e.g. {missing[:3]}")
    raw = raw.loc[expected]
    raw.rename_axis("instance").to_csv(results / f"raw_results_batch_{tag}.csv")

    export_perf_matrix(
        {key: values.to_dict() for key, values in raw.iterrows()},
        csv_path=results / f"perf_matrix_batch_{tag}.csv",
        features_path=features_csv,
        sense="min",
        method="ratio",
    )

    per_algo = runs.groupby("algo").agg(
        runs=("key", "size"),
        unserved=("third_party", "sum"),
        mean_seconds=("elapsed_s", "mean"),
        max_seconds=("elapsed_s", "max"),
    )
    print(per_algo.round(1).to_string())
    ratio = raw["SA"] / raw["GR"]
    print(f"SA better than GR on {(raw['SA'] < raw['GR']).sum()}/{len(raw)} keys; "
          f"SA/GR cost ratio: mean {ratio.mean():.2f}, min {ratio.min():.2f}, max {ratio.max():.2f}")
    print(f"Wrote {results / f'perf_matrix_batch_{tag}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
