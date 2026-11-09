"""
collect_batch.py - ISA instance set: turn the solver results of one batch into its performance matrix.

    python collect_batch.py --batch 1 [--work /work3/$USER/isa]

Reads   <work>/solver/batch_XX/results/<key>_<SA|SA<cap>|GR>_s<seed>.txt   the solver output, one file per run
        <work>/results/features_batch_XX.csv                       the feature rows (instance order and keys)
Writes  <work>/results/solver_runs_batch_XX.csv    one row per run: key, algo, seed, served, third party, URT, seconds
        <work>/results/raw_results_{mean,best}_batch_XX.csv   per key: SA URT (mean over the seeds / best seed), GR URT
        <work>/results/perf_matrix_{mean,best}_batch_XX.csv   normalized scores (ratio: best / value; best = 1.0)

Three performance measures, each scored with SA by the mean of its seeds and by its best seed:
  total    best_total_user_ride_time: total ride time of all passengers, a passenger handed to a third party
           counts as 4 x the direct travel time (files keep the names perf_matrix_{mean,best}_batch_XX.csv)
  served   share of the passengers served by the own fleet (perf_matrix_served_{mean,best}_batch_XX.csv)
  quality  ride time / direct time averaged over the accepted passengers (perf_matrix_quality_{mean,best}_...)
For small fleets the total is dominated by the third-party penalty; served and quality describe those cases.
quality is only comparable between two algorithms that accept about the same passengers (the printed count).
SA<cap> (e.g. SA2000) is SA with that many counted moves per call (make_solver_jobs.py --sa-variants).
Every algorithm found becomes a column; the script prints the ISA_ALGORITHMS value the instance space pipeline needs.

The first line of a result file starts with the instance path, which contains spaces ("Capital Region of
Denmark"), so the numbers are located from the token that ends in ".csv", not from the start of the line.
"""

import argparse
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd  # noqa: E402

NAME = re.compile(r"^(?P<key>.+)_(?P<algo>SA\d*|GR)_s(?P<seed>\d+)\.txt$")
# Position of each number after the instance path, as written at the end of odbrp-parallel.cpp main().
FIELDS = {"served": 1, "third_party": 2, "avg_extra_s": 5, "avg_ratio": 6, "best_total_urt": 10}
# What is compared: column of the runs table, which direction is better, and a short description.
MEASURES = {
    "total": ("best_total_urt", "min", "total ride time, a third-party passenger counts 4 x the direct time"),
    "served": ("served_share", "max", "share of the passengers served by the own fleet"),
    "quality": ("avg_ratio", "min", "ride time / direct time, averaged over the passengers the fleet accepted"),
}


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
    expected = pd.read_csv(features_csv)["instance"]
    # Columns of the performance matrix: SA, then its variants by move cap, then GR. perf_matrix_export reads
    # the list from ISA_ALGORITHMS when it is imported, so set it first.
    found = set(runs["algo"])
    sa_algos = sorted((a for a in found if a.startswith("SA")), key=lambda a: int(a[2:] or 0))
    algos = sa_algos + sorted(found - set(sa_algos))
    os.environ["ISA_ALGORITHMS"] = ",".join(algos)
    import importlib
    import perf_matrix_export
    importlib.reload(perf_matrix_export)  # picks up ISA_ALGORITHMS even if the module was imported before
    export_perf_matrix = perf_matrix_export.export_perf_matrix

    runs["served_share"] = runs["served"] / (runs["served"] + runs["third_party"])
    complete = set(runs.groupby(["key", "algo"])["best_total_urt"].mean().unstack("algo").dropna().index)
    missing = [key for key in expected if key not in complete]
    if missing:
        raise SystemExit(f"{len(missing)} keys lack results of some of {algos}, e.g. {missing[:3]}")
    runs.to_csv(results / f"solver_runs_batch_{tag}.csv", index=False)

    per_algo = runs.groupby("algo").agg(
        runs=("key", "size"),
        third_party=("third_party", "sum"),
        mean_seconds=("elapsed_s", "mean"),
        max_seconds=("elapsed_s", "max"),
    )
    print(per_algo.round(1).to_string())
    for measure, (column, sense, description) in MEASURES.items():
        print(f"\n{measure}: {description} ({'lower' if sense == 'min' else 'higher'} is better)")
        grouped = runs.groupby(["key", "algo"])[column]
        # SA is reported two ways: the mean over its seeds and its best seed (GR has one run).
        best_of_seeds = grouped.min() if sense == "min" else grouped.max()
        for scoring, values in (("mean", grouped.mean()), ("best", best_of_seeds)):
            raw = values.unstack("algo").loc[expected]
            suffix = f"{scoring}" if measure == "total" else f"{measure}_{scoring}"  # total keeps the first names
            raw.rename_axis("instance").to_csv(results / f"raw_results_{suffix}_batch_{tag}.csv")
            export_perf_matrix(
                {key: row.to_dict() for key, row in raw.iterrows()},
                csv_path=results / f"perf_matrix_{suffix}_batch_{tag}.csv",
                features_path=features_csv,
                sense=sense,
                method="ratio",
            )
            for algo in sa_algos:
                wins = (raw[algo] < raw["GR"]) if sense == "min" else (raw[algo] > raw["GR"])
                ratio = raw[algo] / raw["GR"]
                print(f"  {algo} ({scoring} of seeds) better than GR on {wins.sum()}/{len(raw)} keys; "
                      f"{algo}/GR: mean {ratio.mean():.2f}, min {ratio.min():.2f}, max {ratio.max():.2f}")
            print(f"  -> perf_matrix_{suffix}_batch_{tag}.csv")
    served = runs.groupby(["key", "algo"])["served_share"].mean().unstack("algo").loc[expected]
    print()
    for algo in sa_algos:
        comparable = ((served[algo] - served["GR"]).abs() < 0.02).sum()
        print(f"quality of {algo} against GR is like with like only where both serve about the same "
              f"passengers: {comparable}/{len(served)} keys differ by less than 2 points in served share")
    print(f"\nrun the instance space with ISA_ALGORITHMS={','.join(algos)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
