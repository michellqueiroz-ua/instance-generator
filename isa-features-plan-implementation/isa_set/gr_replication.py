"""
gr_replication.py - ISA instance set: how much does the randomised GR construction vary with its seed?

    python gr_replication.py --batch 4 [--work /work3/$USER/isa]

Reads   <work>/solver/batch_XX/results/<key>_GR_s<seed>.txt   GR runs of several seeds (make_solver_jobs.py --gr-extra-seeds)
Writes  <work>/results/gr_replication_batch_XX.csv   per key with at least 2 seeds: n seeds, mean, sd, min, max of
        best_total_urt and of the served share, and the coefficient of variation (sd / mean) of best_total_urt
Prints  for how many keys the seed-1 value is within one sd of the mean, and the median coefficient of variation.
The result files are read with collect_batch.read_runs, the same parser as the performance matrices.
Standard library, numpy and pandas only.
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

from collect_batch import read_runs  # noqa: E402


def summarise(runs):
    """(table with one row per key that has >= 2 GR seeds, seed-1 best_total_urt per key). sd is the sample sd (ddof=1)."""
    gr = runs[runs["algo"] == "GR"].copy()
    gr["served_share"] = gr["served"] / (gr["served"] + gr["third_party"])
    counts = gr.groupby("key")["seed"].nunique()
    gr = gr[gr["key"].isin(counts[counts >= 2].index)]
    if gr.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    stats = gr.groupby("key").agg(
        n_seeds=("seed", "nunique"),
        urt_mean=("best_total_urt", "mean"), urt_sd=("best_total_urt", "std"),
        urt_min=("best_total_urt", "min"), urt_max=("best_total_urt", "max"),
        served_mean=("served_share", "mean"), served_sd=("served_share", "std"),
        served_min=("served_share", "min"), served_max=("served_share", "max"),
    )
    stats["urt_cv"] = stats["urt_sd"] / stats["urt_mean"]
    seed1 = gr[gr["seed"] == 1].set_index("key")["best_total_urt"]
    return stats, seed1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--work", default=os.environ.get("ISA_WORK", f"/work3/{os.environ.get('USER', 'user')}/isa"))
    args = parser.parse_args(argv)

    tag = f"{args.batch:02d}"
    work = Path(args.work)
    runs, unreadable = read_runs(work / "solver" / f"batch_{tag}" / "results")
    if unreadable:
        raise SystemExit(f"{len(unreadable)} result files cannot be read, e.g. {unreadable[:3]}")
    if runs.empty:
        raise SystemExit("no result files")
    stats, seed1 = summarise(runs)
    if stats.empty:
        raise SystemExit("no key has GR results of two or more seeds")
    results = work / "results"
    results.mkdir(parents=True, exist_ok=True)
    out = results / f"gr_replication_batch_{tag}.csv"
    stats.rename_axis("key").to_csv(out)

    within = [abs(seed1[k] - r["urt_mean"]) <= r["urt_sd"] for k, r in stats.iterrows() if k in seed1.index]
    print(f"{len(stats)} keys with 2 or more GR seeds (seeds per key: min {int(stats['n_seeds'].min())}, "
          f"max {int(stats['n_seeds'].max())})")
    print(f"seed-1 total ride time within one sd of the mean over the seeds: {sum(within)}/{len(within)} keys")
    print(f"median coefficient of variation of the total ride time: {stats['urt_cv'].median():.4f}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
