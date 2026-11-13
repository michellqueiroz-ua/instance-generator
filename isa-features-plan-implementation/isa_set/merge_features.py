"""
merge_features.py - ISA instance set: merge the part files of a parallel features run into features_batch_XX.csv.

    python merge_features.py --batch 2 [--work /work3/$USER/isa]

Reads   <work>/results/features_batch_XX_part*.csv   one file per parallel job (features_batch.py --out ...)
        <work>/results/features_batch_XX.csv         rows already there (for example from a test run), if any
Writes  <work>/results/features_batch_XX.csv         all rows, one per key (a later part replaces an earlier row)

features_batch.py rewrites its csv completely on every flush, so parallel jobs must not share one file: each job
writes its own part and this script merges them afterwards. The merged file is validated like the single-job output.
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (HERE.parent, HERE.parent.parent):
    sys.path.insert(0, str(path))

import pandas as pd  # noqa: E402

from features_export import validate_features_csv  # noqa: E402


def merge(results_dir, tag):
    parts = sorted(results_dir.glob(f"features_batch_{tag}_part*.csv"))
    merged = results_dir / f"features_batch_{tag}.csv"
    files = ([merged] if merged.is_file() else []) + parts
    if not parts:
        raise SystemExit(f"no part files features_batch_{tag}_part*.csv in {results_dir}")
    frames = [pd.read_csv(f) for f in files]
    frame = pd.concat(frames, ignore_index=True)
    key = frame.columns[0]
    frame = frame.drop_duplicates(subset=key, keep="last").sort_values(key)
    frame.to_csv(merged, index=False)
    return merged, len(parts), len(frame)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--work", default=os.environ.get("ISA_WORK", f"/work3/{os.environ.get('USER', 'user')}/isa"))
    args = parser.parse_args(argv)
    merged, n_parts, n_rows = merge(Path(args.work) / "results", f"{args.batch:02d}")
    print(f"{n_parts} part files merged into {merged}: {n_rows} rows")
    problems = validate_features_csv(merged)
    for problem in problems:
        print(f"  problem: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
