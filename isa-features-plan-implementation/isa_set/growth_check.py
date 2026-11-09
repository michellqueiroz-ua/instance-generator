"""
growth_check.py - ISA instance set: does a new batch still add new territory to the instance space?

    python growth_check.py --new <work>/results/features_batch_02.csv
                           [--existing <csv> ...]   (default: every other features_batch_*.csv next to --new)
                           [--far-quantile 0.95] [--margin 0.03] [--cap 3000]

Works in the standardized feature space (mean and standard deviation of the existing set), so it does not
depend on a projection that changes from batch to batch. A new row is "far" when its nearest existing row
is farther than the ``far-quantile`` (default 95%) of the nearest-neighbour distances the existing rows have
to each other. A batch that fills the same space therefore has about 5% far rows by construction (the
baseline); a batch in new territory has far more. The verdict is STOP when the far share is at most the
baseline plus ``margin`` (default 3 points, about two standard errors for a batch of 200), or the total
number of rows reaches ``cap``; otherwise CONTINUE (draw another batch).

Also reported: the share of rows outside the existing range of at least one feature, and per feature.
Columns that are empty or constant in the existing set are left out. Rows are (instance, fleet) pairs.
Exit code: 0 = CONTINUE, 1 = STOP, 2 = nothing to compare.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = [c for c in reader.fieldnames if c != "instance"]
        keys, data = [], []
        for row in reader:
            keys.append(row["instance"])
            data.append([float(row[c]) if row[c] not in ("", "nan", "NaN") else np.nan for c in columns])
    return columns, keys, np.array(data, dtype=float)


def usable_columns(existing):
    """Columns with a value in every existing row and a spread."""
    finite = np.isfinite(existing).all(axis=0)
    spread = np.nanstd(existing, axis=0) > 0
    return finite & spread


def growth(existing, new, far_quantile):
    """Share of new rows far from the existing set, share outside its range, and the numbers behind them."""
    mean, std = existing.mean(axis=0), existing.std(axis=0)
    old_z, new_z = (existing - mean) / std, (new - mean) / std
    tree = cKDTree(old_z)
    own = tree.query(old_z, k=2)[0][:, 1]  # k=1 is the row itself
    threshold = float(np.quantile(own, far_quantile))
    distance = tree.query(new_z, k=1)[0]
    far = distance > threshold
    low, high = existing.min(axis=0), existing.max(axis=0)
    outside = (new < low) | (new > high)
    return {
        "spacing": float(np.median(own)),
        "threshold": threshold,
        "far_share": float(far.mean()),
        "outside_share": float(outside.any(axis=1).mean()),
        "outside_by_column": outside.mean(axis=0),
        "distance": distance,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--new", required=True, help="features csv of the batch to judge")
    parser.add_argument("--existing", nargs="*", help="features csvs of the earlier batches")
    parser.add_argument("--far-quantile", type=float, default=0.95)
    parser.add_argument("--margin", type=float, default=0.03, help="allowed far share above the baseline")
    parser.add_argument("--cap", type=int, default=3000, help="stop when the set reaches this many rows")
    args = parser.parse_args(argv)

    new_path = Path(args.new)
    existing_paths = [Path(p) for p in args.existing] if args.existing is not None else sorted(
        p for p in new_path.parent.glob("features_batch_*.csv") if p.resolve() != new_path.resolve()
    )
    if not existing_paths:
        print("no earlier batch to compare with: nothing to judge, draw another batch")
        return 2

    columns, _, new = read_rows(new_path)
    old_parts = []
    for path in existing_paths:
        old_columns, _, part = read_rows(path)
        if old_columns != columns:
            raise SystemExit(f"{path} has other feature columns than {new_path}")
        old_parts.append(part)
    existing = np.vstack(old_parts)

    keep = usable_columns(existing)
    names = [c for c, k in zip(columns, keep) if k]
    new = new[:, keep]
    existing = existing[:, keep]
    bad = ~np.isfinite(new).all(axis=1)
    if bad.any():
        print(f"{int(bad.sum())} new rows have a missing value in a used feature and are left out")
        new = new[~bad]
    if not len(new):
        print("no usable new rows")
        return 2

    result = growth(existing, new, args.far_quantile)
    baseline = 1 - args.far_quantile
    total = len(existing) + len(new)
    print(f"existing rows {len(existing)}, new rows {len(new)}, features used {len(names)} of {len(columns)}")
    print(f"typical spacing (median nearest-neighbour distance, standardized): {result['spacing']:.3f}")
    print(f"new rows farther than {result['threshold']:.3f} from the existing set "
          f"(the {args.far_quantile:.0%} quantile of the existing rows' own distances): {result['far_share']:.1%} "
          f"(baseline {baseline:.0%})")
    print(f"new rows outside the existing range of some feature: {result['outside_share']:.1%}")
    worst = np.argsort(result["outside_by_column"])[::-1][:3]
    listed = [f"{names[i]} {result['outside_by_column'][i]:.0%}" for i in worst if result["outside_by_column"][i] > 0]
    print("features most often outside the existing range: " + (", ".join(listed) or "none"))

    if total >= args.cap:
        print(f"STOP: {total} rows reach the cap of {args.cap}")
        return 1
    limit = baseline + args.margin
    if result["far_share"] <= limit:
        print(f"STOP: far share {result['far_share']:.1%} is within {limit:.0%} (baseline plus margin)")
        return 1
    print(f"CONTINUE: far share {result['far_share']:.1%} is above {limit:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
