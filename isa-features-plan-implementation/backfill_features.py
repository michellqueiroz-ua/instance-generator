"""
backfill_features.py — recompute features.csv for already-generated instances

The T12 hook writes one features.csv row as each instance is generated.  This
script does the same for instances that already exist on disk — the ones
generated before the hook, or after editing ``fleet_config.json``, since
features 11-14 change with the fleet.

    python isa-features-plan-implementation/backfill_features.py
    python isa-features-plan-implementation/backfill_features.py --dry-run
    python isa-features-plan-implementation/backfill_features.py --pattern "Maastricht*/json_format/*.json"

Rows are keyed by the instance file stem and replace any existing row for the
same stem, so re-running is safe and idempotent.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

_THIS_DIR = Path(__file__).resolve().parent
# The checkout root goes first so network pickles unpickle with this
# checkout's REQreate rather than an older installed one.
for _path in (_THIS_DIR.parent, _THIS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from compute_features_scaffold import compute_features
from csv_table import instance_stem, repo_root
from features_export import (
    append_features_rows,
    resolve_features_csv_path,
    validate_features_csv,
)
from fleet_config import load_fleet_config, resolve_fleet, resolve_fleet_config_path
from reqreate_adapter import load_reqreate_instance


LOGGER = logging.getLogger(__name__)

DEFAULT_PATTERN = "*/json_format/**/*.json"


def find_instance_files(pattern: str = DEFAULT_PATTERN, root: Optional[Path] = None) -> List[Path]:
    """Generated instance JSONs under the repository, in a stable order."""
    base = root or repo_root()
    return sorted(path for path in base.glob(pattern) if path.is_file())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute data/results/features.csv from generated instances."
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"glob for instance JSONs, relative to the repo root (default: {DEFAULT_PATTERN})",
    )
    parser.add_argument("--out", help="features.csv path (default: data/results/)")
    parser.add_argument("--fleet-config", help="fleet config JSON (default: fleet_config.json)")
    parser.add_argument("--limit", type=int, help="stop after N instances")
    parser.add_argument(
        "--max-requests",
        type=int,
        help=(
            "skip instances above this request count; the O(n^2) features "
            "(spatial_cluster_coeff, tw_overlap_ratio) get slow past ~2000"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be written without touching features.csv",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint: recompute feature rows for instances already on disk."""
    args = _build_arg_parser().parse_args(argv)

    features_path = resolve_features_csv_path(args.out)
    fleet_config = load_fleet_config(args.fleet_config)
    instance_files = find_instance_files(args.pattern)

    if args.limit:
        instance_files = instance_files[: args.limit]

    if not instance_files:
        print(f"No instance JSON matched {args.pattern!r} under {repo_root()}")
        return 1

    print(f"features.csv : {features_path}")
    print(f"fleet config : {resolve_fleet_config_path(args.fleet_config)}"
          f"{'' if fleet_config else ' (absent - fleet features stay NaN)'}")
    print(f"instances    : {len(instance_files)}")
    print()

    items = []
    skipped: List[str] = []

    for path in instance_files:
        stem = instance_stem(path)
        started = time.time()
        try:
            instance = load_reqreate_instance(path)
        except Exception as exc:
            LOGGER.warning("skipping %s: %s", path.name, exc)
            skipped.append(f"{stem} ({exc})")
            continue

        n_requests = len(instance.get("requests", []))
        if args.max_requests and n_requests > args.max_requests:
            skipped.append(f"{stem} ({n_requests} requests > --max-requests)")
            continue

        instance["vehicles"] = resolve_fleet(
            stem, n_requests=n_requests, config=fleet_config
        )

        features = compute_features(instance)
        items.append((stem, features))
        print(
            f"  {stem}: {n_requests} requests, "
            f"{len(instance['vehicles'])} vehicles, {time.time() - started:.1f}s"
        )

    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for entry in skipped:
            print(f"  - {entry}")

    if args.dry_run:
        print(f"\nDry run: {len(items)} row(s) would be written to {features_path}")
        return 0

    if not items:
        print("\nNothing to write.")
        return 1

    append_features_rows(items, csv_path=features_path)

    problems = validate_features_csv(features_path)
    if problems:
        print(f"\nfeatures.csv has {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"\nWrote {len(items)} row(s) to {features_path}; all checks pass.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(main())
