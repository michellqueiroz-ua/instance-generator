"""
features_batch.py - ISA instance set: compute the ISA features of one batch, one row per instance and fleet.

    python features_batch.py --batch 1 [--work /work3/$USER/isa] [--first 1 --last 100]

Reads   <work>/design/batch_XX_fleets.json                 the fleets per instance
        <work>/<network>/json_format/<instance>.json       the generated instances
        <work>/<network>/travel_time_matrix/station_travel_time_matrix.npy   (attached by the loader)
Writes  <work>/results/features_batch_XX.csv               key = the fleet key, e.g. ISA_b01_i001_1_fleet2

The fleet features (11-14) depend on the fleet, so each instance gives one row per fleet; the other
features repeat. The instance is loaded once and the network pickle once per process. Rows are written every
10 instances, so a killed job keeps its progress; rerunning replaces rows by key.
"""

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMPLEMENTATION = HERE.parent
# The checkout root goes first so network pickles unpickle with this checkout's REQreate.
for path in (IMPLEMENTATION, IMPLEMENTATION.parent):
    sys.path.insert(0, str(path))

import json  # noqa: E402

from compute_features_scaffold import compute_features  # noqa: E402
from features_export import append_features_rows, validate_features_csv  # noqa: E402
from fleet_config import vehicles_from_spec  # noqa: E402
from reqreate_adapter import load_reqreate_instance  # noqa: E402

FLUSH_EVERY = 10


def peak_memory_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024  # KB on Linux
    except ImportError:
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--work", default=os.environ.get("ISA_WORK", f"/work3/{os.environ.get('USER', 'user')}/isa"))
    parser.add_argument("--network", default="Capital Region of Denmark")
    parser.add_argument("--first", type=int, default=1, help="first instance number")
    parser.add_argument("--last", type=int, default=10**6, help="last instance number")
    parser.add_argument("--out", help="features csv (default: <work>/results/features_batch_XX.csv)")
    args = parser.parse_args(argv)

    work = Path(args.work)
    tag = f"{args.batch:02d}"
    fleets = json.loads((work / "design" / f"batch_{tag}_fleets.json").read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else work / "results" / f"features_batch_{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    by_instance = {}
    for key, fleet in fleets.items():
        by_instance.setdefault(fleet["instance"], []).append((key, fleet))

    items, done, skipped = [], 0, []
    started = time.time()
    for name in sorted(by_instance):
        number = int(name.split("_i")[1].split("_")[0])
        if not args.first <= number <= args.last:
            continue
        path = work / args.network / "json_format" / f"{name}.json"
        if not path.is_file():
            skipped.append(name)
            continue
        began = time.time()
        instance = load_reqreate_instance(path)
        if "travel_time_matrix_sec" not in instance:
            raise SystemExit(f"no station matrix attached to {name}: run station_matrix.sh first")
        n_requests = len(instance["requests"])
        for key, fleet in sorted(by_instance[name]):
            instance["vehicles"] = vehicles_from_spec(fleet, n_requests=n_requests)
            items.append((key, compute_features(instance)))
        done += 1
        print(f"{name}: {n_requests} requests, {len(by_instance[name])} fleets, {time.time() - began:.1f} s, "
              f"peak memory {peak_memory_mb()} MB", flush=True)
        if done % FLUSH_EVERY == 0:
            append_features_rows(items, csv_path=out)
            items = []

    if items:
        append_features_rows(items, csv_path=out)
    problems = validate_features_csv(out)
    print(f"{done} instances in {time.time() - started:.0f} s, {len(skipped)} skipped (no json), file {out}")
    for problem in problems:
        print(f"  problem: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
