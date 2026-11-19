"""
check_design.py - does a generated batch match isa_set/design.json?

    python tools/check_design.py --design-json isa-features-plan-implementation/isa_set/design.json \
        --batch-dir <work folder> --batch 4 [--n 100] [--seed 1]

--batch-dir is the folder that make_batch.py wrote to (its --work): it holds design/batch_NN.csv,
design/batch_NN_fleets.json and configs/batch_NN/. A flat folder with the three files side by side also works.

Checks (PASS or FAIL per line, exit code 1 if any FAIL):
  instances  row count, every varied setting inside its range, Latin hypercube (one point per slice per setting),
             fixed settings (start_hour, requests_per_hour, requests) as in the design
  fleets     n * draws_per_instance entries, fleet percent inside its range, exactly draws_per_instance per
             instance, one fleet per equal stratum of the fleet-size range (when assignment is "stratified"),
             8-seat share inside its range, composition sums to 100
  configs    number of configs, and in the first and one random config: max_walking has loc = the fixed
             walking time and scale 0, requests as in the design

Only the standard library is used.
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

RANGE_TOL = 1e-3     # the csv rounds settings to 4 decimals
PERCENT_TOL = 0.006  # the fleet percent is rounded to 2 decimals (task brief)
SHARE_TOL = 0.06     # the 8-seat share (in percent) is rounded to 1 decimal
MAX_LISTED = 5       # how many offenders a FAIL line names


class Report:
    def __init__(self):
        self.lines = []

    def add(self, name, ok, detail=""):
        self.lines.append((name, bool(ok), detail))

    @property
    def failed(self):
        return [line for line in self.lines if not line[1]]

    def text(self):
        out = [f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else "")
               for name, ok, detail in self.lines]
        out.append(f"{len(self.lines) - len(self.failed)} PASS, {len(self.failed)} FAIL")
        return "\n".join(out)


def listed(items):
    items = list(items)
    shown = ", ".join(str(i) for i in items[:MAX_LISTED])
    return shown + (f" (and {len(items) - MAX_LISTED} more)" if len(items) > MAX_LISTED else "")


def find_file(batch_dir, relative_candidates):
    for relative in relative_candidates:
        path = batch_dir / relative
        if path.exists():
            return path
    return None


def check_instances(design, rows, n, report):
    report.add("instances: row count", len(rows) == n, f"{len(rows)} rows, expected {n}")
    names = [r["instance"] for r in rows]
    report.add("instances: names unique", len(set(names)) == len(names), f"{len(set(names))} distinct of {len(names)}")
    varied = {k: v for k, v in design["instance"].items() if isinstance(v, list)}
    fixed = {k: v for k, v in design["instance"].items() if not isinstance(v, list)}
    for name, (low, high) in varied.items():
        if rows and name not in rows[0]:
            report.add(f"instances: {name} in range", False, "column missing in the csv")
            continue
        out = [(r["instance"], r[name]) for r in rows if not low - RANGE_TOL <= float(r[name]) <= high + RANGE_TOL]
        report.add(f"instances: {name} in [{low}, {high}]", not out,
                   f"{len(rows) - len(out)} of {len(rows)} inside" + (f"; outside: {listed(out)}" if out else ""))
    for name, (low, high) in varied.items():
        if rows and name not in rows[0]:
            continue
        slices = Counter()
        for r in rows:
            u = (float(r[name]) - low) / (high - low)
            slices[min(n - 1, max(0, int(u * n)))] += 1
        empty = [s for s in range(n) if slices[s] == 0]
        crowded = [s for s in range(n) if slices[s] > 1]
        report.add(f"instances: {name} Latin hypercube", not empty and not crowded,
                   f"{n - len(empty)} of {n} slices held, {len(crowded)} slices with more than one point"
                   + (f"; empty slices {listed(empty)}, crowded {listed(crowded)}" if empty or crowded else ""))
    for name, value in fixed.items():
        if rows and name in rows[0]:
            bad = [r["instance"] for r in rows if abs(float(r[name]) - value) > RANGE_TOL]
            report.add(f"instances: {name} fixed at {value}", not bad,
                       f"{len(rows) - len(bad)} of {len(rows)} equal" + (f"; differ: {listed(bad)}" if bad else ""))
        else:
            report.add(f"instances: {name} fixed at {value}", False, "column missing in the csv")
    expected_requests = max(1, round(float(design["instance"]["requests_per_hour"]) * design["period_hours"])) \
        if not isinstance(design["instance"].get("requests_per_hour"), list) else None
    if expected_requests is not None and rows and "requests" in rows[0]:
        bad = [r["instance"] for r in rows if int(float(r["requests"])) != expected_requests]
        report.add(f"instances: requests = {expected_requests}", not bad,
                   f"{len(rows) - len(bad)} of {len(rows)} equal" + (f"; differ: {listed(bad)}" if bad else ""))


def stratum_matches(percents, low, high, k, tol):
    """True if the k values can be put one per equal stratum of [low, high] (strata are ordered, so sorting is enough)."""
    width = (high - low) / k
    return all(low + s * width - tol <= v <= low + (s + 1) * width + tol for s, v in enumerate(sorted(percents)))


def check_fleets(design, fleets, instance_names, n, report):
    fleet = design["fleet"]
    k = fleet["draws_per_instance"]
    report.add("fleets: count", len(fleets) == n * k, f"{len(fleets)} entries, expected {n} x {k} = {n * k}")
    low, high = fleet["percent_of_requests"]
    out = [key for key, f in fleets.items()
           if not low - PERCENT_TOL <= float(f["fleet_percent_of_requests"]) <= high + PERCENT_TOL]
    report.add(f"fleets: percent in [{low}, {high}]", not out,
               f"{len(fleets) - len(out)} of {len(fleets)} inside" + (f"; outside: {listed(out)}" if out else ""))
    per_instance = defaultdict(list)
    for key, f in fleets.items():
        per_instance[f["instance"]].append(float(f["fleet_percent_of_requests"]))
    expected_instances = {f"{name}_1" for name in instance_names}
    unknown = sorted(set(per_instance) - expected_instances)
    missing = sorted(expected_instances - set(per_instance))
    report.add("fleets: instances match the csv", not unknown and not missing,
               f"{len(per_instance)} instances with fleets" + (f"; unknown {listed(unknown)}" if unknown else "")
               + (f"; without fleets {listed(missing)}" if missing else ""))
    wrong = [f"{i}:{len(v)}" for i, v in sorted(per_instance.items()) if len(v) != k]
    report.add(f"fleets: exactly {k} per instance", not wrong,
               f"{len(per_instance) - len(wrong)} of {len(per_instance)} instances" + (f"; wrong: {listed(wrong)}" if wrong else ""))
    if fleet.get("assignment", "random") == "stratified":
        bad = [i for i, v in sorted(per_instance.items()) if len(v) == k and not stratum_matches(v, low, high, k, PERCENT_TOL)]
        report.add(f"fleets: one fleet per stratum ({k} strata of {low}-{high}%)", not bad and not wrong,
                   f"{len(per_instance) - len(bad) - len(wrong)} of {len(per_instance)} instances span all strata"
                   + (f"; do not: {listed(bad)}" if bad else ""))
    s_low, s_high = (100 * v for v in fleet["share_8_seats"])
    bad_share, bad_sum = [], []
    for key, f in fleets.items():
        comp = {c["capacity"]: float(c["percent"]) for c in f["composition"]}
        if not s_low - SHARE_TOL <= comp.get(8, -1) <= s_high + SHARE_TOL:
            bad_share.append(key)
        if abs(sum(comp.values()) - 100) > SHARE_TOL:
            bad_sum.append(key)
    report.add(f"fleets: 8-seat share in [{s_low:g}, {s_high:g}]%", not bad_share,
               f"{len(fleets) - len(bad_share)} of {len(fleets)} inside" + (f"; outside: {listed(bad_share)}" if bad_share else ""))
    report.add("fleets: composition sums to 100%", not bad_sum,
               f"{len(fleets) - len(bad_sum)} of {len(fleets)}" + (f"; off: {listed(bad_sum)}" if bad_sum else ""))


def check_configs(design, config_dir, n, seed, report):
    paths = sorted(p for p in config_dir.glob("*.json"))
    report.add("configs: count", len(paths) == n, f"{len(paths)} config files, expected {n}")
    if not paths:
        return
    walk = design["fixed"]["max_walking_s"]
    walk_low, walk_high = (walk, walk) if not isinstance(walk, list) else walk
    # UNSURE: the brief says scale 0; make_batch.py writes scale = high - low, which is 0 only while the range is a point.
    expected_scale = 0
    picks = [paths[0]]
    others = paths[1:]
    if others:
        picks.append(random.Random(seed).choice(others))
    for path in picks:
        config = json.loads(path.read_text(encoding="utf-8"))
        attrs = {a["name"]: a for a in config.get("attributes", [])}
        pdf = (attrs.get("max_walking") or {}).get("pdf", [{}])[0]
        ok = pdf.get("loc") == walk_low and pdf.get("scale") == expected_scale and walk_high == walk_low
        report.add(f"configs: {path.name} max_walking loc = {walk_low}, scale = 0", ok,
                   f"loc {pdf.get('loc')}, scale {pdf.get('scale')}")
        expected_requests = None
        if not isinstance(design["instance"].get("requests_per_hour"), list):
            expected_requests = max(1, round(float(design["instance"]["requests_per_hour"]) * design["period_hours"]))
        if expected_requests is not None:
            report.add(f"configs: {path.name} requests = {expected_requests}", config.get("requests") == expected_requests,
                       f"requests {config.get('requests')}")


def run(design_json, batch_dir, batch, n=100, seed=1):
    design = json.loads(Path(design_json).read_text(encoding="utf-8"))
    batch_dir = Path(batch_dir)
    report = Report()
    tag = f"batch_{batch:02d}"
    csv_path = find_file(batch_dir, [f"design/{tag}.csv", f"{tag}.csv"])
    fleets_path = find_file(batch_dir, [f"design/{tag}_fleets.json", f"{tag}_fleets.json"])
    config_dir = find_file(batch_dir, [f"configs/{tag}", tag])
    for label, path in (("instance table", csv_path), ("fleets file", fleets_path)):
        if path is None:
            report.add(f"files: {label} found", False, f"no {tag} file under {batch_dir}")
    if csv_path is None or fleets_path is None:
        return report
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    fleets = json.loads(fleets_path.read_text(encoding="utf-8"))
    check_instances(design, rows, n, report)
    check_fleets(design, fleets, [r["instance"] for r in rows], n, report)
    if config_dir is None:
        report.add("configs: folder found", False, f"no configs/{tag} under {batch_dir}")
    else:
        check_configs(design, config_dir, n, seed, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check a generated batch against design.json.")
    parser.add_argument("--design-json", required=True)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--n", type=int, default=100, help="instances expected in the batch")
    parser.add_argument("--seed", type=int, default=1, help="seed for the random config that is checked")
    args = parser.parse_args(argv)
    report = run(args.design_json, args.batch_dir, args.batch, args.n, args.seed)
    print(report.text())
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
