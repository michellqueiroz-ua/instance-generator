#!/usr/bin/env python
"""
fix_unservable.py - post-generation filter: every request of a batch must be servable by the solver.

The generator sets latest_arrival = earliest_departure + direct_travel_time (node to node) + reaction_time + slack and
never tests the station pairs the solver actually rides between. A request is UNSERVABLE (same test as
local_only/edge_check.py section C and ODBRP_P model/check_requests.py) when

    earliest_departure + max(1, best) > latest_arrival,   best = min over the stop pairs (stops_orgn x stops_dest)
                                                          of the station-to-station matrix time (s, whole seconds)

(walking is NOT part of this test: in the static solver latest_departure stands in for the walk, docs/06 and docs/07).
For every such request:   new latest_arrival = earliest_departure + best + MARGIN   (MARGIN = 300 s, --margin).
Requests that pass are not touched. A request with earliest_departure > latest_departure (cannot be repaired by
latest_arrival) or without stops is only reported, never changed.
--include-walking (off by default) makes `best` the cheapest (walk to pickup + ride + walk from drop-off) over the stop pairs,
walking seconds = stops_*_walking_distance / walk_speed; the unservable test stays the one above, only the new value is larger.

Reads   <work>/<network>/csv_format/ISA_bNN_iXXX_1.csv                 the generated instances (all of batch NN)
        <work>/<network>/travel_time_matrix/station_travel_time_matrix.npy   (the .csv next to it if the .npy is missing)
Writes  with --apply only: csv_format/orig/<file> (copy of the original, never overwritten), the csv in place (ONLY the
        latest_arrival cell of the changed rows; every other byte stays), <work>/design/batch_NN_unservable_fix.csv
        (instance, reqid, old latest_arrival, new latest_arrival, best station time, earliest_departure, walk, applied_at).
Idempotent: a second --apply finds nothing to change, keeps the first log and only touches its modification time (the gate in
hpc_submit_next_batch.sh refuses while an instance csv is newer than the primary log). A second --apply that DOES find
requests to change (e.g. an instance was regenerated) appends them to batch_NN_unservable_fix_2.csv (same columns, applied_at
tells when) and touches the primary log; the gate reads the primary log.
json_format/*.json is NOT rewritten (decision 2026-11-18, option b). Consequence, exact: the feature code reads the csv-derived
values from the json. tw_end comes from latest_departure first (reqreate_adapter.convert_request), so the time-window features are
unchanged, but compute_features_scaffold._compute_geographic_dispersion (line ~1515) takes latest_arrival from the json
(request["latest_arrival"]); for the changed requests (about 1-8 of 5000 per instance) it keeps the OLD, too early value.
That value enters only the omega (detour) term of geographic_dispersion: the destination sorting and the windows
|la_i - la_j| < 10 min, |ed_i - la_j| < 10 min, |la_i - ed_j| < 10 min; mu and every other feature do not use it. The effect is
a tiny change of one feature in a few instances; the chapter must say it. The solver reads the csv only.

  python fix_unservable.py --batch 4 [--dry-run | --apply] [--work DIR] [--margin 300] [--include-walking]
One core, memory below 1 GB (matrix 3515 x 3515 int32 = 50 MB, one instance csv at a time).
"""
import argparse
import csv
import datetime
import io
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

MARGIN_S = 300
LOG_HEADER = ["instance", "reqid", "old_latest_arrival", "new_latest_arrival", "best_station_time_s",
              "earliest_departure", "walk_s_included", "applied_at"]
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def load_matrix(network):
    folder = Path(network) / "travel_time_matrix"
    npy = folder / "station_travel_time_matrix.npy"
    if npy.is_file():
        return np.load(npy, mmap_mode="r")
    cs = folder / "station_travel_time_matrix.csv"
    if cs.is_file():   # pandas layout: header row + index column, floats
        return np.loadtxt(cs, delimiter=",", skiprows=1, usecols=None)[:, 1:].astype(np.int64)
    raise SystemExit(f"no station matrix in {folder}: run station_matrix.sh first")


def ints(text):
    return [int(float(x)) for x in NUM.findall(text)]


def floats(text):
    return [float(x) for x in NUM.findall(text)]


def best_time(matrix, row, include_walking):
    """(matrix_min, best, walk) for one request dict, or None if it has no stops. matrix_min = matrix minimum over the stop
    pairs (the unservable test); best = the value used for the new window (= matrix_min, or with walking the cheapest total)."""
    so, sd = ints(row["stops_orgn"]), ints(row["stops_dest"])
    if not so or not sd:
        return None
    n = matrix.shape[0]
    if max(so) >= n or max(sd) >= n or min(so) < 0 or min(sd) < 0:
        return None
    block = np.asarray(matrix[np.ix_(so, sd)], dtype=np.int64)
    mmin = int(block.min())
    if not include_walking:
        return mmin, mmin, 0
    ws = float(row["walk_speed"])
    wo, wd = floats(row["stops_orgn_walking_distance"]), floats(row["stops_dest_walking_distance"])
    if ws <= 0 or len(wo) != len(so) or len(wd) != len(sd):
        return mmin, mmin, 0
    total = (np.array(wo)[:, None] / ws) + block + (np.array(wd)[None, :] / ws)
    i, j = np.unravel_index(int(np.argmin(total)), total.shape)
    walk = int(np.ceil(wo[i] / ws + wd[j] / ws))
    return mmin, int(block[i, j]) + walk, walk


def scan_file(path, matrix, margin, include_walking):
    """Returns (lines, changes, info). lines: raw text lines with their endings; changes: list of dicts
    {line_index, reqid, old, new, best, ed, walk}; info: counts."""
    raw = Path(path).read_bytes().decode("utf-8")
    lines = raw.splitlines(keepends=True)
    info = {"rows": 0, "no_stops": 0, "ed_gt_ld": 0, "unservable": 0}
    if not lines:
        return lines, [], info, None
    header = next(csv.reader([lines[0]]))
    col = {name: i for i, name in enumerate(header)}
    need = ["earliest_departure", "latest_departure", "latest_arrival", "stops_orgn", "stops_dest", "reqid"]
    if include_walking:
        need += ["walk_speed", "stops_orgn_walking_distance", "stops_dest_walking_distance"]
    missing = [c for c in need if c not in col]
    if missing:
        raise SystemExit(f"{path}: columns missing {missing}")
    changes = []
    for k in range(1, len(lines)):
        if not lines[k].strip():
            continue
        fields = next(csv.reader([lines[k]]))
        if len(fields) != len(header):
            raise SystemExit(f"{path}: line {k + 1} has {len(fields)} fields, header {len(header)}")
        row = {name: fields[i] for name, i in col.items()}
        info["rows"] += 1
        bt = best_time(matrix, row, include_walking)
        if bt is None:
            info["no_stops"] += 1
            continue
        mmin, best, walk = bt
        ed, ld, la = int(float(row["earliest_departure"])), int(float(row["latest_departure"])), int(float(row["latest_arrival"]))
        if ed > ld:
            info["ed_gt_ld"] += 1
            continue
        if ed + max(1, mmin) > la:
            info["unservable"] += 1
            changes.append({"line": k, "reqid": int(float(row["reqid"])), "old": la, "new": ed + best + margin,
                            "best": best, "ed": ed, "walk": walk})
    return lines, changes, info, col["latest_arrival"]


def rewrite_line(line, col_index, new_value):
    """Same line with only the latest_arrival cell replaced (same line ending, same quoting rules as csv QUOTE_MINIMAL)."""
    ending = line[len(line.rstrip("\r\n")):]
    fields = next(csv.reader([line[:len(line) - len(ending)]]))
    fields[col_index] = str(new_value)
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(fields)
    return buf.getvalue() + ending


def apply_file(path, orig_dir, lines, changes, col_index):
    orig_dir.mkdir(parents=True, exist_ok=True)
    backup = orig_dir / Path(path).name
    if not backup.exists():
        shutil.copy2(path, backup)
    out = list(lines)
    for c in changes:
        out[c["line"]] = rewrite_line(lines[c["line"]], col_index, c["new"])
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        f.write("".join(out).encode("utf-8"))
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--work", default=os.environ.get("ISA_WORK", "/work3/micqu/isa"))
    ap.add_argument("--network", default="Capital Region of Denmark")
    ap.add_argument("--margin", type=int, default=MARGIN_S, help="seconds added to the best station time (default 300)")
    ap.add_argument("--include-walking", action="store_true", help="best time includes the walk to and from the stations")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="default: list, write nothing")
    mode.add_argument("--apply", action="store_true", help="rewrite the csv files in place (original copied to csv_format/orig/)")
    args = ap.parse_args(argv)

    work = Path(args.work)
    network = work / args.network
    tag = f"{args.batch:02d}"
    files = sorted((network / "csv_format").glob(f"ISA_b{tag}_i*_1.csv"))
    if not files:
        raise SystemExit(f"no ISA_b{tag}_i*_1.csv in {network / 'csv_format'}")
    matrix = load_matrix(network)
    print(f"{'APPLY' if args.apply else 'DRY RUN'}: batch {tag}, {len(files)} instance files, matrix {matrix.shape}, "
          f"margin {args.margin} s, walking {'included' if args.include_walking else 'not included (checker definition)'}")

    applied_at = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    log_rows, tot = [], {"rows": 0, "unservable": 0, "no_stops": 0, "ed_gt_ld": 0}
    affected = 0
    for path in files:
        lines, changes, info, col_index = scan_file(path, matrix, args.margin, args.include_walking)
        for key in tot:
            tot[key] += info[key]
        name = path.stem
        if changes:
            affected += 1
            print(f"{name}: {len(changes)} of {info['rows']} requests unservable")
            for c in changes:
                print(f"    reqid {c['reqid']}: latest_arrival {c['old']} -> {c['new']} (earliest_departure {c['ed']}, "
                      f"best station time {c['best']} s)")
                log_rows.append([name, c["reqid"], c["old"], c["new"], c["best"], c["ed"], c["walk"], applied_at])
            if args.apply:
                apply_file(path, path.parent / "orig", lines, changes, col_index)
        if info["ed_gt_ld"] or info["no_stops"]:
            print(f"{name}: NOT changed (cannot be fixed by latest_arrival): {info['ed_gt_ld']} with earliest_departure > "
                  f"latest_departure, {info['no_stops']} without usable stops")
    print(f"TOTAL: {tot['rows']} requests, {tot['unservable']} unservable in {affected} instances "
          f"({100.0 * tot['unservable'] / max(1, tot['rows']):.4f} %); not fixable: {tot['ed_gt_ld']} (ed > ld), {tot['no_stops']} (no stops)")
    if args.apply:
        log = work / "design" / f"batch_{tag}_unservable_fix.csv"
        log2 = work / "design" / f"batch_{tag}_unservable_fix_2.csv"
        log.parent.mkdir(parents=True, exist_ok=True)
        if not log.exists():
            with open(log, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(LOG_HEADER)
                w.writerows(log_rows)
            print(f"log written: {log} ({len(log_rows)} changes)")
        elif log_rows:
            new_file = not log2.exists()
            with open(log2, "a", encoding="utf-8", newline="") as f:
                w = csv.writer(f, lineterminator="\n")
                if new_file:
                    w.writerow(LOG_HEADER)
                w.writerows(log_rows)
            os.utime(log)   # the primary log must stay newer than the csv files just rewritten (gate need_fixlog)
            print(f"second apply: {len(log_rows)} changes appended to {log2}; the first log {log} is kept (modification time refreshed)")
        else:
            os.utime(log)   # nothing to change: the log is current for the csv files as they are now
            print(f"nothing to change; the existing log {log} is kept (modification time refreshed)")
        print(f"originals in {network / 'csv_format' / 'orig'}")
        print(f"NOTE: json_format/*.json was NOT rewritten. For the {tot['unservable']} changed requests the json keeps the old "
              "latest_arrival. The solver reads the csv (fixed). The features read the json: only geographic_dispersion "
              "(omega term, latest_arrival windows) sees the old value; time-window features use latest_departure and are unchanged. "
              "Say so in the chapter.")
        print("FIXWIN DONE")
    else:
        print("dry run: nothing written (add --apply to rewrite the csv files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
