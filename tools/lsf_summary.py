"""
lsf_summary.py - read LSF job logs and compare them with the pass table of the probe.

    python tools/lsf_summary.py <log files or folders> [--kind solver|features|generation|other] [--rows-per-block 10]

Per log (one line per job; a file holding several "Resource usage summary" blocks gives one line per block):
status (Successfully completed / Exited with exit code N), any TERM_ line, CPU time, Run time, utilisation
(CPU time / Run time), Max Memory in GB, Max Swap. A field that is not in the log is printed as "missing";
nothing is guessed.

Per kind: jobs, failed jobs, utilisation, median and maximum Run time, maximum Max Memory, and PASS or FAIL
against the pass table (local_only/FIXED_BATCH_PLAN.md, update paragraphs b and d):
  solver    Max Memory <= 1.7 GB, CPU time / Run time >= 0.90, Run time of a block (10 rows) <= 90 min,
            longest row <= 1100 s (a log only gives the block time: the mean row time is checked, the longest row
            stays "missing" and needs the result files)
  features  Run time <= 2 h, Max Memory <= 1.7 GB
  generation, other: no pass table; only failed jobs make the verdict FAIL
A job that failed (exit code, TERM_ line) or has no readable summary makes the kind FAIL.
--kind sets the kind for every log; without it the kind comes from the file name (rows_* solver, features* features,
gen* generation, otherwise other). Folders are searched for *.out and *.log.

Exit code 1 if any kind is FAIL. Only the standard library is used.
"""

import argparse
import re
import statistics
import sys
from pathlib import Path

KINDS = ("solver", "features", "generation", "other")
MAX_MEMORY_GB = 1.7  # UNSURE: LSF counts 1024 MB per GB, so 1.7 GB is 1740.8 MB
MIN_UTILISATION = 0.90
SOLVER_BLOCK_MAX_S = 90 * 60
LONGEST_ROW_MAX_S = 1100
FEATURES_MAX_S = 2 * 3600

SUMMARY = re.compile(r"Resource usage summary")
STATUS = re.compile(r"Successfully completed|Exited with exit code (\d+)")
TERM = re.compile(r"\bTERM_[A-Z_]+")
UNITS = {"KB": 1 / 1024 ** 2, "MB": 1 / 1024, "GB": 1.0, "TB": 1024.0}


def number(text, label):
    m = re.search(rf"^\s*{label}\s*:\s*([\d.]+)\s*(\w*)", text, re.M)
    return (float(m.group(1)), m.group(2)) if m else None


def swap_gb(summary):
    """Max Swap in GB; '-' when the log prints a dash (LSF: no swap reported); None when the line is absent."""
    value = number(summary, "Max Swap")
    if value:
        return to_gb(value)
    return "-" if re.search(r"^\s*Max Swap\s*:\s*-", summary, re.M) else None


def to_gb(value):
    return value[0] * UNITS.get(value[1].upper(), UNITS["MB"]) if value else None


def parse_block(segment, summary):
    """One job: ``segment`` is the log text up to the end of its summary, ``summary`` the summary block itself."""
    statuses = list(STATUS.finditer(segment))
    status, ok = "missing", None
    if statuses:
        last = statuses[-1]
        ok = last.group(0).startswith("Successfully")
        status = "completed" if ok else f"exit {last.group(1)}"
    cpu = number(summary, "CPU time")
    run = number(summary, "Run time")
    return {"status": status, "ok": ok, "term": sorted(set(TERM.findall(segment))),
            "cpu": cpu[0] if cpu else None, "run": run[0] if run else None,
            "mem_gb": to_gb(number(summary, "Max Memory")), "swap_gb": swap_gb(summary)}


def parse_log(text):
    """A list of jobs, one per 'Resource usage summary' block. A text without a summary gives one job of missing
    fields when it has a status or TERM_ line, and none otherwise (it is not an LSF log)."""
    starts = [m.start() for m in SUMMARY.finditer(text)]
    if not starts:
        return [parse_block(text, "")] if STATUS.search(text) or TERM.search(text) else []
    jobs, previous_end = [], 0
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        following = STATUS.search(text, start, end)  # the next job's status line sits before its own summary
        block_end = following.start() if following else end
        jobs.append(parse_block(text[previous_end:block_end], text[start:block_end]))
        previous_end = block_end
    return jobs


def kind_of(path):
    name = path.name.lower()
    if name.startswith("rows_"):
        return "solver"
    if name.startswith("features"):
        return "features"
    if name.startswith("gen"):
        return "generation"
    return "other"


def collect(paths):
    files, explicit = [], set()
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(list(p.rglob("*.out")) + list(p.rglob("*.log")))
        else:
            files.append(p)
            explicit.add(p)
    return files, explicit


def fmt(value, spec):
    return "missing" if value is None else format(value, spec)


def utilisation(job):
    if job["cpu"] is None or not job["run"]:
        return None
    return job["cpu"] / job["run"]


def failed(job):
    return job["ok"] is False or bool(job["term"])


def unreadable(job):
    return job["ok"] is None or job["run"] is None


def verdict(kind, jobs, rows_per_block):
    """('PASS' | 'FAIL' | 'NO TABLE', reasons). A field that is missing is never judged, but an unreadable job fails."""
    reasons = []
    for label, job in jobs:
        if failed(job):
            reasons.append(f"{label}: failed ({' '.join([job['status']] + job['term'])})")
        elif unreadable(job):
            reasons.append(f"{label}: no readable status or Run time")
        if kind in ("solver", "features"):
            if job["mem_gb"] is not None and job["mem_gb"] > MAX_MEMORY_GB:
                reasons.append(f"{label}: Max Memory {job['mem_gb']:.2f} GB > {MAX_MEMORY_GB} GB")
        if kind == "solver":
            u = utilisation(job)
            if u is not None and u < MIN_UTILISATION:
                reasons.append(f"{label}: utilisation {u:.2f} < {MIN_UTILISATION:.2f}")
            if job["run"] is not None and job["run"] > SOLVER_BLOCK_MAX_S:
                reasons.append(f"{label}: Run time {job['run']:.0f} s > {SOLVER_BLOCK_MAX_S} s")
            if job["run"] is not None and job["run"] / rows_per_block > LONGEST_ROW_MAX_S:
                reasons.append(f"{label}: mean row time {job['run'] / rows_per_block:.0f} s > {LONGEST_ROW_MAX_S} s")
        elif kind == "features":
            if job["run"] is not None and job["run"] > FEATURES_MAX_S:
                reasons.append(f"{label}: Run time {job['run']:.0f} s > {FEATURES_MAX_S} s")
    if reasons:
        return "FAIL", reasons
    return ("PASS" if kind in ("solver", "features") else "NO TABLE"), reasons


def kind_summary(kind, jobs, rows_per_block):
    runs = [j["run"] for _, j in jobs if j["run"] is not None]
    mems = [j["mem_gb"] for _, j in jobs if j["mem_gb"] is not None]
    both = [j for _, j in jobs if j["cpu"] is not None and j["run"]]
    cpu_total, run_total = sum(j["cpu"] for j in both), sum(j["run"] for j in both)
    v, reasons = verdict(kind, jobs, rows_per_block)
    lines = [f"{kind}: {len(jobs)} jobs, {sum(failed(j) for _, j in jobs)} failed, "
             f"{sum(unreadable(j) for _, j in jobs)} without a readable status or Run time",
             f"  utilisation (total CPU time / total Run time): {fmt(cpu_total / run_total if run_total else None, '.2f')}",
             f"  Run time median {fmt(statistics.median(runs) if runs else None, '.0f')} s, "
             f"maximum {fmt(max(runs) if runs else None, '.0f')} s",
             f"  Max Memory maximum {fmt(max(mems) if mems else None, '.2f')} GB"]
    if kind == "solver":
        lines.append(f"  mean row time (Run time / {rows_per_block}) maximum "
                     f"{fmt(max(runs) / rows_per_block if runs else None, '.0f')} s; "
                     f"longest single row: missing (not in an LSF log; read the result files)")
    lines.append(f"  {v}" + (" (no pass table for this kind)" if kind in ("generation", "other") else " against the pass table"))
    lines += [f"    - {r}" for r in reasons]
    return lines, v


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarise LSF logs against the pass table.")
    parser.add_argument("paths", nargs="+", help="log files or folders")
    parser.add_argument("--kind", choices=KINDS, help="kind of every log (default: from the file name)")
    parser.add_argument("--rows-per-block", type=int, default=10, help="solver rows in one job (default 10)")
    args = parser.parse_args(argv)

    files, explicit = collect(args.paths)
    by_kind, table = {}, []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            table.append((path.name, "?", f"unreadable ({error.__class__.__name__})"))
            continue
        jobs = parse_log(text)
        if not jobs:
            if path not in explicit:
                continue  # a folder search also finds logs that are not LSF logs
            jobs = [parse_block("", "")]
        kind = args.kind or kind_of(path)
        for i, job in enumerate(jobs, start=1):
            label = path.name if len(jobs) == 1 else f"{path.name}#{i}"
            by_kind.setdefault(kind, []).append((label, job))
            table.append((label, kind,
                          f"{job['status']:<10} {','.join(job['term']) or '-':<12} "
                          f"CPU {fmt(job['cpu'], '.0f'):>7} s  Run {fmt(job['run'], '.0f'):>7} s  "
                          f"util {fmt(utilisation(job), '.2f'):>7}  Mem {fmt(job['mem_gb'], '.2f'):>7} GB  "
                          f"Swap {job['swap_gb'] if job['swap_gb'] == '-' else fmt(job['swap_gb'], '.2f'):>7} GB"))
    if not table:
        print("no LSF logs found")
        return 1
    width = max(len(r[0]) for r in table)
    for label, kind, text in table:
        print(f"{label:<{width}}  {kind:<10} {text}")
    print()
    code = 0
    for kind in KINDS:
        if kind in by_kind:
            lines, v = kind_summary(kind, by_kind[kind], args.rows_per_block)
            print("\n".join(lines))
            code = 1 if v == "FAIL" else code
    return code


if __name__ == "__main__":
    sys.exit(main())
