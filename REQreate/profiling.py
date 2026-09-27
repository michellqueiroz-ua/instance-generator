"""Opt-in profiling, for finding out where a generation actually spends its time.

Off unless REQREATE_PROFILE is set to something other than 0/false/no. When off,
`stage` is a no-op context manager and `tick` returns immediately, so leaving the
calls in the hot paths costs nothing in a normal run.

Two kinds of measurement, because wall-clock alone does not explain this code:

    stage("walk.dijkstra")   how long a phase took
    tick("request.restart")  how many times a loop went round

The counters matter as much as the timers. Request generation is rejection
sampling - a request that fails any feasibility check is thrown away and redrawn
- so "slow" and "drew 40,000 candidates to keep 20" are different diagnoses with
different fixes, and only a counter tells them apart.

Results go to stdout at exit and to a JSON file, so a smoke test can attach them.
"""

import atexit
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

__all__ = ["ENABLED", "stage", "tick", "note", "dump", "reset"]


def _truthy(value):
    return value.strip().lower() not in ("", "0", "false", "no", "off")


ENABLED = _truthy(os.environ.get("REQREATE_PROFILE", ""))

# Where the JSON lands. Relative paths are resolved against the working
# directory, which is where the generator already writes its output.
OUTPUT_PATH = os.environ.get("REQREATE_PROFILE_OUT", "reqreate_profile.json")

_elapsed = defaultdict(float)
_calls = defaultdict(int)
_counters = defaultdict(int)
_notes = {}
_started_at = time.time()
_dumped = False


def reset():
    """Forget everything measured so far. Only useful in tests."""
    _elapsed.clear()
    _calls.clear()
    _counters.clear()
    _notes.clear()
    global _started_at, _dumped
    _started_at = time.time()
    _dumped = False


if ENABLED:

    @contextmanager
    def stage(name):
        """Accumulate wall-clock time spent inside this block under `name`."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _elapsed[name] += time.perf_counter() - t0
            _calls[name] += 1

    def tick(name, n=1):
        """Count an event - a loop iteration, a rejected candidate, a retry."""
        _counters[name] += n

    def note(name, value):
        """Record a one-off fact, such as a node count or a matrix shape."""
        _notes[name] = value

else:

    @contextmanager
    def stage(name):
        yield

    def tick(name, n=1):
        pass

    def note(name, value):
        pass


def _summary():
    total = time.time() - _started_at
    return {
        "total_seconds": round(total, 3),
        "stages": {
            name: {
                "seconds": round(secs, 3),
                "share_of_total": round(secs / total, 4) if total else None,
                "calls": _calls[name],
            }
            for name, secs in sorted(_elapsed.items(), key=lambda kv: -kv[1])
        },
        "counters": dict(sorted(_counters.items(), key=lambda kv: -kv[1])),
        "notes": _notes,
    }


def dump(path=None, to_stdout=True):
    """Write the summary to JSON and optionally print it. Idempotent per run."""
    global _dumped
    if not ENABLED or _dumped:
        return None
    _dumped = True

    data = _summary()
    target = path or OUTPUT_PATH
    try:
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except OSError as exc:                       # never let profiling break a run
        print(f"[profile] could not write {target}: {exc}", file=sys.stderr)
        target = None

    if to_stdout:
        _print(data, target)
    return data


def _print(data, target):
    total = data["total_seconds"]
    out = sys.stdout
    print("\n" + "=" * 64, file=out)
    print("REQREATE PROFILE", file=out)
    print("=" * 64, file=out)
    print(f"total wall clock: {total / 60:.1f} min ({total:.1f} s)\n", file=out)

    if data["stages"]:
        print(f"{'stage':<34}{'minutes':>9}{'share':>8}{'calls':>10}", file=out)
        print("-" * 64, file=out)
        for name, s in data["stages"].items():
            share = "" if s["share_of_total"] is None else f"{s['share_of_total']:6.1%}"
            print(f"{name:<34}{s['seconds'] / 60:9.2f}{share:>8}{s['calls']:>10,}", file=out)
        accounted = sum(s["seconds"] for s in data["stages"].values())
        print("-" * 64, file=out)
        print(f"{'accounted for':<34}{accounted / 60:9.2f}"
              f"{(accounted / total if total else 0):7.1%}", file=out)
        print(f"{'unaccounted':<34}{(total - accounted) / 60:9.2f}"
              f"{((total - accounted) / total if total else 0):7.1%}", file=out)

    if data["counters"]:
        print(f"\n{'counter':<44}{'count':>18}", file=out)
        print("-" * 64, file=out)
        for name, n in data["counters"].items():
            print(f"{name:<44}{n:>18,}", file=out)

    if data["notes"]:
        print("\nnotes", file=out)
        print("-" * 64, file=out)
        for name, value in data["notes"].items():
            print(f"  {name}: {value}", file=out)

    if target:
        print(f"\nwritten to {target}", file=out)
    print("=" * 64 + "\n", file=out)


if ENABLED:
    # A generation that dies partway is exactly when the numbers matter most, so
    # dump on the way out rather than only on a clean finish.
    atexit.register(dump)
