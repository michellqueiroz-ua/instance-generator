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

__all__ = ["ENABLED", "stage", "tick", "note", "dump", "reset",
           "instrument_osmnx", "CPROFILE", "cprofile_run"]


def _truthy(value):
    return value.strip().lower() not in ("", "0", "false", "no", "off")


_MODE = os.environ.get("REQREATE_PROFILE", "").strip().lower()
ENABLED = _truthy(_MODE)
# Whole-run cProfile. Slower, but it accounts for 100% by construction rather
# than only for the stages someone remembered to instrument - which is exactly
# how two rounds of hand-placed timers left half the runtime in the dark.
CPROFILE = ENABLED and _MODE in ("cprofile", "all", "full")

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


def instrument_osmnx():
    """Time every osmnx call that the generator leans on, at the source.

    These are called from a dozen places - plot_graph alone from eight modules,
    twice per zone in retrieve_zones - so wrapping the library functions once
    beats threading a stage() through every call site, and cannot miss one.

    plot_graph is here because it is not a side show: a single full-network
    render of a 26k-node graph takes seconds, and several call sites ask for
    dpi=1440, which is an 11520x11520 canvas.
    """
    if not ENABLED:
        return
    try:
        import osmnx as ox
    except ImportError:
        return

    for name in ("plot_graph", "nearest_nodes", "graph_from_place",
                 "graph_from_point", "features_from_place", "features_from_point"):
        original = getattr(ox, name, None)
        if original is None or getattr(original, "_reqreate_timed", False):
            continue

        def make(call_name, func):
            def wrapper(*args, **kwargs):
                with stage("osmnx." + call_name):
                    return func(*args, **kwargs)
            wrapper._reqreate_timed = True
            wrapper.__name__ = call_name
            wrapper.__doc__ = getattr(func, "__doc__", None)
            return wrapper

        setattr(ox, name, make(name, original))
        tick("osmnx.wrapped." + name, 0)


def cprofile_run(func, *args, **kwargs):
    """Run func under cProfile when REQREATE_PROFILE=cprofile, else call it.

    Writes the raw stats next to the output as reqreate_profile.prof, so it can
    be loaded with pstats later, and puts the top callers by cumulative and by
    own time into the printed report and the JSON.
    """
    if not CPROFILE:
        return func(*args, **kwargs)

    import cProfile
    import io
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        return func(*args, **kwargs)
    finally:
        profiler.disable()

        prof_path = os.environ.get("REQREATE_PROFILE_PROF", "reqreate_profile.prof")
        try:
            profiler.dump_stats(prof_path)
            note("cprofile_stats_file", prof_path)
        except OSError as exc:
            print(f"[profile] could not write {prof_path}: {exc}", file=sys.stderr)

        for sort_key, label in (("cumulative", "cumulative"), ("tottime", "own_time")):
            buf = io.StringIO()
            try:
                pstats.Stats(profiler, stream=buf).sort_stats(sort_key).print_stats(40)
            except Exception as exc:                # a report is never worth a crash
                _notes["cprofile_" + label] = f"(failed: {exc})"
                continue
            text = buf.getvalue()
            _notes["cprofile_top_" + label] = text
            print(f"\n--- cProfile: top 40 by {label} ---", file=sys.stdout)
            print(text, file=sys.stdout)
