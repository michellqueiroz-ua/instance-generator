# 02. Tests and results

All numbers are from runs in this project. "Reported by session" means the figure comes from
the working session and the raw log may have been overwritten; the file location is given
where the raw data exists.

## T1. Pilot on Copenhagen Municipality

Design (`pilot/run_pilot.py`): 12 configs x 2 replicas = 24 instances, each with 3 fleets = 72
ISA instances. Requests 200/500/1000, peak spread sd 15/30 min, two time-window settings (tight:
about 10 min reaction and 15 min arrival slack; loose: about 30 and 45 min), fleet 10/20/30% of
requests with 60% 4-seat and 40% 8-seat vehicles.

Raw data: `pilot/work/results/` (`features.csv`, `perf_matrix.csv`, `instance_space.csv`).

| Finding | Value |
|---|---|
| Instance space after the first pilot | Large gap in the middle (author, from the plot) |
| Cause found | Seed bug: the per-request seeds did not depend on the config seed, so all configs drew the same locations. `od_distance_mean` took only 6 distinct values over 72 instances. |
| After the seed fix | `od_distance_mean` takes 24 distinct values |
| SA vs GR (ratio to best), after fix | SA better on 70 of 72 instances; GR is competitive only on tight windows with the smallest fleet (8 of 72) |
| Remaining gap | Along the time-window axis, which has only two levels in the grid. This is why the full set uses continuous ranges (D8). |

Reading: SA dominates GR almost everywhere in the pilot, so the pilot alone gives a thin
algorithm-selection story. A wider design (D6-D8) and possibly a third algorithm (D3) are
needed. **OPEN**: whether SA-vs-GR stays this one-sided on the wide design.

## T2. Checker for the Gurobi model (`ODBRP_P/model/check_model.py`)

The checker has no Gurobi dependency. It builds hand-made routes and tests them against the
model constraints, with 11 scenarios (single request, shared ride, capacity, time windows,
third party, and others).

| Model | Scenario mismatches against the heuristic's feasibility rules |
|---|---|
| v1 (original) | 9 of 11 |
| v2 (rewritten) | 0 of 11 |

Defects found in v1: infeasible by construction, capacity effectively C-2, travel-time columns
shifted by one, no third-party option, and the objective did not match the heuristic.

## T3. Exact model v2 on small instances (HPC, 1 hour limit)

| Requests | Result (reported by session) |
|---|---|
| 5 | Optimal in 2.5 s after adding valid ride lower bounds (before: gap 98% at the time limit) |
| 10 | 25% optimality gap at the limit |
| 15 | Not solved to optimality |

What changed between the first and the second run: valid lower bounds on the ride time of each
request, arc pruning, node-level time and load variables, minimum arc time of 1 s, and a start
route given as a MIP start. Raw (first run, before the bounds): `pilot/work/v2_tests/v2_results.txt`.

Observation by the author: SA is not optimal on the small instances, but the exact model may
look better partly because SA uses fewer vehicles. The exact model tends to serve requests
alone (shorter ride, more driving). This is a reason to keep ride time as the only objective
(D1) and to consider operator cost as a post-hoc measure.

SA against v2 (from `../pilot/settings_from_papers.md`): on 5 requests SA is 4.6% above the
proven optimum; on 10 requests SA is at least 20% above v2's best solution (v2 is not proven
optimal there). SA run time over the pilot: mean 15.5 s, maximum 73 s (200-1000 requests).

**OPEN**: a table of SA objective vs v2 objective vs bound for the 5- and 10-request instances,
with the number of vehicles used by each. Not tabulated yet.

## T4. Matheuristic (MIP-based LNS)

Implemented (`ODBRP_P/model/matheuristic.py`, `submit_mh.sh`): parts = a seed vehicle, its nearest
vehicles and nearby third-party requests; each part is re-solved with v2 starting from the SA
solution, so it cannot get worse. Tested locally with a mock model only.
**OPEN**: HPC run and its results (see 06).

## T5. Network build for the Capital Region

| Item | Result |
|---|---|
| Stations | 3515 (lon 11.85-12.68) |
| Stations on Bornholm | 0 (first build: 1 of 3516 leaked in; fixed, D17) |
| Job | 1 core, 6 GB, 12 h limit |
| Measured | CPU time 3439 s, run time 12374 s (3.4 h), max memory 4466 MB (73% of 6 GB), CPU use 28% (waiting for Overpass) |

An earlier build job reserved 4 cores and 64 GB and used 7.7% CPU and 4.4 GB; it was cut to 1
core and 6 GB (D22).

## T6. Test generation job (instance ISA_b00_i001, with the station matrix)

First attempt: LSF job array, exit code 2, the config argument was empty (cause unknown). Replaced
by one job per config with the path as argument (`generate_batch.sh`, `generate_one.sh`).

Second attempt (job 29846018, 1 core, 8 GB, 6 h limit): after 5.3 h the job had used 19443 s CPU
(about 1 core, so computing and not blocked) and 7.1 GB of 8 GB (89.95%), and had written
**no output files**. The limit could not be raised (`bmod` is admin-only).
**OPEN**: final result of this job (killed at the limit, or finished). Expected to be killed at
about 12:33 on 2026-10-29.

## T7. Profile of a 6-request generation (`profile_one.sh`, no station matrix)

| Item | Value |
|---|---|
| Run time / CPU / memory | 250 s / 192 s / 1593 MB (19% of 8 GB) |
| Time per request | 16-18 s; 2-3 sampling attempts per request (so rejection is **not** the cause) |
| Where the time went | `osmnx.nearest_nodes`: 136 calls, 117.8 s cumulative, each call running `graph_to_gdfs` on the whole graph (0.8 s). About 23 calls per request. |
| Other fixed costs | about 60 s of imports, about 27 s for the request plot |

Hypothesis by the author that was tested and **rejected**: stops too close to each other (under
10 min) so that requests cannot be created. The attempt counts show 2-3 attempts per request.

Consequence: a cached nearest-node index was added (D18, `REQreate/nearest.py`). Unit test
`tests/test_nearest.py` checks it returns the same node as osmnx on 30 random points.
**OPEN**: rerun of the profile after the fix, to measure the speed-up (expected from about 20 s to
about 1 s per request).

**OPEN**: the ~4 h of the test job that the requests do not explain (218 requests x 20 s is
about 1.2 h). Suspected: the station travel-time matrix (3515 x 3515 = 12.4 million pairs, 3515
single-source searches on the full drive graph, with the row cache growing). Not measured yet.

## T8. Test suites

- New tests for `features_in_area`, `exclude_from_polygon`, the CLI `--exclude` option and
  `nearest_node` pass.
- 7 web-app / job-queue tests fail. They also fail on the commit before these changes: separate
  processes pick up the REQreate of the main checkout, plus some timeouts. Not caused by this work.
