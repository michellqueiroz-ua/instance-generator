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
Rerun after the fix (same config, 6 requests, no station matrix; log `logs/profile2.out`):

| Item | Before | After |
|---|---|---|
| Request 0 | 16.23 s (2 attempts) | 1.61 s (2 attempts) |
| Request 5 | 18.06 s (3 attempts) | 0.54 s (3 attempts) |
| CPU time of the whole run | 192 s | 72 s |
| Run time of the whole run | 250 s | 115 s (CPU use 63%) |
| Max memory reported by LSF | 1593 MB | 67 MB (**suspicious**, see 06 OPEN-G6) |

Speed-up per request: about 10x for the first request (which builds the index) and about 33x
for later ones. Request time is no longer a bottleneck: 1000 requests take roughly 10 minutes
at 0.5 s each. The attempts per request are unchanged, so the fix does not change which
requests are drawn.

Reading the code (`passenger_requests.py` from line 1217, `network_class.py:160`) gives a likely
cause of the heavy test job, not yet confirmed by a measurement. After the requests, REQreate
builds the station matrix with a Python double loop over all 3515 x 3515 pairs. Each new origin
runs one shortest-path search over the whole drive graph, and the on-demand matrix keeps every
row (one value per graph node, up to 5000 rows), so memory grows with each station (the job
reached 7.1 GB of 8 GB). The matrix is then converted to nested lists, written into the instance
JSON with `indent=4`, and written again as CSV. Unreachable pairs would stay uninitialised
(`np.ndarray` without a fill value); the drive graph is a strongly connected component, so none
are expected.

**OPEN**: the ~4 h of the test job that the requests do not explain (218 requests x 20 s is
about 1.2 h). Suspected: the station travel-time matrix (3515 x 3515 = 12.4 million pairs, 3515
single-source searches on the full drive graph, with the row cache growing). Not measured yet.

## T7b. Station matrix as its own job (`station_matrix.py`, finished 2026-10-29)

| Item | Value |
|---|---|
| Stations / drive graph | 3515 stations, 59115 nodes |
| Job | 1 core, 4 GB, 4 h limit |
| Measured | run time 97 s, CPU time 68 s (70% of one core), max memory 74 MB as reported by LSF (**suspicious**, as in OPEN-G6) |
| Output | `station_travel_time_matrix.csv` (81 MB) and `.npy` (48 MB, int32) in `travel_time_matrix/` |

The same matrix took more than 5 h inside the generation job (no output written, 7.1 GB). The
drive graph is small (59115 nodes), so the shortest-path searches are cheap; the cost was the
Python double loop and the conversion to nested lists and JSON. Resource waste of the first
attempt: 4 h limit and 4 GB reserved for 97 s of work.

## T8. Test suites

- New tests for `features_in_area`, `exclude_from_polygon`, the CLI `--exclude` option and
  `nearest_node` pass.
- 7 web-app / job-queue tests fail. They also fail on the commit before these changes: separate
  processes pick up the REQreate of the main checkout, plus some timeouts. Not caused by this work.

## T9. Batch 1, first five instances (2026-11-02)

Reserved 4 GB, 1 core each. i001 to i004 were killed (`TERM_MEMLIMIT`, exit 137, 4096 MB); i005 finished (CPU 128.5 s, run 163 s).

Rerun of i002 and i003 at 8 GB with the row cache capped (commit 2ffbeae4) and peak memory in the log:

| Instance | Requests | Peak at request 0 | Peak at end | Growth per request | CPU | Run time | CPU / run |
|---|---|---|---|---|---|---|---|
| ISA_b01_i002 | 435 | 1614 MB | 5256 MB | 8.5 MB | 293 s | 302 s | 97% |
| ISA_b01_i003 | 685 | 1615 MB | 7199 MB | 8.2 MB | 490 s | 499 s | 98% |

LSF `Max Memory` (5160 and 7280 MB) matched the logged peak, so it was reliable for these long jobs.
The growth is linear in the number of requests, so the cap of 2ffbeae4 had no effect. Cause: the cap was not applied to matrices restored from the network pickle (see 04, change 9). Result of the fix: **OPEN**, HPC down on 2026-11-02.

## T10. Batch 1 generation and features (2026-11-06)

100 instances generated at 1 core and 4 GB each, none killed; highest LSF Max Memory 2849 MB (cache plateau). Feature test on i001 to i005 (15 rows, 1 core): peak memory 1564 MB, 119 s of which about 114 s is the one-time network pickle read; CPU 21 s of run time 132 s (16%, disk wait).

| Instance | Requests | tw_tightness_factor | Fleet sizes (3 fleets) |
|---|---|---|---|
| ISA_b01_i001_1 | 579 | 1.58 | 163, 154, 186 |
| ISA_b01_i002_1 | 435 | 1.00 | 117, 80, 81 |
| ISA_b01_i003_1 | 685 | 0.49 | 53, 180, 151 |
| ISA_b01_i004_1 | 561 | 2.09 | 73, 165, 199 |
| ISA_b01_i005_1 | 144 | 2.03 | 53, 9, 46 |

tw_tightness spreads over a factor of four and fleet sizes differ per fleet, as the design intends. Not compared against the old direct-travel-time fallback. Full 100-instance run: **OPEN** (submitted 2026-11-06).

## T11. Solver on the largest instance (2026-11-09)

ISA_b01_i024 (995 requests), fleet 1, default build (`maxstations 4000`, 1 core): GR 38 s, SA 39-42 s (CPU/run 96-98%); peak resident memory 277-279 MB (GNU time). Instance i001 (579 requests): SA 8-9 s, GR 2 s. Memory is far below the 6 GB the old build reserved. SA code reading: `number_clusters` is 1 (one thread); the run is static: all requests are queued before the main loop (line 16666), the constructive phase inserts them and re-queues rejected ones, and the main loop (`while(algo_iterations < 90)`, line 16709) then calls SA once per outer iteration, up to 90 times. Each SA call stops after 200 counted moves (`count_static_it`, line 15625), a budget that does not depend on the instance size (about 18,000 counted moves in total, which is 20 per request at 1000 requests and 180 per request at 100). So `--maxnrep`, `--increase_rep` and `--lambda` never take effect and the temperature stays at 1.3. New flag `--max_static_iterations` (default 200) makes the cap a setting. Jobs bundle one instance (12 runs, about 5-8 min) per LSF job at 1 GB.

## T12. Batch 1 results (2026-11-09)

1200 runs (100 instances x 3 fleets x (3 SA seeds + GR)), 100 LSF jobs of 12 runs, 1 core and 1 GB each: no failed run, peak memory 407 MB, CPU 8878 s of 11959 s run time (74%; each run reads the 81 MB travel-time file). GR 4.3 s on average (max 36 s), SA 7.5 s (max 40 s).

Total objective, 300 (instance, fleet) keys: SA beats GR on 232 keys scored by its mean over the seeds and on 263 keys scored by its best seed; SA/GR cost ratio 0.70 (mean score) and 0.68 (best seed), range 0.35-1.14. The ratio correlates -0.21 with the number of requests (SA relatively better on larger instances), so the fixed SA budget does not show up as a size effect in this data.

Third-party share (GR, share of the passengers not served by the own fleet) by fleet size as a percentage of the requests:

| Fleet size | Keys | Mean third-party share | Max |
|---|---|---|---|
| 5-10% | 42 | 55.7% | 73.9% |
| 10-15% | 44 | 32.4% | 48.5% |
| 15-20% | 43 | 16.7% | 33.7% |
| 20-25% | 43 | 4.5% | 16.1% |
| 25-30% | 43 | 0.9% | 4.5% |
| 30-40% | 85 | 0.2% | 5.1% |

Below a fleet of about 20% of the requests the total objective is dominated by the 4 x direct time penalty. Decision (author, 2026-11-09): the small fleets stay in the design; for them the algorithms are compared on the accepted requests (served share and ride time ratio of the accepted passengers) in addition to the total. First instance space (pilot-analytic, 16 features): variance explained 0.45 (features) and 0.39 (performance); at footprint threshold 0.80 SA is good on 300/300 keys and GR on 116/300.

## T13. SA with more moves per call, largest instance (2026-11-10)

ISA_b01_i024 (995 requests), 3 fleets, 3 seeds each, mean over the seeds. One LSF job of 27 runs: run time 6523 s, CPU 6486 s (99.4%), peak memory 387 MB, no failed run.

| Fleet | Algorithm | Total cost | Served share | Ride time / direct (accepted) | Run time per run |
|---|---|---|---|---|---|
| 1 (tight) | GR | 1527149 | 0.48 | 2.47 | 36 s |
| | SA (200 moves) | 1689896 | 0.57 | 2.32 | 39 s |
| | SA1000 | 1720765 | 0.63 | 2.09 | 43 s |
| | SA2000 | 1658281 | 0.63 | 1.99 | 58 s |
| | SA10000 | 1337167 | 0.64 | 1.54 | 171 s |
| 2 (ample) | GR | 3051719 | 1.00 | 2.31 | 0.5 s |
| | SA (200 moves) | 1381448 | 1.00 | 1.00 | 17 s |
| | SA10000 | 1380076 | 1.00 | 1.00 | 698 s |
| 3 (ample) | GR | 3044730 | 1.00 | 2.29 | 0.4 s |
| | SA (200 moves) | 1382072 | 1.00 | 1.00 | 17 s |
| | SA10000 | 1379823 | 1.00 | 1.00 | 745 s |

Tight fleet: the ride time ratio of the accepted passengers falls steadily with the moves (2.32, 2.09, 1.99, 1.54), the total only drops below GR at 10000 moves (-12% against GR, -21% against SA with 200 moves). Ample fleets: SA with 200 moves already gives ratio 1.00 (every passenger rides directly) and the total sits at the sum of the direct times (checked against the instance file: OPEN until the sum is printed); 10000 moves change it by 0.1% and take 40 times as long, so these fleets are 86% of the job's run time. SA serves more passengers than GR but the total can be worse because the third-party penalty is weighted by the direct time of the unserved passengers (a count-based served share does not show this). `make_solver_jobs.py --variants-max-fleet-percent 25` limits the variants to fleets of at most 25 vehicles per 100 requests.

## T14. SA with 1000, 2000 and 10000 moves on the tight fleets of batch 1 (2026-11-10)

SA variants were run for the 174 of 300 keys whose fleet has at most 25 vehicles per 100 requests (3 seeds each, 522 runs per variant, 2766 result files in total). Total cost against GR (ratio of costs, SA scored by the mean of its seeds / by its best seed):

| Algorithm | Run time per run (mean / max) | Better than GR on keys (mean score) | Cost / GR (mean score) | Better than GR on keys (best seed) | Cost / GR (best seed) |
|---|---|---|---|---|---|
| SA (200 moves) | 7.5 s / 40 s | 106 / 174 | 0.85 | 137 / 174 | 0.82 |
| SA1000 | 17.8 s / 80 s | 128 / 174 | 0.82 | 152 / 174 | 0.79 |
| SA2000 | 29.8 s / 160 s | 142 / 174 | 0.80 | 159 / 174 | 0.77 |
| SA10000 | 125 s / 812 s | 155 / 174 | 0.76 | 166 / 174 | 0.73 |

Served share against GR: SA serves more passengers on 159 to 163 of 174 keys, ratio 1.07 to 1.08 (no change with more moves). Ride-time ratio of the accepted passengers against GR: 0.77 (200 moves) falling to 0.67 (10000 moves), mean score. The quality comparison is only like for like where both serve about the same passengers (served shares within 2 points): 39 to 50 of 174 keys. The cost of GR is still lower than SA10000 on 19 keys (mean score) and 8 keys (best seed), with a worst ratio of 1.13. **OPEN**: instance space with the five algorithms (ISA_ALGORITHMS=SA,SA1000,SA2000,SA10000,GR) on `features_batch_01_partial.csv`.
