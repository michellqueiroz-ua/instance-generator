# 06. Open items (unresolved, from the working sessions)

Marked in order of what blocks the next step.

## Blocking the full instance set

| ID | Item | State |
|---|---|---|
| OPEN-G1 | Result of test job 29846018 (ISA_b00_i001, with station matrix). It reached the 6 h limit with no output files; expected to be killed at about 12:33 on 2026-10-29. | Waiting. If killed, nothing is saved. |
| OPEN-G2 | Rerun of the 6-request profile after the nearest-node fix. | **Done 2026-10-29**: 0.5-1.6 s per request instead of 16-18 s (see 02, T7). |
| OPEN-G6 | The rerun log reports Max Memory 67 MB (run time 115 s, completed), against 1593 MB before. 67 MB cannot hold the Capital Region network, so it is probably an LSF sampling artefact; do not size memory from it. | Open |
| OPEN-G3 | (Decision D14b: the station matrix gets its own job.) Where the other ~4 h of the test job went. Requests explain about 1.2 h of 5.3 h. Suspected: the station matrix (3515 searches on the full drive graph; the row cache grows, memory 7.1 GB of 8 GB). | **Done 2026-10-29**: the matrix job takes 97 s (02, T7b). The slow part of the old job was the in-generation loop and JSON, not the searches. |
| OPEN-G4 | Feature step that attaches the shared station matrix (see 05). | **Done 2026-11-02** (04, change 10); untested on real HPC data. |
| OPEN-G5 | Size `MEM_GB` and `WALL_H` for batch 1. Measured before the fix: 1.6 GB + 8.3 MB per request (02, T9). Rerun with commit 5434c45d is needed; the HPC was down on 2026-11-02. | Waiting for HPC |

## Algorithms

| ID | Item | State |
|---|---|---|
| OPEN-A1 | Third ISA algorithm: SA variants (neighbourhood configurations) or max-ALNS (D3). Which exact variants, and where the code lives. | Author to decide. |
| OPEN-A2 | HPC test of the matheuristic (copy `MDHODBRP_model_v2.cpp`, `matheuristic.py`, `audit_sa.py`, `submit_mh.sh`, `submit_audit.sh` to `~/ip_model`, rebuild, first run `submit_audit.sh` on 20 SA results, then `submit_mh.sh`). Phase A code written 2026-11-10 (02, T15), not run. | Not run. |
| OPEN-A3 | SA/GR job memory: 16 GB reserved in the pilot, about 6 GB used. `make_solver_jobs.py` defaults to 8 GB (`MEM_GB`); measure on the first 4 jobs with the 4000-station build. | Measure |
| OPEN-A4 | Rebuild the solver with `maxstations 4000`; the change in ODBRP_P is not committed. | Author to commit. |
| OPEN-A5 | Table of SA vs v2 objective and vehicles used on the 5-request and 10-request instances. | Not tabulated. |
| OPEN-A6 | Does SA's one-sided advantage over GR (70 of 72) persist on the wide design? | Needs the full set. |

## Settings

See OPEN-S1 to S4 in `03_recommended_settings.md`. The author makes the final choice.

## Documentation and housekeeping

| ID | Item |
|---|---|
| OPEN-D1 | Done: `REQreate/rank_model.py` builds the 25 x 25 zone grid (always, at network build), the POI density of the zones (downloads OSM POIs; only with `method_pois`) and the zone ranks and travel probabilities between zones (only with `zone_probabilities`). The ISA set uses none of these: depots and hotspots are ignored (D-decisions). |
| OPEN-D2 | 7 web-app tests fail on the branch and on the commit before; cause is the old main-checkout REQreate in separate processes, plus timeouts. Not fixed. |
| OPEN-D3 | The cause of the first generation job's exit code 2 (empty config argument from the job array) was never found; replaced by one job per config. |
| OPEN-D4 | The exit of the LSF job reminder: the `/jobs` mod tracks only `bsub` commands the assistant runs, not the author's own HPC window. |

## Lessons for the write-up

1. A grid design with two levels per factor leaves gaps; continuous Latin hypercube draws fix this (D8).
2. A seed bug can silently collapse the instance space; check the number of distinct values of key features.
3. Profile before scaling up: one hidden call (`osmnx.nearest_nodes`) dominated request generation.
4. Reserve resources from measured usage; early jobs over-reserved memory and cores.

## Growth check settings (OPEN-G7)

The stop rule is not the literal "fewer than 5% far instances": with any fixed distance threshold a same-distribution batch has a far share that depends on the tails of the feature distribution (13.5% at 2 x the typical spacing in a test). The rule is calibrated on the existing rows instead (see 05). Author to decide: the quantile (95%), the margin (3 points), the cap (3000 rows), and whether to judge in the feature space (now) or in the 2-D projection (footprints in `src/isa_pipeline.py`). Nothing has run on real data yet: it needs at least two batches.

## Incomplete requests in batch 1 (OPEN-G8)

Found 2026-11-09 when the solver job writer failed on an empty `stops_orgn`. Cause and fix: 04, change 11. Batch 1 instances with such requests are regenerated with `isa_set/hpc/regenerate_incomplete.sh 1 --redo` (old files are moved to `<work>/incomplete_backup/batch_01`), then the features are recomputed. The earlier feature run counted incomplete requests in `n_requests` (they have a time stamp). **OPEN** until regenerated and the check finds none.

## SA budget and small fleets (OPEN-A7, OPEN-S5), 2026-11-09

- OPEN-A7: the SA budget is fixed (90 outer iterations x 200 counted moves, `--max_static_iterations`), so SA gets fewer moves per request on large instances. This may couple the SA/GR gap to instance size. Check the correlation of SA/GR cost ratio with n_requests in batch 1 before interpreting the instance space. SA is also skipped when the fleet has at least as many vehicles as requests (`k > total_number_vehicles + 1`).
- OPEN-S5: small fleets (down to 6% of the requests) hand up to 56% of the passengers to a third party (02, T12). Author decision 2026-11-09: keep them and compare on the accepted requests as well; collect_batch.py writes perf_matrix_served_* and perf_matrix_quality_* (quality is like-for-like only where both algorithms serve about the same passengers). Open: how to combine the three measures in one instance space (separate spaces per measure, or only the total for fleets of at least 20%).
  Update 2026-11-10 (D27): a fleet that leaves about 55% out is too small. Options: filter the keys of batch 1 by fleet share (no rerun; about 86% of the keys have at least 10 vehicles per 100 requests) and raise `fleet.percent_of_requests` from [5, 40] for new batches. The threshold is OPEN.
- Footprints at the default threshold 0.80: SA is good on 300/300 (it is within 20% of the best everywhere), GR on 116/300, so the footprint carries no selection signal. A tighter threshold (0.95) is the candidate.

## D28: SA without the +3600 s (2026-11-10)

- Edited in `odbrp-parallel.cpp` (ODBRP_P, author commits): rebuild `a.out` with `maxstations 4000` (OPEN-A4) and rerun batch 1 for SA, SA1000/2000/10000 and GR. All results in 02 (T12-T14) and the perf matrices are void until then. Expect ample fleets to no longer sit exactly at the sum of the direct times if some requests become unservable in time.
- Run `model/check_requests.py` on the batch 1 instances (pure Python) to count requests that the matrix cannot serve in their window and requests whose file direct time differs from the matrix. Decide whether the generator should drop or repair such requests (OPEN).

## D42/D43: objective without penalty, soft latest_departure (2026-11-11)

- The audit of the batch 90/91 scaling test (12 results, CSV deadlines): no violation of drop-off window, stop choice, pairing, spacing, capacity or ride-time rules, and printed ride times equal the times in the routes. 1215-2296 pickups per result arrive after `latest_departure` (D43: allowed). The reported total is lower than ride times + 4 x direct time by about 4 x the direct time of the third-party passengers (exact match only where there are none), i.e. the reported total seems to be the ride time of served passengers only (D42). **OPEN**: confirm with the rerun of `audit_scaling.sh` (prints `reported - ride time`, expected 0). Call-graph check of `odbrp-parallel.cpp` (comments stripped, `model/active_calls.py`): `serve_passenger_third_party_vehicle` (the only place with `user_ride_time = 4 x direct`) is defined but never called, so the SA and GR in this source never apply the penalty; unserved passengers keep `user_ride_time` 0 and are printed `UNSERVED` without `(3rd party)`. The reported total is then the ride sum of the served passengers, as intended. **OPEN**: author to say whether another, newer version of the SA exists (path), otherwise this file is the one.
- `isa_set/collect_batch.py` documents `total` as "a third-party passenger counts 4 x the direct time". With D42 the `total` of SA and GR is the ride time of the served passengers; the comparison of two algorithms at different served shares is not like for like. **OPEN**: define the performance measures again (served share, ride time at equal service) before the instance space.
- Docs 02 (T12-T14), 03 and the LaTeX sections still describe the penalty and need a revision.

## Features that do not vary in batch 2 (2026-11-12)

Of the 21 features, 5 are constant over the 300 rows of batch 2 and are dropped before the instance space:
`n_requests` (5000 everywhere), `demand_density` and `station_density` (same network and service area), `avg_station_degree`
(same network), and `avg_request_urgency`. The last one is **0 by construction, not a bug**: it is the mean of
`tw_start - issue_time`, that is `earliest_departure - time_stamp`, and the generator sets `earliest_departure = time_stamp`.
The urgency of the generator (the reaction time, 5-40 min) is the width of the time window and is already carried by
`tw_width_mean` (5.2-39.8 min) and `tw_width_std`. `request_flux_dynamism` (0.15-0.42) and `static_to_dynamic_ratio`
(2-34) are computed from the spread of the time stamps, not from real dynamism (the study is static): they describe how
spread out the demand curve is (peak sd) and should be named and interpreted as temporal spread in the report. Sixteen
features vary.

## 2026-11-12 Phase A pricing audit (b90 and b91, fleet 3, SA, 200 vehicles each)
- b90: 200/200 loadable. b91: 199/200. Not loadable: vehicle 813, cause is the model's 1 s minimum between two nodes at the same station (route has slack 0). Fix: `--same_station_gap 0` (only with `--fix_start`). Test pending. OPEN-A3.
- Model cost vs SA cost with fixed routes: b91 137 equal, 36 dearer (max +8 s), 27 cheaper (down to -1070 s). Hypothesis: the SA respects latest_departure, the model does not (D43). OPEN-A4, not tested.
- LATER (author, 2026-11-12): latest_departure exists because walking to the station is ignored in the static case (latest_departure + shortest bus trip = latest_arrival). A `--hard_latest_departure` comparison is not needed now. In the dynamic version walking times must be accounted for.
- 2 b91 requests (rows 4663, 4804) are unservable: matrix minimum exceeds the whole window (generator vs matrix inconsistency).
- 2026-11-12 RESOLVED OPEN-A3: with `--same_station_gap 0` b91 fleet 3 SA prices 200/200 loadable, vehicle 813 model cost 6616 = SA cost 6616. Pricing runs use GAP=0 from now on.
