# 04. Changes to the instance generator (REQreate) and the solver

"Original" = the upstream REQreate on `master` at the point where this branch started (merge
base committed 2026-10-12), plus the earlier ISA work. The branch also merged several `master`
releases (0.3.0 to 0.4.1); those are listed separately so the ISA-specific work is clear.

## A. Changes made for the ISA study

| Change | Where | Why | Effect |
|---|---|---|---|
| Per-request seeds follow the config seed: `seed_offset = (base_seed - 1) * 1000003` added to the location and attribute seeds | `REQreate/passenger_requests.py`; commit `9eb5e160` | All configs drew the same locations, so the instance space had only 6 distinct mean OD distances in 72 instances | 24 distinct values after the fix. Different seeds now give different instances. **Changes generated instances: results from before the fix are not comparable.** |
| `exclude_places` option in the config (list of place names cut out of the network polygon); CLI: `reqreate network PLACE --exclude X` (repeatable) | `retrieve_network.py`, `input_json.py`, `schema/config.schema.json`, `cli.py`, `docs/CONFIG_REFERENCE.md` | The Capital Region includes Bornholm, which is an island | Network built without Bornholm |
| Stations, schools and hospitals are downloaded for the polygon, not for the place name (`osm_features.features_in_area`) | `osm_features.py`, `retrieve_bus_stations.py`, `retrieve_schools.py`, `retrieve_hospitals.py` | The place-name query ignored the exclusion; one station on Bornholm leaked in | 0 stations on Bornholm, 3515 stations |
| Cached nearest-node index (`nearest_node`) in request sampling | `REQreate/nearest.py`, `passenger_requests.py`, `network_class.py` | Profile: 117 s of 154 s were spent in `osmnx.nearest_nodes`, which rebuilds the graph for every lookup | Same node as osmnx (unit test). Expected large speed-up; **not yet measured**, see 06 |
| ISA feature export at generation time | `REQreate/isa_features_hook.py`, `isa-features-plan-implementation/` | The features (axes of the instance space) are computed when an instance is generated; one row per instance in a CSV (`ISA_FEATURES_CSV`) | Feature computation never stops generation (failures are logged) |
| `avg_station_degree` from the drive road network | commit `a1cdb815` | Feature definition | |
| Scalar taken from size-1 samples in request generation | commit `bf3dafcd` | Bug in attribute sampling | |
| Debug prints removed from request generation | commit `2e05d236` | Noise | |
| Pilot v2: real 7:30 peak, replicas, three fleet sizes | `pilot/run_pilot.py` | The first pilot flattened the peak ("dynamism" replaced the pdf) | |

| `travel_times_between()` for the station matrix: chunked searches that keep only the station columns | `REQreate/compute_distance_matrix.py`, `isa_set/station_matrix.py`, `isa_set/hpc/station_matrix.sh` | The in-generation matrix caches one full-graph row per station (memory) and loops in Python (time). Same whole-second values as the on-demand matrix (unit test `tests/test_station_matrix.py`). | Station matrix is computed once in its own job (D14b); **not yet run on the HPC** |

## B. Tooling added for the full set

| File | Purpose |
|---|---|
| `isa_set/design.json` | Ranges of every setting |
| `isa_set/make_batch.py` | Draws one batch (Latin hypercube) and writes configs, a design CSV and fleets |
| `isa_set/hpc/build_network.sh` | Builds the network once (1 core, 6 GB) |
| `isa_set/hpc/generate_batch.sh`, `generate_one.sh` | One single-core LSF job per config |
| `isa_set/hpc/profile_one.sh` | Times the phases of a small generation run (cProfile) |
| `pilot/make_solver_jobs.py`, `collect_results.py` | Solver job bundle and collection into `perf_matrix.csv` |

## C. Solver changes (separate repository ODBRP_P, `MDHODBRP-P`)

| Change | Why |
|---|---|
| `--greedy_only` option: GR is the SA binary without annealing | Second ISA algorithm without a second code base |
| Travel-time matrix read as given (row = from, column = to), no symmetrisation | Real travel times are asymmetric; symmetrising gave wrong data |
| Type-3 request reader converts walking distances (metres) to seconds with the walk speed | Time windows were wrong by the unit |
| Station limit `maxstations` 2000 to 4000 | The Capital Region has 3515 stations. **Uncommitted in ODBRP_P: author commits it.** |
| New exact model `model/MDHODBRP_model_v2.cpp` and checker `check_model.py` | v1 was infeasible by construction (see 02, T2) |
| Matheuristic `model/matheuristic.py` | Shows where SA is weak (D4) |

## D. Inherited from `master` (merged into this branch, not ISA-specific)

From the commit subjects: distance matrices computed on demand per origin instead of all pairs
(the all-pairs matrix needed about 75 GB for a 26k-node city); points of interest downloaded
on demand; batched nearest-node lookups for OSM features; Overpass queries sent to a live
mirror; separate network building from instance generation; a persistent single-worker queue
in the web app; licensing cleanup (trip-data scripts moved to `legacy/`); Windows installer for
versions 0.3.0 to 0.4.1; fix for ODBRP crash on `stops()`.

The on-demand distance matrix matters for this study: request generation runs one shortest-path
search per drawn origin, never the full matrix. The **station-to-station** matrix is a separate
output (`travel_time_matrix` in the config) needed by the solver.

## E. Another ISA feature (not examined here)

`REQreate/rank_model.py` (new, 90 lines) is part of the ISA work: the author confirmed it is
another feature. It was written outside this working session and is not described here.
**OPEN**: add one line on what it computes.

## Change 9. Distance row cache capped by bytes (also for old pickles)

`LazyDistanceMatrix` kept up to 5000 cached rows, one number per graph node. On the Capital Region walk graph a row is about 4 MB, and a request adds about two rows, so memory grew 8 MB per request (7.2 GB at 680 requests). The cache is now also capped at 400 MB per matrix (`MAX_CACHE_BYTES`); evicted rows are recomputed (one search, tens of milliseconds). The pickled network stores `max_rows`, so `__setstate__` applies the cap again on load (commit 5434c45d); no network rebuild is needed. The request log now prints the process peak memory.

## Change 10. Feature step uses the shared station matrix

`reqreate_adapter.load_reqreate_instance` looks for `<network>/travel_time_matrix/station_travel_time_matrix.npy` (written by `isa_set/station_matrix.py`) and attaches it, memory-mapped, as `travel_time_matrix_sec`; instances that carry their own matrix keep it. A matrix whose size differs from the number of stations is rejected. `compute_features_scaffold` looks up numpy matrices by station index. `tw_tightness` then uses the driving time between the two stations instead of each request's direct travel time. Tests: `test_t15_station_matrix_feature.py`.

## Change 11. Reaction times are drawn against the constraint's lower bound

The reaction times of an instance are drawn once for all requests and handed to the requests; only negative values were redrawn. The ISA configs set the constraint `reaction_time >= 60`, so a drawn value between 0 and 60 s could never pass the per-request check. After 100 failed attempts the generator printed "Request N failed" and wrote the half-built request (only time stamp and reaction time, everything else empty). In batch 1, 2 to 75 such warnings per instance appeared in about 40 of 100 logs (each failed request prints the warning twice). `passenger_requests.attribute_minimum` reads the lower bound from the constraints and the draw is repeated until the value respects it. Test: `tests/test_attribute_minimum.py`. The same defect could still occur for other pre-drawn attributes (time stamps with a dynamism setting); the batch 1 logs showed only reaction-time cases where checked (i002 request 239).
