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
| OPEN-A2 | HPC test of the matheuristic (copy `MDHODBRP_model_v2.cpp`, `matheuristic.py`, `submit_mh.sh` to `~/ip_model`, rebuild, run `submit_mh.sh ... sa15`). | Not run. |
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
