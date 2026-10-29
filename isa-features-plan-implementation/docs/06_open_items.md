# 06. Open items (unresolved, from the working sessions)

Marked in order of what blocks the next step.

## Blocking the full instance set

| ID | Item | State |
|---|---|---|
| OPEN-G1 | Result of test job 29846018 (ISA_b00_i001, with station matrix). It reached the 6 h limit with no output files; expected to be killed at about 12:33 on 2026-10-29. | Waiting. If killed, nothing is saved. |
| OPEN-G2 | Rerun of the 6-request profile after the nearest-node fix. Expected about 1 s per request instead of about 20 s. | Commit `f67f8723` pushed? Rerun not done. |
| OPEN-G3 | (Decision D14b: the station matrix gets its own job.) Where the other ~4 h of the test job went. Requests explain about 1.2 h of 5.3 h. Suspected: the station matrix (3515 searches on the full drive graph; the row cache grows, memory 7.1 GB of 8 GB). | Not measured. Plan: time the matrix in its own job; if slow, compute it separately and write it once. |
| OPEN-G4 | Feature step for the batches that attaches the shared station matrix (see 05). | Not written. |
| OPEN-G5 | Size `MEM_GB` and `WALL_H` for batch 1 from the measured full instance. | After G2/G3. |

## Algorithms

| ID | Item | State |
|---|---|---|
| OPEN-A1 | Third ISA algorithm: SA variants (neighbourhood configurations) or max-ALNS (D3). Which exact variants, and where the code lives. | Author to decide. |
| OPEN-A2 | HPC test of the matheuristic (copy `MDHODBRP_model_v2.cpp`, `matheuristic.py`, `submit_mh.sh` to `~/ip_model`, rebuild, run `submit_mh.sh ... sa15`). | Not run. |
| OPEN-A3 | SA/GR job memory: 16 GB reserved, about 6 GB used. | Resize. |
| OPEN-A4 | Rebuild the solver with `maxstations 4000`; the change in ODBRP_P is not committed. | Author to commit. |
| OPEN-A5 | Table of SA vs v2 objective and vehicles used on the 5-request and 10-request instances. | Not tabulated. |
| OPEN-A6 | Does SA's one-sided advantage over GR (70 of 72) persist on the wide design? | Needs the full set. |

## Settings

See OPEN-S1 to S4 in `03_recommended_settings.md`. The author makes the final choice.

## Documentation and housekeeping

| ID | Item |
|---|---|
| OPEN-D1 | `REQreate/rank_model.py`: another ISA feature (confirmed); add one line on what it computes. |
| OPEN-D2 | 7 web-app tests fail on the branch and on the commit before; cause is the old main-checkout REQreate in separate processes, plus timeouts. Not fixed. |
| OPEN-D3 | The cause of the first generation job's exit code 2 (empty config argument from the job array) was never found; replaced by one job per config. |
| OPEN-D4 | The exit of the LSF job reminder: the `/jobs` mod tracks only `bsub` commands the assistant runs, not the author's own HPC window. |

## Lessons for the write-up

1. A grid design with two levels per factor leaves gaps; continuous Latin hypercube draws fix this (D8).
2. A seed bug can silently collapse the instance space; check the number of distinct values of key features.
3. Profile before scaling up: one hidden call (`osmnx.nearest_nodes`) dominated request generation.
4. Reserve resources from measured usage; early jobs over-reserved memory and cores.
