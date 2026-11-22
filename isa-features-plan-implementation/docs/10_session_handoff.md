# 10. Session hand-off (written 2026-11-13, end of the batch 2 session)

Read this first in a new session, then `11_hpc_runbook.md` (how to run things on the HPC), then `01_decisions.md`
(D1-D47, the authority) and `06_open_items.md`. The researcher is Michell Queiroz (DTU); she wants next steps first as a
numbered list, one command per code block, exact names, and no solver or generation runs on her PC (HPC only).

## Where things are

| What | Where |
|---|---|
| Code and docs (this repo, branch `isa-merge-041` locally, `isa-features` on GitHub) | `isa-features-plan-implementation/` (docs, `isa_set/`, `src/isa_pipeline.py` is at the repo root `src/`) |
| Generator (REQreate) | `REQreate/` at the repo root |
| Data and results on the HPC | `/work3/micqu/isa/` (see the runbook) |
| Solver and exact model (separate repo, she commits there) | `C:\Users\micqu\Desktop\ODBRP_P\MDHODBRP-P` (`odbrp-parallel.cpp`, `model/`), built on the HPC |
| Chapter (Overleaf, git bridge) | `C:\Users\micqu\Desktop\Overleaf Projects\isa-hodbrp-chapter` (branch `main`, remote `overleaf`; `HANDOFF.md` inside is the writing session's own hand-off) |
| Local GTFS work (not on GitHub, uncommitted, gitignored) | `scratch_gtfs_router.py`, `scratch_osm_to_gtfs_stops.py`, `scratch_osm_gtfs_all_pairs.py`, `scratch_odbrp_vs_gtfs.py`, `scratch_gtfs_features.py`, folders `scratch_odbrp_gtfs/`, `scratch_gtfs_features/`, data in `data/2026-11-10/` at the repo root (main checkout); copy on the HPC in `/work3/micqu/isa/gtfs/` |

## State at the end of the session

- **Study design:** instance-space analysis of the static H-ODBRP on the Capital Region of Denmark (without Bornholm). Batch 1 (uniform
  demand, 2 h, 100-1000 requests) is only the homogeneous corner and its results are obsolete (D28). **Batch 2 is the final set**:
  100 instances of 5000 requests, 4 h from 14:00, zone weights from OpenStreetMap homes and attractions with drawn `inbound_share`
  [0, 1] and `concentration` [0.1, 1.5], Copenhagen and Frederiksberg excluded (`central_threshold` 0.1), zone-level distance decay
  L = 3 km, direct time 300-3600 s, 3 fleets per instance with 10-40% of the requests as vehicles (D29-D41, D47). See `08_demand_plan.md`.
- **Done for batch 2:** generation (100/100, 0 failed requests), request summary (`plot_requests.py --batch 2 --summary`), features
  (300 rows, `results/features_batch_02.csv`; 5 constant features dropped, 16 vary), solver rows (1200 runs: SA seeds 1-3 and GR on 300
  fleets, all with results), collection (`collect_batch.py --batch 2`), first instance space of SA against GR (table in
  `09_batch2_handoff.md`): SA beats GR on 287 of 300 keys (total), 299 (served), 300 (quality); GR competitive only on the served share
  where the fleet is ample (purity 0.85); on total cost its 32 good keys are scattered (purity 0.21).
- **Running or unknown at hand-off:** (a) SA1000 timing probe on batch 91 (rows 7-8, results in `/work3/micqu/isa/solver/batch_91/`);
  (b) the 10 jobs of the public transport features (`/work3/micqu/isa/gtfs/`): the first attempt failed because `pyarrow` was missing
  in `~/venvs/reqreate`; `pip install pyarrow` and the resubmission were given to the user and may not have been done: check `bjobs` and
  `ls /work3/micqu/isa/gtfs/features`. Ask her.
- **Unpushed commits** in the code repo (5 at hand-off): `git push origin isa-merge-041:isa-features`.

## Next tasks, in order

1. **Close the open jobs.** Read the SA1000 probe (time, memory) and decide whether the SA variants (1000, 2000, 10000 moves) are affordable at 5000
   requests (batch 1 on 1000 requests: SA10000 mean 125 s, max 812 s). Finish the PT features: install `pyarrow`, resubmit, merge
   (`scratch_gtfs_features.py --merge`), join to the feature table by the instance key (the fleet key `ISA_b02_iNNN_1_fleetK` starts with it),
   check that `validate_features_csv` accepts extra columns, else keep an extra file and merge in the pipeline step.
2. **Decide the performance measures after D42.** The objective is now the sum of the ride times of the served passengers (no 4 x penalty), so
   total ride time is only comparable at the same number of served requests. `isa_set/collect_batch.py` still describes `total` as including the
   penalty and ranks by it; its measures are total, served share and quality (ride time / direct time of the accepted passengers). Re-define
   the primary measure (served share first, then ride time: the lexicographic order of D42), update the docstring, the docs and the chapter
   (section 3 has a TODO about it).
3. **Third algorithm for the instance space.** SA beats GR almost everywhere, so the analysis needs contrast: SA variants (more moves, tight fleets,
   `make_solver_jobs.py --sa-variants ... --variants-max-fleet-percent ...`), and/or SA plus the exact model (`07_model_neighbourhood_search_plan.md`,
   D23-D27, D46; the SA + model session did the audits and the scaling test; Phase A of the plan is not started), and the PT coverage features.
   Then `collect_batch.py --partial`, the instance space with `ISA_ALGORITHMS=SA,SA1000,...,GR`, footprints.
4. **PT coverage as ISA features** (author decision: yes): `pt_no_stop_share`, `pt_direct_share`, `pt_transfer_share`, `pt_no_connection_share`, mean door-to-door
   minutes and transfers per instance (definitions in `scratch_gtfs_features.py`; one planned Tuesday timetable, no delays). Data may go to the HPC by scp, not to GitHub.
5. **ODBRP against the fixed lines** on batch 2 results (D45: same leaving moment, wait included; D46: `Departure from home` unused): download one instance csv and
   its result (for example `ISA_b90_i001`, in `/work3/micqu/isa/solver/batch_90/results/`) into `scratch_odbrp_gtfs/hpc/` and run `scratch_odbrp_vs_gtfs.py`
   (pilot result on central Copenhagen in the morning: public transport faster for most passengers; pickups after the latest departure 51%).
6. **Chapter.** Overleaf project connected (remote `overleaf`); sections 0-4 drafted by the writing session, 5-8 are outlines. Send it measured numbers (it must not quote any it was not given).
   Uncommitted in that folder at hand-off: `HANDOFF.md`, `springer_template/`, edits in `sections/02-literature.tex`. The user pushes (token); never push for her.
7. **Housekeeping:** raise the solver's 6000-passenger cap (`matrixPS[6000+1]`, line 46) and make the SA exit non-zero on the station-id error (batch 92 with 10000 requests
   gives empty result files while the job reports success); OPEN-A4 (model prices 27 of 200 sampled b91 routes up to 1070 s cheaper than the SA cost); the growth-check cap is undecided
   (OPEN-S4); names in the report: `request_flux_dynamism` and `static_to_dynamic_ratio` measure temporal spread of the demand in this static study.

## Rules the researcher set (apply them)

- Never run the solver, the generator or long jobs on her PC; HPC only. Short local Python and unit tests are fine (`pytest` or `python -m unittest` in `isa-features-plan-implementation`; osmnx and shapely are not installed locally, so generator tests skip).
- Do not search outside the repository; ask for a path instead. (The Overleaf folder and the GTFS data were named by her explicitly.)
- HPC jobs must use what they reserve: one core per single-threaded job, memory is per slot, state cores and memory with every submit command, and after a run report CPU time, run time and max memory with the utilisation.
- Never add Co-Authored-By lines or mention Claude or Anthropic in commits or pull requests. The user commits in `ODBRP_P` herself.
- After she says a step is done, check the pasted output step by step ("expected X, got Y, OK / NOT OK") before giving the next step.
