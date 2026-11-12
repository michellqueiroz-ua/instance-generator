# 09. Batch 2: hand-off for the SA + model session (2026-11-11)

Read this first if your context stops at batch 1. Decisions are in `01_decisions.md` (D29-D41), the plan in
`08_demand_plan.md`; the model plan is `07_model_neighbourhood_search_plan.md`.

## What changed between batch 1 and batch 2

| | Batch 1 | Batch 2 |
|---|---|---|
| Period | 2 h | 4 h (D36) |
| Requests per instance | 100-1000 | **5000** fixed (D35) |
| Demand in space | uniform | zone weights from OSM homes and attractions; `inbound_share` and `concentration` drawn (D30, D31) |
| Central Copenhagen and Frederiksberg | no rule | excluded, `central_threshold` 0.1 (D37, D41) |
| Trip length | ring 1.7-20 km | zone-level distance decay L = 3 km, direct time 300-3600 s (D33, D39) |
| Start | [12, 19.5] drawn | 14:00 fixed |
| Fleet | 5-40% of requests | 10-40%, provisional (fleets can be redrawn, `design/batch_02_fleets.json`) |
| Stop search of the generator | missed zones north and south | fixed (D38) |

Fleet percent of 5000 requests means 500-2000 vehicles. The solver at 5000 requests is **not measured**.

## Smoke instances (generated, in `/work3/micqu/isa/Capital Region of Denmark/csv_format/`)

| Instance | Requests | inbound_share | concentration | Generation |
|---|---|---|---|---|
| `ISA_b90_i001_1.csv` | 5000 | 1.0 | 1.5 | 1129 s, 2.8 GB |
| `ISA_b91_i001_1.csv` | 5000 | 0.0 | 0.1 | 2250 s, 2.85 GB |
| `ISA_b92_i001_1.csv` | 10000 | 0.5 | 0.8 | 3012 s, 2.86 GB |

Their fleets are in `/work3/micqu/isa/design/batch_90_fleets.json` (and 91, 92). Trip statistics and maps:
`python isa_set/plot_requests.py ISA_b90_i001` writes `/work3/micqu/isa/results/maps/<name>.png`.
Effective drive speed of the network is 27-28 km/h (D34), not the 20 km/h in `design.json`.

## First job: solver scaling test (settles the solver budget, D35 risk)

Question: how long and how much memory do GR and SA (200 moves) need at 5000 and 10000 requests with 500-2000
vehicles? One run per job so the logs give per-run `CPU time`, `Run time`, `Max Memory`; one seed. On the HPC login node,
venv active, solver built in `~/solver` (check which `a.out`: D28 removes the +3600 s slack from `odbrp-parallel.cpp`;
timings are comparable either way, served shares are not):

```bash
for b in 90 91 92; do python ~/instance-generator/isa-features-plan-implementation/isa_set/make_solver_jobs.py --batch $b --sa-seeds 1; done
```
```bash
for b in 90 91 92; do SOLVER=~/solver PER_JOB=1 MEM_GB=4 WALL_H=8 bash /work3/micqu/isa/solver/batch_$b/submit.sh; done
```

Each batch gets 6 rows (3 fleets x SA seed 1 + GR), so 18 jobs, 1 core and 4 GB each, 8 h limit; the memory and time
are the measurement, so the reservation is generous. After the runs: read the logs, report per run `CPU time`,
`Run time`, `Max Memory` and the share served by third parties, and then size the full batch (rows, `MEM_GB`,
`WALL_H`, the SA move budget: 200 moves per call is 3.6 moves per request at 5000 requests).

## Then

1. Full batch 2 solver rows only after the scaling test (the generation of the 100 instances runs meanwhile).
2. Fleet lower bound (D27) from batch 2 data, not batch 1.
3. SA + model (07): the model works on 6-8 requests of a solution, so the instance size matters only for the SA start.

Everything in `/work3/micqu/isa/results/` from batch 1 used the +3600 s deadline (D28) and is to be rerun.

## Result of the solver scaling test (2026-11-12, from the SA + model session)

Instances ISA_b90_i001 and ISA_b91_i001 (5000 requests), fleets 1-3, SA (200 moves, seed 1) and GR, solver without the
+3600 s slack (D28), one core and 2 GB reserved per run: all 12 runs completed, CPU use 88-99%, run time 35-663 s
(longest: b91 fleet 1 GR, 11 min), peak memory 1.09-1.60 GB, so 2 GB per job is enough. Estimate for the whole batch
(300 fleets x 3 SA seeds + GR): about 90 core-hours. Third-party share SA / GR: b90 4.6/11.4%, 0.6/3.8%, 0.0/0.7% (fleets of
902, 1288, 1985 vehicles); b91 25.0/32.6%, 0.9/3.6%, 9.2/17.2% (732, 1914, 1122 vehicles). Audits (model v2 rules): reported
cost equals the sum of the ride times of the served passengers in all 12 runs; no violation except the known 1 s
same-station artefact; 2 of 5000 requests of b91 are unservable (the matrix minimum exceeds the time window).
**Limit:** the solver caps the number of passengers at 6000 (`matrixPS[6000+1]`, odbrp-parallel.cpp line 46), so
instances of 10000 requests (batch 92) give empty result files although the job reports success; batch 2 (5000) is not
affected. To fix: raise the cap and make the SA exit non-zero on the station-id error (deferred). Open: the model prices
27 of 200 sampled b91 routes up to 1070 s cheaper than the SA cost (OPEN-A4).
