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

## Batch 2 solver results, first look (2026-11-13, `collect_batch.py --batch 2`)

1200 runs (300 fleets: SA seeds 1-3 and GR), all with a result file. Mean run time 206 s for SA and 191 s for GR (longest 801 s and
869 s), about 67 core-hours in total. Third-party requests summed over the runs: GR 111,063 in 300 runs (about 7.4% of 5000 per
run), SA 113,216 in 900 runs (about 2.5%). Total ride time with the 4 x direct time penalty (the old measure, to be replaced
after D42): SA better than GR on 287 of 300 keys by the mean of the seeds (288 by the best seed), SA/GR cost ratio mean 0.76,
min 0.53, max 1.09. Served share: SA better on 299 of 300 (ratio 1.06, max 1.25). Ride quality of the accepted passengers
(ride time / direct time): SA better on 300 of 300 (ratio 0.65). Like with like (served share differing by less than 2
points): 109 of 300 keys. Batch 1 for comparison (old solver rule): SA better than GR on 232 of 300 keys.
Consequence for the instance space: SA beats GR almost everywhere, so a footprint of GR rests on about 13 keys; the
informative contrast may need the SA variants, SA plus the model, or the served share and quality measures.

## Batch 2 instance space, first fit (2026-11-13, SA against GR, footprint threshold 0.95, 300 keys, 16 varying features)

| Measure | Variance explained (features / performance) | SA good | GR good (purity) |
|---|---|---|---|
| total ride time (SA by the mean of seeds) | 0.53 / 0.31 | 296 of 300 (purity 0.99) | 32 of 300 (0.21) |
| served share | 0.50 / 0.86 | 300 of 300 (1.00) | 176 of 300 (0.85) |
| ride quality of the accepted passengers | 0.54 / 0.46 | 300 of 300 (1.00) | 0 of 300 (no footprint) |

Reading: SA is good almost everywhere; GR is competitive on the served share where the fleet is ample (a well-defined footprint,
purity 0.85, and the projection explains 86% of the performance variance), but on the total cost its 32 keys are scattered
(purity 0.21) and on the quality of the accepted passengers it is never within 5% of SA. The contrast the analysis needs has to
come from a third algorithm (SA variants, SA plus the model) or from the PT coverage features.
