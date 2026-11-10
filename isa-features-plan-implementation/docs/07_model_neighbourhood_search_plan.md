# 07. Plan: exact-model repair of SA solutions (PLAN ONLY, nothing implemented)

Status 2026-11-10. Author's idea: split the vehicles into groups of nearby vehicles as the parallel SA does
(k-medoids), free 6-8 requests of a group, keep everything else fixed and let the exact model re-optimise only
those: a destroy-and-repair move with an exact repair, accepted only if it improves. Advice from an advisor
agent (Fable) is merged in; the advisor read `matheuristic.py` and grepped the model but did not read the SA code,
and its line references are unverified.

## Relation to the existing matheuristic (`ODBRP_P/MDHODBRP-P/model/matheuristic.py`)

It frees whole vehicles (every request of a group). With about 8 or more requests per vehicle in tight fleets a
part shrinks to one vehicle plus up to 3 unserved requests, so there is no exchange between vehicles. The new idea
frees a fixed number of requests and keeps the rest as a skeleton. Extend `matheuristic.py` (parsing, solver call,
part log, acceptance loop); add a request-based part builder and a partial-fix option to model v2 (today: fix
everything with `--fix_start`, or nothing).

## Design (decisions D23-D26 in 01)

- Neighbourhood: a seed vehicle and its 2-3 nearest vehicles (4 in total), 6-8 freed requests, about half served and half unserved in tight fleets.
- Operators (all tried, gain logged per operator): related requests (pickup/drop-off distance, time-window overlap), worst ride-time ratio, unserved requests with the largest 4 x direct time whose window fits the group's routes. Part size adapts: +1 after a part proven optimal in under 5 s, -1 after a timeout; a set that was solved is not solved again.
- Fixed requests: vehicle, stop and relative order fixed; times and interleaving free. In the model: delete arcs of fixed nodes on other vehicles, backward arcs between fixed nodes and forward arcs that skip a fixed node, and the non-chosen stops of fixed requests. Arcs per vehicle fall from about (F+m)^2 to about F*m + m^2 (F fixed nodes, m freed). Faster alternative if the model is slow: set partitioning over enumerated insertions of at most 3-4 freed requests per vehicle.
- Preprocessing priority: time-window tightening and arc elimination; limit on candidate stops per request; skeleton arc removal; symmetry breaking only for empty identical vehicles.
- Search: improvements only (threshold 0.5), 10-20 s and 1 thread per part, disjoint vehicle groups in parallel rounds (k-medoids with a new seed each round so borders move; every unserved request belongs to exactly one group), one part per core so no reserved core idles. Stop on a core-seconds budget or after about 3 x (number of vehicles) parts without gain. Measure process start-up (Gurobi environment, travel-time matrix); keep a persistent solver process if it exceeds about 20% of a part.
- Fair baseline: SA with 10000 moves at the same total core-seconds, and SA plus model against SA given the same extra time.

## Phases and go/no-go

| Phase | Work | Go criterion |
|---|---|---|
| A (2-3 days) | Fix the known issues of `matheuristic.py` (below) and audit model/heuristic consistency on 20 instances | 100% of SA routes feasible in the model, costs equal |
| B (3-4 days) | Partial-fix option in v2, request-based part builder, serial | 8 freed requests in 4 vehicles proven optimal in 20 s median |
| C (2-3 days) | Parallel disjoint rounds, adaptive size, operator logging; CPU and memory utilisation measured | |
| D (about 100 core-hours) | 6 tight-fleet instances (about 100, 300, 1000 requests), starts from SA 200 and SA 10000 | At least 2% below SA 10000 at equal compute on at least 4 of 6, and at least 70% of parts proven optimal in the limit. No-go: below 0.5% |
| E (only on go) | Set-partitioning repair if B is slow; full tight-fleet batch; "SA plus model" as a performance matrix column (D25) | |

Phase A fixes: infeasible start routes are skipped and their cost left out (find the cause); idle vehicles are missing from the SA output (they absorb unserved requests cheaply); passenger id mapping assumes ids 0..n-1; the parts size, vehicles per part and seed cannot be set from `submit_mh.sh`; `text.split()[10]` breaks when the instance path contains spaces (the Capital Region path does).

## Risks

- Any SA route that is infeasible in the model invalidates every comparison. Check the waiting rule, walking times and stop choice in the time windows, depot start and return, the drop-before-pick order at a shared stop, and the truncation of travel times to integers.
- Improvements only is a local search: expect a plateau; report it as a polishing step.
- Ample fleets are a sanity check only: SA already reaches the sum of the direct times (see 02, T13), so any gain there signals a bug.
- Gurobi licence tokens with many parallel processes; every process loads the full travel-time matrix (memory per slot).
