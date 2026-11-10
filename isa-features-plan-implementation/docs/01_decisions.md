# 01. Decisions

Format: decision, reason, source. Dates are 2026. "Author" is the researcher (Michell);
"agreed" means proposed by the assistant and accepted by the author.

## Study scope

| # | Decision | Reason | Source |
|---|---|---|---|
| D1 | Performance measure = total passenger ride time, plus 4 x direct travel time for every request served by a third party. Operator cost (driving time, vehicles used) is **not** optimised. | Keeps the three algorithms comparable. Operator cost may be added after the fact as extra performance measures in the ISA. | Author, 10-26 |
| D2 | Algorithms in the performance matrix: SA and GR. The exact Gurobi model is used only to validate SA/GR on 5-15 requests. | The exact model only solves about 5 requests to optimality and reaches a 25% gap at 10 (see 02). It cannot be a third ISA algorithm. | Agreed, 10-26 |
| D3 | A third ISA algorithm is wanted: different versions of the SA (other neighbourhood configurations, and so on) or a max-ALNS variant. The exact model and the matheuristic are not candidates. | Two algorithms give a single ratio axis; a third gives a richer instance-space analysis. Which variant exactly is **OPEN**, see 06. | Author, 10-29 |
| D4 | The matheuristic (MIP-based LNS) exists mainly to show **where SA is weak**, not to be an ISA algorithm. | It re-solves parts of SA solutions with the exact model; a part can never get worse. | Author, 10-26 |

## Instance set

| # | Decision | Reason | Source |
|---|---|---|---|
| D5 | Network = whole Capital Region of Denmark, **without Bornholm**. Includes inner and outer municipalities. | Author wanted suburban areas. Bornholm is an island, not connected by road, and would distort distances. | Author, 10-27 |
| D6 | Planning period = 2 hours, placed in the low-demand part of the day: start between 12:00 and 19:30. | Public transport is already full at peak hours; on-demand buses are interesting when demand is low. The 19:30 limit keeps latest arrivals before midnight, because the solver treats 86400 s specially. | Author, 10-27 |
| D7 | 50-500 requests per hour, so 100-1000 requests per instance. | Covers small to large instances. | Agreed, 10-27 |
| D8 | Instances are drawn with a **Latin hypercube** over continuous ranges, not a factorial grid. | The pilot (grid with 2-3 levels per factor) left large gaps in the instance space. | Agreed, 10-27 |
| D9 | The set is grown in **batches** until the instance space stops growing (share of new instances far from existing ones below about 5%, or a cap). | Answers "how do we know the space is large enough". | Agreed, 10-27 |
| D10 | Depots and hotspots are **not** varied during generation. The depot is chosen after generation (a station no request uses). | Author: a depot can be any node, chosen later. Hotspots: the generator only measures geographic dispersion, it cannot control it. | Author, 10-27 |
| D11 | Each generated instance gets **3 fleets** drawn with a second Latin hypercube (5-40% of requests as vehicles, 0-100% share of 8-seat vehicles). Fleets cost nothing to generate. | Fleet is a solver input, not part of the instance; it adds features 11-14 without regenerating requests. | Agreed, 10-27 |
| D12 | Walking time limit 550-600 s, walk speed 1.38889 m/s, vehicle speed 20 km/h, direct trip time between 300 s and 3600 s. | Taken from the author's earlier papers (see `pilot/settings_from_papers.md`). The upper trip limit is 3600 s because suburban trips are longer than in Copenhagen Municipality. | Agreed, 10-27 |
| D13 | The station-to-station travel-time matrix is written **once** for the network, not per instance. | It is identical for every instance (same stations, same speed). | Agreed, 10-28 |
| D14a | Request range stays 50-500 per hour for now (papers go to 3000 per hour). | Fine for now; may be raised later. | Author, 10-29 |
| D14b | The station travel-time matrix is computed in its own HPC job, once, not inside an instance's generation. | It is the same for all instances and may be the slow part of generation. | Author, 10-29 |
| D14 | The settings may still change. The author will make the final decisions. | Ranges are in one file (`isa_set/design.json`) and each batch reads it. | Author |

## Generator and solver

| # | Decision | Reason | Source |
|---|---|---|---|
| D15 | The earlier IP model (v1) is replaced by a new model v2. | v1 was infeasible by construction (see 02, checker). | Author asked for the check, 10-25 |
| D16 | Request seeds follow the config seed. | In the first pilot all configs drew the same locations (6 distinct mean OD distances in 72 instances); see 04 and 02. | Author, 10-25 |
| D17 | Station, school and hospital lookups are restricted to the network polygon. | A station outside the cut-out area (Bornholm) leaked into the network. | Found in the test, 10-28 |
| D18 | A cached nearest-node index replaces the per-call `osmnx.nearest_nodes` in request sampling. | The profile showed 117 s of 154 s were spent rebuilding the graph for each lookup (see 02). | 10-29 |
| D19 | Heuristic fixes: the travel-time file is no longer symmetrised (row = from, column = to); walking distances in metres are converted to seconds with the walk speed. | Wrong data in the first pilot solver runs; those results were discarded and rerun. | 10-26 |
| D20 | Hybrid-feasibility (HF) code is kept but deprecated. | Not part of the main ISA. | Earlier session |

## Working rules for compute

| # | Decision | Source |
|---|---|---|
| D21 | Solvers and long generation runs are **not run on the author's PC**. They run on the DTU HPC. Short local Python checks and unit tests are fine. | Author |
| D22 | HPC jobs reserve only what they use. One core per single-threaded job. On LSF memory is per slot. After each run, report CPU time, run time and max memory. | Author (admins kill over-reserving jobs); an early network job reserved 4 cores / 64 GB and used 7.7% CPU / 4.4 GB |

## Decisions of 2026-11-10 (model-based neighbourhood search, see 07)

| # | Decision | Source |
|---|---|---|
| D23 | In the model-based repair, requests that stay fixed keep their vehicle, their chosen stop and their relative order; their times may shift as long as the route stays feasible, and the freed requests may be inserted between their stops. | Author |
| D24 | The 6-8 freed requests are chosen with three operators that are all tried and logged (related requests in space and time, worst ride-time ratio, unserved requests with the largest penalty); unserved requests are included. | Author |
| D25 | "SA plus model" becomes a column of the performance matrix. This revises D4 (the matheuristic was only to show where SA is weak). | Author |
| D26 | Budget of the first experiment: about 100 core-hours (6 tight-fleet instances, starts from SA with 200 and with 10000 moves). | Author |
| D27 | A fleet that leaves about 55% of the passengers to third parties is too small for the design. The new lower bound of the fleet size is **OPEN** (see 06, OPEN-S5). | Author |

## Decisions of 2026-11-10 (demand pattern of batch 2, see 08)

| # | Decision | Reason | Source |
|---|---|---|---|
| D28 | Batch 1 (uniform demand: origins uniform over the region, destinations on a 1.7-20 km ring) is kept as the homogeneous corner. Batch 2 replaces it for the final set. | Inner Copenhagen got as many requests as farmland; spatial structure was not a real axis of the instance set. | Author |
| D29 | Scenarios a (feeder, inbound), b (all-day, both ways) and c (night, outbound) are two continuous design factors, not three categories: `inbound_share` in [0, 1] and `concentration` in [0, 2]. a is share 1, c is share 0, b is about 0.5; the label is derived and only used in reports. | Uses every instance for the analysis; b is a mixture of a and c. | Agreed (Fable review) |
| D30 | Where requests are: zone weights from OpenStreetMap counts per zone of the 25 x 25 grid. Origins of inbound trips follow residential buildings, destinations follow attractions (restaurants, cafes, bars, pubs, shops, cinemas, theatres, tourism, leisure); outbound is the mirror. `concentration` raises the shares to a power (0 = uniform, 2 = packed). No metro is modelled; the inner area simply gets few origins. Counted once by `isa_set/zone_weights.py` (hpc/zone_weights.sh), not during generation. | Author chose real POIs. The generator's own `density_pois` mixes all POI types and is dominated by buildings, so attractions are counted separately. | Author |
| D31 | Time of day is a label only: `start_hour` is fixed at 14.0 and leaves the hypercube. | For a static solver with a constant travel-time matrix, shifting all times changes nothing; a fixed afternoon also keeps latest arrivals far from midnight (86400 s). Crossing midnight needs a solver check first (OPEN). | Agreed |
| D32 | `max_dtt_s` is 5400 (90 min) and `min_dtt_s` stays 300. With zone weights the generator bypasses the dtt ring around the origin, so the cap is the only brake on trip length; deleting it makes destinations uniform over the region. | Verified in `passenger_requests.py` (subset_zones branch) and `input_json.py` (ring only when both min and max exist). The cap is one number and can be loosened after the smoke test. | Proposed, author asked whether it is needed |
| D33 | Speed stays 20 km/h for batch 2. | Edge times are fixed when the network is built; a new speed needs a rebuilt network and station matrix. Triangle inequality holds for shortest-path matrices; the old violations came from the symmetrisation removed in D19. | Fable review |
| D34 | The volume anchor (requests per hour, requests per km2) waits for the author's report. Until then 50-500 requests per hour stay. | Author | Author |
| D28 | The +3600 s that `odbrp-parallel.cpp` added to every `latest_arrival` ("ensure feasibility") is removed. SA, GR and model v2 use the time windows of the instance file. A request that the travel-time matrix cannot serve in its window stays with the third party. Every SA/GR result produced before this change (batch 1, T12-T14, the perf matrices) used the relaxed deadline and must be rerun. Reason: the shift only existed because old instances were rejected; the generator's windows are the intended problem. Origin and destination must be reachable in the matrix (OSM can report different times in generation and in the matrix): `model/check_requests.py` counts such requests. | Author, 2026-11-10 |
