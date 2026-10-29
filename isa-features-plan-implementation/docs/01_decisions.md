# 01. Decisions

Format: decision, reason, source. Dates are 2026. "Author" is the researcher (Michell);
"agreed" means proposed by the assistant and accepted by the author.

## Study scope

| # | Decision | Reason | Source |
|---|---|---|---|
| D1 | Performance measure = total passenger ride time, plus 4 x direct travel time for every request served by a third party. Operator cost (driving time, vehicles used) is **not** optimised. | Keeps the three algorithms comparable. Operator cost may be added after the fact as extra performance measures in the ISA. | Author, 10-26 |
| D2 | Algorithms in the performance matrix: SA and GR. The exact Gurobi model is used only to validate SA/GR on 5-15 requests. | The exact model only solves about 5 requests to optimality and reaches a 25% gap at 10 (see 02). It cannot be a third ISA algorithm. | Agreed, 10-26 |
| D3 | A third ISA algorithm is still wanted. | Two algorithms give a single ratio axis; a third gives a richer instance-space analysis. Candidates: the MIP-based LNS (matheuristic) or another heuristic. **OPEN**, see 06. | Open |
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
