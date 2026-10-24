# Feature List — H-ODBRP Instance Features

Supervisor review: May 30.  
Last updated: May 27, 2026.

> **Note on static vs dynamic:** The experimental protocol uses the **static, offline** H-ODBRP (all requests known in advance). Features marked *(dynamic only)* take a constant value for static instances and are only discriminating in dynamic/online variants — included here for completeness and future ISA work on dynamic H-ODBRP.

---

## 1 — Demand features

| # | Name | Formula / algorithm | Range | Motivation |
|---|------|---------------------|-------|------------|
| 1 | `n_requests` | Count of requests in the instance | [5000, 20000] | Directly controls problem size and algorithmic complexity; larger instances are harder for exact methods. |
| 2 | `demand_density` | n_requests / service_area_km2 | depends on network | High density concentrates requests spatially, making route-sharing more feasible and benefiting constructive heuristics. |
| 3 | `spatial_cluster_coeff` | Average nearest-neighbour ratio of request origins (Clark–Evans index) | [0, 1] | Captures geographic clustering of demand; clustered demand favours route-sharing and may benefit the SA over greedy. |
| 4 | `od_distance_mean` | Mean of origin–destination driving distances (OpenStreetMap network) | depends on network | Longer trips increase URT and constrain time window feasibility; affects how tight a given TW width actually is. |
| 5 | `od_distance_std` | Std of origin–destination driving distances | depends on network | High variance means the instance mixes short and long trips, which complicates vehicle assignment for heterogeneous fleets. |

---

## 2 — Time window features

| # | Name | Formula / algorithm | Range | Motivation |
|---|------|---------------------|-------|------------|
| 6 | `tw_width_mean` | Mean of (tw_end − tw_start) in minutes | depends on config | Average slack available for scheduling; wider windows give the solver more flexibility. |
| 7 | `tw_width_std` | Std of (tw_end − tw_start) in minutes | depends on config | High variance means some requests are very flexible while others are very constrained; mixed instances are harder to schedule. |
| 8 | `tw_overlap_ratio` | Fraction of request pairs with overlapping time windows | [0, 1] | High overlap means requests compete for the same vehicles in the same time slots, increasing scheduling difficulty. *Note: equivalent to "Temporal Demand Overlap" in the literature.* |
| 9 | `tight_tw_fraction` | Fraction of requests with TW width < 20 min | [0, 1] | Quick proxy for how many requests are hard to schedule; high values strongly correlate with infeasibility in IP. |
| 10 | `tw_tightness_factor` | Mean of (tw_width / min_trip_time) per request, where min_trip_time = shortest driving time from origin to destination | [1.1, 2.5+] | Normalizes window width by the trip's own minimum travel time (the f-factor from DARP literature). A TW of 20 min means something very different for a 5-min trip vs a 60-min trip. More principled than raw width. |

---

## 3 — Fleet features

| # | Name | Formula / algorithm | Range | Motivation |
|---|------|---------------------|-------|------------|
| 11 | `fleet_size` | Number of vehicles Proportional Fleet to the number of requests in % | [15, 25] | Directly bounds parallelism; fewer vehicles force more ride-sharing and harder scheduling. |
| 12 | `capacity_cv` | std(vehicle_capacities) / mean(vehicle_capacities) | [0, ∞) | Scale-independent measure of fleet heterogeneity. High CV means a mix of small and large vehicles, which complicates assignment. *Note: "Fleet Heterogeneity Ratio" in CSV uses only std; CV is preferred as it is scale-independent.* |
| 13 | `fleet_utilization_ratio` | sum(vehicle_capacities) / n_requests | [0.2, 1.5+] | Ratio of total fleet capacity to total demand. Values < 1 indicate the fleet cannot serve all requests simultaneously; values > 1 indicate slack. *Note: equivalent to "Demand-to-Capacity Ratio" (inverted) from CSV.* |
| 14 | `request_vehicle_ratio` | n_requests / fleet_size | depends on config | Average load per vehicle if distributed evenly; high values signal that vehicles will be very full and scheduling is tight. |

---

## 4 — Spatial / Network features

| # | Name | Formula / algorithm | Range | Motivation |
|---|------|---------------------|-------|------------|
| 15 | `station_density` | Number of bus stops / service_area_km2 | [0.5, 12.0] stations/km² | Denser stop networks offer more route options, reducing detour costs and making constructive heuristics more effective. |
| 16 | `bsa_flexibility` | Mean number of potential pick-up or drop-off stops available per passenger (Bus Stop Assignment flexibility) | [1, 10+] | When passengers can board/alight at multiple nearby stops, the solver has more routing options; high flexibility tends to reduce URT but increases combinatorial complexity. |
| 17 | `avg_station_degree` | Average degree (connectivity) of bus stops in the road graph | depends on network | Well-connected stops offer more direct routes; low-degree stops can create bottlenecks that force detours. |
| 18 | `coverage_ratio` | Fraction of service area within 500 m of a bus stop | [0, 1] | Low coverage means some requests require long access walks, effectively shrinking the practical service window. |

---

## 5 — Dynamic / Temporal features *(mostly relevant for dynamic H-ODBRP)*

> For the static ISA study these features are constant across all instances (Flux Dynamism = 0, Static-to-Dynamic Ratio = 100%). Include in the feature matrix only if running ISA on dynamic variants.

| # | Name | Formula / algorithm | Range | Motivation |
|---|------|---------------------|-------|------------|
| 19 | `request_flux_dynamism` | Periodicity/concentration of request arrival times during the planning period. 0 = all requests revealed at start (batch / static); 1 = requests arrive continuously and uniformly. | [0, 1] | Governs how much future information the algorithm has; high dynamism means the solver must act on incomplete information. |
| 20 | `static_to_dynamic_ratio` | Percentage of requests known before the planning period starts, relative to total requests | [0%, 100%] | Complement of dynamism; 100% = purely static (this study). Low values require online/re-optimisation methods. |
| 21 | `avg_request_urgency` | Mean lead time = mean(earliest_departure_time − request_issue_time) in minutes | [0, 60+ min] | Short lead times leave little time for optimisation before a vehicle must be dispatched; primarily a dynamic-case feature. |

---

## 6 — Network features (future implementation)

> Not implemented in the current study — would require realistic traffic data. Listed for future dynamic/stochastic variants.

| # | Name | Formula / algorithm | Range | Motivation |
|---|------|---------------------|-------|------------|
| 22 | `congestion_factor` | Mean(actual travel time / free-flow travel time) across all used arcs | [0.5, 1.0] | Values below 1 indicate congestion; affects the reliability of time window feasibility under real traffic. |
| 23 | `triangle_ineq_violation` | Fraction of stop triplets (A, B, C) where t(A,C) > t(A,B) + t(B,C) due to time-dependency | [0.0, 0.2] | Time-dependent travel times can violate the triangle inequality, making distance-based heuristics unreliable and increasing solver difficulty. |

---

## Feature selection notes

For PILOT projection, target **5–8 features** for a meaningful 2D instance space. Candidates to drop first if correlations are high (>0.95):
- `tw_width_std` if correlated with `tw_width_mean`
- `tight_tw_fraction` if redundant with `tw_tightness_factor`
- `od_distance_std` if correlated with `od_distance_mean`
- Features 19–23 (constant in static case or not implemented)
- bsa_flexibility = 1 means only 1 stop is available and this should be treated as the passenger choosing the closest stop to their location

**Features confirmed in scope for static ISA study (Phase C):** #1–#18  
**Features deferred to dynamic ISA (future work):** #19–#23
