# Implementation Context — Instance Generator Feature Computation & MATILDA Pipeline

> **Purpose of this file:** Briefing document for a coding agent implementing feature computation inside the instance generator and the MATILDA ISA pipeline (`src/isa_pipeline.py`). Features must be computed at generation time and stored alongside each instance.  
> **Last updated:** May 27, 2026

---

## 1. Instance JSON Schema

This is the **definitive schema** for a REQreate instance JSON. All extractor code must be written against this structure.

```json
{
  "service_area_km2": 25.0,

  "requests": [
    {
      "origin":      { "x": 4.40123, "y": 51.21456 },
      "destination": { "x": 4.41789, "y": 51.22001 },
      "tw_start":    480,
      "tw_end":      510,
      "demand":      2,
      "issue_time":  420
    }
  ],

  "vehicles": [
    {
      "capacity": 8,
      "depot":    { "x": 4.39000, "y": 51.20000 }
    }
  ],

  "stops": [
    {
      "id":   "stop_001",
      "x":    4.40050,
      "y":    51.21400,
      "degree": 3
    }
  ],

  "road_network": {
    "nodes": [
      { "id": "n1", "x": 4.40050, "y": 51.21400 }
    ],
    "edges": [
      { "u": "n1", "v": "n2", "travel_time_min": 2.5, "distance_km": 0.4 }
    ]
  },

  "travel_time_matrix": {
    "stop_001:stop_002": 3.2
  }
}
```

### Field notes

| Field | Type | Notes |
|---|---|---|
| `service_area_km2` | float | May be absent in older instances — fall back to bounding box of origins |
| `requests[].origin/destination` | `{x, y}` | **WGS84 longitude/latitude** (x = lon, y = lat). NOT kilometres. Convert before computing distances. |
| `requests[].tw_start/tw_end` | int | Minutes from midnight (e.g., 480 = 08:00) |
| `requests[].demand` | int | Number of passengers in this request |
| `requests[].issue_time` | int | Minutes from midnight when request was submitted — used only for dynamic features; absent in static instances |
| `vehicles[].depot` | `{x, y}` | Same coordinate system as requests |
| `stops` | list | Bus stops available in the network. May be absent in older instances. |
| `stops[].degree` | int | Number of road edges connecting this stop in the graph |
| `road_network` | object | Embedded graph. May be absent — if so, use `osmnx` to fetch from OpenStreetMap using the bounding box of the instance. |
| `travel_time_matrix` | object | Precomputed stop-to-stop travel times in minutes. Key format: `"stop_A:stop_B"`. If absent, compute from `road_network` edges or estimate from Euclidean distance. |

### Coordinate conversion
`x,y` are WGS84 degrees. To convert to kilometres for distance calculations, use the **Haversine formula** or approximate as:

```python
import math

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))
```

> ⚠️ Use Haversine (see helper above) for all distance calculations — `x,y` are degrees, not kilometres. Do not use raw Euclidean distance on degree coordinates.

---

## 2. Features to Implement in the Generator

All 21 features (#1–#21 from `docs/feature_list.md`) must be computed **at instance generation time**. The generator should call a `compute_features(instance: dict) -> dict` function and write one row to `data/results/features.csv` immediately after each instance is generated (preferred), or append the feature dict to the instance JSON under a `"features"` key.

| # | Name | JSON fields needed | Difficulty |
|---|---|---|---|
| 1 | `n_requests` | `requests` | Trivial |
| 2 | `demand_density` | `requests`, `service_area_km2` | Easy |
| 3 | `spatial_cluster_coeff` | `requests[].origin` | Easy |
| 4 | `od_distance_mean` | `requests[].origin/destination` | Easy — use Haversine |
| 5 | `od_distance_std` | same | Easy |
| 6 | `tw_width_mean` | `requests[].tw_start/tw_end` | Trivial |
| 7 | `tw_width_std` | same | Trivial |
| 8 | `tw_overlap_ratio` | `requests[].tw_start/tw_end` | Easy — O(n²), fine for n ≤ 200 |
| 9 | `tight_tw_fraction` | `requests[].tw_start/tw_end` | Trivial |
| 10 | `tw_tightness_factor` | `requests[].tw_start/tw_end` + `travel_time_matrix` | Medium — needs travel times; Haversine fallback if absent |
| 11 | `fleet_size` | `vehicles` | Trivial |
| 12 | `capacity_cv` | `vehicles[].capacity` | Easy |
| 13 | `fleet_utilization_ratio` | `vehicles[].capacity`, `requests[].demand` | Easy |
| 14 | `request_vehicle_ratio` | `requests`, `vehicles` | Trivial |
| 15 | `station_density` | `stops`, `service_area_km2` | Easy |
| 16 | `bsa_flexibility` | `stops`, `requests[].origin/destination` | Easy-Medium |
| 17 | `avg_station_degree` | `stops[].degree` (or `road_network`) | Easy |
| 18 | `coverage_ratio` | `stops`, `requests[].origin/destination` | Medium — grid approximation |
| 19 | `request_flux_dynamism` | `requests[].issue_time` | Easy (constant = 0 for static) |
| 20 | `static_to_dynamic_ratio` | `requests[].issue_time` | Easy (constant = 100% for static) |
| 21 | `avg_request_urgency` | `requests[].issue_time`, `tw_start` | Easy (constant for static) |

> Features 22 (`congestion_factor`) and 23 (`triangle_ineq_violation`) are deferred — require real traffic data. Do not implement yet.

---

## 3. Implementation Instructions

### Feature 10 — `tw_tightness_factor`
**Definition:** Mean over all requests of `tw_width / min_trip_time`.

`min_trip_time` = shortest travel time (minutes) from the request's origin stop to its destination stop.

```python
def compute_tw_tightness_factor(requests, travel_time_matrix):
    factors = []
    for r in requests:
        key_fwd = f"{r['origin_stop']}:{r['destination_stop']}"
        key_rev = f"{r['destination_stop']}:{r['origin_stop']}"
        min_tt = travel_time_matrix.get(key_fwd) or travel_time_matrix.get(key_rev)
        if min_tt is None or min_tt <= 0:
            continue  # skip if no travel time available
        tw_width = r["tw_end"] - r["tw_start"]
        factors.append(tw_width / min_tt)
    return float(np.mean(factors)) if factors else float("nan")
```

**Fallback** if `travel_time_matrix` is absent: estimate `min_trip_time` from Haversine distance divided by an assumed average speed (e.g., 30 km/h). Log a warning.

---

### Feature 15 — `station_density`
**Definition:** `len(stops) / service_area_km2`

```python
def compute_station_density(stops, service_area_km2):
    return len(stops) / service_area_km2 if service_area_km2 > 0 else 0.0
```

**Fallback** if `stops` is absent: return `float("nan")` and log a warning. Do not fabricate the value.

---

### Feature 16 — `bsa_flexibility`
**Definition:** Mean number of candidate stops within walking distance for each request origin **and** each request destination.

Walking distance threshold: **500 m** (configurable, default 500 m).

A stop `s` is a candidate for request endpoint `p` if `haversine_km(p.x, p.y, s.x, s.y) <= walk_threshold_km`.

```python
def compute_bsa_flexibility(requests, stops, walk_threshold_km=0.5):
    if not stops:
        return float("nan")
    stop_coords = np.array([(s["x"], s["y"]) for s in stops])
    counts = []
    for r in requests:
        for pt in [r["origin"], r["destination"]]:
            n_near = sum(
                haversine_km(pt["x"], pt["y"], s["x"], s["y"]) <= walk_threshold_km
                for s in stops
            )
            counts.append(n_near)
    return float(np.mean(counts)) if counts else float("nan")
```

> **Note from feature_list.md:** `bsa_flexibility = 1` means only 1 stop is accessible. This effectively means the passenger has no choice (they must use the closest stop). This is the degenerate case that should be noted when interpreting results — it is a structural property of the network, not of the demand.

---

### Feature 17 — `avg_station_degree`
**Definition:** Mean of `stop["degree"]` across all stops.

```python
def compute_avg_station_degree(stops):
    if not stops:
        return float("nan")
    degrees = [s.get("degree", 0) for s in stops]
    return float(np.mean(degrees))
```

**Fallback** if `degree` key is absent in stops: compute from `road_network.edges` using `networkx`:

```python
import networkx as nx

def build_graph(road_network):
    G = nx.DiGraph()
    for e in road_network["edges"]:
        G.add_edge(e["u"], e["v"], weight=e["travel_time_min"])
    return G

def compute_avg_station_degree_from_graph(stops, road_network):
    G = build_graph(road_network)
    stop_node_ids = [s["id"] for s in stops if s["id"] in G]
    if not stop_node_ids:
        return float("nan")
    degrees = [G.degree(n) for n in stop_node_ids]
    return float(np.mean(degrees))
```

---

### Feature 18 — `coverage_ratio`
**Definition:** Fraction of the service area within 500 m walking distance of at least one stop.

Approximate via a grid:
1. Build a grid of points over the bounding box of the instance (e.g., 50×50 grid).
2. For each grid point, check if any stop is within 500 m.
3. `coverage_ratio = (grid points with at least one nearby stop) / (total grid points)`.

```python
def compute_coverage_ratio(stops, requests, walk_threshold_km=0.5, grid_n=50):
    if not stops:
        return float("nan")
    all_pts = (
        [r["origin"] for r in requests] +
        [r["destination"] for r in requests]
    )
    xs = [p["x"] for p in all_pts]
    ys = [p["y"] for p in all_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    covered = 0
    total = grid_n * grid_n
    for i in range(grid_n):
        for j in range(grid_n):
            gx = x_min + (x_max - x_min) * i / (grid_n - 1)
            gy = y_min + (y_max - y_min) * j / (grid_n - 1)
            if any(haversine_km(gx, gy, s["x"], s["y"]) <= walk_threshold_km for s in stops):
                covered += 1
    return covered / total
```

---

### Features 19–21 — Dynamic features
For the **static** case (this study), these are all constant. Still compute them so the CSV is complete; they will simply have zero variance and will be dropped in feature selection.

```python
def compute_dynamic_features(requests):
    # For static instances, issue_time is absent or equal to 0 for all requests
    issue_times = [r.get("issue_time", 0) for r in requests]
    tw_starts = [r["tw_start"] for r in requests]
    n = len(requests)

    # Flux dynamism: std of issue times, normalized by planning horizon
    if max(issue_times) == min(issue_times):
        request_flux_dynamism = 0.0
    else:
        horizon = max(tw_starts) - min(issue_times)
        request_flux_dynamism = float(np.std(issue_times) / horizon) if horizon > 0 else 0.0

    # Static-to-dynamic ratio: fraction of requests with issue_time == 0
    static_to_dynamic_ratio = float(sum(1 for t in issue_times if t == 0) / n)

    # Average urgency: mean(tw_start - issue_time) in minutes
    urgencies = [s - t for s, t in zip(tw_starts, issue_times)]
    avg_request_urgency = float(np.mean(urgencies))

    return {
        "request_flux_dynamism": request_flux_dynamism,
        "static_to_dynamic_ratio": static_to_dynamic_ratio,
        "avg_request_urgency": avg_request_urgency,
    }
```

---

## 4. `compute_features()` — Expected Return Contract

The function must return a flat dictionary with **exactly these keys** (all numeric). `float("nan")` is acceptable for features that cannot be computed due to missing data in the instance — MATILDA will handle NaN columns during feature selection.

```python
{
    # Demand
    "n_requests":             int or float,
    "demand_density":         float,
    "spatial_cluster_coeff":  float,
    "od_distance_mean":       float,   # km
    "od_distance_std":        float,   # km
    # Time window
    "tw_width_mean":          float,   # minutes
    "tw_width_std":           float,   # minutes
    "tw_overlap_ratio":       float,   # [0, 1]
    "tight_tw_fraction":      float,   # [0, 1]
    "tw_tightness_factor":    float,   # [1.1, 2.5+]
    # Fleet
    "fleet_size":             int or float,
    "capacity_cv":            float,
    "fleet_utilization_ratio":float,
    "request_vehicle_ratio":  float,
    # Spatial / network
    "station_density":        float,   # stops/km²
    "bsa_flexibility":        float,   # mean candidate stops per endpoint
    "avg_station_degree":     float,
    "coverage_ratio":         float,   # [0, 1]
    # Dynamic (constant for static study)
    "request_flux_dynamism":  float,
    "static_to_dynamic_ratio":float,
    "avg_request_urgency":    float,   # minutes
}
```

**Do not add or remove keys without updating:**
1. `docs/feature_list.md` (canonical feature registry)
2. This document (Section 4)

---

## 5. Features CSV Format (input to MATILDA)

The file `data/results/features.csv` must conform to this format:

```
instance,n_requests,demand_density,spatial_cluster_coeff,...,avg_request_urgency
antwerp_001,45,1.8,0.72,...,240.0
antwerp_002,78,3.1,0.55,...,240.0
...
```

Rules:
- First column: `instance` — the instance filename stem (no extension), used as the join key with `perf_matrix.csv`
- Subsequent columns: one per feature, in any order, all numeric
- No string columns, no units in headers
- `float("nan")` → write as empty cell (pandas default when writing CSV with `NaN`)
- One row per instance; rows must match 1-to-1 with rows in `perf_matrix.csv`

Generate with:
```python
df.to_csv("data/results/features.csv", index=True, index_label="instance")
```

---

## 6. Performance Matrix CSV Format (input to MATILDA)

The file `data/results/perf_matrix.csv` must conform to this format:

```
instance,SA,IP,GR
antwerp_001,0.97,1.00,0.82
antwerp_002,0.88,0.95,0.71
...
```

Rules:
- First column: `instance` — must match `features.csv` exactly
- One column per algorithm: `SA` (simulated annealing), `IP` (integer programming), `GR` (greedy)
- Values are **normalized scores** in [0, 1]: `score(i, a) = (best_known(i) - urt(i,a)) / best_known(i)` — higher = better; best algorithm on instance i gets 1.0
- See `docs/experimental_protocol.md` for the normalization formula

---

## 7. How MATILDA Consumes These Files

MATILDA (https://github.com/andremun/InstanceSpace) is the University of Melbourne toolbox implementing the full ISA pipeline including PILOT projection and footprint analysis.

> **Corrected May 2026 (T14).** This section previously documented `pip install matilda` and an `InstanceSpace(features=..., performance=...).fit()` API. Both were wrong. The package name and the API below were verified against an actual install (`instancespace` 0.2.1).

### Python interface (preferred)

The correct package is **`instancespace`** — the Melbourne group's Python port. The PyPI package named `matilda` is an unrelated project and must not be installed for this.

**It cannot share an environment with REQreate.** `instancespace` 0.2.1 pins `numpy<2` and `pandas<3`; installing it alongside REQreate downgrades both and breaks osmnx. Use a dedicated virtualenv:

```bash
python -m venv .venv-isa
.venv-isa/Scripts/pip install instancespace     # pulls numpy 1.26.x, pandas 2.x
.venv-isa/Scripts/python src/isa_pipeline.py --method package
```

The package does **not** take features and performance frames directly. It takes a `Metadata` object built from a single wide table whose *column-name prefixes* carry the meaning:

| column | meaning |
|---|---|
| `Instances` | instance label (our `instance` stem) |
| `Source` | optional grouping label |
| `feature_<name>` | one column per feature |
| `algo_<name>` | one column per algorithm score |

```python
import pandas as pd
from instancespace import InstanceSpace, InstanceSpaceOptions
from instancespace.data.metadata import Metadata
from instancespace.data.options import PerformanceOptions

features = pd.read_csv("data/results/features.csv", index_col="instance")
performance = pd.read_csv("data/results/perf_matrix.csv", index_col="instance")
assert list(features.index) == list(performance.index), "Instance mismatch!"

frame = pd.DataFrame({"Instances": features.index.astype(str)})
for name in features.columns:
    frame[f"feature_{name}"] = features[name].to_numpy()
for name in performance.columns:
    frame[f"algo_{name}"] = performance[name].to_numpy()

metadata = Metadata.from_data_frame(frame)

# CRITICAL: the package minimises performance by default. Our perf_matrix
# scores are normalised so higher = better, so max_perf MUST be True or
# PRELIM marks every algorithm as bad and the run is meaningless.
options = InstanceSpaceOptions.default(
    parallel=None, auto=None, bound=None, norm=None, selvars=None,
    sifted=None, pilot=None, cloister=None, pythia=None, trace=None, outputs=None,
    perf=PerformanceOptions.default(max_perf=True, abs_perf=True, epsilon=0.80),
)

model = InstanceSpace(metadata, options).build()   # all 7 stages
projection = model.pilot.z          # ndarray (n, 2)
footprints = model.trace            # TraceOut: space, good, best, hard, summary
```

Stage order is PREPROCESSING → PRELIM → SIFTED → PILOT → PYTHIA → CLOISTER → TRACE. To stop after the projection, iterate `space.run_iter()` and break once `pilot_z` appears — note `run_until_stage(PilotStage)` stops *before* PILOT, not after it. `src/isa_pipeline.py` does this so it can skip PYTHIA's SVM, which is slow and fails on small datasets.

### MATILDA input requirements
- Features must be numeric (no NaN columns — drop or impute before passing)
- Features should have variance > 0 (constant features will cause PILOT to fail — drop them)
- Performance scores must be in [0, 1] — verify normalization before calling `build()`
- Set `PerformanceOptions.max_perf=True` for our higher-is-better scores
- Minimum recommended: ~100 instances for meaningful projection; target is 500+
- SIFTED keeps only features that correlate significantly with performance, and raises `NotEnoughFeatureError` if fewer than 2 survive. Too few instances or noisy performance data trips this — it is a data problem, not a configuration one.

### Feature pre-processing before MATILDA
```python
# Drop constant features (variance == 0)
features = features.loc[:, features.std() > 0]

# Drop features with >10% NaN
features = features.loc[:, features.isna().mean() < 0.1]

# Impute remaining NaN with column median
features = features.fillna(features.median())

# Drop dynamic features (constant in static study)
drop_dynamic = ["request_flux_dynamism", "static_to_dynamic_ratio", "avg_request_urgency"]
features = features.drop(columns=[c for c in drop_dynamic if c in features.columns])
```

---

## 8. Repository Conventions

| Convention | Rule |
|---|---|
| Instance files | `data/instances/<city>_<id>.json` e.g. `antwerp_001.json` |
| Feature CSV | `data/results/features.csv` — written row-by-row at generation time |
| Performance CSV | `data/results/perf_matrix.csv` |
| Instance space coordinates | `data/results/instance_space.csv` |
| Figures | `figures/footprint_<ALG>.pdf`, `figures/instance_space.pdf` |
| Python environment | `.venv/` at repo root; `pip install -r requirements.txt` |
| Code style | Standard Python; no type annotation required unless adding a new public function |

---

## 9. Open Questions / Decisions Pending

These need to be resolved before implementation is finalised. Check with Michell before assuming.

| # | Question | Impact |
|---|---|---|
| OQ1 | Do real REQreate instances have a `stops` list and `travel_time_matrix` in the JSON? Or are they separate files? | Features 10, 15–18 depend on this. | Answer: REQreate have these files in csv format. They can be changed if needed.
| OQ2 | Are `x,y` coordinates WGS84 (lon/lat) or projected (e.g., Dutch RD, Belgian Lambert)? | Affects Haversine vs Euclidean vs proper CRS transform. | Answer: lon,lat
| OQ3 | Does `service_area_km2` match the actual bounding box or is it a pre-defined administrative area? | Affects `demand_density`, `station_density`. | Answer: Unsure, depends on openstreetmaps
| OQ4 | What is the walking distance threshold for `bsa_flexibility` and `coverage_ratio`? | Default is 500 m; confirm with Michell. | Answer: if I remember correctly, this can be input by the user when generating instances, but usually used 10 min walk as default.
| OQ5 | Should `od_distance_mean` use straight-line (Haversine) or network driving distance? | Network is more accurate but requires the embedded graph; straight-line is the current fallback. | Answer: driving distance
| OQ6 | MATILDA Python vs MATLAB interface — which is being used? | The `requirements.txt` does not list `matilda` yet; confirm the correct package name. | Answer: Probably python. should be easier.

### T01 assumptions block

```python
# Assumptions for ISA-HODBRP static feature pipeline:
# - REQreate stores stops and travel-time inputs as CSV files.
# - Coordinates are WGS84 lon/lat.
# - OD distance features use driving distance as the primary source.
# - Walking threshold defaults to 10 minutes and stays configurable.
# - MATILDA is implemented through the Python interface.
# - service_area_km2 is derived from reachable geometry when available;
#   convert reachable nodes/edges into a polygon and calculate area in km2.
#   Use bounding-box area only as a documented last-resort fallback.
```

### Implementation note

`service_area_km2` should be computed from the reachable service geometry whenever the generator/network data exposes enough information. Prefer the existing polygon-building logic already used for partitioning or coverage-related operations, and calculate the area from that polygon in km2. If the reachable geometry is not available, fall back to the instance bounding box and mark the value as approximate.

---

## 10. Quick Smoke Test

After implementing `compute_features()` in the generator, verify it with a minimal instance dict in a Python REPL or quick script:

```python
from src.generate_instances import compute_features  # adjust import to actual location

minimal = {
    "service_area_km2": 100.0,
    "requests": [
        {"origin": {"x": 4.40, "y": 51.21}, "destination": {"x": 4.42, "y": 51.22},
         "tw_start": 480, "tw_end": 510, "demand": 2},
        {"origin": {"x": 4.41, "y": 51.20}, "destination": {"x": 4.43, "y": 51.23},
         "tw_start": 490, "tw_end": 540, "demand": 1},
    ],
    "vehicles": [
        {"capacity": 8,  "depot": {"x": 4.39, "y": 51.20}},
        {"capacity": 12, "depot": {"x": 4.45, "y": 51.24}},
    ],
    "stops": [
        {"id": "s1", "x": 4.401, "y": 51.211, "degree": 3},
        {"id": "s2", "x": 4.420, "y": 51.220, "degree": 2},
    ],
}

feats = compute_features(minimal)
assert len(feats) == 21, f"Expected 21 features, got {len(feats)}"
assert all(isinstance(v, (int, float)) for v in feats.values()), "Non-numeric feature"
print(feats)
```

All 21 feature keys must be present and numeric. `float("nan")` is acceptable for features that depend on absent optional fields (e.g., `stops`).
