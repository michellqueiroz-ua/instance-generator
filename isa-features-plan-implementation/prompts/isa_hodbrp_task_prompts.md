# ISA-HODBRP Task Prompts (ADHD Sprint Format)

How to use each prompt:
1. Set a timer for the suggested sprint length.
2. Execute only the listed scope.
3. Stop at the done criteria, write 3-line progress note, then take a short break.

---

## ISA-HODBRP-T01
Suggested sprint: 25-30 min

Prompt:
You are implementing task ISA-HODBRP-T01.
Goal: lock assumptions from answered OQs for the static H-ODBRP feature pipeline.
Decisions to encode:
- Stops and travel-time inputs are CSV-based in REQreate outputs.
- Coordinates are lon/lat (WGS84).
- od_distance_mean and od_distance_std should use driving distance as primary source.
- Walking threshold should default to 10 minutes and remain configurable.
- MATILDA interface should use Python.
- service_area_km2 should be derived from reachable geometry when available by converting reachable nodes/edges into a polygon and measuring area in km2, with bounding-box fallback only if geometry is unavailable.
Output:
- One compact assumptions block in code comments and one in implementation notes.
Done criteria:
- No unresolved assumption remains for feature coding.
- Fallback policy for service_area_km2 is explicit.

---

## ISA-HODBRP-T02
Suggested sprint: 35-45 min

Prompt:
You are implementing task ISA-HODBRP-T02.
Build compute_features(instance: dict) skeleton with exact 21 output keys and numeric values.
Add helper functions for:
- safe mean/std
- NaN-safe numeric casting
- missing-field guards
- basic schema checks
Do not implement full feature formulas yet; this is scaffold-only.
Done criteria:
- compute_features returns all required keys.
- Output types are numeric or NaN.
- Minimal smoke call works.

---

## ISA-HODBRP-T03
Suggested sprint: 45-60 min

Prompt:
You are implementing task ISA-HODBRP-T03.
Create CSV loaders for REQreate files needed by features:
- stops/stations CSV
- travel-time CSV (.tt.drive.csv and travel_time_* variants)
Design stable return structures for downstream feature functions.
Validate loader behavior against Maastricht network files.
Done criteria:
- Loader functions return parsed structures without manual edits.
- Basic validation errors are explicit and actionable.

---

## ISA-HODBRP-T04
Suggested sprint: 30-40 min

Prompt:
You are implementing task ISA-HODBRP-T04.
Implement features 1-3:
- n_requests
- demand_density
- spatial_cluster_coeff
Use fallback when service_area_km2 is missing or unreliable.
Log fallback usage in a concise way.
Done criteria:
- All 3 metrics compute on toy and real sample data.
- Fallback path is tested once.

---

## ISA-HODBRP-T05
Suggested sprint: 40-50 min

Prompt:
You are implementing task ISA-HODBRP-T05.
Implement features 4-5 with driving distance policy:
- od_distance_mean
- od_distance_std
Primary source must be driving distance matrix/network data.
Only use fallback if driving data is unavailable.
Done criteria:
- Features use driving distance by default.
- Unit sanity checks pass on sample instances.

---

## ISA-HODBRP-T06
Suggested sprint: 30-40 min

Prompt:
You are implementing task ISA-HODBRP-T06.
Implement features 6-9:
- tw_width_mean
- tw_width_std
- tw_overlap_ratio
- tight_tw_fraction
Include robust handling for empty/malformed request arrays.
Done criteria:
- Metrics match formulas.
- Edge cases do not crash and return valid numeric/NaN outputs.

---

## ISA-HODBRP-T07
Suggested sprint: 40-55 min

Prompt:
You are implementing task ISA-HODBRP-T07.
Implement feature 10 tw_tightness_factor:
- mean(tw_width / min_trip_time)
- min_trip_time should come from driving travel-time source
- skip invalid/non-positive travel-time rows safely
- only fallback when primary travel-time source missing
Done criteria:
- Formula is correct.
- Missing data handling is explicit.
- One example run returns plausible range.

---

## ISA-HODBRP-T08
Suggested sprint: 30-35 min

Prompt:
You are implementing task ISA-HODBRP-T08.
Implement fleet features 11-14:
- fleet_size
- capacity_cv
- fleet_utilization_ratio
- request_vehicle_ratio
Handle zero division and empty inputs safely.
Done criteria:
- All outputs numeric/NaN as appropriate.
- Degenerate cases are covered by quick checks.

---

## ISA-HODBRP-T09
Suggested sprint: 45-60 min

Prompt:
You are implementing task ISA-HODBRP-T09.
Implement features 15-17:
- station_density
- bsa_flexibility
- avg_station_degree
Walking rule:
- default threshold = 10 minutes walk
- convert to distance with walk speed input where available
- use documented fallback speed if missing
Done criteria:
- All 3 features compute from CSV inputs.
- bsa_flexibility interpretation for value=1 is noted.

---

## ISA-HODBRP-T10
Suggested sprint: 45-60 min

Prompt:
You are implementing task ISA-HODBRP-T10.
Implement feature 18 coverage_ratio using grid approximation.
Requirements:
- configurable grid_n
- walking threshold based on 10-minute default rule
- avoid excessive runtime
Done criteria:
- coverage_ratio computed deterministically.
- runtime acceptable on sample network.

---

## ISA-HODBRP-T11
Suggested sprint: 20-30 min

Prompt:
You are implementing task ISA-HODBRP-T11.
Implement static-study placeholders for features 19-21:
- request_flux_dynamism
- static_to_dynamic_ratio
- avg_request_urgency
Keep behavior future-compatible for dynamic runs.
Done criteria:
- Features always present.
- Values are numeric and stable for static data.

---

## ISA-HODBRP-T12
Suggested sprint: 40-55 min

Prompt:
You are implementing task ISA-HODBRP-T12.
Integrate compute_features into generation pipeline.
Write one row per instance to data/results/features.csv with instance stem key.
Add protections:
- no duplicate columns
- no malformed rows
- stable append behavior
Done criteria:
- Repeated generation appends valid rows.
- Schema stays consistent.

---

## ISA-HODBRP-T13
Suggested sprint: 30-40 min

Prompt:
You are implementing task ISA-HODBRP-T13.
Build data/results/perf_matrix.csv export with SA/IP/GR normalized scores in [0,1].
Add checks:
- index alignment with features.csv
- score range validity
Done criteria:
- perf_matrix.csv created and passes checks.

---

## ISA-HODBRP-T14
Suggested sprint: 50-75 min

Prompt:
You are implementing task ISA-HODBRP-T14.
Update src/isa_pipeline.py for MATILDA (Python):
- load features.csv and perf_matrix.csv
- align indices
- drop constant columns
- handle NaN (drop high-NaN, median-impute remainder)
- drop static dynamic columns if needed
- fit InstanceSpace and export instance_space.csv and footprint figures
Done criteria:
- End-to-end run completes.
- Output files are generated in expected locations.

---

## ISA-HODBRP-T15
Suggested sprint: 35-50 min

Prompt:
You are implementing task ISA-HODBRP-T15.
Run QA and prepare supervisor-ready summary.
Include:
- implemented coverage for features 1-21
- deferred scope note for features 22-23
- assumptions and caveats
- reproducibility notes and next steps
Done criteria:
- Smoke tests pass.
- Review summary is concise and presentation-ready.
