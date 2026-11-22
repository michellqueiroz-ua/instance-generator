# 08. Demand pattern of batch 2 (plan, 2026-11-10)

Decisions D29-D38 are in `01_decisions.md`. This file is the plan and the checks.

## What batch 1 had

- Origins uniform over the region polygon (`passenger_requests.py`, `_get_random_coord(network.polygon)`).
- Destinations on a ring of 1.7-20 km around the origin (both `min_dtt` and `max_dtt` exist, so the generator
  draws a distance in the ring).
- A bus stop within 550-600 s walking of both ends; the two ends must not share a stop.
- No link between demand and where people live or go. Inner Copenhagen (served by metro and S-train) got
  the same density as farmland.

## What batch 2 changes

| Setting | Batch 1 | Batch 2 |
|---|---|---|
| Origin and destination | uniform, destination on a ring | zone weights from OpenStreetMap counts (D31) |
| `inbound_share` | none | [0, 1], drawn |
| `concentration` | none | [0, 2], drawn |
| `start_hour` | [12, 19.5], drawn | 14.0, fixed (D32) |
| `max_dtt_s` | 3600 | 3600 (D33, back to the batch 1 value) |
| `period_hours` | 2 | 4 (D36) |
| `requests_per_hour` | 50-500, drawn | 1250 fixed = 5000 per instance (D35) |
| central municipalities | no rule | excluded (D37) |
| `peak_sd_min` | [10, 60] | [20, 120] |
| `concentration` | none | [0.1, 1.5] |
| fleet % of requests | [5, 40] | [10, 40], provisional (fleets can be redrawn) |
| stop search of the generator | offsets that miss north/south zones | fixed (D38) |
| speed, walking, urgency, slack | | unchanged (D34: speed becomes a factor in a later batch) |

Weights per zone (`make_batch.od_weights`): residential share r and attraction share a of the zone, each raised
to `concentration` and normalised. Origin weight = p r + (1 - p) a, destination weight = p a + (1 - p) r, with
p = `inbound_share`. `concentration` 0 is uniform over the zones.

## Steps and checks

1. `hpc/zone_weights.sh` (one job, 50 Overpass queries, restartable). Check: the log ends with the zone count,
   residential and attraction totals, and the zones without any attraction; no zone has weight 0 in both columns
   that also has stops.
2. `make_batch.py --batch 2 --n 100` (needs step 1). Check: `design/batch_02.csv` has `inbound_share`,
   `concentration`, `scenario`; `start_hour` is 14.0 everywhere.
3. Smoke test (done 2026-11-10, results in D40): three instances (batches 90-92) made with `make_batch.py --set`, 5000 requests
   (10000 for one) and inbound_share / concentration 1 / 1.5, 0 / 0.1, 0.5 / 0.8. Check: origins/destinations plot, "failed after" lines in the logs (a request
   returned incomplete after 100 attempts), generation time per request, `direct_travel_time` distribution.
4. Full batch 2 with the usual `generate_batch.sh`; incomplete requests are found by `regenerate_incomplete.sh`.

## Risks (from the review)

- A zone with many homes but no stop within walking range is rejected by `len(stops_orgn) > 0`; with high
  concentration this is rare, with low concentration it is the same rejection as in batch 1.
- Zones are drawn independently for origin and destination, so home-to-home and attraction-to-attraction trips
  occur when `inbound_share` is between 0 and 1.
- `stops()` searches neighbouring zones at offsets +-1 and +-16..24 but never +-25 (batch 1 too), so stops
  just across a north-south zone border can be missed (to check).
- The attraction count mixes shops with tourism and leisure; the weights can be changed later without a new
  download (the counts are in `zone_weights.csv`).
- Crossing midnight needs the solver checked for 86400 (OPEN-S6).
