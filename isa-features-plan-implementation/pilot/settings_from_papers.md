# Instance settings from the two H-ODBRP papers, for the ISA instance set

Sources:
- **[H]** Queiroz & Sörensen, *Evaluating a Simulated Annealing Metaheuristic for the Heterogeneous
  On-Demand Bus Routing Problem with a novel experimental setup* (R1, 2026). Chicago, 5280 instances,
  1 h planning period, dynamic.
- **[P]** Queiroz & Sörensen, *Parallel Simulated Annealing metaheuristic for the Heterogeneous
  On-Demand Bus Routing Problem*, Parallel Computing 129 (2026) 103212. Four demand scenarios,
  5000–20 000 requests.

"Pilot" = the T14 pilot (Copenhagen, static, 72 instances). "ISA check" = what the ISA study can
confirm or contradict.

| # | Setting | Used in the papers | What the papers found (results, managerial insights) | Pilot | Suggested ISA range | ISA check |
|---|---|---|---|---|---|---|
| 1 | Number of requests | [H] 300–3000 per hour, step 300; exact model 5–14. [P] 5000, 10 000, 15 000, 20 000 | [H] Rejections fall with size (F1: 8.8% at 300 → 0.4% at 3000), URT falls (3123 → 2682 s). SA within 10% of the IP for most sizes 5–14 (2.8–23.6%). [P] Larger instances gain most from parallel SA (≈16% vs ≈8% at 5000) → sequential SA is weaker on large instances | 200, 500, 1000 in 2 h | 100–3000 per hour (state as demand per hour, not raw count) | Does SA's advantage over GR, and the matheuristic gain, change with size? [P] predicts SA leaves more room on large instances |
| 2 | Planning period / scenario | [H] 1 h. [P] 2–7 h per scenario: concert 17–19 h / 22:30–23:30, festival 11–18 h, nightlife 21–24 h / 3–6 h, commuting 6:30–8:30 / 17–19 h | [P] Commuting and nightlife (requests spread over the city) parallelise best (speed-up 7× / 5.6×, improvement up to 24–30%); concert and festival (converging on one venue) less (3–3.5×, ≈10%) | 6:30–8:30 commuting peak | Keep 1–2 h windows; add the scenario shape as a setting (row 4) | Do "converging" instances form their own region with different algorithm behaviour? |
| 3 | Dynamism | [H] 0–100%, step 10 | [H] Only 0% differs: lower URT (≈2120 vs ≈2255 s), less deadheading, higher occupancy (1.35 vs 1.25). 10–100% no pattern | none (static) | Not used: the ISA study is static | — |
| 4 | Temporal demand shape | [H] uniform within the hour. [P] commuting: 40% uniform 6:30–8:30 + 60% normal 7–8 h; concert: normal before, gamma after; festival: 20% uniform + two normal peaks; nightlife: normal | [P] see row 2 | normal around 7:30, sd 15 / 30 min, no uniform part | Peak sd 10–60 min **and** uniform share 0–60% (the [P] commuting mix is sd ≈15 min + 40% uniform) | Is peak concentration a performance driver, or absorbed by time windows? |
| 5 | Urgency (reaction time = latest pickup − request time) | [H] mean {5, 10, 15, 30} min, sd {0, 5, 10} min | [H] 30 min mean: lower URT (≈2120 vs 2250–2350 s), +0.4% accepted, ≈10 min less deadheading per vehicle, but less sharing (lower shared time and occupancy). 5–15 min: no clear pattern. Larger sd → more accepted. Insight: encourage early booking | mean 10 / 30 min, sd 2 / 5 min | mean 5–40 min, sd 0–10 min (continuous) | Is there a threshold between 15 and 30 min, as [H] suggests, or a gradual effect? This is the axis with the gap in the pilot |
| 6 | Arrival slack (latest arrival − earliest arrival) | not varied (generator default) | — | 15 / 45 min | 5–60 min | New: the papers have no result. Check whether it matters separately from urgency |
| 7 | Geographic dispersion | [H] 4 intervals: [1000,1100], [2100,2200], [3000,3200], [4300,4400] (REQreate measure) | [H] Deadheading rises with dispersion (≈3510 → 3960 s), shared time (≈900 → 4500 s), occupancy (0.65 → 1.65), full-capacity events (≈100 → 275). Acceptance not monotonic (best at the 3rd interval). Insight: low dispersion → push sharing; high dispersion → prefer nearby vehicles | not controlled (uniform over the city) | Target dispersion continuously over the [H] span (≈1000–4400) | Do our spatial features (od_distance_mean, spatial_cluster_coeff) reproduce the [H] trend? |
| 8 | Fleet size (% of requests) | [H] unlimited-fleet run used 21% (300 req) → 10% (3000 req); fleets F1 15%, F2 12%, F3 10%; final FF 20%. [P] fleet fixed per instance | [H] Smaller fleets → more rejections and longer rides (F3 10%: 22% rejected at 300 req). Even 15% too small for small instances → 20%. Fleet needed per request falls with size | 10 / 20 / 30% | 5–40%, continuous | Pilot: GR is only competitive at 10% + tight windows. Does that corner match [H]'s "fleet too small" region? |
| 9 | Fleet composition | [H] unlimited run: types 4, 8, 12, 16, 25, >25 seats; fleets F1 60/40/0, F2 40/45/15, F3 30/45/25 (% of 4/8/12 seats); final 60/40/0 | [H] >12 seats almost never used (<5%); 12-seat buses average <4 on board; 4+8 mix (F1) has the fewest rejections and lowest URT. Insight: large buses not needed | fixed 60% 4-seat / 40% 8-seat | 8-seat share 0–100%, optional 12-seat share 0–25% | Does composition matter at all once fleet size is known (capacity_cv feature)? [H] says 12-seat buses add little |
| 10 | Spatial demand concentration | [H] 70% of requests in one area, 30% in another (fleet-deployment test) | [H] Sending large buses to the busy area gives slightly higher occupancy but higher URT; random choice better for URT | none | Optional: share of demand in a hotspot 50–90% | Hotspot instances vs uniform ones in the space |
| 11 | Depots | [H] several depots (vehicles return to their own) | — | 1 depot (station 423) | 1–4 depots, later | — |
| 12 | Walking / candidate stops | [H] max walking time u_r (generator default); cites Melis & Sörensen: no stop choice → +24% URT | — (cited result only) | 550–600 s | 300–900 s → number of candidate stops | bsa_flexibility feature: does more stop choice widen SA's lead? |
| 13 | City / network | [H] Chicago. [P] OpenStreetMap city (scenarios as in Muelas et al., San Francisco) | [H] future work: other network layouts | Copenhagen | Copenhagen + 1 more city | Do cities separate in the space? |

## Algorithm results to compare against

| Result | Papers | ISA pilot so far |
|---|---|---|
| SA vs exact model | [H] SA 2.8–23.6% above the IP optimum (5–14 requests, IP proves ≤14) | v2 proves 5 requests optimal in 2.5 s; SA 4.6% above. 10 requests: SA ≥20% above v2's best (v2 not proven) |
| SA run time | [H] <4 s for 5–14 requests | SA 15.5 s mean, 73 s max for 200–1000 requests |
| Room left by sequential SA | [P] parallel SA improves 4–30%, more on large instances | Matheuristic test pending |
