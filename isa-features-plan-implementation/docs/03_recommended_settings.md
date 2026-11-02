# 03. Recommended settings so far (PROPOSED)

These are the values in `isa_set/design.json` today. They are a recommendation, not a final
choice: the author decides. Comparison with the two earlier papers is in
`../pilot/settings_from_papers.md` ([H] = Queiroz and Sorensen, R1 2026; [P] = Parallel Computing 129,
2026).

## Instance settings (drawn per instance, Latin hypercube)

| Setting | Range | Paper range | Why this range | Status |
|---|---|---|---|---|
| Network | Capital Region of Denmark without Bornholm | Chicago [H]; city scenarios [P] | Includes suburbs (author's wish) | PROPOSED |
| Planning period | 2 h | 1 h [H]; 2-7 h [P] | Long enough for time-window overlap; 2 h is the pilot value | PROPOSED |
| Start of period | 12:00-19:30 | commuting and event scenarios [P] | Low-demand windows, not the commuting peak (D6). 19:30 keeps latest arrivals before midnight. | PROPOSED |
| Requests per hour | 50-500 (100-1000 per instance) | 300-3000 [H]; 5000-20000 per scenario [P] | Pilot used 200-1000 per 2 h. Smaller than the papers because the exact comparison is not possible above about 15 requests and generation cost grows with size. | PROPOSED, see OPEN-S1 |
| Peak spread (sd of the demand curve) | 10-60 min | sd about 15 min plus 40% uniform [P] | A large sd approaches uniform demand | PROPOSED |
| Urgency mean (reaction time) | 5-40 min | 5, 10, 15, 30 [H] | [H] found a difference only at 30 min; the pilot had two levels, which left a gap | PROPOSED |
| Urgency sd | 1-10 min | 0, 5, 10 [H] | 0 is excluded because the reaction time is a truncated normal with a 60 s minimum | PROPOSED |
| Arrival slack | 5-60 min | not varied | New axis; no paper result to compare with | PROPOSED |
| Max walking time | 550-600 s | generator default | Fixed (almost); varying it would change stop choice | PROPOSED, see OPEN-S2 |
| Walk speed | 1.38889 m/s | same | Fixed | PROPOSED |
| Direct trip time | 300-3600 s | same lower bound | Upper bound raised for suburban trips | PROPOSED |
| Vehicle speed | 20 km/h | same | Fixed | PROPOSED |

## Fleet settings (3 draws per instance, second Latin hypercube)

| Setting | Range | Paper range | Status |
|---|---|---|---|
| Fleet size | 5-40% of requests | 10-21% [H] | PROPOSED. The pilot found GR competitive only at 10% with tight windows. |
| Share of 8-seat vehicles | 0-100% (rest 4-seat) | 4+8 seat mix best; 12-seat rarely used [H] | PROPOSED |
| Depot | one station no request uses, chosen after generation | several depots [H] | D10 |

## Settings from the papers that are NOT in the design

| Setting | Why left out | What it costs |
|---|---|---|
| Geographic dispersion | The generator only measures it; it cannot be set (D10) | The ISA shows it only through the spatial features (`od_distance_mean`, `spatial_cluster_coeff`) |
| Hotspot / demand concentration | Same | Same |
| Uniform share of the demand curve | Not implemented as a setting | Peak spread covers part of it |
| Dynamism | The study is static | None for this study |
| Several depots | Chosen after generation | OPEN-S3 |
| More than one city | One network for the first study | Cities cannot be compared |

## Recommendation on the algorithm side

1. Use SA and GR as the performance matrix now (D2).
2. Use the exact model v2 only to validate SA and GR on 5-10 requests. Report SA's gap (4.6% above the optimum on 5 requests; at least 20% on 10 requests, where v2 is not proven).
3. Third algorithm: a variant of the SA (other neighbourhood configuration) or a max-ALNS variant (D3). The exact variant is OPEN-A1. The matheuristic is used to show where SA is weak, not as an ISA algorithm.

## Open settings questions

- **S1 (decided for now, D14a)** 50-500 requests per hour. The papers go to 3000 per hour; the
  upper end can be raised later, now that generation is cheaper (D18).
- **OPEN-S2** Should walking time vary (the number of candidate stops changes the problem)?
- **OPEN-S3** Do several depots matter for the performance differences?
- **OPEN-S4** Number of instances: batches of 100 until the space stops growing; the cap is not set.
