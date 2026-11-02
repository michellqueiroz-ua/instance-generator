# ISA-HODBRP study: documentation index

Working documentation of the instance-space analysis (ISA) study for the HODBRP
(heterogeneous on-demand bus routing with request-dependent stops). Written during the
work so that it can later be turned into a LaTeX document and slides.

Status of this set: **draft, updated 2026-10-29** (branch `isa-merge-041`, pushed to `isa-features`).
Anything marked **OPEN** is unresolved and needs a decision or a result. Anything marked
**PROPOSED** is a recommendation that the author has not yet decided.

| File | What it holds |
|---|---|
| [01_decisions.md](01_decisions.md) | Every decision taken so far, with the reason and who took it |
| [02_tests_and_results.md](02_tests_and_results.md) | Tests run, numbers measured, what each showed |
| [03_recommended_settings.md](03_recommended_settings.md) | Settings recommended for the full instance set (PROPOSED) |
| [04_generator_changes.md](04_generator_changes.md) | Changes to the instance generator (REQreate) compared with the original, and why |
| [05_hpc_and_reproduction.md](05_hpc_and_reproduction.md) | How the set is produced on the DTU HPC, resource use, commands |
| [06_open_items.md](06_open_items.md) | Unresolved questions and pending work |

Related files that already existed:

- `../feature_list.md`: the instance features (the ISA axes).
- `../pilot/settings_from_papers.md`: settings taken from the two earlier papers, with the paper results.
- `../isa_set/design.json`: the ranges used to draw the full instance set.

## One-paragraph summary of the study design

Instances are generated on the whole Capital Region of Denmark (without Bornholm), for a 2-hour
planning period in the low-demand part of the day (start between 12:00 and 19:30). Each
generated instance is paired with several fleets. The algorithms to compare are a simulated
annealing heuristic (SA), a greedy constructive heuristic (GR) and, for validation on very small
instances only, an exact Gurobi model. The performance measure is total passenger ride time
(plus a penalty for requests handed to a third party). Instances are described by features
(`../feature_list.md`) and projected into an instance space. The set is grown in Latin hypercube
batches until the instance space stops growing.
