You are writing a LaTeX report on the instance-space analysis (ISA) study for the heterogeneous on-demand bus routing problem (H-ODBRP). The author will use it for presentations to a supervisor and colleagues. Work inside the repository checkout, folder `isa-features-plan-implementation/`.

## Sources (read all of them first; they are the only facts you may use)

1. `docs/README.md`, `docs/01_decisions.md`, `docs/02_tests_and_results.md`, `docs/03_recommended_settings.md`, `docs/04_generator_changes.md`, `docs/05_hpc_and_reproduction.md`, `docs/06_open_items.md`
2. `feature_list.md` (the instance features, i.e. the axes of the instance space)
3. `pilot/settings_from_papers.md` (settings and results of the author's two earlier papers)
4. `isa_set/design.json` (ranges of the full instance set)
5. Pilot data, for tables and figures: `pilot/work/results/` (`features.csv`, `perf_matrix.csv`, `instance_space.csv`, `footprint_summary.csv`, `solver_runs.csv`, `raw_results.csv`, `figures/`), and `pilot/work/v2_tests/`

## Output

A folder `docs/latex/` with `main.tex`, one `.tex` file per section (`\input`), `references.bib`, a `figures/` folder (PDF figures made with a script `make_figures.py` that reads the CSVs, so they can be regenerated), and a `Makefile` or a one-line build command. Use `article` class, `booktabs` tables, `siunitx` for numbers, `hyperref`. Compile it (`latexmk -pdf`) and fix all errors and overfull boxes before you finish. Running Python for figures and LaTeX locally is fine. Do **not** run solvers, generation or anything heavy on this computer.

## Structure

1. Summary (half a page): what the study is, what was done, the main preliminary findings, what is still open.
2. Study design: network, planning period, instance parameters and ranges, fleets, performance measure, algorithms. One table of settings with a column "paper range" and a column "status" (decided / proposed / open).
3. What was implemented: generator changes against the original (table: change, file, why, effect), solver changes, exact model v2 and its checker, matheuristic, HPC tooling.
4. Tests and preliminary results: pilot (instance space, SA vs GR), model checker (v1 vs v2), exact model on small instances, network build, profile of the generation, resource use.
5. Decisions log (table from `01_decisions.md`, shortened).
6. Open questions and next steps: one table, with ID, item, blocking or not. Copy every item marked OPEN in the docs; do not drop any.
7. Reproduction: the main HPC commands (short, in a `listings` or `verbatim` block).

## Figures and tables to make (from real data only)

- Instance space: scatter of the pilot instances in the 2-D projection (`instance_space.csv`), coloured by SA/GR performance ratio (`perf_matrix.csv`). Mark the gap along the time-window axis described in the docs.
- Distinct values of `od_distance_mean` before and after the seed fix (6 vs 24). Use the data if both versions exist under `pilot/work/` (`pilot_v1`, `pilot_v2_old`, `results`); if you cannot reproduce the before value from data, state the number from the docs in text and do not draw a fake bar.
- SA vs GR: ratio per fleet size and time-window setting (grouped bars or a heat map), supporting "SA better on 70 of 72".
- Exact model v2: table of 5, 10, 15 requests with the objective, bound, gap, time. Use the numbers in `docs/02_tests_and_results.md`; state which are from the first run and which from the rerun.
- Profile of generation: horizontal bar chart of the cumulative time per phase from the numbers in `docs/02_tests_and_results.md` (nearest-node lookups 117.8 s of 154.3 s for requests; imports about 60 s; request plot about 27 s). Add the "after the fix" bar only if the docs contain a measured value.
- Resource table: phase, cores, memory reserved, memory used, CPU use, run time (from `docs/05_hpc_and_reproduction.md`).

## Rules

- Do not invent numbers, results, citations or dates. If a value is not in the sources, write it as an open item, do not estimate it. Mark every OPEN or PROPOSED item in the text with a visible label (for example a small `[OPEN]` tag in bold) so the author can find them.
- Where a source says "reported by session" (no raw log), say so in a table footnote.
- Cite the two papers in `references.bib`: Queiroz and Sorensen, "Evaluating a Simulated Annealing Metaheuristic for the Heterogeneous On-Demand Bus Routing Problem with a novel experimental setup" (R1, 2026), and Queiroz and Sorensen, "Parallel Simulated Annealing metaheuristic for the Heterogeneous On-Demand Bus Routing Problem", Parallel Computing 129 (2026) 103212. Leave other references out unless the docs name them.
- Keep the language plain and short. Tables over prose where possible. Figures need captions that say what to see and which file the data came from.
- At the end, print a list of: every figure with its data file, every number you could not verify, and anything in the sources that contradicts something else.
