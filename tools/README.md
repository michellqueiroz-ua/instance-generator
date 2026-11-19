# tools/

Small checks we run after every HPC step. Standard library only. Tests: `python -m pytest -q tools` from the repository root.

| Tool | What it does |
|---|---|
| `check_design.py` | Checks a generated batch against `design.json`: instance ranges, Latin hypercube, fixed settings, fleet count, ranges and strata, walking time in the configs. `python tools/check_design.py --design-json isa-features-plan-implementation/isa_set/design.json --batch-dir <work folder> --batch N [--n 100]`. Exit code 1 on any FAIL. |
| `lsf_summary.py` | Reads LSF logs (files or folders) and prints status, CPU time, Run time, utilisation and Max Memory per job, then a PASS or FAIL per kind (solver, features) against the pass table of the probe. `python tools/lsf_summary.py <logs> [--kind solver\|features\|generation\|other] [--rows-per-block 10]`. Exit code 1 on any FAIL. |
