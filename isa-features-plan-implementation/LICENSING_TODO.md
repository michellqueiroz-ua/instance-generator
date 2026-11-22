# Licensing to-do before merging to master

The MATILDA step (T14, `src/isa_pipeline.py`) uses
[`instancespace`](https://pypi.org/project/instancespace/) (CI still pins 0.2.1), which is
now licensed **PolyForm Noncommercial 1.0.0** (0.2.1 was GPL-3.0; the chapter runs used 0.4.1). REQreate is MIT.

Before this branch is merged, the main README must say:

- The ISA / MATILDA step is **optional** and not part of the core install.
  Generating instances and writing `features.csv` (T04–T13) does not need it.
- It requires `instancespace` (PolyForm Noncommercial 1.0.0, Python 3.12 only), installed
  separately: `pip install "instancespace==0.4.1"`.
- Users who distribute software combining REQreate with `instancespace`
  must follow the instancespace licence terms (no commercial use).

Keep `instancespace` out of `dependencies` in `pyproject.toml`. If it is ever
added, use an optional extra (e.g. `isa`), never a core dependency.
