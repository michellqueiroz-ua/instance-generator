# Licensing to-do before merging to master

The MATILDA step (T14, `src/isa_pipeline.py`) uses
[`instancespace`](https://pypi.org/project/instancespace/) 0.2.1, which is
licensed **GPL-3.0**. REQreate is MIT.

Before this branch is merged, the main README must say:

- The ISA / MATILDA step is **optional** and not part of the core install.
  Generating instances and writing `features.csv` (T04–T13) does not need it.
- It requires `instancespace` (GPL-3.0, Python 3.12 only), installed
  separately: `pip install "instancespace==0.2.1"`.
- Users who distribute software combining REQreate with `instancespace`
  must follow the GPL-3.0 terms for that combination.

Keep `instancespace` out of `dependencies` in `pyproject.toml`. If it is ever
added, use an optional extra (e.g. `isa`), never a core dependency.
