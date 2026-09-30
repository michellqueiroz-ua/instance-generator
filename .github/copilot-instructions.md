# Copilot instructions

REQreate generates instances for on-demand transport problems, including the
Dial-a-Ride Problem (DARP) and On-demand Bus Routing Problem (ODBRP), from
OpenStreetMap networks. The main entry point is
`REQreate.input_json.input_json`; the CLI is `REQreate/cli.py`, and the
Streamlit UI is `REQreate/webapp/app.py`.

Python 3.11 or newer is required. Project dependencies are declared in
`pyproject.toml`.

Tests use `unittest` under `tests/`. Run them with
`python -m pytest tests` or `python -m unittest discover tests`.

Network access to OpenStreetMap/Overpass is not available in the agent
environment. Tests must build small synthetic `networkx.MultiDiGraph` graphs
with `x` and `y` node attributes, `crs='epsg:4326'`, and `length` and
`travel_time` edge attributes instead of downloading networks.

Leave `REQreate/REQreate*.py`, `*.slurm` files, and the `trip_patterns_*.py`
research scripts alone unless an issue names them. Keep changes minimal and
match the surrounding code style.
