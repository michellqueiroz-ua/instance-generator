# Temporary Windows installer

`windows/install_and_run_reqreate.bat` installs REQreate and opens the desktop
app with a double-click. It is a **temporary** convenience for testing, not the
supported install method. The supported way is `pip install "reqreate[app,desktop]"`
(see the main [README](../README.md)).

> **Only run it if you trust the developer.** It downloads code from GitHub
> and PyPI and runs it on your computer. Read the `.bat` file first; it is
> short and plain text. Windows may show a SmartScreen warning because the
> file is not signed.

## What it does

First run (asks you to type `Y` before doing anything):

1. Finds Python 3.12, 3.11 or 3.13 on your computer. It does **not** install
   Python: if none is found it tells you to get Python 3.12 from
   [python.org](https://www.python.org/downloads/).
2. Creates a private Python environment in
   `%LOCALAPPDATA%\reqreate\venv-<version>`, separate from any other Python
   or conda setup you have.
3. Installs REQreate from the matching GitHub release
   (`https://github.com/michellqueiroz-ua/instance-generator/archive/refs/tags/v<version>.zip`)
   and its dependencies from PyPI: osmnx, networkx, geopandas, shapely, pandas,
   numpy, scipy, scikit-learn, matplotlib, geopy, jsonschema, streamlit, folium,
   streamlit-folium, plotly and pywebview, plus what those need. About 1 GB;
   several minutes.
4. Opens the desktop app.

Later runs skip straight to step 4.

Instances are saved in `%USERPROFILE%\Documents\REQreate`. OpenStreetMap
downloads are cached in `%LOCALAPPDATA%\reqreate\osm-cache`.

## Uninstall

Delete `%LOCALAPPDATA%\reqreate` (the environment and download cache).
Your instances in `Documents\REQreate` are kept unless you delete them too.

## Overrides (for testing)

| Variable | Default |
|---|---|
| `REQREATE_SOURCE` | the GitHub release zip for this version |
| `REQREATE_VENV` | `%LOCALAPPDATA%\reqreate\venv-<version>` |
| `REQREATE_WORKDIR` | `%USERPROFILE%\Documents\REQreate` |

Pass `--install-only` to set up without opening the app.

Data downloaded through REQreate comes from OpenStreetMap, © OpenStreetMap
contributors, ODbL. See [NOTICE](../NOTICE).
