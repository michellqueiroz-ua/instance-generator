# REQreate 🚌

REQreate is a tool to generate instances for on-demand transportation problems. Such problems consist of optimizing the routes of vehicles according to passengers' demand for transportation under space and time restrictions (requests). REQreate is flexible and can be configured to generate instances for a large number of problems in this problem class. For example, the Dial-a-Ride Problem (DARP) and On-demand Bus Routing Problem (ODBRP). The tool makes use of real-life networks from OpenStreetMaps to generate instances for an extensive catalogue of existing and upcoming on-demand transportation problems.

## Installation

Requires **Python 3.11 or newer** (osmnx 2.x does not support older versions).

```bash
pip install "reqreate[app]"
```

If you use conda, create a fresh environment rather than reusing an old one —
environments built for earlier versions of this tool usually have a Python
older than 3.11 and pinned packages that conflict:

```bash
conda create -n reqreate python=3.11 -y
conda activate reqreate
pip install "reqreate[app]"
```

Check it worked:

```bash
reqreate --version
```

### Optional extras

| Extra | Installs | Needed for |
|---|---|---|
| `app` | streamlit, folium, plotly | the web interface (`reqreate app`) |
| `analysis` | sqlalchemy | the taxi-dataset trip-pattern and `uber_movement` modules |
| `parallel` | ray | distributing the distance matrices over cores - see below |

```bash
pip install "reqreate[app,analysis]"
```

The `parallel` extra installs ray. Until recently `compute_distance_matrix`
shadowed the imported module with a dummy and ran sequentially regardless; that
is fixed, so the distance-matrix work now really is handed to ray when the extra
is installed. Results are identical either way (verified value-by-value and
dtype-by-dtype on a synthetic graph).

**It is not necessarily faster, and we have not measured a speedup on a real
city.** On a 4-core machine a synthetic 1500-node graph took 20s sequentially
and 33s through ray, because starting ray and shipping the graph to the workers
costs more than the shortest paths save at that size. Larger networks have more
work to amortise that over, but until someone measures one, treat the extra as
experimental and leave it out if in doubt.

## Quick start: the web interface

The output is written to the directory you run the command from, so start in an
empty folder:

```bash
mkdir my-instances
cd my-instances
reqreate app
```

Your browser opens at `http://localhost:8501`. Then:

1. Choose a **problem type** (DARP, ODBRP, or Patient Transport)
2. Enter a **location**, for example `Aachen, Germany`
3. Set the **number of requests**
4. Click **Generate**, and leave the browser tab open

**Start with a small number of requests (around 20) the first time you try a
new city.** Almost all of the running time goes into processing the street
network rather than into the requests themselves, so a 20-request run costs
about the same as a large one. What it buys you is a check that the pipeline
works for that location before you commit to a full-size run.

**Expect hours, not minutes.** Downloading the network takes a few minutes, but
the distance and travel-time matrices that follow are the expensive part, and
they scale with the size of the city rather than the number of requests. A
20-request run for `Aachen, Germany` took about 90 minutes end to end on a
single core, most of it in those matrices. The `parallel` extra may help on
large networks, but no speedup has been measured; see the note above.

Runs of the same location afterwards are quicker, because the OpenStreetMap
responses are cached in a `cache/` folder next to the output.

### Where the output goes

Files are written to a folder named after the location you entered, inside the
directory you launched from:

```
my-instances/
└── Aachen, Germany/
    ├── csv_format/           generated instances
    ├── json_format/
    ├── csv/                  network data: stops, zones, POIs, matrices
    ├── graphml_format/       street network
    ├── travel_time_matrix/
    ├── images/               maps and heatmaps
    ├── logs/
    └── pickle/
```

The interface also offers the whole folder as a single ZIP download.

## Command line

Generate from a JSON configuration file without the interface:

```bash
reqreate generate my_config.json
```

`reqreate --help` lists every option.

Attributes and parameters that define an instance are described in a
configuration file given as input to REQreate. The syntax used is JSON. Each
configuration file can generate one or more instances. See
[examples/basic_examples](https://github.com/michellqueiroz-ua/instance-generator/tree/master/examples/basic_examples)
for example configuration files.

## Troubleshooting

### "Connection refused" or other Overpass API errors

If generation fails with a connection error mentioning `overpass-api.de`, the
public OpenStreetMap query servers are refusing or rate-limiting the request.
**This is unrelated to the location you asked for** — the same run usually
succeeds if you retry a few minutes later.

REQreate retries and fails over across several mirrors automatically. If none
of them are reachable, point it at an Overpass instance you can reach:

```bash
# Linux / macOS
export REQREATE_OVERPASS_URL=https://overpass.kumi.systems/api/interpreter

# Windows
set REQREATE_OVERPASS_URL=https://overpass.kumi.systems/api/interpreter
```

Any instance you set this way must hold planet-wide data. A regional Overpass
mirror answers a query for a place it does not cover with an empty result
rather than an error, which would produce an instance with no network in it.

To check whether Overpass is reachable from your machine at all:

```bash
python -c "import requests; print(requests.get('https://overpass-api.de/api/status', timeout=30).text[:300])"
```

### The install fails, or `reqreate` is not found

Almost always a Python version problem. Check with `python --version`; it must
be 3.11 or newer. If you are in a conda environment created for an older
version of this tool, make a fresh one as shown under Installation.

## Running it hosted

The repository includes a `Dockerfile` for deploying the interface to
[Hugging Face Spaces](https://huggingface.co/spaces) — see
[the deployment guide](docs/DEPLOYMENT_GUIDE.md).

Running locally is recommended over a hosted deployment. OpenStreetMap
downloads then leave from your own IP address rather than one shared with
every other user of a hosting platform, and a shared address is what gets
requests refused or rate-limited by the public Overpass API.

## Features

- 🌍 **Real-world networks** from OpenStreetMap
- 🚍 **Flexible configuration** for various on-demand transport problems
- 📊 **Multiple output formats** (CSV, JSON, GraphML, Pickle)
- 🎯 **Realistic request patterns** based on POI density
- ⏱️ **Accurate travel times** using actual road speeds
- 🗺️ **Interactive maps** with request distribution and demand heatmaps
- 🖥️ **User-friendly web interface** for easy instance generation

## Output files

REQreate generates comprehensive datasets including:

- Bus station locations
- Network topology (walk + drive)
- Passenger requests with time windows
- [Travel time matrices](docs/TRAVEL_TIME_MATRICES.md)
- Zone definitions
- Points of Interest (POIs)
- Network visualizations

## Development

To work on REQreate itself, install from a checkout in editable mode:

```bash
git clone https://github.com/michellqueiroz-ua/instance-generator.git
cd instance-generator
pip install -e ".[app]"
```

# Questions? Do not hesitate to send an e-mail to:

michell.queiroz@uantwerpen.be
