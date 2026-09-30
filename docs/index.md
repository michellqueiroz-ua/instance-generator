# REQreate 🚌

REQreate generates instances for on-demand transportation problems, such as the
Dial-a-Ride Problem (DARP) and On-demand Bus Routing Problem (ODBRP). It uses
real-world street networks from OpenStreetMap and can be configured for a range
of transportation problems.

## Installation

Requires **Python 3.11 or newer**.

```bash
pip install "reqreate[app]"
```

Check that the command is available:

```bash
reqreate --version
```

## Quick start

Run the web interface from an empty folder so generated files are kept together:

```bash
mkdir my-instances
cd my-instances
reqreate app
```

Open `http://localhost:8501` in your browser, choose a problem type and location,
set the number of requests, then click **Generate**. Start with around 20
requests when trying a new city. Generation can take hours for a large city
because processing the street network is the most time-consuming step.

For more details, see the [web interface guide](WEBAPP.md), the
[deployment guide](DEPLOYMENT_GUIDE.md), and the guide to
[travel-time matrices](TRAVEL_TIME_MATRICES.md).
