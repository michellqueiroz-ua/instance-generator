# Basic examples

These configurations illustrate common ways to combine request attributes, distributions, expressions, location subsets, and network data. See the [JSON configuration reference](../../docs/CONFIG_REFERENCE.md) for field-by-field documentation and notes about legacy spellings in some examples.

* [`example_0/00_example.json`](example_0/00_example.json) — custom places/zones, a time window, and location subsets; uses an alternate `locations` schema.
* [`example_1/01_example.json`](example_1/01_example.json) — custom locations and zones, selected depots/schools, walking access, and a travel-time matrix.
* [`example_2/02_example.json`](example_2/02_example.json) — DARP requests with a POI-based origin/destination distance distribution and derived travel time/distance.
* [`example_3/03_example.json`](example_3/03_example.json) — dynamic request timestamps, reaction times, and derived pickup/drop-off windows.
* [`example_4/04_example.json`](example_4/04_example.json) — reachable-stop arrays with constraints and an additional sampled service attribute; uses an alternate `locations` schema.
* [`example_5/05_example.json`](example_5/05_example.json) — a second POI-based DARP configuration with alternate request and time-window settings.
