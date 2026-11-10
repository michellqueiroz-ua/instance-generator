"""
zone_weights.py - ISA instance set: how many homes and attractions each zone of the network holds.

    python zone_weights.py [--work /work3/$USER/isa] [--place "Capital Region of Denmark"]

Reads   <work>/<place>/csv/<place>.zones.csv      the 25 x 25 zone grid (written when the network is built)
Writes  <work>/<place>/csv/<place>.zone_weights.csv   one row per zone: name, n_stations, n_residential, n_attraction
        <work>/<place>/csv/zone_weights_cache/strip_*.json   one file per strip of the grid (a restart skips them)

n_residential counts residential buildings (where people live: origins of inbound trips, destinations of
outbound trips). n_attraction counts places people travel to in their free time: restaurants, cafes, bars,
pubs, shops, cinemas, theatres, tourism and leisure sites. make_batch.py turns the two counts into the origin
and destination weights of the zones (inbound share and concentration factors).

The download is one OpenStreetMap query per strip of the grid and per group, so memory stays small and a
failed run continues where it stopped. Nothing here needs the network graph.
"""

import argparse
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESIDENTIAL_TAGS = {"building": ["apartments", "detached", "dormitory", "house", "residential",
                                 "semidetached_house", "terrace"]}
ATTRACTION_TAGS = {
    "amenity": ["bar", "cafe", "restaurant", "pub", "fast_food", "biergarten", "food_court", "ice_cream",
                "cinema", "theatre", "nightclub", "night_club", "casino", "arts_centre", "community_centre",
                "events_venue", "library", "marketplace", "social_centre"],
    "shop": True,
    "tourism": ["aquarium", "attraction", "gallery", "museum", "theme_park", "zoo", "hotel", "hostel"],
    "leisure": ["fitness_centre", "sports_centre", "swimming_pool", "stadium", "water_park", "bowling_alley"],
}


def read_zones(zones_csv):
    """[{name, bounds (west, south, east, north), n_stations}] from the zones csv written at network build."""
    from shapely import wkt
    frame = pd.read_csv(zones_csv)
    zones = []
    for _, row in frame.iterrows():
        stations = row["stations"]
        count = len(ast.literal_eval(stations)) if isinstance(stations, str) else 0
        zones.append({"name": row["name"], "bounds": wkt.loads(row["polygon"]).bounds, "n_stations": count})
    return zones


def count_points(zones, lats, lons):
    """Number of points inside each zone's rectangle (west and south edges inclusive)."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    counts = []
    for zone in zones:
        west, south, east, north = zone["bounds"]
        counts.append(int(((lons >= west) & (lons < east) & (lats >= south) & (lats < north)).sum()))
    return counts


def strips(zones):
    """Zones grouped by their southern edge: one horizontal strip of the grid each, south to north."""
    grouped = {}
    for zone in zones:
        grouped.setdefault(round(zone["bounds"][1], 7), []).append(zone)
    return [grouped[key] for key in sorted(grouped)]


def download_points(bounds, tags):
    """Centroids (lats, lons) of the OpenStreetMap features with these tags inside (west, south, east, north)."""
    import osmnx as ox
    from shapely.geometry import box
    from REQreate.nearest import centroid_coordinates
    try:
        features = ox.features_from_polygon(box(*bounds), tags=tags)
    except Exception as error:  # osmnx raises InsufficientResponseError when nothing matches
        if type(error).__name__ == "InsufficientResponseError":
            return [], []
        raise
    return centroid_coordinates(features)


def strip_counts(strip, download=download_points):
    """{zone name: [n_residential, n_attraction]} for one strip of the grid."""
    bounds = (min(z["bounds"][0] for z in strip), min(z["bounds"][1] for z in strip),
              max(z["bounds"][2] for z in strip), max(z["bounds"][3] for z in strip))
    residential = count_points(strip, *download(bounds, RESIDENTIAL_TAGS))
    attraction = count_points(strip, *download(bounds, ATTRACTION_TAGS))
    return {zone["name"]: [r, a] for zone, r, a in zip(strip, residential, attraction)}


def build(zones, cache_dir, download=download_points, log=print):
    cache_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    all_strips = strips(zones)
    for i, strip in enumerate(all_strips, start=1):
        cache = cache_dir / f"strip_{i:03d}.json"
        if cache.exists():
            counts.update(json.loads(cache.read_text()))
            log(f"strip {i}/{len(all_strips)}: cached")
            continue
        result = strip_counts(strip, download)
        cache.write_text(json.dumps(result))
        counts.update(result)
        log(f"strip {i}/{len(all_strips)}: {sum(v[0] for v in result.values())} residential, "
            f"{sum(v[1] for v in result.values())} attraction points in {len(strip)} zones")
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--work", default=os.environ.get("ISA_WORK", f"/work3/{os.environ.get('USER', 'user')}/isa"))
    parser.add_argument("--place", default="Capital Region of Denmark")
    args = parser.parse_args(argv)

    csv_dir = Path(args.work) / args.place / "csv"
    zones = read_zones(csv_dir / f"{args.place}.zones.csv")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # REQreate lives at the repository root
    from REQreate.overpass_config import configure_overpass
    configure_overpass(timeout=1800)
    counts = build(zones, csv_dir / "zone_weights_cache")
    out = pd.DataFrame([{"name": z["name"], "n_stations": z["n_stations"],
                         "n_residential": counts[z["name"]][0], "n_attraction": counts[z["name"]][1]}
                        for z in zones])
    path = csv_dir / f"{args.place}.zone_weights.csv"
    out.to_csv(path, index=False)
    print(f"{len(out)} zones written to {path}: {out['n_residential'].sum()} residential, "
          f"{out['n_attraction'].sum()} attraction points; zones without any attraction: "
          f"{(out['n_attraction'] == 0).sum()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
