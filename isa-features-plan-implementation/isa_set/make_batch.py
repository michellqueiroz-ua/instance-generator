"""
make_batch.py - ISA instance set: draw one batch of generator configs.

    python make_batch.py --batch 1 --n 100 [--work /work3/$USER/isa] [--seed 1]
    python make_batch.py --batch 0 --n 1 --with-matrix        # one test instance

Reads   design.json (next to this file): the ranges of every setting.
Writes  <work>/configs/batch_XX/ISA_bXX_iNNN.json   one REQreate config per instance
        <work>/configs/batch_XX/list.txt            the config paths, one per line (for the job array)
        <work>/design/batch_XX.csv                  the drawn settings per instance
        <work>/design/batch_XX_fleets.json          the fleets per instance (draws_per_instance each)

Instances are drawn with a Latin hypercube over design.json "instance", so every setting is
spread over its whole range (no grid, no gaps from two-level factors); a setting given as a single
number is held fixed. Fleets cost nothing to generate, so each instance gets several fleets, drawn
with a second Latin hypercube.

Where requests are (batch 2 on): when design.json draws "inbound_share" and "concentration", origins and
destinations are drawn zone by zone with the weights of od_weights(), built from the homes and attractions
that zone_weights.py counted per zone. The batch csv then has a "scenario" label (a_inbound, b_both_ways,
c_outbound) derived from the inbound share.

The station travel-time matrix is the same for every instance of the network (same stations),
so configs leave it out; --with-matrix adds it, for the one test instance that writes it once.

Only the standard library is used, so this runs on the HPC login node.
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGN_JSON = HERE / "design.json"


def latin_hypercube(n, dims, rng):
    """n points in [0, 1)^dims, one in each of the n slices of every dimension."""
    columns = []
    for _ in range(dims):
        order = list(range(n))
        rng.shuffle(order)
        columns.append([(order[i] + rng.random()) / n for i in range(n)])
    return [[columns[d][i] for d in range(dims)] for i in range(n)]


def scale(unit, low_high):
    low, high = low_high
    return low + unit * (high - low)


def read_zone_weights(path):
    """[(zone name, n_residential, n_attraction, central_share)] from the csv that zone_weights.py writes.

    central_share is 0 for every zone when the csv has no such column (see has_central_share).
    """
    with open(path, encoding="utf-8", newline="") as f:
        return [(r["name"], float(r["n_residential"]), float(r["n_attraction"]), float(r.get("central_share") or 0.0))
                for r in csv.DictReader(f)]


def has_central_share(path):
    with open(path, encoding="utf-8", newline="") as f:
        return "central_share" in (csv.DictReader(f).fieldnames or [])


CENTRAL_THRESHOLD = 0.5  # a zone whose rectangle is mostly inside the central municipalities counts as central


def od_weights(zones, inbound_share, concentration, central_factor=1.0):
    """Origin and destination weight of every zone (same order as ``zones``; each list sums to 1).

    Residential and attraction counts become shares, sharpened by ``concentration`` (count ** concentration):
    0 gives the same weight to every zone (uniform demand, as in batch 1), 2 concentrates the demand in the
    densest zones. ``inbound_share`` 1 draws every origin from the residential shares and every destination
    from the attraction shares (people travel to where things happen); 0 is the opposite direction (night
    trips home); in between, each end of a trip follows a mix of the two.

    ``central_factor`` multiplies the counts of the zones that are mostly inside the central municipalities
    (Copenhagen and Frederiksberg, served by metro and S-train): 0 removes them from the service area, 1 keeps
    them. Zones are tuples (name, residential, attraction[, central_share]); a zone drawn as a whole then
    receives its points uniformly, so a boundary zone below the threshold leaks a few points into the centre.
    """
    multiplier = [central_factor if len(z) > 3 and z[3] >= CENTRAL_THRESHOLD else 1.0 for z in zones]

    def shares(counts):
        powered = [0.0 if m == 0 else (count * m) ** concentration for count, m in zip(counts, multiplier)]
        total = sum(powered)
        if total <= 0:
            raise ValueError("a zone count column is zero everywhere; run zone_weights.py first")
        return [p / total for p in powered]

    home = shares([z[1] for z in zones])
    place = shares([z[2] for z in zones])
    origin = [inbound_share * h + (1 - inbound_share) * a for h, a in zip(home, place)]
    destination = [inbound_share * a + (1 - inbound_share) * h for h, a in zip(home, place)]
    return origin, destination


def scenario_label(inbound_share):
    """Names the corners of the inbound-share axis; only a label for reports, not a design factor."""
    if inbound_share >= 0.75:
        return "a_inbound"
    if inbound_share <= 0.25:
        return "c_outbound"
    return "b_both_ways"


def make_config(design, settings, seed, with_matrix, zones=None):
    fixed = design["fixed"]
    start_s = settings["start_hour"] * 3600
    end_s = start_s + design["period_hours"] * 3600
    centre_s = (start_s + end_s) / 2
    walk_low, walk_high = fixed["max_walking_s"]
    config = {
        "seed": seed,
        "network": design["network"],
        "problem": design["problem"],
        "set_fixed_speed": {"vehicle_speed_data": fixed["vehicle_speed_kmh"], "vehicle_speed_data_unit": "kmh"},
        "replicas": 1,
        "requests": settings["requests"],
        "instance_filename": ["network", "problem", "requests"],
        "parameters": [
            {"name": "min_early_departure", "type": "float", "value": start_s / 3600, "time_unit": "h"},
            {"name": "max_early_departure", "type": "float", "value": end_s / 3600, "time_unit": "h"},
            {"name": "min_dtt", "type": "integer", "value": fixed["min_dtt_s"], "time_unit": "s"},
            {"name": "max_dtt", "type": "integer", "value": fixed["max_dtt_s"], "time_unit": "s"},
        ],
        "attributes": [
            {
                # demand around the middle of the period; a large sd makes it close to uniform
                "name": "time_stamp", "type": "integer", "time_unit": "s",
                "pdf": [{"type": "normal", "loc": round(centre_s), "scale": round(settings["peak_sd_min"] * 60)}],
                "constraints": [
                    "time_stamp >= 0",
                    "time_stamp >= min_early_departure",
                    "time_stamp <= max_early_departure",
                ],
            },
            {
                "name": "reaction_time", "type": "integer", "time_unit": "s",
                "pdf": [{"type": "normal", "loc": round(settings["urgency_mean_min"] * 60),
                         "scale": round(settings["urgency_sd_min"] * 60)}],
                "constraints": ["reaction_time >= 60"],
            },
            {
                "name": "earliest_departure", "type": "integer", "time_unit": "s",
                "expression": "time_stamp",
                "constraints": ["earliest_departure >= 0", "earliest_departure >= min_early_departure"],
            },
            {
                "name": "latest_departure", "type": "integer", "time_unit": "s",
                "expression": "time_stamp + reaction_time",
            },
            {
                "name": "latest_arrival", "type": "integer", "time_unit": "s",
                "expression": "earliest_departure + direct_travel_time + reaction_time + "
                              f"{round(settings['arrival_slack_min'] * 60)}",
            },
            {"name": "origin", "type": "location"},
            {"name": "destination", "type": "location"},
            {
                "name": "stops_orgn", "type": "array_primitives", "expression": "stops(origin)",
                "constraints": ["len(stops_orgn) > 0"],
            },
            {
                "name": "stops_dest", "type": "array_primitives", "expression": "stops(destination)",
                "constraints": ["len(stops_dest) > 0", "not (set(stops_orgn) & set(stops_dest))"],
            },
            {
                "name": "max_walking", "type": "integer", "time_unit": "s",
                "pdf": [{"type": "uniform", "loc": walk_low, "scale": walk_high - walk_low}],
            },
            {
                "name": "walk_speed", "type": "real", "speed_unit": "mps",
                "pdf": [{"type": "uniform", "loc": fixed["walk_speed_mps"], "scale": 0}],
            },
            {
                "name": "direct_travel_time", "type": "integer", "time_unit": "s",
                "expression": "dtt(origin,destination)",
                "constraints": ["direct_travel_time >= min_dtt", "direct_travel_time <= max_dtt"],
            },
            {
                "name": "direct_distance", "type": "integer", "length_unit": "m",
                "expression": "dist_drive(origin,destination)",
            },
        ],
    }
    if zones is not None:
        # Origins and destinations are drawn zone by zone with the weights of od_weights(); the generator then
        # picks a random point in the zone (it ignores the dtt ring, so max_dtt only rejects long trips).
        origin_w, destination_w = od_weights(zones, settings["inbound_share"], settings["concentration"],
                                             design.get("spatial", {}).get("central_factor", 1.0))
        config["parameters"].append({"name": "zones_all", "type": "array_zones", "size": len(zones),
                                     "value": [z[0] for z in zones]})
        for attribute in config["attributes"]:
            if attribute["name"] == "origin":
                attribute.update({"subset_zones": "zones_all", "weights": [round(w, 8) for w in origin_w]})
            elif attribute["name"] == "destination":
                attribute.update({"subset_zones": "zones_all", "weights": [round(w, 8) for w in destination_w]})
                decay_km = design.get("spatial", {}).get("trip_decay_km")
                if decay_km:
                    # zone-level distance decay of the generator: short trips are favoured without extra searches
                    attribute["distance_decay_m"] = round(decay_km * 1000)
    if with_matrix:
        config["travel_time_matrix"] = ["bus_stations"]
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--n", type=int, required=True, help="instances in this batch")
    parser.add_argument("--work", default=os.environ.get("ISA_WORK", f"/work3/{os.environ.get('USER', 'user')}/isa"))
    parser.add_argument("--seed", type=int, help="random seed of the draw (default: the batch number)")
    parser.add_argument("--with-matrix", action="store_true", help="also write the station travel-time matrix")
    parser.add_argument("--zone-weights", help="csv written by zone_weights.py (default: in the network folder of "
                        "--work); needed when design.json draws inbound_share and concentration")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="hold an instance setting fixed instead of drawing it, e.g. --set requests_per_hour=2000 "
                             "--set inbound_share=1 (for smoke and scaling tests); may be repeated")
    args = parser.parse_args(argv)

    design = json.loads(DESIGN_JSON.read_text(encoding="utf-8"))
    zones = None
    if "inbound_share" in design["instance"]:
        place = design["network"]
        weights_csv = Path(args.zone_weights or Path(args.work) / place / "csv" / f"{place}.zone_weights.csv")
        if not weights_csv.is_file():
            raise SystemExit(f"{weights_csv} not found: run hpc/zone_weights.sh first (or --zone-weights)")
        zones = read_zone_weights(weights_csv)
        if design.get("spatial", {}).get("central_factor", 1.0) < 1.0 and not has_central_share(weights_csv):
            raise SystemExit(f"{weights_csv} has no central_share column: run "
                             f"'python isa_set/zone_weights.py --central-only' first")
    rng = random.Random(args.seed if args.seed is not None else args.batch)
    tag = f"b{args.batch:02d}"
    work = Path(args.work)
    config_dir = work / "configs" / f"batch_{args.batch:02d}"
    design_dir = work / "design"
    config_dir.mkdir(parents=True, exist_ok=True)
    design_dir.mkdir(parents=True, exist_ok=True)
    if any(config_dir.glob("*.json")):
        raise SystemExit(f"{config_dir} already has configs; use a new batch number")

    # A range [low, high] is drawn; a single number is a fixed setting (it stays out of the hypercube).
    names = [n for n, v in design["instance"].items() if isinstance(v, list)]
    held = {n: v for n, v in design["instance"].items() if not isinstance(v, list)}
    for assignment in args.set:
        name, _, value = assignment.partition("=")
        if name not in design["instance"]:
            raise SystemExit(f"--set {assignment}: {name} is not an instance setting of design.json "
                             f"({', '.join(design['instance'])})")
        names = [n for n in names if n != name]
        held[name] = float(value)
    rows, configs = [], []
    for i, unit in enumerate(latin_hypercube(args.n, len(names), rng), start=1):
        settings = {name: scale(u, design["instance"][name]) for name, u in zip(names, unit)}
        settings.update(held)
        settings["requests"] = max(1, round(settings["requests_per_hour"] * design["period_hours"]))
        name = f"ISA_{tag}_i{i:03d}"
        seed = args.batch * 1000 + i  # unique per instance over all batches
        path = config_dir / f"{name}.json"
        path.write_text(json.dumps(make_config(design, settings, seed, args.with_matrix, zones), indent=2),
                        encoding="utf-8")
        configs.append(str(path))
        row = {"instance": name, "seed": seed, **{k: round(v, 4) for k, v in settings.items()}}
        if zones is not None:
            row["scenario"] = scenario_label(settings["inbound_share"])
        rows.append(row)

    fleet = design["fleet"]
    draws = latin_hypercube(args.n * fleet["draws_per_instance"], 2, rng)
    fleets = {}
    for i, row in enumerate(rows):
        for k in range(fleet["draws_per_instance"]):
            percent_u, share_u = draws[i * fleet["draws_per_instance"] + k]
            percent = round(scale(percent_u, fleet["percent_of_requests"]), 2)
            share8 = round(100 * scale(share_u, fleet["share_8_seats"]), 1)
            # REQreate names the instance <config>_<replica>; replicas is 1
            fleets[f"{row['instance']}_1_fleet{k + 1}"] = {
                "instance": f"{row['instance']}_1",
                "fleet_percent_of_requests": percent,
                "composition": [{"capacity": 4, "percent": round(100 - share8, 1)},
                                {"capacity": 8, "percent": share8}],
            }

    (config_dir / "list.txt").write_text("\n".join(configs) + "\n", encoding="utf-8")
    with open(design_dir / f"batch_{args.batch:02d}.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (design_dir / f"batch_{args.batch:02d}_fleets.json").write_text(json.dumps(fleets, indent=2), encoding="utf-8")
    print(f"batch {args.batch}: {len(rows)} configs in {config_dir}, {len(fleets)} fleets in {design_dir}")


if __name__ == "__main__":
    main()
