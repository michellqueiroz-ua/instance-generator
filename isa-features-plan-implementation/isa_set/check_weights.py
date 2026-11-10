"""
check_weights.py - ISA instance set: what trips do the zone weights imply? (no generation, takes seconds)

    python check_weights.py [--work /work3/$USER/isa] [--samples 200000]

Reads   <work>/<place>/csv/<place>.zone_weights.csv and <place>.zones.csv (zone centres)
        design.json (central_factor, trip bounds, speed)
For a grid of inbound_share and concentration it draws origin and destination zones with make_batch.od_weights,
places each end uniformly inside a square of the zone's size around its centre, and applies the generator's rule
(direct drive time between min_dtt_s and max_dtt_s; drive distance taken as 1.3 x straight line, an assumption).
Prints per setting: how many zones carry the demand, the share of draws the rule accepts (the generation cost is
about 1 / acceptance per request), the trip length distribution of the accepted trips against the 2024 mobility
report (car trips up to 20 km, renormalised), and the share of trips with an end in a central zone.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_batch  # noqa: E402

DRIVE_FACTOR = 1.3
# Report, share of car trips by length (0-2, 2-5, 5-10, 10-15, 15-20 km, over 20 km omitted), 2035 scenario
REPORT_CAR = {"<5 km": 45.0, "5-10 km": 23.0, "10-15 km": 11.0, "15-20 km": 7.0}


def effective_zones(weights):
    w = np.asarray(weights, dtype=float)
    return float(1.0 / np.sum(w ** 2)) if w.sum() > 0 else 0.0


def simulate(zones, centres, half_km, inbound, concentration, central_factor, bounds_s, speed_mps, samples, rng):
    origin_w, destination_w = make_batch.od_weights(zones, inbound, concentration, central_factor)
    n = len(zones)
    o = rng.choice(n, size=samples, p=np.asarray(origin_w) / sum(origin_w))
    d = rng.choice(n, size=samples, p=np.asarray(destination_w) / sum(destination_w))
    lat0 = np.radians(centres[:, 1].mean())
    km_per_lat, km_per_lon = 111.0, 111.0 * np.cos(lat0)
    jitter = rng.uniform(-half_km, half_km, size=(samples, 4))
    ox = centres[o, 0] * km_per_lon + jitter[:, 0]
    oy = centres[o, 1] * km_per_lat + jitter[:, 1]
    dx = centres[d, 0] * km_per_lon + jitter[:, 2]
    dy = centres[d, 1] * km_per_lat + jitter[:, 3]
    straight = np.hypot(ox - dx, oy - dy)
    drive_s = straight * DRIVE_FACTOR * 1000.0 / speed_mps
    ok = (drive_s >= bounds_s[0]) & (drive_s <= bounds_s[1])
    drive_km = straight[ok] * DRIVE_FACTOR
    central = np.array([len(z) > 3 and z[3] >= make_batch.CENTRAL_THRESHOLD for z in zones])
    return {
        "eff_origin": effective_zones(origin_w), "eff_dest": effective_zones(destination_w),
        "accept": float(ok.mean()),
        "mean_km": float(drive_km.mean()) if drive_km.size else float("nan"),
        "under5": float((drive_km < 5).mean() * 100) if drive_km.size else float("nan"),
        "km5_10": float(((drive_km >= 5) & (drive_km < 10)).mean() * 100) if drive_km.size else float("nan"),
        "km10_15": float(((drive_km >= 10) & (drive_km < 15)).mean() * 100) if drive_km.size else float("nan"),
        "over15": float((drive_km >= 15).mean() * 100) if drive_km.size else float("nan"),
        "central_ends": float((central[o][ok] | central[d][ok]).mean() * 100) if ok.any() else float("nan"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--work", default=os.environ.get("ISA_WORK", f"/work3/{os.environ.get('USER', 'user')}/isa"))
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    design = json.loads((HERE / "design.json").read_text(encoding="utf-8"))
    place = design["network"]
    csv_dir = Path(args.work) / place / "csv"
    weights_csv = csv_dir / f"{place}.zone_weights.csv"
    zones = make_batch.read_zone_weights(weights_csv)
    with open(csv_dir / f"{place}.zones.csv", encoding="utf-8", newline="") as f:
        centre_of = {r["name"]: (float(r["center_x"]), float(r["center_y"])) for r in csv.DictReader(f)}
    centres = np.array([centre_of[z[0]] for z in zones])
    # a zone is a rectangle of about (east-west span / 25) by (north-south span / 25)
    half_km = 1.25
    fixed = design["fixed"]
    bounds_s = (fixed["min_dtt_s"], fixed["max_dtt_s"])
    speed = fixed["vehicle_speed_kmh"] / 3.6
    central_factor = design.get("spatial", {}).get("central_factor", 1.0)
    rng = np.random.default_rng(args.seed)

    print(f"{len(zones)} zones, central_factor {central_factor}, trip bounds {bounds_s} s at {fixed['vehicle_speed_kmh']} km/h, "
          f"drive distance = {DRIVE_FACTOR} x straight line (assumption), ends jittered +-{half_km} km (assumption)")
    print("report, car trips up to 20 km (share of all trips): " + ", ".join(f"{k} {v:.0f}%" for k, v in REPORT_CAR.items())
          + f"; renormalised to 100% of the trips up to 20 km: <5 km {45 / 86 * 100:.0f}%, 5-10 {23 / 86 * 100:.0f}%, "
            f"10-15 {11 / 86 * 100:.0f}%, 15-20 {7 / 86 * 100:.0f}%")
    header = ("inbound conc | eff.zones o/d | accepted | mean km | <5km  5-10  10-15  >15 (%) | central ends %")
    print(header)
    for inbound in (0.0, 0.5, 1.0):
        for concentration in (0.1, 0.8, 1.5):
            r = simulate(zones, centres, half_km, inbound, concentration, central_factor, bounds_s, speed, args.samples, rng)
            print(f"{inbound:7.1f} {concentration:4.1f} | {r['eff_origin']:5.0f} /{r['eff_dest']:5.0f}   | {r['accept'] * 100:6.1f}% | "
                  f"{r['mean_km']:6.1f}  | {r['under5']:4.0f} {r['km5_10']:5.0f} {r['km10_15']:6.0f} {r['over15']:5.0f}      | {r['central_ends']:5.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
