"""
plot_requests.py - ISA instance set: look at the requests of generated instances (map and trip statistics).

    python plot_requests.py ISA_b90_i001 [ISA_b91_i001 ...] [--work /work3/$USER/isa]

Reads   <work>/<place>/csv_format/<instance>_1.csv   the solver csv of the instance
Writes  <work>/results/maps/<instance>.png            origins (blue) and destinations (red), a sample of the trips as lines
Prints  requests, mean direct drive distance, share of trips by length against the 2024 mobility report, the share of
        ends inside the rough bounding box of the central municipalities, and the busiest 5 x 5 km cells.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PLACE = "Capital Region of Denmark"
CENTRAL_BOX = (12.453, 55.613, 12.734, 55.733)  # lon min, lat min, lon max, lat max of Copenhagen + Frederiksberg
BINS = [(0, 5, "<5 km"), (5, 10, "5-10"), (10, 15, "10-15"), (15, 20, "15-20"), (20, 1e9, ">20")]
REPORT = "report, trips of 2-20 km renormalised: car 41/33/16/10 %, public transport 28/39/21/12 % (<5, 5-10, 10-15, 15-20)"


def parse_point(text):
    """(lat, lon) from a cell like '(55.67, 12.56)'."""
    numbers = re.findall(r"-?\d+\.\d+", str(text))
    return float(numbers[0]), float(numbers[1])


def load(csv_path):
    frame = pd.read_csv(csv_path)
    origin = np.array([parse_point(v) for v in frame["origin"]])
    destination = np.array([parse_point(v) for v in frame["destination"]])
    return frame, origin, destination


def central_polygon(work):
    """Exact boundary of Copenhagen + Frederiksberg (cached as WKT next to the zone weights)."""
    from shapely import wkt
    cache = Path(work) / PLACE / "csv" / "central_polygon.wkt"
    if cache.exists():
        return wkt.loads(cache.read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import zone_weights
    polygon = zone_weights.download_central_polygon(log=lambda message: None)
    cache.write_text(polygon.wkt)
    return polygon


def share_inside(polygon, points):
    """Percent of the (lat, lon) points inside the polygon."""
    import shapely
    if hasattr(shapely, "contains_xy"):
        return float(shapely.contains_xy(polygon, points[:, 1], points[:, 0]).mean() * 100)
    from shapely.geometry import Point
    return float(np.mean([polygon.contains(Point(lon, lat)) for lat, lon in points]) * 100)


def trip_shares(distance_km):
    return [float(((distance_km >= low) & (distance_km < high)).mean() * 100) for low, high, _ in BINS]


def in_box(points, box=CENTRAL_BOX):
    return (points[:, 1] >= box[0]) & (points[:, 0] >= box[1]) & (points[:, 1] <= box[2]) & (points[:, 0] <= box[3])


def busiest_cells(points, size_km=5.0, top=5):
    lat0 = np.radians(points[:, 0].mean())
    x = np.floor(points[:, 1] * 111.0 * np.cos(lat0) / size_km).astype(int)
    y = np.floor(points[:, 0] * 111.0 / size_km).astype(int)
    cells = pd.Series(list(zip(x, y))).value_counts().head(top)
    return [(f"lat {key[1] * size_km / 111.0:.2f}, lon {key[0] * size_km / (111.0 * np.cos(lat0)):.2f}", int(n))
            for key, n in cells.items()]


def plot(name, origin, destination, out_path, sample=300, seed=1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(origin[:, 1], origin[:, 0], s=3, c="tab:blue", alpha=0.4, label="origins")
    ax.scatter(destination[:, 1], destination[:, 0], s=3, c="tab:red", alpha=0.4, label="destinations")
    for i in rng.choice(len(origin), size=min(sample, len(origin)), replace=False):
        ax.plot([origin[i, 1], destination[i, 1]], [origin[i, 0], destination[i, 0]], c="grey", lw=0.3, alpha=0.5)
    ax.add_patch(plt.Rectangle((CENTRAL_BOX[0], CENTRAL_BOX[1]), CENTRAL_BOX[2] - CENTRAL_BOX[0],
                               CENTRAL_BOX[3] - CENTRAL_BOX[1], fill=False, ec="black", ls="--", lw=0.8))
    ax.set_aspect(1 / np.cos(np.radians(origin[:, 0].mean())))
    ax.set_title(f"{name}: {len(origin)} requests (dashed: central bounding box)")
    ax.legend(loc="upper left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def summarize(args):
    """One row per instance: size, trip length shares, central share, effective speed (a check over a whole batch)."""
    polygon = central_polygon(args.work)
    rows = []
    for name in args.instances:
        frame, origin, destination = load(Path(args.work) / PLACE / "csv_format" / f"{name}_1.csv")
        distance_km = frame["direct_distance"].to_numpy(dtype=float) / 1000.0
        speed = distance_km * 1000.0 / frame["direct_travel_time"].to_numpy(dtype=float) * 3.6
        shares = trip_shares(distance_km)
        rows.append({"instance": name, "requests": len(frame), "mean_km": round(distance_km.mean(), 1),
                     **{f"share_{label}": round(share) for (_, _, label), share in zip(BINS, shares)},
                     "median_speed_kmh": round(float(np.median(speed))),
                     "central_origin_pct": round(share_inside(polygon, origin), 1),
                     "central_dest_pct": round(share_inside(polygon, destination), 1),
                     "top_cell_origin_pct": round(busiest_cells(origin, top=1)[0][1] / len(frame) * 100, 1),
                     "top_cell_dest_pct": round(busiest_cells(destination, top=1)[0][1] / len(frame) * 100, 1)})
    table = pd.DataFrame(rows)
    out = Path(args.work) / "results" / f"request_summary_{Path(args.instances[0]).name[4:7]}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table.drop(columns=["instance"]).describe().loc[["min", "50%", "max"]].round(1).to_string())
    print(f"{len(table)} instances; per-instance rows in {out}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("instances", nargs="*", help="instance names, e.g. ISA_b90_i001")
    parser.add_argument("--batch", type=int, help="all instances of this batch (with --summary: one line each)")
    parser.add_argument("--summary", action="store_true",
                        help="one line per instance (no map), written to results/request_summary_batch_XX.csv")
    parser.add_argument("--work", default=os.environ.get("ISA_WORK", f"/work3/{os.environ.get('USER', 'user')}/isa"))
    args = parser.parse_args(argv)
    if args.batch is not None:
        folder = Path(args.work) / PLACE / "csv_format"
        args.instances = sorted(p.name[:-6] for p in folder.glob(f"ISA_b{args.batch:02d}_i*_1.csv"))
    if not args.instances:
        raise SystemExit("give instance names or --batch N")
    print(REPORT)
    if args.summary:
        return summarize(args)
    for name in args.instances:
        csv_path = Path(args.work) / PLACE / "csv_format" / f"{name}_1.csv"
        frame, origin, destination = load(csv_path)
        distance_km = frame["direct_distance"].to_numpy(dtype=float) / 1000.0
        shares = trip_shares(distance_km)
        print(f"\n{name}: {len(frame)} requests, mean direct drive distance {distance_km.mean():.1f} km "
              f"(median {np.median(distance_km):.1f}); direct time {frame['direct_travel_time'].min():.0f}-"
              f"{frame['direct_travel_time'].max():.0f} s")
        print("  trips by length: " + ", ".join(f"{label} {share:.0f}%" for (_, _, label), share in zip(BINS, shares)))
        speed = frame["direct_distance"].to_numpy(dtype=float) / frame["direct_travel_time"].to_numpy(dtype=float) * 3.6
        print(f"  effective drive speed (distance / time): median {np.median(speed):.0f} km/h, 5-95%: "
              f"{np.percentile(speed, 5):.0f}-{np.percentile(speed, 95):.0f} km/h (design.json says 20: that number only "
              f"sets the ring radius, the travel times come from the network)")
        try:
            polygon = central_polygon(args.work)
            print(f"  ends inside the central municipalities (exact boundary): origins {share_inside(polygon, origin):.1f}%, "
                  f"destinations {share_inside(polygon, destination):.1f}% (target: close to 0, a few from boundary zones)")
        except Exception as error:  # needs shapely and, the first time, Nominatim
            print(f"  (exact central share not available: {error})")
        print(f"  ends inside the central bounding box: origins {in_box(origin).mean() * 100:.1f}%, "
              f"destinations {in_box(destination).mean() * 100:.1f}% (the box is rough: it holds some non-central land)")
        print("  busiest 5 km cells (origins): " + "; ".join(f"{k} = {n}" for k, n in busiest_cells(origin)))
        print("  busiest 5 km cells (destinations): " + "; ".join(f"{k} = {n}" for k, n in busiest_cells(destination)))
        out = Path(args.work) / "results" / "maps" / f"{name}.png"
        plot(name, origin, destination, out)
        print(f"  map: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
