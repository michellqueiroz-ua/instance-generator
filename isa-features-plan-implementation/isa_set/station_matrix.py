"""
station_matrix.py - ISA instance set: write the station-to-station travel-time matrix of a network, once.

    python station_matrix.py [--network "Capital Region of Denmark"] [--work /work3/$USER/isa] [--chunk 64]

The matrix is the same for every instance of the network (same stations, same speed), so it is not
part of any instance's generation. Reads the cached network, runs one shortest-path search per
station (chunk origins at a time, keeping only the station columns) and writes, in
<work>/<network>/travel_time_matrix/:

    station_travel_time_matrix.csv   same layout as REQreate's travel_time_matrix_<instance>_0.csv
                                     (index column, header 0..n-1, one row per station in network order)
    station_travel_time_matrix.npy   the same values as int32, for the feature code

Travel times are whole seconds (truncated), the same values REQreate computes. Stops with an error if
any pair of stations is unreachable (the drive graph is a strongly connected component, so none is
expected).
"""

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

from REQreate.compute_distance_matrix import travel_times_between


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--network", default="Capital Region of Denmark")
    parser.add_argument("--work", default=os.environ.get("ISA_WORK", f"/work3/{os.environ.get('USER', 'user')}/isa"))
    parser.add_argument("--chunk", type=int, default=64, help="origins per shortest-path call")
    args = parser.parse_args(argv)

    base = os.path.join(args.work, args.network)
    with open(os.path.join(base, "pickle", f"{args.network}.network.class.pkl"), "rb") as f:
        network = pickle.load(f)
    nodes = [int(n) for n in network.bus_stations["osmid_drive"]]
    print(f"{len(nodes)} stations, drive graph with {network.G_drive.number_of_nodes()} nodes", flush=True)

    started = time.time()

    def progress(done, total):
        print(f"  {done}/{total} origins, {time.time() - started:.0f} s", flush=True)

    matrix = travel_times_between(network.G_drive, nodes, "travel_time", chunk=args.chunk, progress=progress)
    unreachable = int((matrix < 0).sum())
    if unreachable:
        raise SystemExit(f"{unreachable} station pairs are unreachable; not writing the matrix")

    out_dir = os.path.join(base, "travel_time_matrix")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "station_travel_time_matrix.npy"), matrix)
    pd.DataFrame(matrix.astype(float)).to_csv(os.path.join(out_dir, "station_travel_time_matrix.csv"))
    print(f"wrote {matrix.shape[0]} x {matrix.shape[1]} matrix to {out_dir} in {time.time() - started:.0f} s; "
          f"mean {matrix.mean():.0f} s, max {matrix.max()} s", flush=True)


if __name__ == "__main__":
    main()
