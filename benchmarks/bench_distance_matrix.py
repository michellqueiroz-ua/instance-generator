"""Offline benchmark for walk-distance matrix generation."""

import argparse
import os
import tempfile
import time

import networkx as nx
import pandas as pd

from REQreate import compute_distance_matrix as cdm


def synthetic_road_graph(rows, cols):
    graph = nx.MultiDiGraph(crs="epsg:4326")
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            graph.add_node(node, x=col * 0.001, y=row * 0.001)

            for neighbor in (
                node + 1 if col + 1 < cols else None,
                node + cols if row + 1 < rows else None,
            ):
                if neighbor is None:
                    continue
                length = float(75 + (node * 17 + neighbor * 31) % 126)
                travel_time = length / 4.5
                for origin, destination in ((node, neighbor), (neighbor, node)):
                    graph.add_edge(
                        origin,
                        destination,
                        length=length,
                        travel_time=travel_time,
                    )
    return graph


def time_get_distance_matrix(graph, stops):
    drive = nx.MultiDiGraph()
    drive.add_nodes_from(range(40))
    for node in range(39):
        for origin, destination in ((node, node + 1), (node + 1, node)):
            drive.add_edge(
                origin, destination, length=100.0, travel_time=10.0
            )

    with tempfile.TemporaryDirectory() as save_dir:
        start = time.perf_counter()
        cdm._get_distance_matrix(
            graph, drive, pd.DataFrame({"osmid_walk": stops}), save_dir, "bench"
        )
        return time.perf_counter() - start


def time_update_distance_matrix_walk(graph, stops):
    with tempfile.TemporaryDirectory() as save_dir:
        csv_dir = os.path.join(save_dir, "csv")
        os.makedirs(csv_dir)
        pd.DataFrame([{"osmid_origin": -1}]).to_csv(
            os.path.join(csv_dir, "bench.dist.walk.csv")
        )
        start = time.perf_counter()
        cdm._update_distance_matrix_walk(graph, stops, save_dir, "bench")
        return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=141)
    parser.add_argument("--cols", type=int, default=142)
    parser.add_argument("--stops", type=int, default=300)
    args = parser.parse_args()

    graph = synthetic_road_graph(args.rows, args.cols)
    node_count = len(graph)
    stop_count = min(args.stops, node_count)
    stops = [
        round(index * (node_count - 1) / max(stop_count - 1, 1))
        for index in range(stop_count)
    ]
    print(
        f"Graph: {node_count:,} nodes, {graph.number_of_edges():,} directed edges; "
        f"{stop_count} bus stops"
    )
    print(
        f"_get_distance_matrix: "
        f"{time_get_distance_matrix(graph, stops):.3f} s"
    )
    print(
        f"_update_distance_matrix_walk: "
        f"{time_update_distance_matrix_walk(graph, stops):.3f} s"
    )


if __name__ == "__main__":
    main()
