"""Snapping points onto a street network, in batches rather than one at a time.

ox.nearest_edges rebuilds the graph's edge GeoDataFrame and an R-tree index on
every call, and building that frame means constructing one shapely LineString
per edge - roughly 39,000 of them for Aachen's walk network. Called once per
point, as three of the retrieval modules used to, that work is repeated for
every point: a profiled Aachen run spent 2,642 s inside 2,528 nearest_edges
calls, which between them created 98 million LineStrings.

osmnx is vectorised and says so in its own docstring - pass arrays and the
index is built once. Reading its source confirms the scalar path is the array
path with a one-element array, the same STRtree and the same query, so batching
returns exactly what the per-point calls returned.

The endpoint rule is the callers' own, kept here so all of them share one copy:
of the nearest edge's two endpoints, take whichever lies closer to the point as
the crow flies.

nearest_nodes has the same shape of problem: it builds a BallTree over every node
in the graph - 26,276 of them for Aachen's walk network - on each call, and the
POI retrieval called it once per POI, about 21,000 times in that run.
"""

import numpy as np
import osmnx as ox
from osmnx.distance import great_circle

from . import profiling

__all__ = ["nearest_edge_endpoints", "nearest_nodes"]


def nearest_edge_endpoints(G, xs, ys, label=""):
    """Snap each (x, y) to a node of its nearest edge. One index build, not len(xs).

    xs are longitudes and ys are latitudes, in the graph's CRS. Returns a list of
    node ids, one per point, in the order given.
    """
    xs = list(xs)
    ys = list(ys)
    if len(xs) != len(ys):
        raise ValueError(f"xs and ys must be the same length, got {len(xs)} and {len(ys)}")
    if not xs:
        return []

    with profiling.stage("snap.nearest_edges" + (("." + label) if label else "")):
        edges = ox.nearest_edges(G, xs, ys)
        nodes = G.nodes
        picked = []
        for (u, v, _key), x, y in zip(edges, xs, ys):
            picked.append(
                min((u, v), key=lambda n: great_circle(y, x, nodes[n]["y"], nodes[n]["x"]))
            )
        return picked


def nearest_nodes(G, xs, ys, label=""):
    """Snap each (x, y) to the graph's nearest node. One index build, not len(xs).

    xs are longitudes and ys are latitudes, in the graph's CRS. Returns a list of
    node ids as plain ints, one per point, in the order given - which is what the
    scalar form of ox.nearest_nodes returns, while its vectorised form hands back
    numpy int64.
    """
    xs = list(xs)
    ys = list(ys)
    if len(xs) != len(ys):
        raise ValueError(f"xs and ys must be the same length, got {len(xs)} and {len(ys)}")
    if not xs:
        return []

    with profiling.stage("snap.nearest_nodes" + (("." + label) if label else "")):
        return [int(n) for n in np.atleast_1d(ox.nearest_nodes(G, xs, ys))]
