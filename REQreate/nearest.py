"""Batched nearest-node lookups for OpenStreetMap features.

``ox.nearest_edges`` / ``ox.nearest_nodes`` build a spatial index of the
whole graph on every call. Calling them once per bus stop, school or POI
rebuilt that index hundreds or thousands of times (over an hour for
Aachen's bus stops). These helpers make one call per graph for all points,
which gives the same answers.
"""

import numpy as np
import osmnx as ox
from osmnx.distance import great_circle


def centroid_coordinates(features):
    """(lats, lons) of each feature's centroid, in row order."""
    centroids = [geometry.centroid for geometry in features.geometry]
    return [c.y for c in centroids], [c.x for c in centroids]


def nearest_edge_endpoints(G, lats, lons):
    """For each point, the endpoint of its nearest edge closest to the point."""
    if len(lats) == 0:
        return []
    edges = ox.nearest_edges(G, X=np.asarray(lons, dtype=float), Y=np.asarray(lats, dtype=float))
    nodes = []
    for edge, lat, lon in zip(edges, lats, lons):
        u, v = edge[0], edge[1]
        nodes.append(min((u, v), key=lambda n: great_circle(lat, lon, G.nodes[n]['y'], G.nodes[n]['x'])))
    return nodes


_NODE_INDEX = {}


def nearest_node(G, lon, lat):
    """Nearest graph node of one point; same answer as ``ox.nearest_nodes(G, lon, lat)``.

    ``ox.nearest_nodes`` converts the whole graph to a GeoDataFrame and builds a
    new index on every call (0.8 s on the Capital Region drive graph). The index
    is built once per graph here and kept; request sampling asks for ~25 nodes
    per request.
    """
    if ox.projection.is_projected(G.graph["crs"]):
        return ox.nearest_nodes(G, lon, lat)
    entry = _NODE_INDEX.get(id(G))
    if entry is None or entry[0] is not G:  # the stored graph keeps its id from being reused
        from sklearn.neighbors import BallTree
        ids = np.array(list(G.nodes))
        coordinates = np.radians([[G.nodes[n]["y"], G.nodes[n]["x"]] for n in ids])
        entry = (G, ids, BallTree(coordinates, metric="haversine"))
        _NODE_INDEX[id(G)] = entry
    _, ids, tree = entry
    _, index = tree.query(np.radians([[lat, lon]]), k=1)
    return ids[index[0][0]]


def nearest_nodes(G, lats, lons):
    """Nearest graph node for each point."""
    if len(lats) == 0:
        return []
    return list(ox.nearest_nodes(G, X=np.asarray(lons, dtype=float), Y=np.asarray(lats, dtype=float)))
