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


def nearest_nodes(G, lats, lons):
    """Nearest graph node for each point."""
    if len(lats) == 0:
        return []
    return list(ox.nearest_nodes(G, X=np.asarray(lons, dtype=float), Y=np.asarray(lats, dtype=float)))
