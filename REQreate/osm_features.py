"""OpenStreetMap features (bus stops, schools, ...) inside a network's area."""

import osmnx as ox


def features_in_area(place_name, tags, polygon=None):
    """
    Features with these tags inside the network polygon when given, else inside the place.

    The polygon is the area the street graphs were built from, so places removed with
    exclude_places (e.g. an island of a region) get no stops, schools or hospitals either.
    """
    if polygon is not None:
        return ox.features_from_polygon(polygon, tags=tags)
    return ox.features_from_place(place_name, tags=tags)
