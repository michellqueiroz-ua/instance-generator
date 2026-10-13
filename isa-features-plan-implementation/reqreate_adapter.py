"""
reqreate_adapter.py — T12 (ISA-HODBRP)

Convert REQreate generation output into the instance dict that
``compute_features`` expects (implementation_context.md §1).

REQreate writes, per generated instance:

  ``<location>/json_format/<sub>/<name>.json``
      ``requests``            dict keyed by request id
      ``travel_time_matrix``  square list-of-lists in **seconds**, indexed by
                              station index (row order of stations.csv)

  ``<location>/csv/<location>.stations.csv``
      one row per station: osmid_walk, osmid_drive, lat, lon, type

Unit and key mapping applied here:

  ==============================  ========================================
  REQreate field                  instance dict field
  ==============================  ========================================
  originx / originy               origin = {"x": lon, "y": lat}
  earliest_departure      [s]     tw_start [min]
  latest_departure        [s]     tw_end   [min]
  time_stamp              [s]     issue_time [min]
  direct_distance         [m]     direct_distance   (kept in metres)
  direct_travel_time      [s]     direct_travel_time (kept in seconds)
  stops_orgn[0]                   origin_idx       (station index)
  stops_dest[0]                   destination_idx  (station index)
  travel_time_matrix      [s]     travel_time_matrix_sec
  ==============================  ========================================

ODBRP writes ``earliest_departure`` / ``latest_departure`` / ``direct_*``;
DARP and Patient Transport write ``pickup_from`` / ``pickup_to`` /
``drivingDistance`` / ``drivingDuration`` for the same quantities, so both
naming families are accepted (see the ``_*_FIELDS`` alias lists below).

``originnode_drive`` / ``destinationnode_drive`` are deliberately **not**
forwarded: they are OSM node ids, and the travel-time matrix is indexed by
station index, so forwarding them would make every matrix lookup miss and
silently NaN out ``tw_tightness_factor``.
"""

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


LOGGER = logging.getLogger(__name__)

SECONDS_PER_MINUTE = 60.0
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_EQUATOR = 111.320

# Field aliases across REQreate problem types, most specific first.
# ODBRP uses the first name, DARP / Patient Transport the second.
_TW_START_FIELDS = ("earliest_departure", "pickup_from")
_TW_END_FIELDS = ("latest_departure", "pickup_to", "latest_arrival", "dropoff_to")
_DISTANCE_FIELDS = ("direct_distance", "drivingDistance")       # metres
_TRAVEL_TIME_FIELDS = ("direct_travel_time", "drivingDuration")  # seconds


# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------

def find_stations_csv(reference_path: Any) -> Optional[str]:
    """
    Locate ``<location>/csv/<location>.stations.csv`` from any path inside the
    location folder (typically the instance JSON path).

    Returns None when no stations file is found.
    """
    start = Path(str(reference_path)).resolve()
    candidates = [start] if start.is_dir() else []
    candidates.extend(start.parents)

    for parent in candidates:
        csv_dir = parent / "csv"
        if not csv_dir.is_dir():
            continue
        matches = sorted(csv_dir.glob("*.stations.csv"))
        if matches:
            return str(matches[0])

    return None


def load_stops(stations_csv_path: Any) -> List[Dict[str, Any]]:
    """
    Load stations.csv into the ``stops`` list shape used by the feature code.

    Each stop carries both its positional ``station_index`` (the key used by
    the travel-time matrix) and its OSM ids, so downstream lookups resolve
    whichever identifier a request uses.
    """
    from csv_loaders import load_stops as _load_stop_records

    records = _load_stop_records(str(stations_csv_path))
    return [
        {
            "id": record.station_index,
            "station_index": record.station_index,
            "osmid_drive": record.osmid_drive,
            "osmid_walk": record.osmid_walk,
            "x": record.lon,
            "y": record.lat,
        }
        for record in records
    ]


# ---------------------------------------------------------------------------
# Service area
# ---------------------------------------------------------------------------

def service_area_km2_from_points(points_lon_lat: Sequence[Tuple[float, float]]) -> float:
    """
    Area in km2 of the convex hull of the reachable network points (T01 policy).

    Degrees are converted to a local km frame before the area is measured, so
    the result is metric rather than "square degrees".  Falls back to the
    bounding-box area when shapely is unavailable or the hull degenerates.
    """
    points = [
        (float(lon), float(lat))
        for lon, lat in points_lon_lat
        if not (math.isnan(float(lon)) or math.isnan(float(lat)))
    ]
    if len(points) < 3:
        return float("nan")

    lats = [p[1] for p in points]
    lat0 = sum(lats) / len(lats)
    lon_km_per_deg = KM_PER_DEG_LON_EQUATOR * max(math.cos(math.radians(lat0)), 1e-6)
    lon0 = points[0][0]

    local = [
        ((lon - lon0) * lon_km_per_deg, (lat - lat0) * KM_PER_DEG_LAT)
        for lon, lat in points
    ]

    try:
        from shapely.geometry import MultiPoint

        hull = MultiPoint(local).convex_hull
        area = float(getattr(hull, "area", float("nan")))
        if area > 0.0 and math.isfinite(area):
            return area
    except Exception as exc:  # shapely missing or geometry degenerate
        LOGGER.info("service_area convex hull unavailable (%s), using bbox", exc)

    xs = [p[0] for p in local]
    ys = [p[1] for p in local]
    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    return area if area > 0.0 else float("nan")


def service_area_km2_from_graph(graph: Any) -> float:
    """
    Area in km2 covered by a drive graph's nodes (``inst.network.G_drive``).

    Returns NaN when the graph is missing or has no usable coordinates, which
    lets the feature code fall back to its documented bounding-box behaviour.
    """
    if graph is None:
        return float("nan")

    try:
        nodes = graph.nodes(data=True)
    except Exception:
        return float("nan")

    points: List[Tuple[float, float]] = []
    for _, data in nodes:
        if not isinstance(data, dict):
            continue
        lon = data.get("x")
        lat = data.get("y")
        if lon is None or lat is None:
            continue
        try:
            points.append((float(lon), float(lat)))
        except (TypeError, ValueError):
            continue

    if not points:
        return float("nan")

    return service_area_km2_from_points(points)


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _seconds_to_minutes(value: Any) -> float:
    seconds = _to_float(value)
    if math.isnan(seconds) or not math.isfinite(seconds):
        return float("nan")
    return seconds / SECONDS_PER_MINUTE


def _endpoint(raw: Dict[str, Any], prefix: str) -> Optional[Dict[str, float]]:
    """Build ``{"x": lon, "y": lat}`` from originx/originy or the [lat, lon] list."""
    lon = _to_float(raw.get(prefix + "x"))
    lat = _to_float(raw.get(prefix + "y"))

    if math.isnan(lon) or math.isnan(lat):
        pair = raw.get(prefix)
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            # REQreate stores the pair as [lat, lon].
            lat = _to_float(pair[0])
            lon = _to_float(pair[1])

    if math.isnan(lon) or math.isnan(lat):
        return None
    return {"x": lon, "y": lat}


def _first_present(raw: Dict[str, Any], fields: Sequence[str]) -> float:
    """Return the first field in ``fields`` that holds a usable number."""
    for name in fields:
        value = _to_float(raw.get(name))
        if not math.isnan(value) and math.isfinite(value):
            return value
    return float("nan")


def _first_station_index(value: Any) -> Optional[int]:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        return int(value[0])
    except (TypeError, ValueError):
        return None


def convert_request(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one REQreate request record to the feature-code request shape."""
    if not isinstance(raw, dict):
        return None

    request: Dict[str, Any] = {}

    origin = _endpoint(raw, "origin")
    destination = _endpoint(raw, "destination")
    if origin is not None:
        request["origin"] = origin
    if destination is not None:
        request["destination"] = destination

    tw_start = _seconds_to_minutes(_first_present(raw, _TW_START_FIELDS))
    tw_end = _seconds_to_minutes(_first_present(raw, _TW_END_FIELDS))
    if not math.isnan(tw_start):
        request["tw_start"] = tw_start
    if not math.isnan(tw_end):
        request["tw_end"] = tw_end

    issue_time = _seconds_to_minutes(raw.get("time_stamp"))
    if not math.isnan(issue_time):
        request["issue_time"] = issue_time

    # Kept in REQreate's own units; the feature code carries the unit hints.
    distance_m = _first_present(raw, _DISTANCE_FIELDS)
    if not math.isnan(distance_m):
        request["direct_distance"] = distance_m

    travel_time_s = _first_present(raw, _TRAVEL_TIME_FIELDS)
    if not math.isnan(travel_time_s):
        request["direct_travel_time"] = travel_time_s

    for field in (
        "stops_orgn",
        "stops_dest",
        "stops_orgn_walking_distance",
        "stops_dest_walking_distance",
    ):
        value = raw.get(field)
        if isinstance(value, (list, tuple)):
            request[field] = list(value)

    for field in ("walk_speed", "max_walking", "demand"):
        value = _to_float(raw.get(field))
        if not math.isnan(value):
            request[field] = value

    # Station indices drive travel-time matrix lookups (see module docstring).
    origin_idx = _first_station_index(request.get("stops_orgn"))
    destination_idx = _first_station_index(request.get("stops_dest"))
    if origin_idx is not None and destination_idx is not None:
        request["origin_idx"] = origin_idx
        request["destination_idx"] = destination_idx

    return request


def _iter_raw_requests(raw_requests: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_requests, dict):
        return [value for value in raw_requests.values() if isinstance(value, dict)]
    if isinstance(raw_requests, list):
        return [value for value in raw_requests if isinstance(value, dict)]
    return []


# ---------------------------------------------------------------------------
# Instance assembly
# ---------------------------------------------------------------------------

def build_instance_dict(
    raw_json: Dict[str, Any],
    stops: Optional[List[Dict[str, Any]]] = None,
    vehicles: Optional[List[Dict[str, Any]]] = None,
    service_area_km2: Optional[float] = None,
    walk_speed: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Assemble the feature-code instance dict from a parsed REQreate instance JSON.

    ``vehicles`` is optional: REQreate instance files describe demand only, so
    fleet features stay NaN unless the caller supplies the fleet used for the
    experiment.
    """
    if not isinstance(raw_json, dict):
        raise TypeError(f"raw_json must be a dict, got {type(raw_json).__name__}")

    requests = [
        converted
        for converted in (
            convert_request(raw) for raw in _iter_raw_requests(raw_json.get("requests"))
        )
        if converted
    ]

    instance: Dict[str, Any] = {
        "requests": requests,
        "vehicles": list(vehicles) if vehicles else [],
    }

    if stops:
        instance["stops"] = stops

    matrix = raw_json.get("travel_time_matrix")
    if isinstance(matrix, (list, dict)) and matrix:
        # REQreate writes seconds; the explicit key carries that unit hint.
        instance["travel_time_matrix_sec"] = matrix

    if service_area_km2 is not None:
        area = _to_float(service_area_km2)
        if not math.isnan(area) and math.isfinite(area) and area > 0.0:
            instance["service_area_km2"] = area

    if walk_speed is not None:
        speed = _to_float(walk_speed)
        if not math.isnan(speed) and speed > 0.0:
            instance["walk_speed"] = speed

    return instance


def load_reqreate_instance(
    json_path: Any,
    stations_csv_path: Optional[Any] = None,
    vehicles: Optional[List[Dict[str, Any]]] = None,
    service_area_km2: Optional[float] = None,
    walk_speed: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Load a generated REQreate instance JSON as a feature-ready instance dict.

    ``stations_csv_path`` is auto-discovered from the JSON location when not
    given.  A missing or unreadable stations file is not fatal: the spatial
    features fall back to NaN rather than blocking the export.
    """
    path = Path(str(json_path))
    with path.open("rb") as handle:
        raw_json = json.load(handle)

    if stations_csv_path is None:
        stations_csv_path = find_stations_csv(path)

    stops: List[Dict[str, Any]] = []
    if stations_csv_path:
        try:
            stops = load_stops(stations_csv_path)
        except Exception as exc:
            LOGGER.warning("stations.csv unreadable (%s): %s", stations_csv_path, exc)
    else:
        LOGGER.info("no stations.csv found near %s; spatial features will be NaN", path)

    if service_area_km2 is None and stops:
        service_area_km2 = service_area_km2_from_points(
            [(stop["x"], stop["y"]) for stop in stops]
        )
        if math.isnan(service_area_km2):
            service_area_km2 = None

    return build_instance_dict(
        raw_json,
        stops=stops,
        vehicles=vehicles,
        service_area_km2=service_area_km2,
        walk_speed=walk_speed,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_call() -> None:
    root = _THIS_DIR.parent
    candidates = sorted(root.glob("*/json_format/*.json"))
    if not candidates:
        print("No generated REQreate instance JSON found; skipping smoke call.")
        return

    json_path = candidates[0]
    instance = load_reqreate_instance(json_path)

    print(f"Loaded {json_path.name}")
    print(f"  requests : {len(instance['requests'])}")
    print(f"  stops    : {len(instance.get('stops', []))}")
    print(f"  ttm      : {'yes' if 'travel_time_matrix_sec' in instance else 'no'}")
    print(f"  area_km2 : {instance.get('service_area_km2')}")

    if instance["requests"]:
        sample = instance["requests"][0]
        print(f"  sample   : {json.dumps(sample, default=str)[:200]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _smoke_call()
