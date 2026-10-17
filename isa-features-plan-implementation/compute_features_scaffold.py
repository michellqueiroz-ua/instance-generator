import math
import logging
import heapq
import ast
from typing import Any, Dict, Iterable, List, Optional, Tuple


LOGGER = logging.getLogger(__name__)

# Fallback policy for unreliable/missing service area.
MIN_FALLBACK_SERVICE_AREA_KM2 = 1e-6
SERVICE_AREA_RATIO_MIN = 0.05
SERVICE_AREA_RATIO_MAX = 20.0
TIGHT_TW_THRESHOLD_MIN = 20.0
TRAVEL_TIME_INFER_SECONDS_THRESHOLD = 300.0
DEFAULT_WALK_THRESHOLD_MIN = 10.0
DEFAULT_WALK_SPEED_MPS = 1.38889
DEFAULT_COVERAGE_GRID_N = 50
MIN_COVERAGE_GRID_N = 5
MAX_COVERAGE_GRID_N = 120
MAX_COVERAGE_DISTANCE_CHECKS = 2_000_000
TIME_INFER_SECONDS_THRESHOLD = 1000.0


FEATURE_KEYS: List[str] = [
    # Demand
    "n_requests",
    "demand_density",
    "spatial_cluster_coeff",
    "od_distance_mean",
    "od_distance_std",
    # Time window
    "tw_width_mean",
    "tw_width_std",
    "tw_overlap_ratio",
    "tight_tw_fraction",
    "tw_tightness_factor",
    # Fleet
    "fleet_size",
    "capacity_cv",
    "fleet_utilization_ratio",
    "request_vehicle_ratio",
    # Spatial / network
    "station_density",
    "bsa_flexibility",
    "avg_station_degree",
    "coverage_ratio",
    # Dynamic
    "request_flux_dynamism",
    "static_to_dynamic_ratio",
    "avg_request_urgency",
]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def to_float_or_nan(value: Any) -> float:
    """Safely cast values to float, returning NaN for invalid values."""
    if _is_number(value):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_mean(values: Iterable[Any]) -> float:
    """Return mean for numeric values, otherwise NaN for empty/invalid input."""
    nums = [to_float_or_nan(v) for v in values]
    nums = [v for v in nums if not math.isnan(v)]
    if not nums:
        return float("nan")
    return float(sum(nums) / len(nums))


def safe_std(values: Iterable[Any]) -> float:
    """Return population std for numeric values, otherwise NaN for empty/invalid input."""
    nums = [to_float_or_nan(v) for v in values]
    nums = [v for v in nums if not math.isnan(v)]
    if not nums:
        return float("nan")
    mu = safe_mean(nums)
    if math.isnan(mu):
        return float("nan")
    var = sum((x - mu) ** 2 for x in nums) / len(nums)
    return float(math.sqrt(var))


def get_field_or_default(obj: Dict[str, Any], field: str, default: Any) -> Any:
    """Dictionary guard for missing-field access."""
    if not isinstance(obj, dict):
        return default
    return obj.get(field, default)


def validate_instance_schema(instance: Dict[str, Any]) -> List[str]:
    """Basic schema checks used by scaffold-level validation."""
    errors: List[str] = []

    if not isinstance(instance, dict):
        return ["instance must be a dict"]

    requests = get_field_or_default(instance, "requests", [])
    if not isinstance(requests, list):
        errors.append("requests must be a list")

    vehicles = get_field_or_default(instance, "vehicles", [])
    if not isinstance(vehicles, list):
        errors.append("vehicles must be a list")

    if "service_area_km2" in instance and not _is_number(instance["service_area_km2"]):
        errors.append("service_area_km2 must be numeric when provided")

    return errors


def _extract_lon_lat(point: Any) -> Optional[Tuple[float, float]]:
    """Parse point coordinates as (lon, lat) from supported point shapes."""
    if isinstance(point, dict):
        lon = to_float_or_nan(point.get("x"))
        lat = to_float_or_nan(point.get("y"))
    elif isinstance(point, (list, tuple)) and len(point) >= 2:
        # Legacy CSV rows often encode endpoints as [lat, lon].
        lat = to_float_or_nan(point[0])
        lon = to_float_or_nan(point[1])
    else:
        return None

    if math.isnan(lon) or math.isnan(lat):
        return None

    return (lon, lat)


def _collect_request_points(
    requests: List[Dict[str, Any]], fields: Tuple[str, ...]
) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for req in requests:
        if not isinstance(req, dict):
            continue
        for field in fields:
            maybe_point = _extract_lon_lat(req.get(field))
            if maybe_point is not None:
                points.append(maybe_point)
    return points


def _bbox_area_km2(points_lon_lat: List[Tuple[float, float]]) -> float:
    """Approximate area from coordinate envelope in km2 (lon/lat aware)."""
    if not points_lon_lat:
        return float("nan")

    lons = [p[0] for p in points_lon_lat]
    lats = [p[1] for p in points_lon_lat]

    lon_span = max(lons) - min(lons)
    lat_span = max(lats) - min(lats)
    mean_lat_rad = math.radians((max(lats) + min(lats)) / 2.0)

    # Simple geographic conversion good enough for fallback behavior.
    lon_km = lon_span * (111.320 * max(math.cos(mean_lat_rad), 1e-6))
    lat_km = lat_span * 110.574

    return max(lon_km * lat_km, MIN_FALLBACK_SERVICE_AREA_KM2)


def _resolve_service_area_km2(
    instance: Dict[str, Any], points_lon_lat: List[Tuple[float, float]]
) -> Tuple[float, Optional[str]]:
    """
    Resolve service area with fallback.

    Returns
    -------
    (area_km2, fallback_reason)
        fallback_reason is None when raw service_area_km2 is accepted.
    """
    raw_area = to_float_or_nan(get_field_or_default(instance, "service_area_km2", float("nan")))
    bbox_area = _bbox_area_km2(points_lon_lat)

    if not math.isnan(raw_area) and math.isfinite(raw_area) and raw_area > 0.0:
        if not math.isnan(bbox_area):
            ratio = raw_area / bbox_area
            if ratio < SERVICE_AREA_RATIO_MIN or ratio > SERVICE_AREA_RATIO_MAX:
                return bbox_area, "bbox_mismatch"
        return raw_area, None

    if not math.isnan(bbox_area):
        if "service_area_km2" not in instance:
            return bbox_area, "missing"
        return bbox_area, "invalid"

    return float("nan"), "unavailable"


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in km."""
    r = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _to_distance_km(value: Any, unit_hint: Optional[str] = None) -> float:
    """Normalize a distance value to km, returning NaN for invalid values."""
    dist = to_float_or_nan(value)
    if math.isnan(dist) or not math.isfinite(dist) or dist <= 0.0:
        return float("nan")

    if unit_hint == "km":
        return dist
    if unit_hint == "m":
        return dist / 1000.0

    # Conservative auto-inference for unknown units.
    return dist / 1000.0 if dist > 100.0 else dist


def _to_lookup_key(value: Any) -> Optional[Any]:
    """Cast endpoint identifiers to stable lookup keys (int if possible)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value

    as_float = to_float_or_nan(value)
    if not math.isnan(as_float):
        if abs(as_float - round(as_float)) < 1e-9:
            return int(round(as_float))
        return as_float

    if isinstance(value, str) and value.strip():
        return value.strip()

    return None


def _extract_endpoint_pair(req: Dict[str, Any]) -> Optional[Tuple[Any, Any]]:
    """Extract origin/destination identifiers for matrix/network distance lookups."""
    candidate_pairs = [
        ("originnode_drive", "destinationnode_drive"),
        ("origin_node_drive", "destination_node_drive"),
        ("origin_osmid_drive", "destination_osmid_drive"),
        ("originid", "destinationid"),
        ("origin_id", "destination_id"),
        ("origin_idx", "destination_idx"),
        ("origin_index", "destination_index"),
        ("origin_stop", "destination_stop"),
    ]

    for origin_key, destination_key in candidate_pairs:
        if origin_key in req and destination_key in req:
            origin = _to_lookup_key(req.get(origin_key))
            destination = _to_lookup_key(req.get(destination_key))
            if origin is not None and destination is not None:
                return (origin, destination)

    return None


def _resolve_drive_distance_matrix(instance: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    """Return a driving-distance matrix object and unit hint when available."""
    matrix_specs = [
        ("drive_distance_matrix_km", "km"),
        ("driving_distance_matrix_km", "km"),
        ("distance_matrix_km", "km"),
        ("drive_distance_matrix_m", "m"),
        ("driving_distance_matrix_m", "m"),
        ("distance_matrix_m", "m"),
        ("drive_distance_matrix", None),
        ("driving_distance_matrix", None),
        ("distance_matrix", None),
    ]

    for key, unit in matrix_specs:
        value = instance.get(key)
        if isinstance(value, (dict, list)):
            return value, unit

    road_network = instance.get("road_network")
    if isinstance(road_network, dict):
        matrix = road_network.get("distance_matrix")
        if isinstance(matrix, (dict, list)):
            return matrix, None

    return None, None


def _lookup_distance_in_matrix(matrix: Any, origin_key: Any, destination_key: Any) -> float:
    """Lookup raw distance value in a matrix represented as dict/list."""
    if isinstance(matrix, list):
        oi = _to_lookup_key(origin_key)
        di = _to_lookup_key(destination_key)
        if not isinstance(oi, int) or not isinstance(di, int):
            return float("nan")
        if oi < 0 or di < 0 or oi >= len(matrix):
            return float("nan")
        row = matrix[oi]
        if not isinstance(row, list) or di >= len(row):
            return float("nan")
        return to_float_or_nan(row[di])

    if not isinstance(matrix, dict):
        return float("nan")

    direct_pair_keys = [
        f"{origin_key}:{destination_key}",
        f"{destination_key}:{origin_key}",
    ]
    for pair_key in direct_pair_keys:
        if pair_key in matrix:
            return to_float_or_nan(matrix[pair_key])

    row_key_candidates = [origin_key, str(origin_key)]
    col_key_candidates = [destination_key, str(destination_key)]
    for row_key in row_key_candidates:
        row = matrix.get(row_key)
        if isinstance(row, dict):
            for col_key in col_key_candidates:
                if col_key in row:
                    return to_float_or_nan(row[col_key])

    rev_row_key_candidates = [destination_key, str(destination_key)]
    rev_col_key_candidates = [origin_key, str(origin_key)]
    for row_key in rev_row_key_candidates:
        row = matrix.get(row_key)
        if isinstance(row, dict):
            for col_key in rev_col_key_candidates:
                if col_key in row:
                    return to_float_or_nan(row[col_key])

    return float("nan")


def _build_road_graph_adjacency(instance: Dict[str, Any]) -> Dict[Any, List[Tuple[Any, float]]]:
    """Build directed adjacency list from road_network edges using distance attributes."""
    road_network = instance.get("road_network")
    if not isinstance(road_network, dict):
        return {}

    edges = road_network.get("edges")
    if not isinstance(edges, list):
        return {}

    adjacency: Dict[Any, List[Tuple[Any, float]]] = {}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        u = _to_lookup_key(edge.get("u"))
        v = _to_lookup_key(edge.get("v"))
        if u is None or v is None:
            continue

        dist_km = float("nan")
        if "distance_km" in edge:
            dist_km = _to_distance_km(edge.get("distance_km"), "km")
        elif "distance_m" in edge:
            dist_km = _to_distance_km(edge.get("distance_m"), "m")
        elif "length_m" in edge:
            dist_km = _to_distance_km(edge.get("length_m"), "m")
        elif "length" in edge:
            dist_km = _to_distance_km(edge.get("length"), "m")

        if math.isnan(dist_km):
            continue

        adjacency.setdefault(u, []).append((v, dist_km))

    return adjacency


def _dijkstra_distance_km(
    adjacency: Dict[Any, List[Tuple[Any, float]]], source: Any, target: Any
) -> float:
    """Single-pair shortest path in km over directed adjacency list."""
    if source == target:
        return 0.0
    if source not in adjacency:
        return float("nan")

    pq: List[Tuple[float, Any]] = [(0.0, source)]
    best: Dict[Any, float] = {source: 0.0}

    while pq:
        dist_u, u = heapq.heappop(pq)
        if dist_u > best.get(u, float("inf")):
            continue
        if u == target:
            return dist_u

        for v, w in adjacency.get(u, []):
            cand = dist_u + w
            if cand < best.get(v, float("inf")):
                best[v] = cand
                heapq.heappush(pq, (cand, v))

    return float("nan")


def _drive_distance_from_request(req: Dict[str, Any]) -> float:
    """Extract per-request driving distance from known request-level fields."""
    if "direct_distance" in req:
        return _to_distance_km(req.get("direct_distance"), "m")
    if "direct_distance_m" in req:
        return _to_distance_km(req.get("direct_distance_m"), "m")
    if "distance_m" in req:
        return _to_distance_km(req.get("distance_m"), "m")
    if "direct_distance_km" in req:
        return _to_distance_km(req.get("direct_distance_km"), "km")
    if "distance_km" in req:
        return _to_distance_km(req.get("distance_km"), "km")

    return float("nan")


def _compute_od_distance_features(instance: Dict[str, Any]) -> Dict[str, float]:
    requests = get_field_or_default(instance, "requests", [])
    if not isinstance(requests, list):
        return {
            "od_distance_mean": float("nan"),
            "od_distance_std": float("nan"),
        }

    matrix, matrix_unit_hint = _resolve_drive_distance_matrix(instance)
    adjacency = _build_road_graph_adjacency(instance)
    pair_cache: Dict[Tuple[Any, Any], float] = {}

    distances_km: List[float] = []
    fallback_count = 0
    total_reqs = 0

    for req in requests:
        if not isinstance(req, dict):
            continue
        total_reqs += 1

        distance_km = float("nan")

        endpoint_pair = _extract_endpoint_pair(req)
        if endpoint_pair is not None and matrix is not None:
            raw = _lookup_distance_in_matrix(matrix, endpoint_pair[0], endpoint_pair[1])
            distance_km = _to_distance_km(raw, matrix_unit_hint)

        if math.isnan(distance_km):
            distance_km = _drive_distance_from_request(req)

        if math.isnan(distance_km) and endpoint_pair is not None and adjacency:
            cached = pair_cache.get(endpoint_pair)
            if cached is None:
                cached = _dijkstra_distance_km(adjacency, endpoint_pair[0], endpoint_pair[1])
                pair_cache[endpoint_pair] = cached
            distance_km = cached

        if math.isnan(distance_km):
            origin_point = _extract_lon_lat(req.get("origin"))
            destination_point = _extract_lon_lat(req.get("destination"))
            if origin_point is not None and destination_point is not None:
                distance_km = _haversine_km(
                    origin_point[0],
                    origin_point[1],
                    destination_point[0],
                    destination_point[1],
                )
                fallback_count += 1

        if not math.isnan(distance_km) and math.isfinite(distance_km):
            distances_km.append(distance_km)

    if fallback_count > 0:
        LOGGER.info(
            "od_distance fallback used (haversine %d/%d requests)",
            fallback_count,
            total_reqs,
        )

    return {
        "od_distance_mean": safe_mean(distances_km),
        "od_distance_std": safe_std(distances_km),
    }


def _extract_valid_time_windows(requests: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """Return (start, end) windows for well-formed requests only."""
    windows: List[Tuple[float, float]] = []
    for req in requests:
        if not isinstance(req, dict):
            continue

        tw_start = to_float_or_nan(req.get("tw_start"))
        tw_end = to_float_or_nan(req.get("tw_end"))
        if math.isnan(tw_start) or math.isnan(tw_end):
            continue
        if not math.isfinite(tw_start) or not math.isfinite(tw_end):
            continue
        if tw_end < tw_start:
            continue

        windows.append((tw_start, tw_end))

    return windows


def _compute_tw_overlap_ratio(time_windows: List[Tuple[float, float]]) -> float:
    """Fraction of request pairs with overlapping windows."""
    n = len(time_windows)
    if n < 2:
        return float("nan")

    total_pairs = n * (n - 1) / 2
    overlap_pairs = 0
    for i in range(n):
        start_i, end_i = time_windows[i]
        for j in range(i + 1, n):
            start_j, end_j = time_windows[j]
            if max(start_i, start_j) <= min(end_i, end_j):
                overlap_pairs += 1

    return overlap_pairs / total_pairs


def _compute_time_window_features(instance: Dict[str, Any]) -> Dict[str, float]:
    requests = get_field_or_default(instance, "requests", [])
    if not isinstance(requests, list):
        return {
            "tw_width_mean": float("nan"),
            "tw_width_std": float("nan"),
            "tw_overlap_ratio": float("nan"),
            "tight_tw_fraction": float("nan"),
        }

    time_windows = _extract_valid_time_windows(requests)
    widths = [tw_end - tw_start for tw_start, tw_end in time_windows]

    tw_width_mean = safe_mean(widths)
    tw_width_std = safe_std(widths)
    tw_overlap_ratio = _compute_tw_overlap_ratio(time_windows)

    tight_tw_fraction = float("nan")
    if widths:
        tight_count = sum(1 for width in widths if width < TIGHT_TW_THRESHOLD_MIN)
        tight_tw_fraction = tight_count / len(widths)

    return {
        "tw_width_mean": tw_width_mean,
        "tw_width_std": tw_width_std,
        "tw_overlap_ratio": tw_overlap_ratio,
        "tight_tw_fraction": tight_tw_fraction,
    }


def _to_travel_time_min(value: Any, unit_hint: Optional[str] = None) -> float:
    """Normalize travel time to minutes, returning NaN for invalid values."""
    travel_time = to_float_or_nan(value)
    if math.isnan(travel_time) or not math.isfinite(travel_time):
        return float("nan")
    if travel_time <= 0.0:
        return float("nan")

    if unit_hint == "min":
        return travel_time
    if unit_hint == "sec":
        return travel_time / 60.0

    # Conservative default for unknown-unit values.
    if travel_time > TRAVEL_TIME_INFER_SECONDS_THRESHOLD:
        return travel_time / 60.0
    return travel_time


def _resolve_drive_travel_time_matrix(instance: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    """Return a driving travel-time matrix and unit hint when available."""
    matrix_specs = [
        ("drive_travel_time_matrix_min", "min"),
        ("driving_travel_time_matrix_min", "min"),
        ("travel_time_matrix_min", "min"),
        ("drive_travel_time_matrix_sec", "sec"),
        ("driving_travel_time_matrix_sec", "sec"),
        ("travel_time_matrix_sec", "sec"),
        ("drive_travel_time_matrix", None),
        ("driving_travel_time_matrix", None),
        ("travel_time_matrix", None),
    ]

    for key, unit in matrix_specs:
        value = instance.get(key)
        if isinstance(value, (dict, list)):
            return value, unit

    road_network = instance.get("road_network")
    if isinstance(road_network, dict):
        matrix = road_network.get("travel_time_matrix")
        if isinstance(matrix, (dict, list)):
            return matrix, None

    return None, None


def _lookup_travel_time_in_matrix(matrix: Any, origin_key: Any, destination_key: Any) -> float:
    """Lookup raw travel-time value in a matrix represented as dict/list."""
    if isinstance(matrix, list):
        oi = _to_lookup_key(origin_key)
        di = _to_lookup_key(destination_key)
        if not isinstance(oi, int) or not isinstance(di, int):
            return float("nan")
        if oi < 0 or di < 0 or oi >= len(matrix):
            return float("nan")
        row = matrix[oi]
        if not isinstance(row, list) or di >= len(row):
            return float("nan")
        return to_float_or_nan(row[di])

    if not isinstance(matrix, dict):
        return float("nan")

    direct_pair_keys = [
        f"{origin_key}:{destination_key}",
        f"{destination_key}:{origin_key}",
    ]
    for pair_key in direct_pair_keys:
        if pair_key in matrix:
            return to_float_or_nan(matrix[pair_key])

    row_key_candidates = [origin_key, str(origin_key)]
    col_key_candidates = [destination_key, str(destination_key)]
    for row_key in row_key_candidates:
        row = matrix.get(row_key)
        if isinstance(row, dict):
            for col_key in col_key_candidates:
                if col_key in row:
                    return to_float_or_nan(row[col_key])

    rev_row_key_candidates = [destination_key, str(destination_key)]
    rev_col_key_candidates = [origin_key, str(origin_key)]
    for row_key in rev_row_key_candidates:
        row = matrix.get(row_key)
        if isinstance(row, dict):
            for col_key in rev_col_key_candidates:
                if col_key in row:
                    return to_float_or_nan(row[col_key])

    return float("nan")


def _build_road_graph_time_adjacency(instance: Dict[str, Any]) -> Dict[Any, List[Tuple[Any, float]]]:
    """Build directed adjacency list from road_network edges using travel-time attributes."""
    road_network = instance.get("road_network")
    if not isinstance(road_network, dict):
        return {}

    edges = road_network.get("edges")
    if not isinstance(edges, list):
        return {}

    adjacency: Dict[Any, List[Tuple[Any, float]]] = {}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        u = _to_lookup_key(edge.get("u"))
        v = _to_lookup_key(edge.get("v"))
        if u is None or v is None:
            continue

        time_min = float("nan")
        if "travel_time_min" in edge:
            time_min = _to_travel_time_min(edge.get("travel_time_min"), "min")
        elif "travel_time_sec" in edge:
            time_min = _to_travel_time_min(edge.get("travel_time_sec"), "sec")
        elif "travel_time" in edge:
            time_min = _to_travel_time_min(edge.get("travel_time"), None)

        if math.isnan(time_min):
            continue

        adjacency.setdefault(u, []).append((v, time_min))

    return adjacency


def _dijkstra_travel_time_min(
    adjacency: Dict[Any, List[Tuple[Any, float]]], source: Any, target: Any
) -> float:
    """Single-pair shortest travel time (minutes) over directed adjacency list."""
    if source == target:
        return 0.0
    if source not in adjacency:
        return float("nan")

    pq: List[Tuple[float, Any]] = [(0.0, source)]
    best: Dict[Any, float] = {source: 0.0}

    while pq:
        dist_u, u = heapq.heappop(pq)
        if dist_u > best.get(u, float("inf")):
            continue
        if u == target:
            return dist_u

        for v, w in adjacency.get(u, []):
            cand = dist_u + w
            if cand < best.get(v, float("inf")):
                best[v] = cand
                heapq.heappush(pq, (cand, v))

    return float("nan")


def _drive_travel_time_from_request(req: Dict[str, Any]) -> float:
    """Extract request-level driving travel time and normalize to minutes."""
    if "direct_travel_time" in req:
        return _to_travel_time_min(req.get("direct_travel_time"), "sec")
    if "direct_travel_time_sec" in req:
        return _to_travel_time_min(req.get("direct_travel_time_sec"), "sec")
    if "travel_time_sec" in req:
        return _to_travel_time_min(req.get("travel_time_sec"), "sec")
    if "direct_travel_time_min" in req:
        return _to_travel_time_min(req.get("direct_travel_time_min"), "min")
    if "travel_time_min" in req:
        return _to_travel_time_min(req.get("travel_time_min"), "min")

    return float("nan")


def _compute_tw_tightness_factor(instance: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute mean(tw_width / min_trip_time) with driving travel-time priority.

    Policy:
    - Primary source: driving travel-time matrix or road-network edge travel time.
    - If primary source exists, invalid/non-positive rows are skipped.
    - Fallback (request-level direct_travel_time fields) is used only when
      primary source is missing entirely.
    """
    requests = get_field_or_default(instance, "requests", [])
    if not isinstance(requests, list):
        return {"tw_tightness_factor": float("nan")}

    matrix, matrix_unit_hint = _resolve_drive_travel_time_matrix(instance)
    adjacency = _build_road_graph_time_adjacency(instance)
    has_primary_source = (matrix is not None) or bool(adjacency)
    pair_cache: Dict[Tuple[Any, Any], float] = {}

    factors: List[float] = []
    fallback_count = 0
    candidate_rows = 0

    for req in requests:
        if not isinstance(req, dict):
            continue

        tw_start = to_float_or_nan(req.get("tw_start"))
        tw_end = to_float_or_nan(req.get("tw_end"))
        if math.isnan(tw_start) or math.isnan(tw_end):
            continue
        if not math.isfinite(tw_start) or not math.isfinite(tw_end):
            continue
        if tw_end < tw_start:
            continue

        candidate_rows += 1
        tw_width = tw_end - tw_start

        min_trip_time = float("nan")
        endpoint_pair = _extract_endpoint_pair(req)

        if has_primary_source:
            if endpoint_pair is not None and matrix is not None:
                raw = _lookup_travel_time_in_matrix(matrix, endpoint_pair[0], endpoint_pair[1])
                min_trip_time = _to_travel_time_min(raw, matrix_unit_hint)

            if math.isnan(min_trip_time) and endpoint_pair is not None and adjacency:
                cached = pair_cache.get(endpoint_pair)
                if cached is None:
                    cached = _dijkstra_travel_time_min(adjacency, endpoint_pair[0], endpoint_pair[1])
                    pair_cache[endpoint_pair] = cached
                min_trip_time = cached
        else:
            min_trip_time = _drive_travel_time_from_request(req)
            if not math.isnan(min_trip_time):
                fallback_count += 1

        if math.isnan(min_trip_time) or min_trip_time <= 0.0:
            continue

        factors.append(tw_width / min_trip_time)

    if not has_primary_source and fallback_count > 0:
        LOGGER.info(
            "tw_tightness fallback used (request_travel_time %d/%d rows)",
            fallback_count,
            candidate_rows,
        )

    return {"tw_tightness_factor": safe_mean(factors)}


def _compute_fleet_features(instance: Dict[str, Any]) -> Dict[str, float]:
    vehicles = get_field_or_default(instance, "vehicles", [])
    requests = get_field_or_default(instance, "requests", [])

    if not isinstance(vehicles, list):
        return {
            "fleet_size": float("nan"),
            "capacity_cv": float("nan"),
            "fleet_utilization_ratio": float("nan"),
            "request_vehicle_ratio": float("nan"),
        }

    if not isinstance(requests, list):
        n_requests = float("nan")
    else:
        n_requests = float(len(requests))

    fleet_size = float(len(vehicles))

    capacities: List[float] = []
    for vehicle in vehicles:
        if not isinstance(vehicle, dict):
            continue
        capacity = to_float_or_nan(vehicle.get("capacity"))
        if math.isnan(capacity) or not math.isfinite(capacity) or capacity <= 0.0:
            continue
        capacities.append(capacity)

    capacity_mean = safe_mean(capacities)
    capacity_std = safe_std(capacities)
    capacity_cv = float("nan")
    if not math.isnan(capacity_mean) and capacity_mean > 0.0 and not math.isnan(capacity_std):
        capacity_cv = capacity_std / capacity_mean

    fleet_utilization_ratio = float("nan")
    if not math.isnan(n_requests) and n_requests > 0.0:
        fleet_utilization_ratio = sum(capacities) / n_requests

    request_vehicle_ratio = float("nan")
    if not math.isnan(n_requests) and fleet_size > 0.0:
        request_vehicle_ratio = n_requests / fleet_size

    return {
        "fleet_size": fleet_size,
        "capacity_cv": capacity_cv,
        "fleet_utilization_ratio": fleet_utilization_ratio,
        "request_vehicle_ratio": request_vehicle_ratio,
    }


def _parse_maybe_list(value: Any) -> Optional[List[Any]]:
    """Parse list-like values from native lists/tuples or CSV-style string literals."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
    return None


def _count_stations_from_instance(instance: Dict[str, Any]) -> float:
    """Count available stations from stops list or CSV-derived request stop-id lists."""
    stops = instance.get("stops")
    if isinstance(stops, list):
        return float(len(stops))

    requests = get_field_or_default(instance, "requests", [])
    if not isinstance(requests, list):
        return float("nan")

    unique_stop_ids = set()
    for req in requests:
        if not isinstance(req, dict):
            continue
        for field in ("stops_orgn", "stops_dest"):
            parsed = _parse_maybe_list(req.get(field))
            if not parsed:
                continue
            for raw_stop_id in parsed:
                stop_id = _to_lookup_key(raw_stop_id)
                if stop_id is not None:
                    unique_stop_ids.add(stop_id)

    if unique_stop_ids:
        return float(len(unique_stop_ids))
    return float("nan")


def _extract_stop_coord_lookup(instance: Dict[str, Any]) -> Dict[Any, Tuple[float, float]]:
    """Build stop-id -> (lon, lat) lookup from instance['stops'] when available."""
    stops = instance.get("stops")
    if not isinstance(stops, list):
        return {}

    lookup: Dict[Any, Tuple[float, float]] = {}
    for idx, stop in enumerate(stops):
        lon = float("nan")
        lat = float("nan")

        if isinstance(stop, dict):
            lon = to_float_or_nan(stop.get("x", stop.get("lon")))
            lat = to_float_or_nan(stop.get("y", stop.get("lat")))
            stop_ids = [
                stop.get("id"),
                stop.get("station_index"),
                stop.get("index"),
                stop.get("osmid_drive"),
                stop.get("osmid_walk"),
                idx,
            ]
        else:
            lon = to_float_or_nan(getattr(stop, "x", getattr(stop, "lon", float("nan"))))
            lat = to_float_or_nan(getattr(stop, "y", getattr(stop, "lat", float("nan"))))
            stop_ids = [
                getattr(stop, "id", None),
                getattr(stop, "station_index", None),
                getattr(stop, "index", None),
                getattr(stop, "osmid_drive", None),
                getattr(stop, "osmid_walk", None),
                idx,
            ]

        if math.isnan(lon) or math.isnan(lat):
            continue

        for raw_id in stop_ids:
            stop_id = _to_lookup_key(raw_id)
            if stop_id is not None:
                lookup[stop_id] = (lon, lat)

    return lookup


def _estimate_walk_threshold_m(req: Dict[str, Any], instance: Dict[str, Any]) -> Tuple[float, bool]:
    """Return per-request walking distance threshold in meters and speed-fallback flag."""
    speed_candidates = [
        req.get("walk_speed"),
        req.get("walk_speed_mps"),
        instance.get("walk_speed"),
        instance.get("walk_speed_mps"),
    ]
    speed_mps = float("nan")
    for candidate in speed_candidates:
        speed_mps = to_float_or_nan(candidate)
        if not math.isnan(speed_mps) and math.isfinite(speed_mps) and speed_mps > 0.0:
            return DEFAULT_WALK_THRESHOLD_MIN * 60.0 * speed_mps, False

    return DEFAULT_WALK_THRESHOLD_MIN * 60.0 * DEFAULT_WALK_SPEED_MPS, True


def _count_candidates_for_endpoint(
    req: Dict[str, Any],
    instance: Dict[str, Any],
    endpoint_key: str,
    stop_id_field: str,
    stop_dist_field: str,
    stop_coord_lookup: Dict[Any, Tuple[float, float]],
) -> Tuple[Optional[int], bool]:
    """Count candidate stops for one endpoint; returns (count, used_speed_fallback)."""
    threshold_m, used_speed_fallback = _estimate_walk_threshold_m(req, instance)

    distances = _parse_maybe_list(req.get(stop_dist_field))
    if distances:
        count = 0
        for d in distances:
            d_m = to_float_or_nan(d)
            if math.isnan(d_m) or not math.isfinite(d_m) or d_m < 0.0:
                continue
            if d_m <= threshold_m:
                count += 1
        return count, used_speed_fallback

    stop_ids_raw = _parse_maybe_list(req.get(stop_id_field))
    if not stop_ids_raw:
        return None, used_speed_fallback

    endpoint = _extract_lon_lat(req.get(endpoint_key))
    if endpoint is not None and stop_coord_lookup:
        threshold_km = threshold_m / 1000.0
        count = 0
        for raw_id in stop_ids_raw:
            stop_id = _to_lookup_key(raw_id)
            if stop_id is None:
                continue
            coords = stop_coord_lookup.get(stop_id)
            if coords is None:
                continue
            if _haversine_km(endpoint[0], endpoint[1], coords[0], coords[1]) <= threshold_km:
                count += 1
        return count, used_speed_fallback

    # Last-resort CSV fallback when we only have candidate IDs.
    count = 0
    for raw_id in stop_ids_raw:
        if _to_lookup_key(raw_id) is not None:
            count += 1
    return count, used_speed_fallback


def _compute_station_density(instance: Dict[str, Any]) -> float:
    requests = get_field_or_default(instance, "requests", [])
    points = _collect_request_points(requests, ("origin", "destination")) if isinstance(requests, list) else []
    service_area_km2, _ = _resolve_service_area_km2(instance, points)

    station_count = _count_stations_from_instance(instance)
    if math.isnan(station_count) or math.isnan(service_area_km2) or service_area_km2 <= 0.0:
        return float("nan")

    return station_count / service_area_km2


def _compute_bsa_flexibility(instance: Dict[str, Any]) -> float:
    requests = get_field_or_default(instance, "requests", [])
    if not isinstance(requests, list):
        return float("nan")

    stop_coord_lookup = _extract_stop_coord_lookup(instance)

    counts: List[int] = []
    fallback_speed_rows = 0
    for req in requests:
        if not isinstance(req, dict):
            continue

        origin_count, origin_fallback_speed = _count_candidates_for_endpoint(
            req,
            instance,
            "origin",
            "stops_orgn",
            "stops_orgn_walking_distance",
            stop_coord_lookup,
        )
        if origin_count is not None:
            counts.append(origin_count)
            if origin_fallback_speed:
                fallback_speed_rows += 1

        dest_count, dest_fallback_speed = _count_candidates_for_endpoint(
            req,
            instance,
            "destination",
            "stops_dest",
            "stops_dest_walking_distance",
            stop_coord_lookup,
        )
        if dest_count is not None:
            counts.append(dest_count)
            if dest_fallback_speed:
                fallback_speed_rows += 1

    if not counts:
        return float("nan")

    if fallback_speed_rows > 0:
        LOGGER.info(
            "bsa_flexibility walk-speed fallback used (%d endpoints)",
            fallback_speed_rows,
        )

    bsa = safe_mean(counts)
    if not math.isnan(bsa) and abs(bsa - 1.0) < 1e-9:
        LOGGER.info(
            "bsa_flexibility=1.0 indicates one accessible stop per endpoint (no stop choice)"
        )

    return bsa


def _compute_avg_station_degree(instance: Dict[str, Any]) -> float:
    stops = instance.get("stops")
    if isinstance(stops, list) and stops:
        stop_degrees: List[float] = []
        for stop in stops:
            if isinstance(stop, dict):
                degree = to_float_or_nan(stop.get("degree"))
            else:
                degree = to_float_or_nan(getattr(stop, "degree", float("nan")))
            if not math.isnan(degree) and math.isfinite(degree) and degree >= 0.0:
                stop_degrees.append(degree)
        if stop_degrees:
            return safe_mean(stop_degrees)

    # First fallback: derive stop-node degrees from road_network edges.
    road_network = instance.get("road_network")
    if isinstance(stops, list) and stops and isinstance(road_network, dict):
        edges = road_network.get("edges")
        if isinstance(edges, list):
            neighbors: Dict[Any, set] = {}
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                u = _to_lookup_key(edge.get("u"))
                v = _to_lookup_key(edge.get("v"))
                if u is None or v is None:
                    continue
                neighbors.setdefault(u, set()).add(v)
                neighbors.setdefault(v, set()).add(u)

            stop_node_ids: List[Any] = []
            for idx, stop in enumerate(stops):
                if isinstance(stop, dict):
                    candidates = [
                        stop.get("id"),
                        stop.get("osmid_drive"),
                        stop.get("osmid_walk"),
                        stop.get("station_index"),
                        idx,
                    ]
                else:
                    candidates = [
                        getattr(stop, "id", None),
                        getattr(stop, "osmid_drive", None),
                        getattr(stop, "osmid_walk", None),
                        getattr(stop, "station_index", None),
                        idx,
                    ]

                chosen = None
                for raw in candidates:
                    node_id = _to_lookup_key(raw)
                    if node_id in neighbors:
                        chosen = node_id
                        break
                if chosen is not None:
                    stop_node_ids.append(chosen)

            if stop_node_ids:
                degs = [float(len(neighbors.get(node_id, set()))) for node_id in stop_node_ids]
                return safe_mean(degs)

    # Second fallback: matrix reachability degree proxy from CSV matrix input.
    matrix, _ = _resolve_drive_travel_time_matrix(instance)
    if isinstance(matrix, list) and matrix:
        degrees: List[float] = []
        n = len(matrix)
        for i in range(n):
            row = matrix[i]
            if not isinstance(row, list):
                continue
            deg = 0
            for j, value in enumerate(row):
                if i == j:
                    continue
                tt = to_float_or_nan(value)
                if not math.isnan(tt) and math.isfinite(tt) and tt > 0.0:
                    deg += 1
            degrees.append(float(deg))
        if degrees:
            return safe_mean(degrees)

    if isinstance(matrix, dict) and matrix:
        neighbor_sets: Dict[Any, set] = {}
        for key, value in matrix.items():
            if isinstance(value, dict):
                origin = _to_lookup_key(key)
                if origin is None:
                    continue
                for dest_key, raw in value.items():
                    tt = to_float_or_nan(raw)
                    if math.isnan(tt) or not math.isfinite(tt) or tt <= 0.0:
                        continue
                    dest = _to_lookup_key(dest_key)
                    if dest is None or dest == origin:
                        continue
                    neighbor_sets.setdefault(origin, set()).add(dest)
            elif isinstance(key, str) and ":" in key:
                left, right = key.split(":", 1)
                origin = _to_lookup_key(left)
                dest = _to_lookup_key(right)
                tt = to_float_or_nan(value)
                if origin is None or dest is None or origin == dest:
                    continue
                if math.isnan(tt) or not math.isfinite(tt) or tt <= 0.0:
                    continue
                neighbor_sets.setdefault(origin, set()).add(dest)

        if neighbor_sets:
            return safe_mean([float(len(v)) for v in neighbor_sets.values()])

    return float("nan")


def _extract_unique_stop_coords(instance: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Extract unique (lon, lat) stop coordinates from instance stops."""
    coord_lookup = _extract_stop_coord_lookup(instance)
    if not coord_lookup:
        return []

    unique: List[Tuple[float, float]] = []
    seen = set()
    for lon, lat in coord_lookup.values():
        key = (round(lon, 12), round(lat, 12))
        if key in seen:
            continue
        seen.add(key)
        unique.append((lon, lat))
    return unique


def _resolve_coverage_grid_n(instance: Dict[str, Any], n_stops: int) -> int:
    """Resolve configured grid size with hard caps to avoid excessive runtime."""
    raw = to_float_or_nan(instance.get("coverage_grid_n", instance.get("grid_n", DEFAULT_COVERAGE_GRID_N)))
    if math.isnan(raw) or not math.isfinite(raw):
        grid_n = DEFAULT_COVERAGE_GRID_N
    else:
        grid_n = int(round(raw))

    if grid_n < MIN_COVERAGE_GRID_N:
        grid_n = MIN_COVERAGE_GRID_N

    grid_cap = MAX_COVERAGE_GRID_N
    if n_stops > 0:
        budget_cap = int(math.sqrt(MAX_COVERAGE_DISTANCE_CHECKS / max(1, n_stops)))
        if budget_cap > 0:
            grid_cap = min(grid_cap, max(MIN_COVERAGE_GRID_N, budget_cap))

    if grid_n > grid_cap:
        LOGGER.info("coverage_ratio grid_n capped (%d->%d)", grid_n, grid_cap)
        grid_n = grid_cap

    return grid_n


def _resolve_coverage_walk_threshold_km(instance: Dict[str, Any]) -> Tuple[float, bool]:
    """Resolve walking threshold distance in km from 10-min rule and walk-speed input."""
    speed_candidates = [
        instance.get("walk_speed"),
        instance.get("walk_speed_mps"),
    ]

    requests = get_field_or_default(instance, "requests", [])
    if isinstance(requests, list):
        for req in requests:
            if not isinstance(req, dict):
                continue
            speed_candidates.append(req.get("walk_speed"))
            speed_candidates.append(req.get("walk_speed_mps"))

    speeds: List[float] = []
    for candidate in speed_candidates:
        speed = to_float_or_nan(candidate)
        if math.isnan(speed) or not math.isfinite(speed) or speed <= 0.0:
            continue
        speeds.append(speed)

    if speeds:
        walk_speed_mps = safe_mean(speeds)
        return (DEFAULT_WALK_THRESHOLD_MIN * 60.0 * walk_speed_mps) / 1000.0, False

    return (DEFAULT_WALK_THRESHOLD_MIN * 60.0 * DEFAULT_WALK_SPEED_MPS) / 1000.0, True


def _compute_coverage_ratio(instance: Dict[str, Any]) -> float:
    """Approximate covered fraction of the instance bbox with a configurable grid."""
    stop_coords = _extract_unique_stop_coords(instance)
    if not stop_coords:
        return float("nan")

    requests = get_field_or_default(instance, "requests", [])
    bbox_points: List[Tuple[float, float]] = []
    if isinstance(requests, list):
        bbox_points = _collect_request_points(requests, ("origin", "destination"))

    if not bbox_points:
        bbox_points = stop_coords
    if not bbox_points:
        return float("nan")

    threshold_km, used_speed_fallback = _resolve_coverage_walk_threshold_km(instance)
    if used_speed_fallback:
        LOGGER.info("coverage_ratio walk-speed fallback used (%.5f m/s)", DEFAULT_WALK_SPEED_MPS)
    if math.isnan(threshold_km) or threshold_km <= 0.0:
        return float("nan")

    lons = [p[0] for p in bbox_points]
    lats = [p[1] for p in bbox_points]
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    grid_n = _resolve_coverage_grid_n(instance, len(stop_coords))
    if grid_n <= 0:
        return float("nan")

    mean_lat_rad = math.radians((lat_min + lat_max) / 2.0)
    lat_km_per_deg = 110.574
    lon_km_per_deg = 111.320 * max(math.cos(mean_lat_rad), 1e-6)
    lat_thr_deg = threshold_km / lat_km_per_deg
    lon_thr_deg = threshold_km / lon_km_per_deg

    # Spatial binning to keep point-to-stop checks bounded.
    cell_lat = max(lat_thr_deg, 1e-9)
    cell_lon = max(lon_thr_deg, 1e-9)
    bins: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    for slon, slat in stop_coords:
        bx = int((slon - lon_min) / cell_lon)
        by = int((slat - lat_min) / cell_lat)
        bins.setdefault((bx, by), []).append((slon, slat))

    covered = 0
    total = grid_n * grid_n

    for ix in range(grid_n):
        lon = lon_min if grid_n == 1 else lon_min + (lon_max - lon_min) * ix / (grid_n - 1)
        for iy in range(grid_n):
            lat = lat_min if grid_n == 1 else lat_min + (lat_max - lat_min) * iy / (grid_n - 1)

            bx = int((lon - lon_min) / cell_lon)
            by = int((lat - lat_min) / cell_lat)

            is_covered = False
            for dx in (-1, 0, 1):
                if is_covered:
                    break
                for dy in (-1, 0, 1):
                    candidates = bins.get((bx + dx, by + dy), [])
                    for slon, slat in candidates:
                        if abs(slon - lon) > lon_thr_deg or abs(slat - lat) > lat_thr_deg:
                            continue
                        if _haversine_km(lon, lat, slon, slat) <= threshold_km:
                            is_covered = True
                            break
                    if is_covered:
                        break

            if is_covered:
                covered += 1

    return covered / total if total > 0 else float("nan")


def _to_time_min(value: Any) -> float:
    """Normalize generic time values to minutes using conservative unit inference."""
    t = to_float_or_nan(value)
    if math.isnan(t) or not math.isfinite(t):
        return float("nan")
    if t < 0.0:
        return float("nan")
    if t > TIME_INFER_SECONDS_THRESHOLD:
        return t / 60.0
    return t


def _compute_dynamic_placeholder_features(instance: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute features 19-21 with static-study defaults and dynamic compatibility.

    Static-study stable defaults:
    - request_flux_dynamism = 0.0
    - static_to_dynamic_ratio = 100.0
    - avg_request_urgency from tw_start assuming issue_time=0 if missing
    """
    requests = get_field_or_default(instance, "requests", [])
    if not isinstance(requests, list) or not requests:
        return {
            "request_flux_dynamism": 0.0,
            "static_to_dynamic_ratio": 100.0,
            "avg_request_urgency": 0.0,
        }

    issue_times_min: List[float] = []
    tw_starts_min: List[float] = []
    urgencies_min: List[float] = []

    known_before_horizon = 0
    considered_rows = 0
    has_explicit_issue_time = False

    # Planning horizon start is the first valid earliest departure.
    planning_start_candidates = [
        _to_time_min(req.get("tw_start"))
        for req in requests
        if isinstance(req, dict)
    ]
    planning_start_candidates = [
        t for t in planning_start_candidates if not math.isnan(t)
    ]
    planning_start = min(planning_start_candidates) if planning_start_candidates else 0.0

    for req in requests:
        if not isinstance(req, dict):
            continue

        tw_start = _to_time_min(req.get("tw_start"))
        if math.isnan(tw_start):
            continue

        raw_issue = req.get("issue_time")
        issue_time = _to_time_min(raw_issue)
        if raw_issue is not None and not math.isnan(issue_time):
            has_explicit_issue_time = True

        # Static placeholder assumption when issue_time is missing/invalid.
        if math.isnan(issue_time):
            issue_time = 0.0

        considered_rows += 1
        tw_starts_min.append(tw_start)
        issue_times_min.append(issue_time)

        if issue_time <= planning_start:
            known_before_horizon += 1

        urgency = tw_start - issue_time
        if urgency < 0.0:
            urgency = 0.0
        urgencies_min.append(urgency)

    if considered_rows == 0:
        return {
            "request_flux_dynamism": 0.0,
            "static_to_dynamic_ratio": 100.0,
            "avg_request_urgency": 0.0,
        }

    request_flux_dynamism = 0.0
    if has_explicit_issue_time:
        issue_std = safe_std(issue_times_min)
        horizon = max(tw_starts_min) - min(issue_times_min)
        if not math.isnan(issue_std) and horizon > 0.0:
            request_flux_dynamism = issue_std / horizon
            request_flux_dynamism = max(0.0, min(1.0, request_flux_dynamism))

    static_to_dynamic_ratio = (known_before_horizon / considered_rows) * 100.0
    avg_request_urgency = safe_mean(urgencies_min)

    return {
        "request_flux_dynamism": request_flux_dynamism,
        "static_to_dynamic_ratio": static_to_dynamic_ratio,
        "avg_request_urgency": avg_request_urgency,
    }


def _compute_spatial_network_features(instance: Dict[str, Any]) -> Dict[str, float]:
    return {
        "station_density": _compute_station_density(instance),
        "bsa_flexibility": _compute_bsa_flexibility(instance),
        "avg_station_degree": _compute_avg_station_degree(instance),
        "coverage_ratio": _compute_coverage_ratio(instance),
    }


def _spatial_cluster_coeff(
    origin_points_lon_lat: List[Tuple[float, float]], service_area_km2: float
) -> float:
    """
    For each origin, find its nearest other origin and Compute observed average nearest-neighbor distance (observed)
    Expected: Clark-Evans
    
    How to read the returned value:
    Near 0: strongly clustered (origins close together)
    Near 1: close to random or more spread out
    Clark-Evans can be >1 for very dispersed patterns, but this function clips at 1 by design.
    """
    n_points = len(origin_points_lon_lat)
    if n_points < 2 or math.isnan(service_area_km2) or service_area_km2 <= 0.0:
        return float("nan")

    nearest_neighbor_distances_km: List[float] = []
    for i, (lon_i, lat_i) in enumerate(origin_points_lon_lat):
        nearest_km = float("inf")
        for j, (lon_j, lat_j) in enumerate(origin_points_lon_lat):
            if i == j:
                continue
            d_km = _haversine_km(lon_i, lat_i, lon_j, lat_j)
            if d_km < nearest_km:
                nearest_km = d_km
        if nearest_km != float("inf"):
            nearest_neighbor_distances_km.append(nearest_km)

    observed = safe_mean(nearest_neighbor_distances_km)
    if math.isnan(observed):
        return float("nan")

    intensity = n_points / service_area_km2
    if intensity <= 0.0:
        return float("nan")

    expected = 0.5 / math.sqrt(intensity)
    if expected <= 0.0:
        return float("nan")

    ratio = observed / expected
    if math.isnan(ratio):
        return float("nan")

    return float(min(1.0, max(0.0, ratio)))


def _compute_demand_features(instance: Dict[str, Any]) -> Dict[str, float]:
    requests = get_field_or_default(instance, "requests", [])
    if not isinstance(requests, list):
        return {
            "n_requests": float("nan"),
            "demand_density": float("nan"),
            "spatial_cluster_coeff": float("nan"),
        }

    n_requests = float(len(requests))
    origin_points = _collect_request_points(requests, ("origin",))
    endpoint_points = _collect_request_points(requests, ("origin", "destination"))

    service_area_km2, fallback_reason = _resolve_service_area_km2(instance, endpoint_points)
    if fallback_reason is not None:
        if math.isnan(service_area_km2):
            LOGGER.info("service_area_km2 fallback used (%s)", fallback_reason)
        else:
            LOGGER.info(
                "service_area_km2 fallback used (%s, %.3f km2)",
                fallback_reason,
                service_area_km2,
            )

    demand_density = float("nan")
    if n_requests >= 0.0 and not math.isnan(service_area_km2) and service_area_km2 > 0.0:
        demand_density = n_requests / service_area_km2

    spatial_cluster = _spatial_cluster_coeff(origin_points, service_area_km2)

    return {
        "n_requests": n_requests,
        "demand_density": demand_density,
        "spatial_cluster_coeff": spatial_cluster,
    }


def _numeric_scaffold_value(feature_key: str, instance: Dict[str, Any]) -> float:
    """
    Return scaffold placeholder values.

    Only minimal/trivial values are populated at scaffold stage; all other
    feature formulas are intentionally deferred to later tasks.
    """
    if feature_key == "n_requests":
        requests = get_field_or_default(instance, "requests", [])
        return float(len(requests)) if isinstance(requests, list) else float("nan")

    if feature_key == "fleet_size":
        vehicles = get_field_or_default(instance, "vehicles", [])
        return float(len(vehicles)) if isinstance(vehicles, list) else float("nan")

    # Placeholder for deferred feature formula implementation.
    return float("nan")


def compute_features(instance: Dict[str, Any]) -> Dict[str, float]:
    """
    Scaffold-only feature entrypoint.

    Contract:
    - Always returns exactly 21 flat keys (FEATURE_KEYS).
    - Values are numeric floats (including NaN placeholders).
    - Performs lightweight schema validation without failing hard.
    """
    _ = validate_instance_schema(instance)

    demand_features = _compute_demand_features(instance)
    od_features = _compute_od_distance_features(instance)
    tw_features = _compute_time_window_features(instance)
    tw_tightness_feature = _compute_tw_tightness_factor(instance)
    fleet_features = _compute_fleet_features(instance)
    spatial_network_features = _compute_spatial_network_features(instance)
    dynamic_features = _compute_dynamic_placeholder_features(instance)

    features: Dict[str, float] = {}
    for key in FEATURE_KEYS:
        if key in demand_features:
            features[key] = to_float_or_nan(demand_features[key])
        elif key in od_features:
            features[key] = to_float_or_nan(od_features[key])
        elif key in tw_features:
            features[key] = to_float_or_nan(tw_features[key])
        elif key in tw_tightness_feature:
            features[key] = to_float_or_nan(tw_tightness_feature[key])
        elif key in fleet_features:
            features[key] = to_float_or_nan(fleet_features[key])
        elif key in spatial_network_features:
            features[key] = to_float_or_nan(spatial_network_features[key])
        elif key in dynamic_features:
            features[key] = to_float_or_nan(dynamic_features[key])
        else:
            features[key] = to_float_or_nan(_numeric_scaffold_value(key, instance))

    return features


def _smoke_call() -> None:
    minimal_instance = {
        "service_area_km2": 100.0,
        "requests": [
            {
                "origin": {"x": 4.40, "y": 51.21},
                "destination": {"x": 4.42, "y": 51.22},
                "tw_start": 480,
                "tw_end": 510,
                "demand": 2,
            },
            {
                "origin": {"x": 4.41, "y": 51.20},
                "destination": {"x": 4.43, "y": 51.23},
                "tw_start": 490,
                "tw_end": 540,
                "demand": 1,
            },
        ],
        "vehicles": [
            {"capacity": 8, "depot": {"x": 4.39, "y": 51.20}},
            {"capacity": 12, "depot": {"x": 4.45, "y": 51.24}},
        ],
    }

    feats = compute_features(minimal_instance)

    assert len(feats) == len(FEATURE_KEYS), (
        f"Expected {len(FEATURE_KEYS)} features, got {len(feats)}"
    )
    assert list(feats.keys()) == FEATURE_KEYS, "Feature key order/contract mismatch"
    assert all(isinstance(v, (int, float)) for v in feats.values()), "Non-numeric value found"

    print("Smoke call passed.")
    print(f"n_requests={feats['n_requests']}, fleet_size={feats['fleet_size']}")


if __name__ == "__main__":
    _smoke_call()
