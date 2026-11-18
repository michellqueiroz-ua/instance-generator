"""
csv_loaders.py — T03 (ISA-HODBRP)

Loaders for REQreate output CSV files consumed by downstream feature functions.

Two file formats are supported:

  Network-level travel-time matrix
    File:   <location>/csv/<location>.tt.drive.csv
    Rows:   one row per origin stop (osmid_origin column)
    Cols:   one column per destination stop (column name = osmid string)
    Units:  seconds (converted to minutes on load)

  Instance-level travel-time matrix
    File:   <location>/travel_time_matrix/travel_time_matrix_*.csv
    Rows:   one row per origin station index (Unnamed: 0 column = integer index)
    Cols:   one column per destination station index (column name = int string)
    Units:  seconds (converted to minutes on load)

  Stops / stations
    File:   <location>/csv/<location>.stations.csv
    Cols:   Unnamed: 0.2, Unnamed: 0.1, Unnamed: 0, osmid_walk, osmid_drive,
            lat, lon, type
    Note:   The three unnamed columns are identical row-index artefacts from
            successive pandas saves; we treat column index as station_index.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Stop record
# ---------------------------------------------------------------------------

@dataclass
class StopRecord:
    """One row from a REQreate stations.csv file."""
    station_index: int   # 0-based positional index (row order)
    osmid_walk: int      # OSM node ID in the walking network
    osmid_drive: int     # OSM node ID in the driving network
    lat: float           # WGS-84 latitude
    lon: float           # WGS-84 longitude
    type: int            # stop type flag (0 = regular stop)


# ---------------------------------------------------------------------------
# Travel-time matrix wrappers
# ---------------------------------------------------------------------------

class NetworkTravelTimeMatrix:
    """
    Network-level .tt.drive.csv loaded as a square DataFrame.

    Keys are osmid_drive integers.  All travel times are in **minutes**.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        # df is indexed by osmid (int), columns are osmid (int)
        self._df: pd.DataFrame = df

    # -- public interface -------------------------------------------------------

    @property
    def osmids(self) -> List[int]:
        """Sorted list of stop osmid_drive values present in the matrix."""
        return sorted(self._df.index.tolist())

    def get_minutes(self, origin_osmid: int, dest_osmid: int) -> float:
        """
        Return driving travel time in minutes between two stops by osmid.

        Returns NaN if either osmid is not in the matrix.
        """
        try:
            return float(self._df.at[origin_osmid, dest_osmid])
        except KeyError:
            return float("nan")

    def as_dataframe(self) -> pd.DataFrame:
        """Return the underlying DataFrame (minutes, osmid-indexed)."""
        return self._df


class InstanceTravelTimeMatrix:
    """
    Instance-level travel_time_matrix_*.csv loaded as a square DataFrame.

    Keys are integer station indices (0-based row order matching stations.csv).
    All travel times are in **minutes**.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        # df indexed by station_index (int), columns are station_index (int)
        self._df: pd.DataFrame = df

    # -- public interface -------------------------------------------------------

    @property
    def station_indices(self) -> List[int]:
        """Sorted list of station indices present in the matrix."""
        return sorted(self._df.index.tolist())

    def get_minutes(self, origin_idx: int, dest_idx: int) -> float:
        """
        Return driving travel time in minutes between two stations by index.

        Returns NaN if either index is not in the matrix.
        """
        try:
            return float(self._df.at[origin_idx, dest_idx])
        except KeyError:
            return float("nan")

    def as_dataframe(self) -> pd.DataFrame:
        """Return the underlying DataFrame (minutes, station-index-indexed)."""
        return self._df


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_stops(path: str) -> List[StopRecord]:
    """
    Load a REQreate stations.csv file.

    Parameters
    ----------
    path : str
        Absolute or relative path to the stations CSV file.

    Returns
    -------
    List[StopRecord]
        One record per row, in file order, with station_index = 0-based row.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If required columns are missing or the file contains no rows.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Stops file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Stops file has no rows: {path}")

    required = {"osmid_walk", "osmid_drive", "lat", "lon", "type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Stops file is missing required columns {sorted(missing)}: {path}"
        )

    records: List[StopRecord] = []
    for idx, row in enumerate(df.itertuples(index=False)):
        records.append(
            StopRecord(
                station_index=idx,
                osmid_walk=int(row.osmid_walk),
                osmid_drive=int(row.osmid_drive),
                lat=float(row.lat),
                lon=float(row.lon),
                type=int(row.type),
            )
        )
    return records


def load_network_ttm(path: str) -> NetworkTravelTimeMatrix:
    """
    Load a REQreate .tt.drive.csv file (network-level, osmid-keyed).

    Travel times in the file are in seconds and are converted to minutes on
    load (divided by 60).

    Parameters
    ----------
    path : str
        Absolute or relative path to the .tt.drive.csv file.

    Returns
    -------
    NetworkTravelTimeMatrix

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the required ``osmid_origin`` column is missing or the file is empty.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Network TTM file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Network TTM file has no rows: {path}")

    if "osmid_origin" not in df.columns:
        raise ValueError(
            f"Network TTM file is missing required column 'osmid_origin': {path}"
        )

    # Drop the unnamed index artefact if present
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Set osmid_origin as the row index (int64 — required for large OSM IDs)
    df = df.set_index("osmid_origin")
    df.index = df.index.astype("int64")

    # Column names are osmid strings → convert to int64 to match index dtype
    df.columns = pd.Index([np.int64(c) for c in df.columns])

    # Convert seconds → minutes
    df = df / 60.0

    return NetworkTravelTimeMatrix(df)


def load_instance_ttm(path: str) -> InstanceTravelTimeMatrix:
    """
    Load a REQreate travel_time_matrix_*.csv file (instance-level, index-keyed).

    Travel times in the file are in seconds and are converted to minutes on
    load (divided by 60).

    Parameters
    ----------
    path : str
        Absolute or relative path to the travel_time_matrix CSV file.

    Returns
    -------
    InstanceTravelTimeMatrix

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the expected index column is missing or the file is empty.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Instance TTM file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Instance TTM file has no rows: {path}")

    # The first unnamed column is the station index
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if not unnamed:
        raise ValueError(
            f"Instance TTM file has no index column ('Unnamed: 0' expected): {path}"
        )

    index_col = unnamed[0]
    df = df.drop(columns=unnamed[1:])   # drop any extra unnamed artefacts
    df = df.set_index(index_col)
    df.index = df.index.astype("int64")
    df.index.name = "station_index"

    # Column names are station index strings → convert to int64
    df.columns = pd.Index([np.int64(c) for c in df.columns])

    # Convert seconds → minutes
    df = df / 60.0

    return InstanceTravelTimeMatrix(df)


# ---------------------------------------------------------------------------
# Convenience: build osmid → station_index lookup from loaded stops
# ---------------------------------------------------------------------------

def build_osmid_to_index(stops: List[StopRecord]) -> Dict[int, int]:
    """
    Return a dict mapping osmid_drive → station_index for each stop.

    Useful for translating network-TTM lookups (osmid-keyed) into
    instance-TTM lookups (station-index-keyed) and vice-versa.
    """
    return {s.osmid_drive: s.station_index for s in stops}


# ---------------------------------------------------------------------------
# Smoke / validation test
# ---------------------------------------------------------------------------

def _smoke_test(
    stops_path: Optional[str] = None,
    network_ttm_path: Optional[str] = None,
    instance_ttm_path: Optional[str] = None,
) -> None:
    """
    Validate loaders against real REQreate files.

    Default paths point to the Maastricht network included in the repository.
    Supply alternative paths to test other locations.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    location = "Maastricht, Netherlands"
    csv_dir = os.path.join(repo_root, location, "csv")

    stops_path = stops_path or os.path.join(csv_dir, f"{location}.stations.csv")
    network_ttm_path = network_ttm_path or os.path.join(
        csv_dir, f"{location}.tt.drive.csv"
    )

    # ---- stops ----
    stops = load_stops(stops_path)
    assert len(stops) > 0, "Expected at least one stop"
    s0 = stops[0]
    assert s0.station_index == 0
    assert isinstance(s0.osmid_drive, int)
    assert -90 <= s0.lat <= 90, f"Suspicious lat: {s0.lat}"
    assert -180 <= s0.lon <= 180, f"Suspicious lon: {s0.lon}"
    print(f"[stops]           loaded {len(stops)} stops from {os.path.basename(stops_path)}")
    print(f"                  first stop: index={s0.station_index}, osmid_drive={s0.osmid_drive}, "
          f"lat={s0.lat:.6f}, lon={s0.lon:.6f}")

    # ---- network TTM ----
    nttm = load_network_ttm(network_ttm_path)
    osmids = nttm.osmids
    assert len(osmids) > 0, "Network TTM has no rows"

    # Pick any osmid present in both index and columns (matrix may not be
    # perfectly square for every network export)
    col_set = set(nttm.as_dataframe().columns.tolist())
    valid_osmids = [o for o in osmids if o in col_set]
    assert valid_osmids, "No osmid found in both index and columns"

    diag = nttm.get_minutes(valid_osmids[0], valid_osmids[0])
    assert diag == 0.0, f"Diagonal should be 0 minutes, got {diag}"

    off_diag = nttm.get_minutes(valid_osmids[0], valid_osmids[1])
    assert off_diag > 0.0, f"Off-diagonal should be positive, got {off_diag}"

    missing_lookup = nttm.get_minutes(-999, -998)
    assert missing_lookup != missing_lookup, "Missing osmid should return NaN"  # NaN != NaN

    print(f"[network_ttm]     loaded {len(osmids)}x{len(osmids)} matrix from "
          f"{os.path.basename(network_ttm_path)}")
    print(f"                  diagonal=0.0 min, sample off-diagonal={off_diag:.3f} min")

    # ---- osmid → index mapping ----
    o2i = build_osmid_to_index(stops)
    assert len(o2i) == len(stops), "osmid-to-index map length mismatch"
    print(f"[osmid->index]    {len(o2i)} entries")

    # ---- instance TTM (optional) ----
    if instance_ttm_path:
        ittm = load_instance_ttm(instance_ttm_path)
        idxs = ittm.station_indices
        assert len(idxs) > 0
        diag_i = ittm.get_minutes(idxs[0], idxs[0])
        assert diag_i == 0.0, f"Instance TTM diagonal should be 0, got {diag_i}"
        off_i = ittm.get_minutes(idxs[0], idxs[1])
        assert off_i > 0.0, f"Instance TTM off-diagonal should be positive, got {off_i}"
        print(f"[instance_ttm]    loaded {len(idxs)}x{len(idxs)} matrix from "
              f"{os.path.basename(instance_ttm_path)}")
        print(f"                  diagonal=0.0 min, sample off-diagonal={off_i:.3f} min")
    else:
        print("[instance_ttm]    skipped (no path supplied)")

    print("\nSmoke test PASSED.")


if __name__ == "__main__":
    import glob
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Auto-discover one Copenhagen instance TTM for the instance-level smoke test
    pattern = os.path.join(
        repo_root,
        "Copenhagen Municipality, Denmark",
        "travel_time_matrix",
        "travel_time_matrix_*.csv",
    )
    discovered = sorted(glob.glob(pattern))
    instance_ttm = discovered[0] if discovered else None

    _smoke_test(instance_ttm_path=instance_ttm)
