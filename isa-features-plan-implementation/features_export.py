"""
features_export.py — T12 (ISA-HODBRP)

Generation-time export of the 21 ISA features to ``data/results/features.csv``.

One row per generated instance, keyed by the instance filename stem, matching
the format consumed by MATILDA (see implementation_context.md §5):

    instance,n_requests,demand_density,...,avg_request_urgency
    Maastricht_Netherlands_ODBRP_13req_1,13,1.83,...,240.0

The CSV mechanics — canonical header, duplicate-column collapsing, malformed
row rejection, idempotent append, atomic rewrite — live in ``csv_table`` and
are shared with the perf_matrix export (T13).  This module owns the feature
schema and the generation-time entrypoint.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from compute_features_scaffold import FEATURE_KEYS, compute_features
from csv_table import (
    ON_DUPLICATE_CHOICES,
    TableReport,
    TableSchema,
    append_rows,
    format_number,
    instance_stem,
    normalize_row,
    read_table,
    repo_root,
    resolve_output_path,
    write_table,
)


LOGGER = logging.getLogger(__name__)

INSTANCE_COLUMN = "instance"
FEATURES_SCHEMA = TableSchema(key_column=INSTANCE_COLUMN, value_columns=tuple(FEATURE_KEYS))
FEATURE_CSV_COLUMNS: List[str] = FEATURES_SCHEMA.columns

ENV_FEATURES_CSV = "ISA_FEATURES_CSV"
DEFAULT_FEATURES_CSV_RELPATH = os.path.join("data", "results", "features.csv")

# Re-exported for callers that only import this module.
CsvSanitationReport = TableReport
format_feature_value = format_number


def resolve_features_csv_path(csv_path: Optional[Any] = None) -> Path:
    """
    Resolve the features.csv destination.

    Precedence: explicit argument > ``ISA_FEATURES_CSV`` env var >
    ``<repo_root>/data/results/features.csv``.
    """
    return resolve_output_path(csv_path, ENV_FEATURES_CSV, DEFAULT_FEATURES_CSV_RELPATH)


def normalize_feature_row(instance_ref: Any, features: Dict[str, Any]) -> Dict[str, str]:
    """
    Build one canonical, fully populated features.csv row.

    Unknown keys in ``features`` are ignored and missing keys become empty
    cells, so a row can never widen or narrow the schema.
    """
    return normalize_row(
        FEATURES_SCHEMA, instance_ref, features, logger=LOGGER, label="features.csv"
    )


def read_features_csv(
    csv_path: Optional[Any] = None,
) -> Tuple[Dict[str, Dict[str, str]], TableReport]:
    """Read features.csv into ``{instance_stem: canonical_row}`` plus a report."""
    return read_table(resolve_features_csv_path(csv_path), FEATURES_SCHEMA)


def write_features_csv(
    rows: Iterable[Dict[str, str]], csv_path: Optional[Any] = None
) -> Path:
    """Rewrite features.csv from scratch with the canonical schema."""
    return write_table(resolve_features_csv_path(csv_path), FEATURES_SCHEMA, rows)


def append_features_row(
    instance_ref: Any,
    features: Dict[str, Any],
    csv_path: Optional[Any] = None,
    on_duplicate: str = "replace",
) -> Path:
    """
    Append (or replace) one instance row in features.csv.

    Parameters
    ----------
    instance_ref
        Instance path, file name, or bare stem — the CSV key.
    features
        Mapping produced by ``compute_features``.
    csv_path
        Destination file; defaults to ``resolve_features_csv_path()``.
    on_duplicate
        ``"replace"`` (default) overwrites an existing row for the same stem,
        ``"skip"`` keeps the stored row, ``"error"`` raises ``ValueError``.
    """
    return append_features_rows(
        [(instance_ref, features)], csv_path=csv_path, on_duplicate=on_duplicate
    )


def append_features_rows(
    items: Iterable[Tuple[Any, Dict[str, Any]]],
    csv_path: Optional[Any] = None,
    on_duplicate: str = "replace",
) -> Path:
    """
    Append several ``(instance_ref, features)`` pairs in a single read/write.

    Same protections as :func:`append_features_row`; later pairs in ``items``
    win over earlier ones for the same stem.
    """
    rows = [normalize_feature_row(ref, feats) for ref, feats in items]
    return append_rows(
        resolve_features_csv_path(csv_path),
        FEATURES_SCHEMA,
        rows,
        on_duplicate=on_duplicate,
        logger=LOGGER,
        label="features.csv",
    )


def export_features_for_instance(
    instance: Dict[str, Any],
    instance_ref: Any,
    csv_path: Optional[Any] = None,
    on_duplicate: str = "replace",
) -> Dict[str, float]:
    """
    Compute the 21 features for ``instance`` and write its features.csv row.

    Returns the computed feature dict so callers can log or reuse it.
    """
    features = compute_features(instance)
    append_features_row(
        instance_ref, features, csv_path=csv_path, on_duplicate=on_duplicate
    )
    return features


def validate_features_csv(csv_path: Optional[Any] = None) -> List[str]:
    """
    Check features.csv against the canonical schema.

    Returns a list of problem descriptions; an empty list means the file is
    clean and ready for MATILDA / perf_matrix alignment (T13).
    """
    path = resolve_features_csv_path(csv_path)
    if not path.is_file():
        return [f"file does not exist: {path}"]

    _, report = read_features_csv(path)
    if not report.file_exists:
        return ["file is empty"]

    return report.describe()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_call() -> None:
    import shutil
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="isa_features_export_"))
    csv_file = tmp_dir / "data" / "results" / "features.csv"

    try:
        instance = {
            "service_area_km2": 100.0,
            "requests": [
                {
                    "origin": {"x": 4.40, "y": 51.21},
                    "destination": {"x": 4.42, "y": 51.22},
                    "tw_start": 480,
                    "tw_end": 510,
                },
            ],
            "vehicles": [{"capacity": 8}],
        }

        export_features_for_instance(instance, "toy_001.json", csv_path=csv_file)
        export_features_for_instance(instance, "toy_002.json", csv_path=csv_file)
        export_features_for_instance(instance, "toy_001.csv", csv_path=csv_file)

        rows, _ = read_features_csv(csv_file)
        assert list(rows) == ["toy_001", "toy_002"], list(rows)
        assert validate_features_csv(csv_file) == []

        with csv_file.open("r", encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")
        assert header == FEATURE_CSV_COLUMNS, header

        print("Smoke call passed.")
        print(f"{len(rows)} rows, {len(FEATURE_CSV_COLUMNS)} columns at {csv_file}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _smoke_call()
