"""
csv_table.py — T13 (ISA-HODBRP)

Shared writer for the keyed numeric CSV tables MATILDA consumes:
``data/results/features.csv`` (T12) and ``data/results/perf_matrix.csv`` (T13).

Both files have the same shape — a string key column followed by numeric value
columns, one row per instance — and need the same protections, so the logic
lives here once:

  Schema stability
    The header is always ``[key_column] + value_columns``.  A file whose header
    drifted (extra, missing, or reordered columns) is migrated on the next
    write, keeping the values of columns that still exist.

  No duplicate columns
    Repeated column names collapse to their first occurrence and the file is
    rewritten with the canonical header.

  No malformed rows
    Rows whose field count does not match the header, or whose key is empty,
    are dropped.  Non-numeric and non-finite values become empty cells, which
    pandas reads back as NaN — never the literal strings "nan"/"inf".

  Stable append behaviour
    Appending is idempotent per key: re-running an instance replaces its row
    instead of adding a second one.  Any write that needs a rewrite goes
    through a temporary file plus ``os.replace``, so an interrupted run cannot
    leave a half-written CSV behind.
"""

import csv
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


LOGGER = logging.getLogger(__name__)

ON_DUPLICATE_CHOICES = ("replace", "skip", "error")

# Suffixes stripped when deriving an instance key from a file name.
INSTANCE_SUFFIXES = (".csv", ".json")


# ---------------------------------------------------------------------------
# Paths and keys
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    """Repository root (parent of this implementation folder)."""
    return Path(__file__).resolve().parent.parent


def resolve_output_path(
    explicit: Optional[Any], env_var: str, default_relpath: str
) -> Path:
    """
    Resolve a results-file destination.

    Precedence: explicit argument > environment variable > repo-root default.
    """
    if explicit is not None:
        return Path(str(explicit)).expanduser()

    from_env = os.environ.get(env_var)
    if from_env and from_env.strip():
        return Path(from_env.strip()).expanduser()

    return repo_root() / default_relpath


def instance_stem(instance_ref: Any) -> str:
    """
    Derive the instance key from a path, file name, or bare stem.

    Only ``.csv`` / ``.json`` suffixes are stripped, so stems that legitimately
    contain dots (``inst_v1.2``) survive intact.
    """
    text = str(instance_ref).strip().strip('"').strip("'")
    name = Path(text.replace("\\", "/")).name

    lowered = name.lower()
    for suffix in INSTANCE_SUFFIXES:
        if lowered.endswith(suffix):
            name = name[: -len(suffix)]
            break

    name = name.strip()
    if not name:
        raise ValueError(f"Cannot derive instance stem from {instance_ref!r}")
    if "\n" in name or "\r" in name:
        raise ValueError(f"Instance stem contains a newline: {instance_ref!r}")

    return name


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TableSchema:
    """Canonical layout of a keyed numeric results table."""

    key_column: str
    value_columns: Tuple[str, ...]

    @property
    def columns(self) -> List[str]:
        return [self.key_column] + list(self.value_columns)


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

def format_number(value: Any) -> str:
    """
    Render one numeric cell.

    NaN, +/-inf, None, booleans and anything non-numeric become an empty cell,
    which pandas reads back as NaN (implementation_context.md §5).
    """
    if value is None or isinstance(value, bool):
        return ""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""

    if math.isnan(number) or math.isinf(number):
        return ""

    return repr(number)


def normalize_row(
    schema: TableSchema,
    key_ref: Any,
    values: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    label: str = "table",
) -> Dict[str, str]:
    """
    Build one canonical, fully populated row.

    Unknown keys in ``values`` are ignored and missing ones become empty cells,
    so a single row can never widen or narrow the schema.
    """
    log = logger or LOGGER
    key = instance_stem(key_ref)

    if not isinstance(values, dict):
        raise TypeError(f"values must be a dict, got {type(values).__name__}")

    unknown = [name for name in values if name not in schema.value_columns]
    if unknown:
        log.warning(
            "%s: dropping %d unknown column(s) for %s: %s",
            label,
            len(unknown),
            key,
            ", ".join(sorted(unknown)),
        )

    missing = [name for name in schema.value_columns if name not in values]
    if missing:
        log.warning(
            "%s: %d column(s) missing for %s, written as empty: %s",
            label,
            len(missing),
            key,
            ", ".join(missing),
        )

    row = {schema.key_column: key}
    for name in schema.value_columns:
        row[name] = format_number(values.get(name))
    return row


# ---------------------------------------------------------------------------
# Read / repair
# ---------------------------------------------------------------------------

@dataclass
class TableReport:
    """What had to be fixed while reading an existing results table."""

    file_exists: bool = False
    header_canonical: bool = True
    duplicate_columns: List[str] = field(default_factory=list)
    unknown_columns: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    malformed_rows: int = 0
    duplicate_keys: List[str] = field(default_factory=list)
    coerced_values: int = 0

    @property
    def needs_rewrite(self) -> bool:
        """True when the file on disk does not already match the canonical form."""
        return bool(
            self.file_exists
            and (
                not self.header_canonical
                or self.duplicate_columns
                or self.unknown_columns
                or self.missing_columns
                or self.malformed_rows
                or self.duplicate_keys
                or self.coerced_values
            )
        )

    def describe(self) -> List[str]:
        """Human-readable list of problems found (empty when the file is clean)."""
        problems: List[str] = []
        if not self.header_canonical:
            problems.append("header does not match the canonical schema")
        if self.duplicate_columns:
            problems.append(f"duplicate columns: {', '.join(self.duplicate_columns)}")
        if self.unknown_columns:
            problems.append(f"unknown columns: {', '.join(self.unknown_columns)}")
        if self.missing_columns:
            problems.append(f"missing columns: {', '.join(self.missing_columns)}")
        if self.malformed_rows:
            problems.append(f"{self.malformed_rows} malformed row(s)")
        if self.duplicate_keys:
            problems.append(f"duplicate instance keys: {', '.join(self.duplicate_keys)}")
        if self.coerced_values:
            problems.append(f"{self.coerced_values} non-numeric value(s)")
        return problems


def read_table(
    path: Any, schema: TableSchema
) -> Tuple[Dict[str, Dict[str, str]], TableReport]:
    """
    Read a results table into ``{key: canonical_row}`` plus a repair report.

    Insertion order of the returned mapping is the file's row order (first
    appearance wins for position, last appearance wins for values).  A missing
    or empty file yields an empty mapping and a clean report.
    """
    csv_path = Path(str(path))
    report = TableReport()
    rows: Dict[str, Dict[str, str]] = {}

    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return rows, report

    report.file_exists = True

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return rows, report

        header = [str(name).strip() for name in header]
        report.header_canonical = header == schema.columns

        # First occurrence of each column name wins; later ones are duplicates.
        column_index: Dict[str, int] = {}
        for position, name in enumerate(header):
            if name in column_index:
                if name not in report.duplicate_columns:
                    report.duplicate_columns.append(name)
                continue
            column_index[name] = position

        report.unknown_columns = [
            name for name in column_index if name not in schema.columns
        ]
        report.missing_columns = [
            name for name in schema.columns if name not in column_index
        ]

        key_position = column_index.get(schema.key_column)

        for raw_row in reader:
            if not raw_row or all(not str(cell).strip() for cell in raw_row):
                continue

            if len(raw_row) != len(header) or key_position is None:
                report.malformed_rows += 1
                continue

            try:
                key = instance_stem(raw_row[key_position])
            except ValueError:
                report.malformed_rows += 1
                continue

            row = {schema.key_column: key}
            for name in schema.value_columns:
                position = column_index.get(name)
                raw_value = raw_row[position] if position is not None else ""
                has_text = bool(str(raw_value).strip())
                cell = format_number(raw_value) if has_text else ""
                if has_text and not cell:
                    report.coerced_values += 1
                row[name] = cell

            if key in rows and key not in report.duplicate_keys:
                report.duplicate_keys.append(key)
            rows[key] = row

    if report.needs_rewrite:
        LOGGER.warning(
            "%s needs repair: %s", csv_path, "; ".join(report.describe())
        )

    return rows, report


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_table(
    path: Any, schema: TableSchema, rows: Iterable[Dict[str, str]]
) -> Path:
    """Rewrite the table from scratch, atomically, with the canonical header."""
    csv_path = Path(str(path))
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        dir=str(csv_path.parent),
        prefix=csv_path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(handle.name)
    try:
        with handle:
            writer = csv.writer(handle)
            writer.writerow(schema.columns)
            for row in rows:
                writer.writerow([row.get(name, "") for name in schema.columns])
        os.replace(str(tmp_path), str(csv_path))
    except BaseException:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise

    return csv_path


def _append_line(csv_path: Path, schema: TableSchema, row: Dict[str, str]) -> None:
    """Append one row, writing the canonical header first if the file is new."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not csv_path.is_file()) or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(schema.columns)
        writer.writerow([row.get(name, "") for name in schema.columns])


def append_rows(
    path: Any,
    schema: TableSchema,
    new_rows: Sequence[Dict[str, str]],
    on_duplicate: str = "replace",
    logger: Optional[logging.Logger] = None,
    label: str = "table",
) -> Path:
    """
    Append canonical rows, repairing the file first when it needs it.

    ``on_duplicate`` is ``"replace"`` (default), ``"skip"``, or ``"error"``.
    """
    if on_duplicate not in ON_DUPLICATE_CHOICES:
        raise ValueError(
            f"on_duplicate must be one of {ON_DUPLICATE_CHOICES}, got {on_duplicate!r}"
        )

    log = logger or LOGGER
    csv_path = Path(str(path))

    stored_rows, report = read_table(csv_path, schema)
    needs_rewrite = report.needs_rewrite
    pending_appends: List[Dict[str, str]] = []

    for row in new_rows:
        key = row[schema.key_column]

        if key in stored_rows:
            if on_duplicate == "error":
                raise ValueError(f"Instance {key!r} already present in {csv_path}")
            if on_duplicate == "skip":
                log.info("%s: keeping existing row for %s", label, key)
                continue
            log.info("%s: replacing existing row for %s", label, key)
            stored_rows[key] = row
            needs_rewrite = True
            continue

        stored_rows[key] = row
        if needs_rewrite:
            continue
        pending_appends.append(row)

    if needs_rewrite:
        return write_table(csv_path, schema, stored_rows.values())

    for row in pending_appends:
        _append_line(csv_path, schema, row)

    if not pending_appends and not csv_path.is_file():
        # Nothing to append (all skipped) and no file yet: still emit the header
        # so downstream readers see a valid, empty table.
        write_table(csv_path, schema, [])

    return csv_path


def read_keys_in_order(path: Any, schema: TableSchema) -> List[str]:
    """Instance keys of a table, in file order (empty list when absent)."""
    rows, _ = read_table(path, schema)
    return list(rows)
