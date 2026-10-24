"""
perf_matrix_export.py — T13 (ISA-HODBRP)

Build ``data/results/perf_matrix.csv``: one row per instance, one column per
algorithm (SA, IP, GR), every score normalized to [0, 1] with higher = better
(implementation_context.md §6).

    instance,SA,IP,GR
    Maastricht_Netherlands_ODBRP_13req_1,0.97,1.0,0.82

Normalization
-------------
implementation_context.md §6 states two things that cannot both hold for a
minimisation objective:

    score(i, a) = (best_known(i) - urt(i, a)) / best_known(i)
    "higher = better; best algorithm on instance i gets 1.0"

With ``urt`` a cost to minimise, ``urt >= best_known``, so that expression is
<= 0 for every algorithm and 0 for the best one — it is the negated relative
gap, not a [0, 1] score.  The two invariants the document actually relies on
are the range and "best gets 1.0", so three normalisations are offered and the
range/best-is-1.0 invariants are what the checks enforce:

  ``gap``  (default)    score = 1 - |raw - best| / |best|, clipped to [0, 1]
       The document's relative-gap expression with the sign corrected, so the
       reported score is literally "1 minus the relative gap to the best".
       Linear in the gap, which spreads mid-range algorithms further apart
       than ``ratio`` does.  Its one weakness: everything at or beyond twice
       the best cost collapses to 0.0, so an algorithm that is routinely much
       worse loses all variance — ``check_score_spread`` reports that.

  ``ratio``             score = best / raw          (sense="min")
                        score = raw / best          (sense="max")
       Best algorithm scores exactly 1.0, an algorithm twice as costly scores
       0.5.  Never clips, so it keeps ordering information no matter how bad
       the worst algorithm is.  Use it if ``gap`` saturates.

  ``minmax``            score = (worst - raw) / (worst - best)   (sense="min")
       Per-instance rescaling: best 1.0, worst 0.0.  Use when raw values are
       not ratio-scaled.  Discards how *much* better the winner was.

``best`` is the best value observed across the three algorithms unless a
per-instance ``best_known`` is supplied (e.g. a proven optimum or a long-run
reference solution), in which case no algorithm need reach 1.0.

Checks
------
``validate_perf_matrix_csv`` enforces, with actionable messages:

  * canonical schema (shared with features.csv via ``csv_table``)
  * every score present, numeric, finite and within [0, 1]
  * index alignment with features.csv — same instance keys, same order, which
    is what ``IS = InstanceSpace(features, performance)`` requires
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from csv_table import (
    TableReport,
    TableSchema,
    append_rows,
    instance_stem,
    normalize_row,
    read_table,
    resolve_output_path,
    write_table,
)
from features_export import (
    FEATURES_SCHEMA,
    resolve_features_csv_path,
)


LOGGER = logging.getLogger(__name__)

INSTANCE_COLUMN = "instance"
ALGORITHM_COLUMNS: Tuple[str, ...] = ("SA", "IP", "GR")
PERF_SCHEMA = TableSchema(key_column=INSTANCE_COLUMN, value_columns=ALGORITHM_COLUMNS)
PERF_CSV_COLUMNS: List[str] = PERF_SCHEMA.columns

ENV_PERF_CSV = "ISA_PERF_MATRIX_CSV"
DEFAULT_PERF_CSV_RELPATH = os.path.join("data", "results", "perf_matrix.csv")

NORMALIZATION_METHODS = ("gap", "ratio", "minmax")
OBJECTIVE_SENSES = ("min", "max")

# Scores are clipped to [0, 1]; drift beyond this is a real error, not rounding.
SCORE_TOLERANCE = 1e-9

# How many offending items to name before summarising in an error message.
_MAX_LISTED = 5


class PerfMatrixError(ValueError):
    """Raised when perf_matrix.csv cannot be built or fails its checks."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def resolve_perf_csv_path(csv_path: Optional[Any] = None) -> Path:
    """
    Resolve the perf_matrix.csv destination.

    Precedence: explicit argument > ``ISA_PERF_MATRIX_CSV`` env var >
    ``<repo_root>/data/results/perf_matrix.csv``.
    """
    return resolve_output_path(csv_path, ENV_PERF_CSV, DEFAULT_PERF_CSV_RELPATH)


def _summarize(items: Sequence[str]) -> str:
    listed = ", ".join(items[:_MAX_LISTED])
    if len(items) > _MAX_LISTED:
        listed += f", ... and {len(items) - _MAX_LISTED} more"
    return listed


def _to_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_instance_scores(
    raw_values: Dict[str, Any],
    sense: str = "min",
    method: str = "gap",
    best_known: Optional[Any] = None,
    instance_key: str = "<instance>",
    algorithms: Sequence[str] = ALGORITHM_COLUMNS,
) -> Dict[str, float]:
    """
    Normalize one instance's raw algorithm results to [0, 1], higher = better.

    Parameters
    ----------
    raw_values
        ``{algorithm: raw objective}``, e.g. average user ride time per
        algorithm.  Every algorithm in ``algorithms`` must be present, finite,
        and — for ``ratio`` — positive.
    sense
        ``"min"`` when lower raw values are better (cost, ride time),
        ``"max"`` when higher is better (requests served, profit).
    method
        ``"gap"`` (default), ``"ratio"``, or ``"minmax"`` — see module docstring.
    best_known
        Optional per-instance reference value.  Defaults to the best value
        among ``algorithms``, which makes the winner score exactly 1.0.
    instance_key
        Used only to make error messages actionable.

    Raises
    ------
    PerfMatrixError
        On a missing/non-finite raw value, an unusable ``best`` reference, or
        an unknown ``sense`` / ``method``.
    """
    if sense not in OBJECTIVE_SENSES:
        raise PerfMatrixError(f"sense must be one of {OBJECTIVE_SENSES}, got {sense!r}")
    if method not in NORMALIZATION_METHODS:
        raise PerfMatrixError(
            f"method must be one of {NORMALIZATION_METHODS}, got {method!r}"
        )
    if not isinstance(raw_values, dict):
        raise PerfMatrixError(
            f"{instance_key}: raw values must be a dict, got {type(raw_values).__name__}"
        )

    values: Dict[str, float] = {}
    missing: List[str] = []
    for algorithm in algorithms:
        value = _to_float(raw_values.get(algorithm))
        if math.isnan(value) or math.isinf(value):
            missing.append(algorithm)
            continue
        values[algorithm] = value

    if missing:
        raise PerfMatrixError(
            f"{instance_key}: no usable result for {', '.join(missing)} "
            f"(got {', '.join(repr(raw_values.get(a)) for a in missing)}); "
            "every algorithm needs a finite raw value"
        )

    observed = list(values.values())
    best = min(observed) if sense == "min" else max(observed)
    worst = max(observed) if sense == "min" else min(observed)

    if best_known is not None:
        reference = _to_float(best_known)
        if math.isnan(reference) or math.isinf(reference):
            raise PerfMatrixError(
                f"{instance_key}: best_known must be a finite number, got {best_known!r}"
            )
        best = reference

    scores: Dict[str, float] = {}

    if method == "minmax":
        span = (worst - best) if sense == "min" else (best - worst)
        for algorithm, value in values.items():
            if abs(span) <= SCORE_TOLERANCE:
                # All algorithms tied: nobody is worse, so everybody scores 1.0.
                scores[algorithm] = 1.0
                continue
            scores[algorithm] = (
                (worst - value) / span if sense == "min" else (value - worst) / span
            )
    else:
        if abs(best) <= SCORE_TOLERANCE:
            raise PerfMatrixError(
                f"{instance_key}: method {method!r} divides by the best value, which is "
                f"{best!r}; supply a non-zero best_known or use method='minmax'"
            )

        for algorithm, value in values.items():
            if method == "ratio":
                if value == 0.0:
                    raise PerfMatrixError(
                        f"{instance_key}: {algorithm} raw value is 0, which method "
                        "'ratio' cannot normalize; use method='minmax'"
                    )
                score = (best / value) if sense == "min" else (value / best)
            else:  # gap
                score = 1.0 - abs(value - best) / abs(best)
            scores[algorithm] = score

    for algorithm, score in scores.items():
        if score > 1.0 + SCORE_TOLERANCE:
            LOGGER.warning(
                "%s: %s scored %.6f above 1.0 — best_known is stale (an algorithm "
                "beat it); clipping to 1.0",
                instance_key,
                algorithm,
                score,
            )
        scores[algorithm] = min(1.0, max(0.0, score))

    return scores


def build_perf_rows(
    raw_results: Dict[Any, Dict[str, Any]],
    sense: str = "min",
    method: str = "gap",
    best_known: Optional[Dict[Any, Any]] = None,
) -> List[Dict[str, str]]:
    """
    Normalize a whole ``{instance: {algorithm: raw}}`` mapping into CSV rows.

    Instance keys are reduced to file stems, so raw results keyed by path or
    file name line up with features.csv.
    """
    best_known_by_stem: Dict[str, Any] = {}
    if best_known:
        best_known_by_stem = {
            instance_stem(key): value for key, value in best_known.items()
        }

    rows: List[Dict[str, str]] = []
    for instance_ref, raw_values in raw_results.items():
        stem = instance_stem(instance_ref)
        scores = normalize_instance_scores(
            raw_values,
            sense=sense,
            method=method,
            best_known=best_known_by_stem.get(stem),
            instance_key=stem,
        )
        rows.append(
            normalize_row(
                PERF_SCHEMA, stem, scores, logger=LOGGER, label="perf_matrix.csv"
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Raw results input
# ---------------------------------------------------------------------------

_LONG_ALGORITHM_COLUMNS = ("algorithm", "alg", "solver", "method")
_LONG_VALUE_COLUMNS = ("urt", "value", "objective", "raw", "avgurt", "cost", "result")
_BEST_KNOWN_COLUMNS = ("best_known", "best", "bks", "optimum")


def read_raw_results_csv(
    path: Any,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Read raw (un-normalized) algorithm results from CSV.

    Two layouts are accepted:

      wide  ``instance,SA,IP,GR[,best_known]``
      long  ``instance,algorithm,urt``   (value column may also be named
            value/objective/raw/avgurt/cost/result)

    Returns ``({instance: {algorithm: raw}}, {instance: best_known})``.
    """
    import csv

    csv_path = Path(str(path))
    if not csv_path.is_file():
        raise PerfMatrixError(f"raw results file does not exist: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [str(name).strip() for name in (reader.fieldnames or [])]
        if not fieldnames:
            raise PerfMatrixError(f"raw results file has no header: {csv_path}")

        lowered = {name.lower(): name for name in fieldnames}
        key_column = lowered.get(INSTANCE_COLUMN) or fieldnames[0]

        algorithm_column = next(
            (lowered[name] for name in _LONG_ALGORITHM_COLUMNS if name in lowered), None
        )
        best_column = next(
            (lowered[name] for name in _BEST_KNOWN_COLUMNS if name in lowered), None
        )

        raw_results: Dict[str, Dict[str, float]] = {}
        best_known: Dict[str, float] = {}

        if algorithm_column is not None:
            value_column = next(
                (lowered[name] for name in _LONG_VALUE_COLUMNS if name in lowered), None
            )
            if value_column is None:
                raise PerfMatrixError(
                    f"{csv_path}: long-format raw results need a value column, one of "
                    f"{', '.join(_LONG_VALUE_COLUMNS)}; got {', '.join(fieldnames)}"
                )

            for record in reader:
                stem = instance_stem(record.get(key_column, ""))
                algorithm = str(record.get(algorithm_column, "")).strip()
                raw_results.setdefault(stem, {})[algorithm] = _to_float(
                    record.get(value_column)
                )
                if best_column:
                    reference = _to_float(record.get(best_column))
                    if not math.isnan(reference):
                        best_known[stem] = reference
        else:
            present = [name for name in ALGORITHM_COLUMNS if name in fieldnames]
            if not present:
                raise PerfMatrixError(
                    f"{csv_path}: wide-format raw results need columns "
                    f"{', '.join(ALGORITHM_COLUMNS)}; got {', '.join(fieldnames)}"
                )

            for record in reader:
                stem = instance_stem(record.get(key_column, ""))
                raw_results[stem] = {
                    algorithm: _to_float(record.get(algorithm))
                    for algorithm in present
                }
                if best_column:
                    reference = _to_float(record.get(best_column))
                    if not math.isnan(reference):
                        best_known[stem] = reference

    if not raw_results:
        raise PerfMatrixError(f"raw results file has no rows: {csv_path}")

    return raw_results, best_known


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def read_perf_matrix_csv(
    csv_path: Optional[Any] = None,
) -> Tuple[Dict[str, Dict[str, str]], TableReport]:
    """Read perf_matrix.csv into ``{instance_stem: canonical_row}`` plus a report."""
    return read_table(resolve_perf_csv_path(csv_path), PERF_SCHEMA)


def write_perf_matrix_csv(
    rows: Iterable[Dict[str, str]], csv_path: Optional[Any] = None
) -> Path:
    """Rewrite perf_matrix.csv from scratch with the canonical schema."""
    return write_table(resolve_perf_csv_path(csv_path), PERF_SCHEMA, rows)


def append_perf_row(
    instance_ref: Any,
    scores: Dict[str, Any],
    csv_path: Optional[Any] = None,
    on_duplicate: str = "replace",
) -> Path:
    """
    Append (or replace) one already-normalized row in perf_matrix.csv.

    Use this to record results incrementally as runs finish; use
    :func:`export_perf_matrix` to build the whole file from raw results.
    """
    row = normalize_row(
        PERF_SCHEMA, instance_ref, scores, logger=LOGGER, label="perf_matrix.csv"
    )
    return append_rows(
        resolve_perf_csv_path(csv_path),
        PERF_SCHEMA,
        [row],
        on_duplicate=on_duplicate,
        logger=LOGGER,
        label="perf_matrix.csv",
    )


def export_perf_matrix(
    raw_results: Dict[Any, Dict[str, Any]],
    csv_path: Optional[Any] = None,
    features_path: Optional[Any] = None,
    sense: str = "min",
    method: str = "gap",
    best_known: Optional[Dict[Any, Any]] = None,
    align_to_features: bool = True,
    validate: bool = True,
) -> Path:
    """
    Normalize raw results and write the complete perf_matrix.csv.

    With ``align_to_features`` (default) the rows are emitted in features.csv
    order, which is what MATILDA's index assertion requires.  With ``validate``
    (default) the finished file is checked and :class:`PerfMatrixError` is
    raised listing every problem found.
    """
    rows = build_perf_rows(
        raw_results, sense=sense, method=method, best_known=best_known
    )

    if align_to_features:
        features_keys = _features_keys(features_path)
        if features_keys:
            by_key = {row[INSTANCE_COLUMN]: row for row in rows}
            ordered = [by_key.pop(key) for key in features_keys if key in by_key]
            # Anything not in features.csv keeps its original relative order and
            # is reported by the alignment check below rather than dropped here.
            ordered.extend(row for row in rows if row[INSTANCE_COLUMN] in by_key)
            rows = ordered

    path = write_perf_matrix_csv(rows, csv_path)

    if validate:
        problems = validate_perf_matrix_csv(path, features_path=features_path)
        if problems:
            raise PerfMatrixError(
                f"perf_matrix.csv failed its checks ({path}):\n  - "
                + "\n  - ".join(problems)
            )

    return path


def align_perf_matrix_to_features(
    csv_path: Optional[Any] = None, features_path: Optional[Any] = None
) -> Path:
    """
    Rewrite perf_matrix.csv in features.csv row order.

    Raises :class:`PerfMatrixError` when the two key sets differ, since
    reordering cannot fix a genuine mismatch.
    """
    path = resolve_perf_csv_path(csv_path)
    rows, _ = read_perf_matrix_csv(path)
    features_keys = _features_keys(features_path)

    if not features_keys:
        raise PerfMatrixError(
            f"cannot align: no usable features.csv at "
            f"{resolve_features_csv_path(features_path)}"
        )

    problems = check_index_alignment(path, features_path, require_order=False)
    if problems:
        raise PerfMatrixError("cannot align:\n  - " + "\n  - ".join(problems))

    return write_perf_matrix_csv([rows[key] for key in features_keys], path)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _features_keys(features_path: Optional[Any] = None) -> List[str]:
    """Instance keys of features.csv in file order (empty when absent)."""
    path = resolve_features_csv_path(features_path)
    if not path.is_file():
        return []
    rows, _ = read_table(path, FEATURES_SCHEMA)
    return list(rows)


def check_score_ranges(csv_path: Optional[Any] = None) -> List[str]:
    """
    Check that every score is present, numeric, finite and inside [0, 1].

    Returns a list of problem descriptions, empty when the scores are valid.
    """
    path = resolve_perf_csv_path(csv_path)
    rows, _ = read_table(path, PERF_SCHEMA)

    missing: List[str] = []
    out_of_range: List[str] = []

    for key, row in rows.items():
        for algorithm in ALGORITHM_COLUMNS:
            cell = row.get(algorithm, "")
            if not cell:
                missing.append(f"{key}/{algorithm}")
                continue
            score = _to_float(cell)
            if math.isnan(score) or math.isinf(score):
                missing.append(f"{key}/{algorithm}")
                continue
            if score < -SCORE_TOLERANCE or score > 1.0 + SCORE_TOLERANCE:
                out_of_range.append(f"{key}/{algorithm}={score:g}")

    problems: List[str] = []
    if missing:
        problems.append(
            f"{len(missing)} missing or non-numeric score(s): {_summarize(missing)}"
        )
    if out_of_range:
        problems.append(
            f"{len(out_of_range)} score(s) outside [0, 1]: {_summarize(out_of_range)} "
            "- re-run the export so normalization is applied"
        )
    return problems


def check_score_spread(
    csv_path: Optional[Any] = None, clipped_fraction: float = 0.2
) -> List[str]:
    """
    Report algorithm columns that carry little information.

    These are warnings, not errors — the file is still valid — but both cases
    hurt the instance space:

      * a column with no variance is dropped by MATILDA preprocessing, so that
        algorithm gets no footprint;
      * a pile of exact 0.0 scores means ``method="gap"`` clipped them (the
        algorithm is more than twice the best cost), which flattens real
        differences.  ``method="ratio"`` never clips.
    """
    path = resolve_perf_csv_path(csv_path)
    rows, _ = read_table(path, PERF_SCHEMA)
    if not rows:
        return []

    warnings: List[str] = []

    for algorithm in ALGORITHM_COLUMNS:
        scores = [
            _to_float(row.get(algorithm, ""))
            for row in rows.values()
            if str(row.get(algorithm, "")).strip()
        ]
        scores = [score for score in scores if not math.isnan(score)]
        if not scores:
            continue

        # Clipping is reported first: an all-zero column is also constant, but
        # "clipped" names the cause and has a fix.
        clipped = sum(1 for score in scores if score <= SCORE_TOLERANCE)
        fraction = clipped / len(scores)
        if fraction > clipped_fraction:
            warnings.append(
                f"{algorithm}: {clipped}/{len(scores)} scores ({fraction:.0%}) are 0.0, "
                "clipped because the algorithm exceeds twice the best cost - "
                "consider --method ratio, which never clips"
            )
            continue

        if max(scores) - min(scores) <= SCORE_TOLERANCE:
            warnings.append(
                f"{algorithm}: every score is {scores[0]:g} - a constant column is "
                "dropped by MATILDA preprocessing, so this algorithm gets no footprint"
            )

    return warnings


def check_index_alignment(
    csv_path: Optional[Any] = None,
    features_path: Optional[Any] = None,
    require_order: bool = True,
) -> List[str]:
    """
    Check perf_matrix.csv against features.csv row-for-row.

    MATILDA asserts ``list(features.index) == list(performance.index)``, so by
    default both the key set and the order must match.
    """
    perf_file = resolve_perf_csv_path(csv_path)
    features_file = resolve_features_csv_path(features_path)

    if not features_file.is_file():
        return [
            f"features.csv not found at {features_file} - generate instances first "
            "(T12) so the index can be checked"
        ]

    perf_rows, _ = read_table(perf_file, PERF_SCHEMA)
    features_keys = _features_keys(features_path)
    perf_keys = list(perf_rows)

    problems: List[str] = []

    missing_in_perf = [key for key in features_keys if key not in perf_rows]
    if missing_in_perf:
        problems.append(
            f"{len(missing_in_perf)} instance(s) in features.csv have no perf_matrix "
            f"row: {_summarize(missing_in_perf)}"
        )

    features_set = set(features_keys)
    extra_in_perf = [key for key in perf_keys if key not in features_set]
    if extra_in_perf:
        problems.append(
            f"{len(extra_in_perf)} instance(s) in perf_matrix.csv are absent from "
            f"features.csv: {_summarize(extra_in_perf)}"
        )

    if require_order and not problems and perf_keys != features_keys:
        first_diff = next(
            i for i, (a, b) in enumerate(zip(features_keys, perf_keys)) if a != b
        )
        problems.append(
            f"row order differs from features.csv at position {first_diff}: "
            f"features.csv has {features_keys[first_diff]!r}, perf_matrix.csv has "
            f"{perf_keys[first_diff]!r} - run align_perf_matrix_to_features()"
        )

    return problems


def validate_perf_matrix_csv(
    csv_path: Optional[Any] = None,
    features_path: Optional[Any] = None,
    require_order: bool = True,
    check_alignment: bool = True,
) -> List[str]:
    """
    Run every perf_matrix.csv check: schema, score range, index alignment.

    Returns a list of problem descriptions; an empty list means the file is
    ready for MATILDA (T14).
    """
    path = resolve_perf_csv_path(csv_path)
    if not path.is_file():
        return [f"file does not exist: {path}"]

    _, report = read_table(path, PERF_SCHEMA)
    if not report.file_exists:
        return ["file is empty"]

    problems = report.describe()
    problems.extend(check_score_ranges(path))
    if check_alignment:
        problems.extend(
            check_index_alignment(path, features_path, require_order=require_order)
        )
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and check data/results/perf_matrix.csv from raw SA/IP/GR results."
        )
    )
    parser.add_argument(
        "--raw",
        help=(
            "Raw results CSV. Wide: instance,SA,IP,GR[,best_known]. "
            "Long: instance,algorithm,urt. Omit to only validate the existing file."
        ),
    )
    parser.add_argument("--out", help="perf_matrix.csv path (default: data/results/)")
    parser.add_argument("--features", help="features.csv path (default: data/results/)")
    parser.add_argument(
        "--sense",
        choices=OBJECTIVE_SENSES,
        default="min",
        help="min when lower raw values are better (default), max otherwise",
    )
    parser.add_argument(
        "--method",
        choices=NORMALIZATION_METHODS,
        default="gap",
        help="normalization method (default: gap)",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="keep raw-results row order instead of features.csv order",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint: build perf_matrix.csv from raw results, or validate it."""
    args = _build_arg_parser().parse_args(argv)

    perf_path = resolve_perf_csv_path(args.out)

    if args.raw:
        raw_results, best_known = read_raw_results_csv(args.raw)
        try:
            export_perf_matrix(
                raw_results,
                csv_path=perf_path,
                features_path=args.features,
                sense=args.sense,
                method=args.method,
                best_known=best_known or None,
                align_to_features=not args.no_align,
            )
        except PerfMatrixError as exc:
            print(str(exc))
            return 1
        print(
            f"Wrote {len(raw_results)} row(s) to {perf_path} "
            f"(sense={args.sense}, method={args.method})"
        )
        _print_warnings(perf_path)
        return 0

    problems = validate_perf_matrix_csv(perf_path, features_path=args.features)
    if problems:
        print(f"perf_matrix.csv has {len(problems)} problem(s) ({perf_path}):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"perf_matrix.csv passes all checks: {perf_path}")
    _print_warnings(perf_path)
    return 0


def _print_warnings(perf_path: Path) -> None:
    """Print non-fatal spread warnings after a successful build or check."""
    warnings = check_score_spread(perf_path)
    if not warnings:
        return
    print(f"\n{len(warnings)} warning(s) - the file is valid, but:")
    for warning in warnings:
        print(f"  - {warning}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
