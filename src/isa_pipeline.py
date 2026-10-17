"""
isa_pipeline.py — T14 (ISA-HODBRP)

End-to-end MATILDA pipeline: read ``data/results/features.csv`` (T12) and
``data/results/perf_matrix.csv`` (T13), preprocess, project to a 2-D instance
space, compute per-algorithm footprints, and write the figures for the chapter.

    python src/isa_pipeline.py
    python src/isa_pipeline.py --report-only
    python src/isa_pipeline.py --keep-dynamic --nan-threshold 0.2

Outputs
-------
    data/results/instance_space.csv     instance, z1, z2, scores, best algorithm
    data/results/footprint_summary.csv  per algorithm: area, purity, density
    figures/instance_space.pdf          scatter coloured by best algorithm
    figures/footprint_<ALG>.pdf         one per algorithm

Projection backends
-------------------
``--method package`` runs the real thing: the University of Melbourne's Python
port, published on PyPI as **instancespace** (the PyPI package called
``matilda`` is an unrelated project).  It executes PREPROCESSING -> PRELIM ->
SIFTED -> PILOT, which includes Box-Cox/z normalisation, outlier removal and
correlation-based feature selection before the projection.  **Use this for
anything going into the thesis.**

The package pins ``numpy<2`` and ``pandas<3``, so it cannot share an
environment with REQreate.  Install it in its own virtualenv and run this
script from there:

    python -m venv .venv-isa
    .venv-isa/Scripts/pip install instancespace
    .venv-isa/Scripts/python src/isa_pipeline.py --method package

``--method pilot`` (default) computes the **analytic PILOT solution**
(Muñoz et al., 2018) with numpy alone, so the pipeline runs in any environment:

    Z = argmin ||X - B Z||^2 + ||Y - C Z||^2

for a rank-2 Z, which is the truncated SVD of the standardized, concatenated
``[features | performance]`` matrix.  It uses performance data, so it is a real
instance space rather than a feature PCA, and it is the closed form MATILDA
starts from before its numerical refinement.  It does **not** reproduce the
package: it skips PRELIM's transformations and SIFTED's feature selection, so
coordinates differ materially (on a 200-instance synthetic set the two agreed
on every algorithm label but their pairwise-distance correlation was only
0.46).  Treat it as a fast approximation for iterating, not as a substitute.

Footprints
----------
An algorithm's footprint is the convex hull of the instances it performs well
on (score >= ``--footprint-threshold``), reported with area, purity (share of
instances inside the hull that are actually good) and density.  MATILDA's own
footprint analysis uses alpha-shapes with purity-driven pruning, so these hulls
are an approximation — deliberately simple, and stated as such in the figures.
"""

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ISA_DIR = _REPO_ROOT / "isa-features-plan-implementation"
if _ISA_DIR.is_dir() and str(_ISA_DIR) not in sys.path:
    sys.path.insert(0, str(_ISA_DIR))

from features_export import resolve_features_csv_path  # noqa: E402
from perf_matrix_export import (  # noqa: E402
    ALGORITHM_COLUMNS,
    check_index_alignment,
    check_score_ranges,
    resolve_perf_csv_path,
)


LOGGER = logging.getLogger(__name__)

INSTANCE_COLUMN = "instance"

DEFAULT_INSTANCE_SPACE_CSV = _REPO_ROOT / "data" / "results" / "instance_space.csv"
DEFAULT_FOOTPRINT_CSV = _REPO_ROOT / "data" / "results" / "footprint_summary.csv"
DEFAULT_FIGURES_DIR = _REPO_ROOT / "figures"

# Constant for the static study (feature_list.md, features 19-21).
DYNAMIC_COLUMNS = (
    "request_flux_dynamism",
    "static_to_dynamic_ratio",
    "avg_request_urgency",
)

DEFAULT_NAN_THRESHOLD = 0.10
DEFAULT_FOOTPRINT_THRESHOLD = 0.80
MIN_RECOMMENDED_INSTANCES = 100
MIN_FEATURES_FOR_PROJECTION = 2

PROJECTION_METHODS = ("pilot", "package")


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot run to completion."""


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_inputs(
    features_path: Optional[Any] = None,
    perf_path: Optional[Any] = None,
    strict: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load features.csv and perf_matrix.csv, aligned on the instance index.

    With ``strict`` (default) any index mismatch is an error, using the T13
    check for the message.  Otherwise the two are reduced to their common
    instances, in features.csv order, and the drop is logged.
    """
    features_file = resolve_features_csv_path(features_path)
    perf_file = resolve_perf_csv_path(perf_path)

    for label, path in (("features.csv", features_file), ("perf_matrix.csv", perf_file)):
        if not path.is_file():
            raise PipelineError(
                f"{label} not found at {path} — run the generator (T12) and "
                f"perf_matrix_export.py (T13) first"
            )

    features = pd.read_csv(features_file, index_col=INSTANCE_COLUMN)
    performance = pd.read_csv(perf_file, index_col=INSTANCE_COLUMN)

    score_problems = check_score_ranges(perf_file)
    if score_problems:
        raise PipelineError(
            "perf_matrix.csv scores are not usable:\n  - " + "\n  - ".join(score_problems)
        )

    alignment_problems = check_index_alignment(perf_file, features_file)
    if alignment_problems:
        if strict:
            raise PipelineError(
                "features.csv and perf_matrix.csv are not aligned:\n  - "
                + "\n  - ".join(alignment_problems)
                + "\n  Re-run perf_matrix_export.py, or pass --no-strict to use "
                "only the instances present in both."
            )

        common = [key for key in features.index if key in set(performance.index)]
        if not common:
            raise PipelineError(
                "features.csv and perf_matrix.csv share no instances; nothing to fit"
            )
        LOGGER.warning(
            "using %d instance(s) present in both files, dropping %d",
            len(common),
            len(set(features.index) | set(performance.index)) - len(common),
        )
        features = features.loc[common]
        performance = performance.loc[common]
    else:
        performance = performance.loc[features.index]

    missing_algorithms = [a for a in ALGORITHM_COLUMNS if a not in performance.columns]
    if missing_algorithms:
        raise PipelineError(
            f"perf_matrix.csv is missing algorithm column(s): "
            f"{', '.join(missing_algorithms)}"
        )

    return features, performance[list(ALGORITHM_COLUMNS)]


# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingReport:
    """Which columns were dropped or imputed, and why."""

    n_instances: int = 0
    n_columns_in: int = 0
    dropped_dynamic: List[str] = field(default_factory=list)
    dropped_high_nan: List[Tuple[str, float]] = field(default_factory=list)
    imputed: List[Tuple[str, int]] = field(default_factory=list)
    dropped_constant: List[str] = field(default_factory=list)
    kept: List[str] = field(default_factory=list)

    def describe(self) -> List[str]:
        lines = [
            f"instances            : {self.n_instances}",
            f"features in          : {self.n_columns_in}",
        ]
        if self.dropped_dynamic:
            lines.append(
                f"dropped dynamic      : {', '.join(self.dropped_dynamic)}"
            )
        if self.dropped_high_nan:
            detail = ", ".join(f"{name} ({share:.0%} NaN)" for name, share in self.dropped_high_nan)
            lines.append(f"dropped high-NaN     : {detail}")
        if self.imputed:
            detail = ", ".join(f"{name} ({count})" for name, count in self.imputed)
            lines.append(f"median-imputed       : {detail}")
        if self.dropped_constant:
            lines.append(
                f"dropped constant     : {', '.join(self.dropped_constant)}"
            )
        lines.append(f"features kept ({len(self.kept)})     : {', '.join(self.kept)}")
        return lines


def preprocess_features(
    features: pd.DataFrame,
    nan_threshold: float = DEFAULT_NAN_THRESHOLD,
    drop_dynamic: bool = True,
    dynamic_columns: Sequence[str] = DYNAMIC_COLUMNS,
) -> Tuple[pd.DataFrame, PreprocessingReport]:
    """
    Apply implementation_context.md §7 preprocessing.

    Order matters and differs slightly from the order §7 lists them in:

      1. drop the static-study dynamic columns (they are constant by design);
      2. drop columns with more than ``nan_threshold`` missing;
      3. median-impute what is left;
      4. drop constant columns **after** imputation — imputing can flatten a
         column that still had variance among its non-missing values, and a
         constant column makes the projection singular.
    """
    if not 0.0 <= nan_threshold <= 1.0:
        raise ValueError(f"nan_threshold must be in [0, 1], got {nan_threshold}")

    frame = features.apply(pd.to_numeric, errors="coerce")
    report = PreprocessingReport(
        n_instances=int(frame.shape[0]), n_columns_in=int(frame.shape[1])
    )

    if drop_dynamic:
        present = [name for name in dynamic_columns if name in frame.columns]
        if present:
            report.dropped_dynamic = present
            frame = frame.drop(columns=present)

    nan_share = frame.isna().mean()
    high_nan = [name for name in frame.columns if nan_share[name] > nan_threshold]
    if high_nan:
        report.dropped_high_nan = [(name, float(nan_share[name])) for name in high_nan]
        frame = frame.drop(columns=high_nan)

    for name in frame.columns:
        missing = int(frame[name].isna().sum())
        if not missing:
            continue
        median = frame[name].median()
        if pd.isna(median):
            # Every value missing: nothing to impute from, drop as high-NaN.
            report.dropped_high_nan.append((name, 1.0))
            frame = frame.drop(columns=[name])
            continue
        frame[name] = frame[name].fillna(median)
        report.imputed.append((name, missing))

    spread = frame.max() - frame.min()
    constant = [name for name in frame.columns if not (spread[name] > 0)]
    if constant:
        report.dropped_constant = constant
        frame = frame.drop(columns=constant)

    report.kept = list(frame.columns)
    return frame, report


def _standardize(matrix: np.ndarray) -> np.ndarray:
    """Z-score columns; zero-variance columns become all-zero rather than NaN."""
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return (matrix - mean) / std


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

@dataclass
class ProjectionResult:
    """2-D instance-space coordinates plus how well they explain the inputs."""

    coordinates: np.ndarray
    feature_weights: np.ndarray
    r2_features: float
    r2_performance: float
    backend: str


def pilot_analytic(
    features: np.ndarray, performance: np.ndarray, n_components: int = 2
) -> ProjectionResult:
    """
    Analytic PILOT: rank-2 truncated SVD of standardized ``[features | performance]``.

    Returns coordinates ``Z`` (n x 2), the least-squares feature weights ``A``
    (m x 2) that map standardized features onto ``Z``, and the share of feature
    and performance variance the projection reconstructs.
    """
    if features.ndim != 2 or performance.ndim != 2:
        raise ValueError("features and performance must both be 2-D")
    if features.shape[0] != performance.shape[0]:
        raise ValueError("features and performance must have the same number of rows")
    if features.shape[0] < n_components:
        raise PipelineError(
            f"need at least {n_components} instances to build a {n_components}-D "
            f"instance space, got {features.shape[0]}"
        )
    if features.shape[1] < MIN_FEATURES_FOR_PROJECTION:
        raise PipelineError(
            f"need at least {MIN_FEATURES_FOR_PROJECTION} usable features after "
            f"preprocessing, got {features.shape[1]}"
        )

    x = _standardize(features.astype(float))
    y = _standardize(performance.astype(float))
    combined = np.hstack([x, y])

    u, singular_values, vt = np.linalg.svd(combined, full_matrices=False)
    rank = min(n_components, singular_values.size)

    # Fix each component's sign so runs are reproducible across platforms.
    for component in range(rank):
        dominant = np.argmax(np.abs(u[:, component]))
        if u[dominant, component] < 0:
            u[:, component] *= -1.0
            vt[component, :] *= -1.0

    coordinates = u[:, :rank] * singular_values[:rank]
    if rank < n_components:  # degenerate input: pad so the output stays 2-D
        coordinates = np.hstack(
            [coordinates, np.zeros((coordinates.shape[0], n_components - rank))]
        )

    reconstruction = coordinates[:, :rank] @ vt[:rank, :]
    x_hat = reconstruction[:, : x.shape[1]]
    y_hat = reconstruction[:, x.shape[1] :]

    feature_weights, *_ = np.linalg.lstsq(x, coordinates, rcond=None)

    return ProjectionResult(
        coordinates=coordinates,
        feature_weights=feature_weights,
        r2_features=_r2(x, x_hat),
        r2_performance=_r2(y, y_hat),
        backend="pilot-analytic",
    )


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    total = float(np.sum(actual**2))
    if total <= 0.0:
        return float("nan")
    residual = float(np.sum((actual - predicted) ** 2))
    return 1.0 - residual / total


def build_metadata_frame(
    features: pd.DataFrame, performance: pd.DataFrame
) -> pd.DataFrame:
    """
    Build the ``metadata.csv`` layout the ``instancespace`` package expects.

    That package keys off column-name prefixes, not position: ``Instances`` for
    the label, ``feature_*`` for features, ``algo_*`` for algorithms (see
    ``instancespace.data.metadata.Metadata.from_data_frame``).
    """
    frame = pd.DataFrame({"Instances": features.index.astype(str)})
    for name in features.columns:
        frame[f"feature_{name}"] = features[name].to_numpy()
    for name in performance.columns:
        frame[f"algo_{name}"] = performance[name].to_numpy()
    return frame


def _run_until_pilot(space: Any) -> Dict[str, Any]:
    """
    Drive ``instancespace`` stage by stage, stopping once PILOT has produced Z.

    ``run_until_stage(PilotStage)`` stops *before* PILOT, and running the full
    ``build()`` would also run PYTHIA (SVM), CLOISTER and TRACE — far slower,
    and they fail on small datasets.  Iterating and stopping at the first
    output carrying the projection avoids depending on the stage ordering.
    """
    collected: Dict[str, Any] = {}
    for annotated in space.run_iter():
        output = getattr(annotated, "output", annotated)
        if hasattr(output, "_asdict"):
            output = output._asdict()
        elif hasattr(output, "__dataclass_fields__"):
            output = {name: getattr(output, name) for name in output.__dataclass_fields__}

        if isinstance(output, dict):
            collected.update(output)
        if any(key in collected for key in ("pilot_z", "z")):
            break

    return collected


def _package_projection(
    features: pd.DataFrame,
    performance: pd.DataFrame,
    package_good_threshold: float = DEFAULT_FOOTPRINT_THRESHOLD,
) -> ProjectionResult:
    """
    Project with the installed ``instancespace`` package (the Melbourne port).

    Runs PREPROCESSING -> PRELIM -> SIFTED -> PILOT and stops there; footprints
    are then computed the same way for both backends so the two are comparable.

    Raises with a precise message rather than silently falling back — a
    different projection would change every figure.
    """
    try:
        from instancespace import InstanceSpace, InstanceSpaceOptions
        from instancespace.data.metadata import Metadata
        from instancespace.data.options import PerformanceOptions
    except ImportError as exc:
        raise PipelineError(
            "the instancespace package is not importable here "
            f"({exc}). It requires numpy<2 and pandas<3, so install it in its own "
            "virtualenv rather than alongside REQreate:\n"
            "    python -m venv .venv-isa\n"
            "    .venv-isa/Scripts/pip install instancespace\n"
            "    .venv-isa/Scripts/python src/isa_pipeline.py --method package\n"
            "Or use --method pilot, which needs no extra dependency."
        ) from exc

    metadata = Metadata.from_data_frame(build_metadata_frame(features, performance))
    options = InstanceSpaceOptions.default(
        parallel=None,
        # perf_matrix.csv scores are normalized so higher = better, but the
        # package minimises by default; without max_perf every algorithm is
        # scored as a cost and PRELIM marks nothing as good.
        perf=PerformanceOptions.default(
            max_perf=True, abs_perf=True, epsilon=package_good_threshold
        ),
        auto=None,
        bound=None,
        norm=None,
        selvars=None,
        sifted=None,
        pilot=None,
        cloister=None,
        pythia=None,
        trace=None,
        outputs=None,
    )

    space = InstanceSpace(metadata, options)
    try:
        outputs = _run_until_pilot(space)
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            f"instancespace failed during projection: {type(exc).__name__}: {exc}\n"
            "  A NotEnoughFeatureError here means its SIFTED stage found too few "
            "features correlated with performance - that needs real algorithm "
            "results and enough instances (~100+), not a pipeline fix."
        ) from exc

    coordinates = np.asarray(_require_output(outputs, ("pilot_z", "z")), dtype=float)
    weights = np.asarray(_require_output(outputs, ("pilot_a", "a")), dtype=float).T
    r2_raw = outputs.get("pilot_r2")

    x = _standardize(features.to_numpy(dtype=float))
    y = _standardize(performance.to_numpy(dtype=float))
    r2_features, r2_performance = _reconstruction_r2(coordinates, x, y)
    if r2_raw is not None:
        LOGGER.info("instancespace reported pilot r2: %s", np.ravel(r2_raw)[:4])

    return ProjectionResult(
        coordinates=coordinates,
        feature_weights=weights,
        r2_features=r2_features,
        r2_performance=r2_performance,
        backend="instancespace-pilot",
    )


def _require_output(outputs: Dict[str, Any], names: Sequence[str]) -> Any:
    """Pull the first present key, naming what was available when none match."""
    for name in names:
        if name in outputs and outputs[name] is not None:
            return outputs[name]
    raise PipelineError(
        f"instancespace returned none of {', '.join(names)}; available keys: "
        f"{', '.join(sorted(outputs))}"
    )


def _reconstruction_r2(
    coordinates: np.ndarray, x: np.ndarray, y: np.ndarray
) -> Tuple[float, float]:
    """How much of the standardized inputs a least-squares fit on Z recovers."""
    design = np.hstack([coordinates, np.ones((coordinates.shape[0], 1))])
    x_hat = design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_hat = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return _r2(x, x_hat), _r2(y, y_hat)


def project(
    features: pd.DataFrame, performance: pd.DataFrame, method: str = "pilot"
) -> Tuple[pd.DataFrame, ProjectionResult]:
    """Project instances into 2-D and return coordinates indexed by instance."""
    if method not in PROJECTION_METHODS:
        raise ValueError(f"method must be one of {PROJECTION_METHODS}, got {method!r}")

    if method == "package":
        result = _package_projection(features, performance)
    else:
        result = pilot_analytic(features.to_numpy(), performance.to_numpy())

    coordinates = pd.DataFrame(
        result.coordinates[:, :2], index=features.index, columns=["z1", "z2"]
    )
    return coordinates, result


# ---------------------------------------------------------------------------
# Footprints
# ---------------------------------------------------------------------------

@dataclass
class Footprint:
    """Convex-hull footprint of one algorithm's good instances."""

    algorithm: str
    n_good: int
    n_total: int
    hull: List[Tuple[float, float]]
    area: float
    purity: float
    density: float


def convex_hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Convex hull by monotone chain, counter-clockwise, without the repeated
    end point.  Collinear or duplicate inputs degrade to a segment or a point.
    """
    unique = sorted(set((float(x), float(y)) for x, y in points))
    if len(unique) <= 2:
        return unique

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: List[Tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def polygon_area(polygon: Sequence[Tuple[float, float]]) -> float:
    """Shoelace area; zero for degenerate (point or segment) hulls."""
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    for index in range(len(polygon)):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % len(polygon)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def point_in_polygon(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> bool:
    """Ray-casting test; points on the boundary count as inside."""
    if len(polygon) < 3:
        return False

    x, y = point
    inside = False
    for index in range(len(polygon)):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % len(polygon)]

        # On-segment check keeps hull vertices inside their own footprint.
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if abs(cross) < 1e-12 and min(x1, x2) - 1e-12 <= x <= max(x1, x2) + 1e-12 \
                and min(y1, y2) - 1e-12 <= y <= max(y1, y2) + 1e-12:
            return True

        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if crossing_x > x:
                inside = not inside

    return inside


def compute_footprints(
    coordinates: pd.DataFrame,
    performance: pd.DataFrame,
    threshold: float = DEFAULT_FOOTPRINT_THRESHOLD,
) -> Dict[str, Footprint]:
    """
    Build one footprint per algorithm from the instances it scores well on.

    ``purity`` is the share of instances inside the hull that are actually
    good, and ``density`` is good instances per unit hull area.
    """
    points = list(zip(coordinates["z1"].to_numpy(), coordinates["z2"].to_numpy()))
    footprints: Dict[str, Footprint] = {}

    for algorithm in performance.columns:
        scores = performance[algorithm].to_numpy()
        good_mask = scores >= threshold
        good_points = [point for point, is_good in zip(points, good_mask) if is_good]

        hull = convex_hull(good_points)
        area = polygon_area(hull)

        enclosed = [
            index for index, point in enumerate(points) if point_in_polygon(point, hull)
        ]
        purity = (
            float(np.mean(good_mask[enclosed])) if enclosed else float("nan")
        )
        density = (len(good_points) / area) if area > 0 else float("nan")

        footprints[str(algorithm)] = Footprint(
            algorithm=str(algorithm),
            n_good=int(good_mask.sum()),
            n_total=int(len(points)),
            hull=hull,
            area=area,
            purity=purity,
            density=density,
        )

    return footprints


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _pyplot():
    """Import pyplot with a non-interactive backend (no display on HPC nodes)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_instance_space(
    coordinates: pd.DataFrame, performance: pd.DataFrame, output_path: Path
) -> Path:
    """Scatter of the instance space, coloured by the best algorithm per instance."""
    plt = _pyplot()

    best = performance.idxmax(axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    for algorithm in performance.columns:
        mask = (best == algorithm).to_numpy()
        if not mask.any():
            continue
        axis.scatter(
            coordinates.loc[mask, "z1"],
            coordinates.loc[mask, "z2"],
            label=f"{algorithm} ({int(mask.sum())})",
            s=42,
            alpha=0.85,
            edgecolors="none",
        )

    axis.set_xlabel("z1")
    axis.set_ylabel("z2")
    axis.set_title(f"H-ODBRP instance space ({len(coordinates)} instances)")
    axis.legend(title="best algorithm", loc="best", frameon=False)
    axis.grid(alpha=0.25, linewidth=0.5)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)

    return output_path


def plot_footprint(
    coordinates: pd.DataFrame,
    performance: pd.DataFrame,
    footprint: Footprint,
    threshold: float,
    output_path: Path,
) -> Path:
    """One figure per algorithm: all instances, its good ones, and its hull."""
    plt = _pyplot()

    scores = performance[footprint.algorithm].to_numpy()
    good_mask = scores >= threshold
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    axis.scatter(
        coordinates.loc[~good_mask, "z1"],
        coordinates.loc[~good_mask, "z2"],
        s=30,
        c="#c9c9c9",
        label=f"score < {threshold:g}",
        edgecolors="none",
    )
    axis.scatter(
        coordinates.loc[good_mask, "z1"],
        coordinates.loc[good_mask, "z2"],
        s=44,
        c="#1f77b4",
        label=f"score >= {threshold:g} ({footprint.n_good})",
        edgecolors="none",
    )

    if len(footprint.hull) >= 3:
        polygon = footprint.hull + [footprint.hull[0]]
        axis.fill(
            [p[0] for p in polygon],
            [p[1] for p in polygon],
            facecolor="#1f77b4",
            alpha=0.14,
            edgecolor="#1f77b4",
            linewidth=1.4,
            label="footprint (convex hull)",
        )

    subtitle = (
        f"area={footprint.area:.2f}  purity={footprint.purity:.2f}  "
        f"density={footprint.density:.2f}"
        if footprint.area > 0
        else "hull is degenerate (fewer than 3 distinct good instances)"
    )
    axis.set_xlabel("z1")
    axis.set_ylabel("z2")
    axis.set_title(f"Footprint - {footprint.algorithm}\n{subtitle}", fontsize=11)
    axis.legend(loc="best", frameon=False, fontsize=9)
    axis.grid(alpha=0.25, linewidth=0.5)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)

    return output_path


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    features_path: Optional[Any] = None,
    perf_path: Optional[Any] = None,
    instance_space_path: Optional[Any] = None,
    footprint_summary_path: Optional[Any] = None,
    figures_dir: Optional[Any] = None,
    method: str = "pilot",
    nan_threshold: float = DEFAULT_NAN_THRESHOLD,
    drop_dynamic: bool = True,
    footprint_threshold: float = DEFAULT_FOOTPRINT_THRESHOLD,
    strict: bool = True,
    make_figures: bool = True,
) -> Dict[str, Any]:
    """
    Run the whole pipeline and write every output file.

    Returns a dict with the projection, footprints, preprocessing report and
    the paths written, so callers and tests can inspect the run.
    """
    features, performance = load_inputs(features_path, perf_path, strict=strict)

    if len(features) < MIN_RECOMMENDED_INSTANCES:
        LOGGER.warning(
            "%d instances is below the ~%d recommended for a meaningful projection "
            "(implementation_context.md section 7)",
            len(features),
            MIN_RECOMMENDED_INSTANCES,
        )

    clean_features, report = preprocess_features(
        features, nan_threshold=nan_threshold, drop_dynamic=drop_dynamic
    )

    if clean_features.shape[1] < MIN_FEATURES_FOR_PROJECTION:
        raise PipelineError(
            f"only {clean_features.shape[1]} feature(s) survived preprocessing "
            f"({', '.join(report.kept) or 'none'}); need at least "
            f"{MIN_FEATURES_FOR_PROJECTION}. Generate more varied instances, or "
            "relax --nan-threshold / pass --keep-dynamic."
        )

    coordinates, projection = project(clean_features, performance, method=method)
    footprints = compute_footprints(coordinates, performance, threshold=footprint_threshold)

    space_path = Path(str(instance_space_path or DEFAULT_INSTANCE_SPACE_CSV))
    space_path.parent.mkdir(parents=True, exist_ok=True)
    instance_space = coordinates.join(performance)
    instance_space["best_algorithm"] = performance.idxmax(axis=1)
    instance_space.to_csv(space_path, index_label=INSTANCE_COLUMN)

    summary_path = Path(str(footprint_summary_path or DEFAULT_FOOTPRINT_CSV))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "algorithm": footprint.algorithm,
                "n_good": footprint.n_good,
                "n_total": footprint.n_total,
                "hull_vertices": len(footprint.hull),
                "area": footprint.area,
                "purity": footprint.purity,
                "density": footprint.density,
            }
            for footprint in footprints.values()
        ]
    ).to_csv(summary_path, index=False)

    written: List[Path] = [space_path, summary_path]

    if make_figures:
        figure_dir = Path(str(figures_dir or DEFAULT_FIGURES_DIR))
        written.append(
            plot_instance_space(coordinates, performance, figure_dir / "instance_space.pdf")
        )
        for algorithm, footprint in footprints.items():
            written.append(
                plot_footprint(
                    coordinates,
                    performance,
                    footprint,
                    footprint_threshold,
                    figure_dir / f"footprint_{algorithm}.pdf",
                )
            )

    return {
        "features": clean_features,
        "performance": performance,
        "coordinates": coordinates,
        "projection": projection,
        "footprints": footprints,
        "report": report,
        "written": written,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit the H-ODBRP instance space and export figures (T14)."
    )
    parser.add_argument("--features", help="features.csv path (default: data/results/)")
    parser.add_argument("--perf", help="perf_matrix.csv path (default: data/results/)")
    parser.add_argument("--out", help="instance_space.csv path")
    parser.add_argument("--footprint-summary", help="footprint_summary.csv path")
    parser.add_argument("--figures", help="figures directory (default: figures/)")
    parser.add_argument(
        "--method",
        choices=PROJECTION_METHODS,
        default="pilot",
        help="pilot = analytic PILOT via numpy (default); package = installed MATILDA port",
    )
    parser.add_argument(
        "--nan-threshold",
        type=float,
        default=DEFAULT_NAN_THRESHOLD,
        help=f"drop features with more than this share missing (default: {DEFAULT_NAN_THRESHOLD})",
    )
    parser.add_argument(
        "--keep-dynamic",
        action="store_true",
        help="keep features 19-21 instead of dropping them as static-constant",
    )
    parser.add_argument(
        "--footprint-threshold",
        type=float,
        default=DEFAULT_FOOTPRINT_THRESHOLD,
        help=f"score at which an algorithm counts as good (default: {DEFAULT_FOOTPRINT_THRESHOLD})",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="use only instances present in both files instead of failing on a mismatch",
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="write CSVs only, skip the figures"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="print the preprocessing report and stop, writing nothing",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint: preprocess, fit, and export the instance space."""
    args = _build_arg_parser().parse_args(argv)

    try:
        if args.report_only:
            features, performance = load_inputs(
                args.features, args.perf, strict=not args.no_strict
            )
            _, report = preprocess_features(
                features,
                nan_threshold=args.nan_threshold,
                drop_dynamic=not args.keep_dynamic,
            )
            print("Preprocessing report")
            print("--------------------")
            for line in report.describe():
                print(f"  {line}")
            return 0

        outcome = run_pipeline(
            features_path=args.features,
            perf_path=args.perf,
            instance_space_path=args.out,
            footprint_summary_path=args.footprint_summary,
            figures_dir=args.figures,
            method=args.method,
            nan_threshold=args.nan_threshold,
            drop_dynamic=not args.keep_dynamic,
            footprint_threshold=args.footprint_threshold,
            strict=not args.no_strict,
            make_figures=not args.no_figures,
        )
    except PipelineError as exc:
        print(str(exc))
        return 1

    report = outcome["report"]
    projection = outcome["projection"]

    print("Preprocessing report")
    print("--------------------")
    for line in report.describe():
        print(f"  {line}")

    print("\nProjection")
    print("----------")
    print(f"  backend              : {projection.backend}")
    print(f"  variance explained   : features {projection.r2_features:.2f}, "
          f"performance {projection.r2_performance:.2f}")

    print("\nFootprints")
    print("----------")
    for footprint in outcome["footprints"].values():
        print(
            f"  {footprint.algorithm:<4} good {footprint.n_good:>4}/{footprint.n_total}"
            f"   area {footprint.area:8.3f}   purity {footprint.purity:.2f}"
        )

    print("\nWritten")
    print("-------")
    for path in outcome["written"]:
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
