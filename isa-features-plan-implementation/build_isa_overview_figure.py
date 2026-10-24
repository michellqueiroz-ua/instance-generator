"""
build_isa_overview_figure.py — the ISA figure used in the thesis-ideas PDF

Two panels:
  (a) the spread of the generated instances across the measured features;
  (b) the resulting 2-D instance space with per-algorithm footprints.

    python isa-features-plan-implementation/build_isa_overview_figure.py \
        --features data/results/features.csv \
        --instance-space data/results/instance_space.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _path in (_THIS_DIR, _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from isa_pipeline import compute_footprints  # noqa: E402

OUTPUT_PATH = _THIS_DIR / "isa_overview.png"

FAMILY_COLOURS = {
    "Demand": "#1f4e79",
    "Time window": "#2e7d32",
    "Fleet": "#b8860b",
    "Spatial / network": "#7b4397",
}
FAMILY_OF = {
    "n_requests": "Demand", "demand_density": "Demand",
    "spatial_cluster_coeff": "Demand", "od_distance_mean": "Demand",
    "od_distance_std": "Demand",
    "tw_width_mean": "Time window", "tw_width_std": "Time window",
    "tw_overlap_ratio": "Time window", "tight_tw_fraction": "Time window",
    "tw_tightness_factor": "Time window",
    "fleet_size": "Fleet", "capacity_cv": "Fleet",
    "fleet_utilization_ratio": "Fleet", "request_vehicle_ratio": "Fleet",
    "station_density": "Spatial / network", "bsa_flexibility": "Spatial / network",
    "avg_station_degree": "Spatial / network", "coverage_ratio": "Spatial / network",
}
ALGORITHM_COLOURS = {"SA": "#1f77b4", "IP": "#ff7f0e", "GR": "#2ca02c"}


def _panel_features(axis, features: pd.DataFrame) -> None:
    """Min-max normalised strip plot: one row per feature, one dot per instance."""
    usable = [
        name for name in features.columns
        if name in FAMILY_OF and features[name].notna().any()
        and features[name].max() > features[name].min()
    ]
    usable.sort(key=lambda name: (list(FAMILY_COLOURS).index(FAMILY_OF[name]), name))

    rng = np.random.default_rng(0)
    for row, name in enumerate(usable):
        column = features[name].astype(float)
        scaled = (column - column.min()) / (column.max() - column.min())
        jitter = rng.uniform(-0.16, 0.16, len(scaled))
        axis.scatter(
            scaled, np.full(len(scaled), row) + jitter,
            s=7, alpha=0.30, linewidths=0,
            color=FAMILY_COLOURS[FAMILY_OF[name]],
        )

    axis.set_yticks(range(len(usable)))
    axis.set_yticklabels(usable, fontsize=7.4)
    axis.set_ylim(-0.8, len(usable) - 0.2)
    axis.invert_yaxis()
    axis.set_xlim(-0.04, 1.04)
    axis.set_xticks([0, 0.5, 1])
    axis.set_xticklabels(["min", "", "max"], fontsize=7.5)
    axis.set_xlabel("normalised feature value", fontsize=8)
    axis.set_title(
        f"(a) {len(features)} generated instances measured on {len(usable)} features",
        fontsize=9.2, loc="left",
    )
    axis.grid(axis="x", alpha=0.2, linewidth=0.5)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=4.5,
                   color=colour, label=family)
        for family, colour in FAMILY_COLOURS.items()
    ]
    axis.legend(handles=handles, fontsize=7, frameon=False, loc="lower right",
                handletextpad=0.3, borderaxespad=0.2)


def _panel_space(axis, space: pd.DataFrame) -> None:
    """Instance space scatter coloured by best algorithm, with footprint hulls."""
    coordinates = space[["z1", "z2"]]
    algorithms = [c for c in ("SA", "IP", "GR") if c in space.columns]
    performance = space[algorithms]
    best = performance.idxmax(axis=1)

    for algorithm in algorithms:
        mask = (best == algorithm).to_numpy()
        if mask.any():
            axis.scatter(
                coordinates.loc[mask, "z1"], coordinates.loc[mask, "z2"],
                s=13, alpha=0.85, linewidths=0,
                color=ALGORITHM_COLOURS.get(algorithm, "#666666"),
                label=f"{algorithm} best ({int(mask.sum())})",
            )

    footprints = compute_footprints(coordinates, performance, threshold=0.80)
    for algorithm, footprint in footprints.items():
        if len(footprint.hull) < 3:
            continue
        polygon = footprint.hull + [footprint.hull[0]]
        axis.plot(
            [p[0] for p in polygon], [p[1] for p in polygon],
            linewidth=1.2, alpha=0.9, linestyle="--",
            color=ALGORITHM_COLOURS.get(algorithm, "#666666"),
        )

    axis.set_xlabel("z1", fontsize=8)
    axis.set_ylabel("z2", fontsize=8)
    axis.tick_params(labelsize=7.5)
    axis.set_title(
        "(b) instance space: where each algorithm wins\n"
        "dashed outline = footprint (score >= 0.8)",
        fontsize=9.2, loc="left",
    )
    axis.legend(fontsize=7, frameon=False, loc="best")
    axis.grid(alpha=0.2, linewidth=0.5)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def build(
    features_path: Path,
    space_path: Path,
    output_path: Path = OUTPUT_PATH,
    caption: Optional[str] = None,
) -> Path:
    features = pd.read_csv(features_path, index_col="instance")
    space = pd.read_csv(space_path, index_col="instance")

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), width_ratios=[1.05, 1.0])
    _panel_features(axes[0], features)
    _panel_space(axes[1], space)

    if caption:
        figure.text(0.5, 0.005, caption, ha="center", fontsize=7.2, color="#5b6470")

    figure.tight_layout(rect=(0, 0.035 if caption else 0, 1, 1))
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the ISA overview figure.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--instance-space", required=True)
    parser.add_argument("--out", default=str(OUTPUT_PATH))
    parser.add_argument("--caption", default=None)
    args = parser.parse_args(argv)

    path = build(
        Path(args.features), Path(args.instance_space), Path(args.out), args.caption
    )
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
