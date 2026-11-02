"""Rebuild report figures from the pilot result CSVs and documented timings."""

import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "pilot" / "work" / "results"
FIGURES = Path(__file__).resolve().parent / "figures"
COLORS = {"tight": "#d95f02", "loose": "#1b9e77"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def seed_count_diagnostic():
    candidates = [
        ("pilot_v1", ROOT / "pilot" / "work" / "pilot_v1" / "results" / "features.csv"),
        ("pilot_v2_old", ROOT / "pilot" / "work" / "pilot_v2_old" / "features.csv"),
        ("results", RESULTS / "features.csv"),
    ]
    for label, path in candidates:
        if path.exists():
            rows = read_csv(path)
            distinct = len({row["od_distance_mean"] for row in rows})
            print(f"{label}: {distinct} distinct od_distance_mean values ({len(rows)} rows)")
    print("The documented pre-fix value of 6 is not reproduced; no seed-count chart is drawn.")


def make_instance_space():
    space = read_csv(RESULTS / "instance_space.csv")
    perf = {row["instance"]: row for row in read_csv(RESULTS / "perf_matrix.csv")}
    ratios = [float(perf[row["instance"]]["SA"]) / float(perf[row["instance"]]["GR"]) for row in space]

    tight = [row for row in space if "_tight_" in row["instance"]]
    loose = [row for row in space if "_loose_" in row["instance"]]
    centroids = {}
    for label, rows in (("tight", tight), ("loose", loose)):
        centroids[label] = (
            sum(float(row["z1"]) for row in rows) / len(rows),
            sum(float(row["z2"]) for row in rows) / len(rows),
        )

    fig, ax = plt.subplots(figsize=(8.0, 5.6), constrained_layout=True)
    mappable = None
    for label, marker in (("loose", "o"), ("tight", "^")):
        selected = [i for i, row in enumerate(space) if f"_{label}_" in row["instance"]]
        scatter = ax.scatter(
            [float(space[i]["z1"]) for i in selected],
            [float(space[i]["z2"]) for i in selected],
            c=[ratios[i] for i in selected],
            cmap="viridis",
            vmin=min(ratios),
            vmax=max(ratios),
            marker=marker,
            s=52,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.92,
        )
        mappable = scatter

    start, end = centroids["loose"], centroids["tight"]
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "linestyle": "--", "color": "#333333", "lw": 1.2},
    )
    midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    ax.annotate(
        "two-level time-window gap",
        xy=midpoint,
        xytext=(7, 9),
        textcoords="offset points",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.85},
    )
    ax.set_xlabel("Projection coordinate $z_1$")
    ax.set_ylabel("Projection coordinate $z_2$")
    ax.set_title("Pilot instance space")
    ax.grid(alpha=0.18)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color="#555555", label="Loose windows"),
            Line2D([], [], marker="^", linestyle="", color="#555555", label="Tight windows"),
        ],
        loc="best",
        frameon=True,
    )
    colorbar = fig.colorbar(mappable, ax=ax, pad=0.02)
    colorbar.set_label("SA / GR normalized performance ratio")
    fig.savefig(FIGURES / "instance_space.pdf", bbox_inches="tight")
    plt.close(fig)


def make_sa_gr_ratio():
    rows = read_csv(RESULTS / "perf_matrix.csv")
    groups = {}
    for row in rows:
        match = re.search(r"_(tight|loose)_\d+_fleet(\d+)$", row["instance"])
        if not match:
            raise ValueError(f"Cannot parse time-window/fleet settings from {row['instance']}")
        setting, fleet = match.groups()
        groups.setdefault((setting, int(fleet)), []).append(float(row["SA"]) / float(row["GR"]))

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    fleets = [10, 20, 30]
    width = 0.34
    positions = list(range(len(fleets)))
    for i, setting in enumerate(("tight", "loose")):
        means = [sum(groups[(setting, fleet)]) / len(groups[(setting, fleet)]) for fleet in fleets]
        errors = [
            math.sqrt(
                sum((value - means[j]) ** 2 for value in groups[(setting, fleet)])
                / (len(groups[(setting, fleet)]) - 1)
            )
            for j, fleet in enumerate(fleets)
        ]
        offset = (i - 0.5) * width
        ax.bar(
            [position + offset for position in positions],
            means,
            width,
            yerr=errors,
            capsize=3,
            color=COLORS[setting],
            label=f"{setting.capitalize()} windows",
        )

    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    ax.set_xticks(positions, [f"{fleet}%" for fleet in fleets])
    ax.set_xlabel("Fleet size as a share of requests")
    ax.set_ylabel("Mean SA / GR ratio (normalized scores)")
    ax.set_title("SA-to-GR performance ratio by pilot setting")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(FIGURES / "sa_gr_ratio.pdf", bbox_inches="tight")
    plt.close(fig)


def make_generation_profile():
    phase_names = ["Nearest-node lookups", "Other request phase", "Imports", "Request plot"]
    phase_values = [117.8, 154.3 - 117.8, 60.0, 27.0]
    labels = ["Before cache", "After cache"]
    runtimes = [250.0, 115.0]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.5), constrained_layout=True)
    order = list(reversed(range(len(phase_names))))
    axes[0].barh([phase_names[i] for i in order], [phase_values[i] for i in order], color="#4c78a8")
    axes[0].set_xlabel("Cumulative time (s)")
    axes[0].set_title("Profile phases")
    axes[0].grid(axis="x", alpha=0.2)
    axes[0].annotate(
        "Part of 154.3 s request phase",
        xy=(117.8, 3),
        xytext=(70, 2.45),
        arrowprops={"arrowstyle": "->", "color": "#444444"},
        fontsize=8,
    )
    axes[1].bar(labels, runtimes, color=["#9ecae9", "#2b6cb0"])
    axes[1].set_ylabel("Wall time (s)")
    axes[1].set_title("Whole profile run")
    axes[1].grid(axis="y", alpha=0.2)
    for index, value in enumerate(runtimes):
        axes[1].text(index, value + 5, f"{value:.0f}", ha="center", va="bottom", fontsize=9)
    fig.savefig(FIGURES / "generation_profile.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    seed_count_diagnostic()
    make_instance_space()
    make_sa_gr_ratio()
    make_generation_profile()
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
