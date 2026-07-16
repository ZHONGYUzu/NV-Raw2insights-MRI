#!/usr/bin/env python3
"""Create box and violin plots from a frame-level metric CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_LABELS = {
    "psnr": "PSNR (dB)",
    "ssim": "SSIM",
    "nrmse": "NRMSE",
    "nmse": "NMSE",
    "mse": "MSE",
    "mae": "MAE",
}


def read_metrics(path: Path, metrics: list[str]) -> dict[str, np.ndarray]:
    with path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = [metric for metric in metrics if metric not in rows[0]]
    if missing:
        raise KeyError(f"{path}: missing columns {missing}; columns={list(rows[0])}")
    values = {}
    for metric in metrics:
        array = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        array = array[np.isfinite(array)]
        if not array.size:
            raise ValueError(f"{path}: metric {metric!r} has no finite values")
        values[metric] = array
    return values


def style_axis(axis: plt.Axes, metric: str, count: int) -> None:
    axis.set_title(METRIC_LABELS.get(metric, metric.upper()), fontweight="bold")
    axis.set_xticks([])
    axis.set_xlabel(f"n={count} frames")
    axis.grid(axis="y", alpha=0.25, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def save_boxplot(values: dict[str, np.ndarray], path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(values), figsize=(3.5 * len(values), 4.5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, (metric, array) in zip(axes, values.items()):
        plot = axis.boxplot(
            [array],
            widths=0.45,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            medianprops={"color": "#D55E00", "linewidth": 2},
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#0072B2"},
        )
        plot["boxes"][0].set_facecolor("#56B4E9")
        plot["boxes"][0].set_alpha(0.8)
        style_axis(axis, metric, array.size)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_violinplot(values: dict[str, np.ndarray], path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(values), figsize=(3.5 * len(values), 4.5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, (metric, array) in zip(axes, values.items()):
        plot = axis.violinplot([array], positions=[1], widths=0.75, showmeans=True, showmedians=True)
        for body in plot["bodies"]:
            body.set_facecolor("#56B4E9")
            body.set_edgecolor("#0072B2")
            body.set_alpha(0.8)
        for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
            if key in plot:
                plot[key].set_color("#D55E00" if key == "cmedians" else "#0072B2")
                plot[key].set_linewidth(1.5)
        style_axis(axis, metric, array.size)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "ssim", "nrmse", "nmse"],
        choices=sorted(METRIC_LABELS),
    )
    parser.add_argument("--title", default="Raw VISTA Acc8 — Frame Metrics")
    args = parser.parse_args()

    values = read_metrics(args.input_csv, args.metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    boxplot = args.output_dir / "metrics_boxplot.png"
    violinplot = args.output_dir / "metrics_violin.png"
    save_boxplot(values, boxplot, args.title)
    save_violinplot(values, violinplot, args.title)
    print(f"Frames: {next(iter(values.values())).size}")
    print(f"Saved: {boxplot}")
    print(f"Saved: {violinplot}")


if __name__ == "__main__":
    main()
