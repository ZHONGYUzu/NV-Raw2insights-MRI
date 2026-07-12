#!/usr/bin/env python3
"""Plot CINE metric distributions from a frame_metrics.csv file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def read_metric_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics CSV: {path}")
    with path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def metric_values_by_label(
    rows: List[Dict[str, str]],
    labels: Sequence[str],
    metric: str,
    label_column: str,
) -> List[np.ndarray]:
    values_by_label = []
    for label in labels:
        values = np.array(
            [float(row[metric]) for row in rows if row[label_column] == label],
            dtype=np.float64,
        )
        values_by_label.append(values[np.isfinite(values)])
    return values_by_label


def validate_columns(rows: List[Dict[str, str]], metrics: Sequence[str], label_column: str) -> None:
    columns = set(rows[0])
    missing = [column for column in [label_column, *metrics] if column not in columns]
    if missing:
        raise ValueError(f"Missing required column(s) in metrics CSV: {', '.join(missing)}")


def save_distribution_plot(
    rows: List[Dict[str, str]],
    metrics: Sequence[str],
    output_path: Path,
    label_column: str,
    kind: str,
) -> None:
    import matplotlib.pyplot as plt

    labels = sorted({row[label_column] for row in rows})
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.5), squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        values = metric_values_by_label(rows, labels, metric, label_column)
        if kind == "violin":
            axis.violinplot(values, showmeans=True, showmedians=True)
        else:
            axis.boxplot(values, showmeans=True, showfliers=False)
        axis.set_title(metric.upper())
        axis.set_xticks(range(1, len(labels) + 1))
        axis.set_xticklabels(labels)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame_metrics_csv", type=Path, help="Path to frame_metrics.csv.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for metric plots. Default: same directory as frame_metrics.csv.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "nrmse", "nmse", "mse", "mae"],
        help="Metric columns to plot.",
    )
    parser.add_argument(
        "--label-column",
        default="acceleration",
        help="CSV column used to group distributions. Default: acceleration.",
    )
    parser.add_argument("--box-name", default="metrics_boxplot.png", help="Output filename for the box plot.")
    parser.add_argument("--violin-name", default="metrics_violin.png", help="Output filename for the violin plot.")
    args = parser.parse_args()

    rows = read_metric_rows(args.frame_metrics_csv)
    validate_columns(rows, args.metrics, args.label_column)

    output_dir = args.output_dir or args.frame_metrics_csv.parent
    boxplot_path = output_dir / args.box_name
    violin_path = output_dir / args.violin_name
    save_distribution_plot(rows, args.metrics, boxplot_path, args.label_column, kind="box")
    save_distribution_plot(rows, args.metrics, violin_path, args.label_column, kind="violin")

    print(f"Saved box plot: {boxplot_path}")
    print(f"Saved violin plot: {violin_path}")


if __name__ == "__main__":
    main()
