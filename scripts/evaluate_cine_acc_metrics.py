#!/usr/bin/env python3
"""Evaluate full-size CINE reconstructions across acceleration settings.

This script compares prediction MAT files against normalized fully sampled
ground truth without cropping. Metrics are computed per 2D frame for every
shared case, slice, and temporal frame.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from inspect_cine_recon_quality import load_gt, read_mat_key


def parse_acc_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Acceleration input must be LABEL=PATH, for example acc8=output/CustomCINEOutputR1/val_img4ranking"
        )
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Acceleration label cannot be empty")
    return label, Path(path)


def collect_cases(acc_inputs: Sequence[Tuple[str, Path]], case_glob: str) -> List[str]:
    case_sets = []
    for label, pred_dir in acc_inputs:
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"{label}: prediction directory does not exist: {pred_dir}")
        cases = {path.stem for path in pred_dir.glob(case_glob)}
        if not cases:
            raise FileNotFoundError(f"{label}: no files matching {case_glob!r} in {pred_dir}")
        case_sets.append(cases)
    return sorted(set.intersection(*case_sets))


def load_prediction(path: Path, key: str):
    import numpy as np

    pred = read_mat_key(path, key).squeeze().astype("float32")
    if pred.ndim != 4:
        raise ValueError(f"{path}: expected 4D prediction, got shape {pred.shape}")
    if not np.isfinite(pred).all():
        raise ValueError(f"{path}: prediction contains NaN or Inf")
    return np.abs(pred)


def frame_metrics(pred_frame, gt_frame) -> Dict[str, float]:
    import numpy as np

    pred_frame = np.asarray(pred_frame, dtype=np.float32)
    gt_frame = np.asarray(gt_frame, dtype=np.float32)
    diff = pred_frame - gt_frame
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    gt_norm = float(np.linalg.norm(gt_frame))
    nrmse = float(np.sqrt(np.sum(diff**2)) / gt_norm) if gt_norm > 0 else float("nan")
    nmse = float(np.sum(diff**2) / np.sum(gt_frame**2)) if np.sum(gt_frame**2) > 0 else float("nan")
    data_range = float(gt_frame.max() - gt_frame.min())
    psnr = float("nan") if mse <= 0 or data_range <= 0 else 20.0 * np.log10(data_range / np.sqrt(mse))
    return {
        "psnr": psnr,
        "nrmse": nrmse,
        "nmse": nmse,
        "mse": mse,
        "mae": mae,
    }


def evaluate_case(case_id: str, label: str, pred_dir: Path, gt_root: Path, key: str) -> List[Dict[str, object]]:
    import numpy as np

    pred_path = pred_dir / f"{case_id}.mat"
    gt_path = gt_root / f"norm_img_{case_id}.npy"
    if not pred_path.is_file():
        raise FileNotFoundError(f"{label} {case_id}: missing prediction {pred_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"{case_id}: missing ground truth {gt_path}")

    pred = load_prediction(pred_path, key)
    gt = load_gt(gt_path)
    if pred.shape != gt.shape:
        raise ValueError(f"{label} {case_id}: pred shape {pred.shape} != GT shape {gt.shape}")
    if not np.isfinite(gt).all():
        raise ValueError(f"{case_id}: ground truth contains NaN or Inf")

    _, _, num_slices, num_times = pred.shape
    rows = []
    for slice_index in range(num_slices):
        for time_index in range(num_times):
            metrics = frame_metrics(
                pred[:, :, slice_index, time_index],
                gt[:, :, slice_index, time_index],
            )
            rows.append(
                {
                    "acceleration": label,
                    "case": case_id,
                    "slice": slice_index,
                    "time": time_index,
                    "shape": "x".join(str(dim) for dim in pred.shape),
                    **metrics,
                }
            )
    return rows


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: List[Dict[str, object]], metrics: Sequence[str]) -> List[Dict[str, object]]:
    import numpy as np

    labels = sorted({str(row["acceleration"]) for row in rows})
    summary = []
    for label in labels:
        label_rows = [row for row in rows if row["acceleration"] == label]
        item: Dict[str, object] = {
            "acceleration": label,
            "num_frames": len(label_rows),
            "num_cases": len({row["case"] for row in label_rows}),
        }
        for metric in metrics:
            values = np.array([float(row[metric]) for row in label_rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            item[f"{metric}_mean"] = float(np.mean(values)) if values.size else float("nan")
            item[f"{metric}_std"] = float(np.std(values)) if values.size else float("nan")
            item[f"{metric}_median"] = float(np.median(values)) if values.size else float("nan")
            item[f"{metric}_q25"] = float(np.percentile(values, 25)) if values.size else float("nan")
            item[f"{metric}_q75"] = float(np.percentile(values, 75)) if values.size else float("nan")
        summary.append(item)
    return summary


def metric_values_by_label(rows: List[Dict[str, object]], labels: Sequence[str], metric: str):
    import numpy as np

    values = []
    for label in labels:
        label_values = np.array(
            [float(row[metric]) for row in rows if row["acceleration"] == label],
            dtype=np.float64,
        )
        values.append(label_values[np.isfinite(label_values)])
    return values


def save_distribution_plot(
    rows: List[Dict[str, object]],
    metrics: Sequence[str],
    output_path: Path,
    kind: str,
) -> None:
    import matplotlib.pyplot as plt

    labels = sorted({str(row["acceleration"]) for row in rows})
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.5), squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        values = metric_values_by_label(rows, labels, metric)
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
    parser.add_argument(
        "--acc",
        action="append",
        type=parse_acc_arg,
        required=True,
        help="Acceleration label and prediction directory as LABEL=PATH. Repeat for each acceleration.",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("/home/students/studxuzho1/dataset_v0/norm_img"),
        help="Directory containing norm_img_<case>.npy ground truth files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output/acc_metrics"),
        help="Directory for metric CSVs and distribution plots.",
    )
    parser.add_argument("--case-glob", default="Sub*.mat", help="Prediction case glob used in every acc directory.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional limit after sorting shared cases.")
    parser.add_argument("--key", default="img4ranking", help="MAT key containing the reconstruction.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "nrmse", "nmse"],
        choices=["psnr", "nrmse", "nmse", "mse", "mae"],
        help="Metrics to include in the box/violin plots.",
    )
    args = parser.parse_args()

    cases = collect_cases(args.acc, args.case_glob)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        parser.error("No shared cases found across acceleration directories")

    rows: List[Dict[str, object]] = []
    for label, pred_dir in args.acc:
        for case_id in cases:
            case_rows = evaluate_case(case_id, label, pred_dir, args.gt_root, args.key)
            rows.extend(case_rows)
            print(f"{label} {case_id}: {len(case_rows)} frame metrics")

    metrics_csv = args.output_dir / "frame_metrics.csv"
    write_csv(rows, metrics_csv)

    summary_rows = summarize_rows(rows, args.metrics)
    summary_csv = args.output_dir / "summary_metrics.csv"
    write_csv(summary_rows, summary_csv)

    boxplot_path = args.output_dir / "metrics_boxplot.png"
    violin_path = args.output_dir / "metrics_violin.png"
    save_distribution_plot(rows, args.metrics, boxplot_path, kind="box")
    save_distribution_plot(rows, args.metrics, violin_path, kind="violin")

    print(f"Evaluated cases: {', '.join(cases)}")
    print(f"Saved frame metrics: {metrics_csv}")
    print(f"Saved summary metrics: {summary_csv}")
    print(f"Saved box plot: {boxplot_path}")
    print(f"Saved violin plot: {violin_path}")


if __name__ == "__main__":
    main()
