#!/usr/bin/env python3
"""Evaluate fastMRI brain MAT predictions against H5 reconstruction_rss targets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from evaluate_cmrxrecon2023_fullsample import metrics, summarize_frames
from plot_frame_metrics_distributions import save_boxplot, save_violinplot


def center_crop(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if value.shape[-2] < shape[0] or value.shape[-1] < shape[1]:
        raise ValueError(f"Cannot crop spatial shape {value.shape[-2:]} to {shape}")
    start_h = (value.shape[-2] - shape[0]) // 2
    start_w = (value.shape[-1] - shape[1]) // 2
    return value[..., start_h : start_h + shape[0], start_w : start_w + shape[1]]


def load_prediction(path: Path) -> np.ndarray:
    values = scipy.io.loadmat(path)
    if "img4ranking" not in values:
        raise KeyError(f"{path}: missing img4ranking")
    repo_layout = np.asarray(values["img4ranking"]).squeeze()
    if repo_layout.ndim != 3:
        raise ValueError(f"{path}: expected (frequency, phase, slice[, 1]), got {repo_layout.shape}")
    prediction = np.abs(repo_layout).transpose(2, 1, 0).astype(np.float32)  # (slice, height, width)
    if not np.isfinite(prediction).all():
        raise ValueError(f"{path}: prediction contains NaN or Inf")
    return prediction


def load_target(path: Path) -> np.ndarray:
    with h5py.File(path, "r", swmr=True) as h5_file:
        if "reconstruction_rss" not in h5_file:
            raise KeyError(f"{path}: missing reconstruction_rss")
        target = np.asarray(h5_file["reconstruction_rss"], dtype=np.float32)
    if target.ndim != 3 or not np.isfinite(target).all():
        raise ValueError(f"{path}: invalid reconstruction_rss shape/content {target.shape}")
    return target


def write_csv(rows: list[Dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_comparison(
    target: np.ndarray,
    prediction: np.ndarray,
    path: Path,
    slice_index: int,
    reconstruction_title: str,
) -> None:
    reference = target[slice_index]
    reconstruction = prediction[slice_index]
    error = np.abs(reconstruction - reference)
    vmax = float(np.percentile(np.stack([reference, reconstruction]), 99.5))
    error_vmax = float(np.percentile(error, 99.5))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = (
        ("fastMRI reconstruction_rss", reference, "gray", vmax),
        (reconstruction_title, reconstruction, "gray", vmax),
        ("Absolute error", error, "magma", error_vmax),
    )
    for axis, (title, image, cmap, limit) in zip(axes, panels):
        rendered = axis.imshow(image, cmap=cmap, vmin=0, vmax=limit)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"Middle slice {slice_index}")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="fastMRI Brain", help="Experiment label used in plots and summary.")
    parser.add_argument(
        "--reconstruction-title",
        default="Model reconstruction",
        help="Title for the reconstructed-image comparison panel.",
    )
    parser.add_argument("--max-cases", type=int, help="Evaluate only the first N manifest cases.")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    cases = manifest.get("cases", [])
    if not cases:
        parser.error(f"No cases found in {args.manifest}")
    if args.max_cases is not None:
        if args.max_cases <= 0:
            parser.error("--max-cases must be positive")
        cases = cases[: args.max_cases]
    ground_truth_dir = args.output_dir / "ground_truth"
    comparison_dir = args.output_dir / "comparisons"
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    frame_rows: list[Dict[str, object]] = []
    case_summaries = []
    for case in cases:
        source = Path(str(case["source"]))
        case_id = source.stem
        prediction_path = args.prediction_dir / f"{case_id}.mat"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"{case_id}: missing prediction {prediction_path}")
        target = load_target(source)
        prediction = load_prediction(prediction_path)
        prediction_shape_before_crop = tuple(int(size) for size in prediction.shape)
        if prediction.shape[0] != target.shape[0]:
            raise ValueError(f"{case_id}: prediction slices {prediction.shape[0]} != target {target.shape[0]}")
        prediction = center_crop(prediction, target.shape[-2:])
        if prediction.shape != target.shape:
            raise ValueError(f"{case_id}: aligned prediction {prediction.shape} != target {target.shape}")

        scipy.io.savemat(
            ground_truth_dir / f"{case_id}_reconstruction_rss.mat",
            {"gt": target.transpose(2, 1, 0)[..., None]},
            do_compression=True,
        )
        case_rows = []
        for slice_index in range(target.shape[0]):
            row: Dict[str, object] = {
                "case": case_id,
                "acquisition": str(case["acquisition"]),
                "slice": slice_index,
                "shape": "x".join(str(size) for size in target[slice_index].shape),
            }
            row.update(metrics(target[slice_index], prediction[slice_index]))
            case_rows.append(row)
        frame_rows.extend(case_rows)
        case_summary = {
            "case": case_id,
            "acquisition": str(case["acquisition"]),
            "num_slices": target.shape[0],
            "input_prediction_shape": list(prediction_shape_before_crop),
            "aligned_shape": list(prediction.shape),
            "frame_metrics_summary": summarize_frames(case_rows),
        }
        case_summaries.append(case_summary)
        save_comparison(
            target,
            prediction,
            comparison_dir / f"{case_id}.png",
            target.shape[0] // 2,
            args.reconstruction_title,
        )
        mean_metrics = case_summary["frame_metrics_summary"]
        print(
            f"{case_id}: slices={target.shape[0]}, aligned={target.shape[1:]}, "
            f"PSNR={mean_metrics['psnr']['mean']:.4f}, "
            f"SSIM={mean_metrics['ssim']['mean']:.6f}, NMSE={mean_metrics['nmse']['mean']:.6g}"
        )

    write_csv(frame_rows, args.output_dir / "frame_metrics.csv")
    plot_values = {
        metric: np.asarray([float(row[metric]) for row in frame_rows], dtype=np.float64)
        for metric in ("psnr", "ssim", "nrmse", "nmse")
    }
    save_boxplot(plot_values, args.output_dir / "frame_metrics_boxplot.png", args.label)
    save_violinplot(plot_values, args.output_dir / "frame_metrics_violin.png", args.label)
    summary = {
        "manifest": str(args.manifest.resolve()),
        "prediction_dir": str(args.prediction_dir.resolve()),
        "experiment_label": args.label,
        "num_cases": len(case_summaries),
        "num_slices": len(frame_rows),
        "aggregate_frame_metrics": summarize_frames(frame_rows),
        "cases": case_summaries,
        "notes": {
            "reference": "fastMRI H5 reconstruction_rss",
            "alignment": "Repo output converted to slice/height/width and center-cropped to target shape.",
            "metrics": "Direct-scale per-slice metrics without independent intensity normalization.",
        },
    }
    (args.output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print(f"Evaluated {len(case_summaries)} cases and {len(frame_rows)} slices")
    print(f"Saved results: {args.output_dir}")


if __name__ == "__main__":
    main()
