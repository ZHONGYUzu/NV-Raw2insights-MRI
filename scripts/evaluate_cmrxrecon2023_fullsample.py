#!/usr/bin/env python3
"""Evaluate a CMRxRecon 2023 reconstruction against full-sampled RSS ground truth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from skimage.metrics import structural_similarity


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    shifted = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(shifted, axes=(-2, -1), norm="ortho")
    return np.fft.fftshift(image, axes=(-2, -1))


def load_compound_kspace(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, "r", swmr=True) as mat_file:
        if key not in mat_file:
            raise KeyError(f"{path}: missing {key!r}; available keys: {list(mat_file.keys())}")
        value = np.asarray(mat_file[key])
    if value.dtype.fields and {"real", "imag"}.issubset(value.dtype.fields):
        kspace = value["real"] + 1j * value["imag"]
    elif np.issubdtype(value.dtype, np.complexfloating):
        kspace = value
    else:
        raise TypeError(f"{path}:{key} is not complex data (dtype={value.dtype})")
    if kspace.ndim != 5:
        raise ValueError(f"{path}:{key} must have shape (time, slice, coil, PE, FE), got {kspace.shape}")
    if not np.isfinite(kspace).all():
        raise ValueError(f"{path}:{key} contains NaN or Inf")
    return kspace.astype(np.complex64, copy=False)


def fullsample_rss(kspace: np.ndarray) -> np.ndarray:
    coil_images = ifft2c(kspace)
    rss = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=2)).astype(np.float32)
    return rss.transpose(3, 2, 1, 0)  # (frequency, phase, slice, time)


def load_prediction(path: Path, key: str) -> np.ndarray:
    values = scipy.io.loadmat(path)
    if key not in values:
        keys = [candidate for candidate in values if not candidate.startswith("__")]
        raise KeyError(f"{path}: missing {key!r}; available keys: {keys}")
    prediction = np.abs(np.asarray(values[key]).squeeze()).astype(np.float32)
    if prediction.ndim != 4:
        raise ValueError(f"{path}:{key} must be 4D, got {prediction.shape}")
    if not np.isfinite(prediction).all():
        raise ValueError(f"{path}:{key} contains NaN or Inf")
    return prediction


def metrics(reference: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    reference = np.asarray(reference, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    diff = prediction - reference
    squared_error = float(np.sum(diff**2, dtype=np.float64))
    reference_squared = float(np.sum(reference**2, dtype=np.float64))
    mse = float(np.mean(diff**2, dtype=np.float64))
    mae = float(np.mean(np.abs(diff), dtype=np.float64))
    nrmse = float(np.sqrt(squared_error / reference_squared)) if reference_squared > 0 else float("nan")
    nmse = float(squared_error / reference_squared) if reference_squared > 0 else float("nan")
    data_range = float(reference.max() - reference.min())
    psnr = float("nan") if mse <= 0 or data_range <= 0 else float(20 * np.log10(data_range / np.sqrt(mse)))
    if reference.ndim == 2:
        ssim = (
            float("nan")
            if data_range <= 0
            else float(structural_similarity(reference, prediction, data_range=data_range))
        )
    elif reference.ndim == 4:
        frame_ssim = []
        for slice_index in range(reference.shape[2]):
            for time_index in range(reference.shape[3]):
                ref_frame = reference[:, :, slice_index, time_index]
                pred_frame = prediction[:, :, slice_index, time_index]
                frame_range = float(ref_frame.max() - ref_frame.min())
                if frame_range > 0:
                    frame_ssim.append(structural_similarity(ref_frame, pred_frame, data_range=frame_range))
        ssim = float(np.mean(frame_ssim)) if frame_ssim else float("nan")
    else:
        raise ValueError(f"Metrics expect a 2D frame or 4D volume, got {reference.shape}")
    return {
        "psnr": psnr,
        "ssim": ssim,
        "nrmse": nrmse,
        "nmse": nmse,
        "mse": mse,
        "mae": mae,
    }


def fit_global_scale(reference: np.ndarray, prediction: np.ndarray) -> tuple[float, np.ndarray]:
    denominator = float(np.sum(prediction * prediction, dtype=np.float64))
    if denominator <= 0:
        return 1.0, prediction.copy()
    scale = float(np.sum(reference * prediction, dtype=np.float64) / denominator)
    return scale, (prediction * scale).astype(np.float32)


def array_stats(value: np.ndarray) -> Dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "p99_5": float(np.percentile(value, 99.5)),
    }


def write_frame_metrics(reference: np.ndarray, prediction: np.ndarray, path: Path) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for slice_index in range(reference.shape[2]):
        for time_index in range(reference.shape[3]):
            row: Dict[str, object] = {
                "slice": slice_index,
                "time": time_index,
                "shape": "x".join(str(dim) for dim in reference.shape[:2]),
            }
            row.update(metrics(reference[:, :, slice_index, time_index], prediction[:, :, slice_index, time_index]))
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize_frames(rows: list[Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {"num_frames": len(rows)}
    for metric in ("psnr", "ssim", "nrmse", "nmse", "mse", "mae"):
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        summary[metric] = {
            "mean": float(values.mean()) if values.size else float("nan"),
            "std": float(values.std()) if values.size else float("nan"),
            "median": float(np.median(values)) if values.size else float("nan"),
            "q25": float(np.percentile(values, 25)) if values.size else float("nan"),
            "q75": float(np.percentile(values, 75)) if values.size else float("nan"),
        }
    return summary


def save_metric_plot(rows: list[Dict[str, object]], path: Path) -> None:
    names = ("psnr", "ssim", "nrmse", "nmse")
    fig, axes = plt.subplots(1, len(names), figsize=(14, 4), constrained_layout=True)
    for axis, name in zip(axes, names):
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        axis.boxplot(values[np.isfinite(values)], showmeans=True, showfliers=False)
        axis.set_title(name.upper())
        axis.set_xticks([])
        axis.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_comparison(
    reference: np.ndarray,
    prediction: np.ndarray,
    path: Path,
    slice_index: int,
    time_index: int,
) -> None:
    ref = reference[:, :, slice_index, time_index]
    pred = prediction[:, :, slice_index, time_index]
    error = np.abs(pred - ref)
    vmax = float(np.percentile(np.stack([ref, pred]), 99.5))
    error_vmax = float(np.percentile(error, 99.5))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = (
        ("Full-sampled RSS GT", ref, "gray", vmax),
        ("Model reconstruction", pred, "gray", vmax),
        ("Absolute error", error, "magma", error_vmax),
    )
    for axis, (title, image, cmap, limit) in zip(axes, panels):
        rendered = axis.imshow(image, cmap=cmap, vmin=0, vmax=limit)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"slice={slice_index}, time={time_index}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recon", type=Path, required=True, help="Prediction MAT containing img4ranking.")
    parser.add_argument("--full-kspace", type=Path, required=True, help="Full-sampled CMRxRecon 2023 MAT file.")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Evaluation output directory.")
    parser.add_argument("--recon-key", default="img4ranking")
    parser.add_argument("--kspace-key", default="kspace_full")
    parser.add_argument("--slice", type=int, default=1, help="Slice for comparison PNG (default: 1).")
    parser.add_argument("--time", type=int, default=6, help="Time for comparison PNG (default: 6).")
    args = parser.parse_args()

    prediction = load_prediction(args.recon, args.recon_key)
    kspace = load_compound_kspace(args.full_kspace, args.kspace_key)
    reference = fullsample_rss(kspace)
    if prediction.shape != reference.shape:
        raise ValueError(f"Prediction shape {prediction.shape} != full-sampled RSS shape {reference.shape}")
    if not 0 <= args.slice < reference.shape[2] or not 0 <= args.time < reference.shape[3]:
        parser.error(f"Requested slice/time ({args.slice}, {args.time}) is outside shape {reference.shape}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(
        args.output_dir / "P001_cine_lax_fullsample_rss.mat",
        {"gt": reference},
        do_compression=True,
    )
    rows = write_frame_metrics(reference, prediction, args.output_dir / "frame_metrics.csv")
    scale, scaled_prediction = fit_global_scale(reference, prediction)
    report = {
        "reference": array_stats(reference),
        "prediction": array_stats(prediction),
        "direct_volume_metrics": metrics(reference, prediction),
        "frame_metrics_summary": summarize_frames(rows),
        "diagnostic_global_scale": scale,
        "diagnostic_scale_fitted_volume_metrics": metrics(reference, scaled_prediction),
        "notes": {
            "primary_metrics": "direct_volume_metrics and frame_metrics_summary",
            "scale_fit": "Diagnostic only; it is not the primary quantitative result.",
            "layout": "frequency, phase, slice, time",
        },
    }
    report_path = args.output_dir / "summary_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    save_metric_plot(rows, args.output_dir / "frame_metrics_boxplot.png")
    save_comparison(reference, prediction, args.output_dir / "comparison.png", args.slice, args.time)

    print(json.dumps(report, indent=2, allow_nan=True))
    print(f"Saved evaluation: {args.output_dir}")


if __name__ == "__main__":
    main()
