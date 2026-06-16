#!/usr/bin/env python3
"""Inspect custom CINE reconstruction outputs against normalized ground truth."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_mat_key(path: Path, key: str):
    import numpy as np

    try:
        import h5py

        with h5py.File(path, "r", swmr=True) as mat_file:
            if key not in mat_file:
                raise KeyError(f"{path}: missing key {key!r}; keys={list(mat_file.keys())}")
            return np.asarray(mat_file[key][()])
    except (ImportError, OSError):
        import scipy.io

        data = scipy.io.loadmat(path)
        if key not in data:
            keys = [candidate for candidate in data if not candidate.startswith("__")]
            raise KeyError(f"{path}: missing key {key!r}; keys={keys}")
        return np.asarray(data[key])


def load_gt(path: Path):
    import numpy as np

    gt = np.load(path)
    if gt.ndim != 5 or gt.shape[-1] != 1:
        raise ValueError(f"{path}: expected GT shape (slice, time, phase, frequency, 1), got {gt.shape}")
    return np.abs(gt[..., 0]).transpose(3, 2, 0, 1).astype(np.float32)


def normalize_percentile(image, percentile: float):
    import numpy as np

    image = np.abs(image).astype(np.float32)
    scale = np.percentile(image, percentile)
    if not np.isfinite(scale) or scale <= 0:
        return image
    return image / scale


def compute_metrics(pred, gt, percentile: float) -> Dict[str, float]:
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    pred_norm = normalize_percentile(pred, percentile)
    gt_norm = normalize_percentile(gt, percentile)
    diff = pred_norm - gt_norm
    denom = np.linalg.norm(gt_norm) ** 2
    nmse = float(np.linalg.norm(diff) ** 2 / denom) if denom > 0 else float("nan")
    data_range = float(gt_norm.max() - gt_norm.min())
    if data_range <= 0:
        psnr = float("nan")
    else:
        psnr = float(peak_signal_noise_ratio(gt_norm, pred_norm, data_range=data_range))

    ssims = []
    for slice_index in range(gt_norm.shape[2]):
        for time_index in range(gt_norm.shape[3]):
            gt_frame = gt_norm[:, :, slice_index, time_index]
            pred_frame = pred_norm[:, :, slice_index, time_index]
            frame_range = float(gt_frame.max() - gt_frame.min())
            if frame_range > 0:
                ssims.append(
                    structural_similarity(gt_frame, pred_frame, data_range=frame_range)
                )
    return {
        "nmse": nmse,
        "psnr": psnr,
        "ssim_mean": float(np.mean(ssims)) if ssims else float("nan"),
        "ssim_min": float(np.min(ssims)) if ssims else float("nan"),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
        "gt_min": float(np.min(gt)),
        "gt_max": float(np.max(gt)),
        "gt_mean": float(np.mean(gt)),
        "gt_std": float(np.std(gt)),
    }


def choose_indices(
    shape: Tuple[int, int, int, int],
    slice_index: Optional[int],
    time_index: Optional[int],
) -> Tuple[int, int]:
    _, _, num_slices, num_times = shape
    if slice_index is None:
        slice_index = num_slices // 2
    if time_index is None:
        time_index = num_times // 2
    if not 0 <= slice_index < num_slices:
        raise ValueError(f"slice index {slice_index} outside [0, {num_slices - 1}]")
    if not 0 <= time_index < num_times:
        raise ValueError(f"time index {time_index} outside [0, {num_times - 1}]")
    return slice_index, time_index


def save_comparison_png(
    pred,
    gt,
    output_path: Path,
    slice_index: int,
    time_index: int,
    percentile: float,
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    pred_frame = np.abs(pred[:, :, slice_index, time_index])
    gt_frame = np.abs(gt[:, :, slice_index, time_index])
    error_frame = np.abs(pred_frame - gt_frame)
    vmax = np.percentile(gt_frame, percentile)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = None

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    panels = [
        ("GT", gt_frame, "gray", vmax),
        ("Pred", pred_frame, "gray", vmax),
        ("Abs error", error_frame, "magma", None),
    ]
    for axis, (title, image, cmap, panel_vmax) in zip(axes[0], panels):
        axis.imshow(image.T, cmap=cmap, origin="lower", vmax=panel_vmax)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{output_path.stem}: slice={slice_index}, time={time_index}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def inspect_case(
    mat_path: Path,
    gt_root: Path,
    output_dir: Path,
    key: str,
    percentile: float,
    slice_index: Optional[int],
    time_index: Optional[int],
) -> Dict[str, object]:
    import numpy as np

    case_id = mat_path.stem
    gt_path = gt_root / f"norm_img_{case_id}.npy"
    if not gt_path.is_file():
        raise FileNotFoundError(f"{case_id}: missing ground truth {gt_path}")

    pred = read_mat_key(mat_path, key).squeeze().astype(np.float32)
    gt = load_gt(gt_path)
    if pred.shape != gt.shape:
        raise ValueError(f"{case_id}: pred shape {pred.shape} != GT shape {gt.shape}")
    if not np.isfinite(pred).all():
        raise ValueError(f"{case_id}: prediction contains NaN or Inf")
    if not np.isfinite(gt).all():
        raise ValueError(f"{case_id}: ground truth contains NaN or Inf")
    if np.count_nonzero(pred) == 0:
        raise ValueError(f"{case_id}: prediction is all zero")

    selected_slice, selected_time = choose_indices(pred.shape, slice_index, time_index)
    png_path = output_dir / f"{case_id}_slice{selected_slice:02d}_time{selected_time:02d}_compare.png"
    save_comparison_png(pred, gt, png_path, selected_slice, selected_time, percentile)

    row = {
        "case": case_id,
        "pred_path": str(mat_path),
        "gt_path": str(gt_path),
        "shape": "x".join(str(dim) for dim in pred.shape),
        "finite": True,
        "nonzero": int(np.count_nonzero(pred)),
        "total": int(pred.size),
        "comparison_png": str(png_path),
    }
    row.update(compute_metrics(pred, gt, percentile))
    return row


def collect_mat_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.mat"))


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pred_path",
        type=Path,
        help="Prediction .mat file or directory containing Sub*.mat files.",
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
        default=Path("output/quality_checks"),
        help="Directory for comparison PNGs and metrics CSV.",
    )
    parser.add_argument("--key", default="img4ranking", help="MAT key to inspect.")
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="Percentile used for display scaling and metric normalization.",
    )
    parser.add_argument("--slice-index", type=int, default=None, help="Slice to visualize.")
    parser.add_argument("--time-index", type=int, default=None, help="Time frame to visualize.")
    args = parser.parse_args()

    mat_files = collect_mat_files(args.pred_path)
    if not mat_files:
        parser.error(f"No .mat files found in {args.pred_path}")

    rows = []
    for mat_path in mat_files:
        row = inspect_case(
            mat_path=mat_path,
            gt_root=args.gt_root,
            output_dir=args.output_dir,
            key=args.key,
            percentile=args.percentile,
            slice_index=args.slice_index,
            time_index=args.time_index,
        )
        rows.append(row)
        print(
            f"{row['case']}: shape={row['shape']} NMSE={row['nmse']:.6g} "
            f"PSNR={row['psnr']:.3f} SSIM={row['ssim_mean']:.4f} "
            f"png={row['comparison_png']}"
        )

    csv_path = args.output_dir / "metrics.csv"
    write_csv(rows, csv_path)
    print(f"Saved metrics: {csv_path}")


if __name__ == "__main__":
    main()
