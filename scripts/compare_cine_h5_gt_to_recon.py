#!/usr/bin/env python3
"""Compare a full-sampled CINE H5 reference against a saved reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import scipy.io


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    shifted = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(shifted, axes=(-2, -1), norm="ortho")
    return np.fft.fftshift(image, axes=(-2, -1))


def read_mat_key(path: Path, key: str) -> np.ndarray:
    try:
        with h5py.File(path, "r", swmr=True) as mat_file:
            if key not in mat_file:
                raise KeyError(f"{path}: missing key {key!r}; keys={list(mat_file.keys())}")
            return np.asarray(mat_file[key][()])
    except OSError:
        data = scipy.io.loadmat(path)
        if key not in data:
            keys = [candidate for candidate in data if not candidate.startswith("__")]
            raise KeyError(f"{path}: missing key {key!r}; keys={keys}")
        return np.asarray(data[key])


def h5_gt_from_kspace_dmap(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5_file:
        kspace = np.asarray(h5_file["kSpace"])  # (slice, coil, time, phase, frequency)
        dmap = np.asarray(h5_file["dMap"])  # (slice, coil, 1, phase, frequency)

    coil_images = ifft2c(kspace)
    numerator = np.sum(coil_images * np.conj(dmap), axis=1)
    denominator = np.sum(np.abs(dmap) ** 2, axis=1) + 1e-8
    combined = numerator / denominator  # (slice, time, phase, frequency)
    magnitude = np.abs(combined).astype(np.float32)
    return magnitude.transpose(3, 2, 0, 1)  # (frequency, phase, slice, time)


def h5_gt_from_dimgc(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5_file:
        dimgc = np.asarray(h5_file["dImgC"])  # (slice, 1, time, phase, frequency)
    magnitude = np.abs(dimgc[:, 0]).astype(np.float32)  # (slice, time, phase, frequency)
    return magnitude.transpose(3, 2, 0, 1)  # (frequency, phase, slice, time)


def normalize_by(value: np.ndarray, mode: str) -> np.ndarray:
    value = value.astype(np.float32, copy=False)
    if mode == "none":
        return value
    if mode == "max":
        scale = float(np.max(np.abs(value)))
    elif mode.startswith("p"):
        scale = float(np.percentile(np.abs(value), float(mode[1:])))
    else:
        raise ValueError(f"Unknown normalization mode {mode!r}")
    if not np.isfinite(scale) or scale <= 0:
        return value
    return value / scale


def fit_scale(reference: np.ndarray, prediction: np.ndarray) -> Tuple[float, np.ndarray]:
    denom = float(np.sum(prediction * prediction))
    if denom <= 0:
        return 1.0, prediction
    scale = float(np.sum(reference * prediction) / denom)
    return scale, prediction * scale


def metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    diff = prediction - reference
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    denom = float(np.sum(reference**2))
    nmse = float(np.sum(diff**2) / denom) if denom > 0 else float("nan")
    nrmse = float(np.sqrt(np.sum(diff**2)) / np.sqrt(denom)) if denom > 0 else float("nan")
    corr = float(np.corrcoef(reference.ravel(), prediction.ravel())[0, 1])
    data_range = float(reference.max() - reference.min())
    psnr = float("nan") if mse <= 0 or data_range <= 0 else float(20.0 * np.log10(data_range / np.sqrt(mse)))
    return {"corr": corr, "nmse": nmse, "nrmse": nrmse, "psnr": psnr, "mse": mse, "mae": mae}


def print_stats(name: str, value: np.ndarray) -> None:
    percentiles = np.percentile(value, [0, 1, 50, 95, 99, 99.5, 99.9, 100])
    print(
        f"{name}: shape={value.shape} dtype={value.dtype} "
        f"finite={np.isfinite(value).all()} min={value.min():.6g} "
        f"max={value.max():.6g} mean={value.mean():.6g} std={value.std():.6g}"
    )
    print(f"{name} percentiles [0,1,50,95,99,99.5,99.9,100]: {percentiles}")


def save_png(reference: np.ndarray, prediction: np.ndarray, output_path: Path, slice_index: int, time_index: int) -> None:
    import matplotlib.pyplot as plt

    ref = reference[:, :, slice_index, time_index]
    pred = prediction[:, :, slice_index, time_index]
    err = np.abs(pred - ref)
    vmax = float(np.percentile(np.stack([ref, pred]), 99.5))
    err_vmax = float(np.percentile(err, 99.5))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = [
        ("H5 kSpace+dMap GT", ref, "gray", vmax),
        ("Repo recon", pred, "gray", vmax),
        ("Abs error", err, "magma", err_vmax),
    ]
    for ax, (title, image, cmap, limit) in zip(axes, panels):
        im = ax.imshow(image, cmap=cmap, vmin=0, vmax=limit)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"slice={slice_index}, time={time_index}")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True, help="Source CINE H5 file.")
    parser.add_argument("--recon", type=Path, required=True, help="Saved repo reconstruction MAT file.")
    parser.add_argument("--key", default="img4ranking", help="MAT key for the saved reconstruction.")
    parser.add_argument(
        "--gt-source",
        choices=("kspace-dmap", "dimgc"),
        default="kspace-dmap",
        help="How to build the fully sampled reference from the H5 file.",
    )
    parser.add_argument(
        "--normalize-gt",
        default="max",
        help="Reference normalization: none, max, or p99.5-style percentile.",
    )
    parser.add_argument(
        "--normalize-recon",
        default="none",
        help="Prediction normalization before raw metrics: none, max, or p99.5-style percentile.",
    )
    parser.add_argument("--slice", type=int, default=6, help="Slice index for optional PNG.")
    parser.add_argument("--time", type=int, default=12, help="Time index for optional PNG.")
    parser.add_argument("--png", type=Path, help="Optional comparison PNG output path.")
    args = parser.parse_args()

    if args.gt_source == "kspace-dmap":
        gt = h5_gt_from_kspace_dmap(args.h5)
    else:
        gt = h5_gt_from_dimgc(args.h5)
    gt = normalize_by(gt, args.normalize_gt)

    recon = np.squeeze(read_mat_key(args.recon, args.key)).astype(np.float32)
    recon = np.abs(recon)
    recon = normalize_by(recon, args.normalize_recon)

    if recon.shape != gt.shape:
        raise ValueError(f"Shape mismatch: GT {gt.shape} vs recon {recon.shape}")

    print_stats("gt", gt)
    print_stats("recon", recon)
    print("raw metrics:", metrics(gt, recon))

    scale, recon_scaled = fit_scale(gt, recon)
    print(f"least-squares scale applied to recon: {scale:.8g}")
    print("scale-fitted metrics:", metrics(gt, recon_scaled))

    if args.png:
        save_png(gt, recon_scaled, args.png, args.slice, args.time)
        print(f"wrote {args.png}")


if __name__ == "__main__":
    main()
