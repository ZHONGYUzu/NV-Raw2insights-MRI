#!/usr/bin/env python3
"""Generate CINE ground-truth NPY files from source H5 files."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    shifted = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(shifted, axes=(-2, -1), norm="ortho")
    return np.fft.fftshift(image, axes=(-2, -1))


def normalize_complex(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return image
    magnitude = np.abs(image)
    if mode == "max":
        scale = float(np.max(magnitude))
    elif mode.startswith("p"):
        scale = float(np.percentile(magnitude, float(mode[1:])))
    else:
        raise ValueError(f"Unknown normalization mode {mode!r}; use none, max, or p99.5-style percentile")
    if not np.isfinite(scale) or scale <= 0:
        return image
    return image / scale


def ground_truth_from_dimgc(h5_file: h5py.File, key: str) -> np.ndarray:
    if key not in h5_file:
        raise KeyError(f"missing H5 key {key!r}; keys={list(h5_file.keys())}")
    dimgc = np.asarray(h5_file[key])
    if dimgc.ndim != 5 or dimgc.shape[1] != 1:
        raise ValueError(f"{key} must have shape (slice, 1, time, phase, frequency), got {dimgc.shape}")
    return dimgc[:, 0]  # (slice, time, phase, frequency)


def ground_truth_from_kspace_dmap(h5_file: h5py.File, kspace_key: str, dmap_key: str) -> np.ndarray:
    for key in (kspace_key, dmap_key):
        if key not in h5_file:
            raise KeyError(f"missing H5 key {key!r}; keys={list(h5_file.keys())}")

    kspace = np.asarray(h5_file[kspace_key])  # (slice, coil, time, phase, frequency)
    dmap = np.asarray(h5_file[dmap_key])  # (slice, coil, 1, phase, frequency)
    if kspace.ndim != 5:
        raise ValueError(f"{kspace_key} must have shape (slice, coil, time, phase, frequency), got {kspace.shape}")
    if dmap.ndim != 5 or dmap.shape[2] != 1:
        raise ValueError(f"{dmap_key} must have shape (slice, coil, 1, phase, frequency), got {dmap.shape}")
    if kspace.shape[0] != dmap.shape[0] or kspace.shape[1] != dmap.shape[1] or kspace.shape[-2:] != dmap.shape[-2:]:
        raise ValueError(f"{kspace_key} shape {kspace.shape} is incompatible with {dmap_key} shape {dmap.shape}")

    coil_images = ifft2c(kspace)
    numerator = np.sum(coil_images * np.conj(dmap), axis=1)
    denominator = np.sum(np.abs(dmap) ** 2, axis=1) + 1e-8
    return numerator / denominator  # (slice, time, phase, frequency)


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.save(tmp_path, array)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def convert_file(
    input_path: Path,
    output_dir: Path,
    source: str,
    normalize: str,
    dimgc_key: str,
    kspace_key: str,
    dmap_key: str,
    prefix: str,
    overwrite: bool,
) -> Path:
    output_path = output_dir / f"{prefix}{input_path.stem}.npy"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}; use --overwrite")

    with h5py.File(input_path, "r") as h5_file:
        if source == "dimgc":
            gt = ground_truth_from_dimgc(h5_file, dimgc_key)
        elif source == "kspace-dmap":
            gt = ground_truth_from_kspace_dmap(h5_file, kspace_key, dmap_key)
        else:
            raise ValueError(f"Unknown source {source!r}")

    gt = normalize_complex(gt, normalize).astype(np.complex64, copy=False)
    gt = gt[..., None]  # (slice, time, phase, frequency, 1)
    save_npy_atomic(output_path, gt)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5-dir", type=Path, required=True, help="Directory containing source CINE H5 files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated norm_img_<case>.npy files.")
    parser.add_argument("--glob", default="*.h5", help="Input filename glob (default: *.h5).")
    parser.add_argument(
        "--source",
        choices=("dimgc", "kspace-dmap"),
        default="kspace-dmap",
        help="Ground-truth source. kspace-dmap recombines full k-space with dMap; dimgc uses the H5 coil-combined image.",
    )
    parser.add_argument(
        "--normalize",
        default="max",
        help="Complex image normalization before saving: none, max, or p99.5-style percentile (default: max).",
    )
    parser.add_argument("--dimgc-key", default="dImgC")
    parser.add_argument("--kspace-key", default="kSpace")
    parser.add_argument("--dmap-key", default="dMap")
    parser.add_argument("--prefix", default="norm_img_", help="Output filename prefix (default: norm_img_).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inputs = sorted(args.input_h5_dir.glob(args.glob))
    if not inputs:
        parser.error(f"No files matched {args.input_h5_dir / args.glob}")

    for input_path in inputs:
        output_path = convert_file(
            input_path=input_path,
            output_dir=args.output_dir,
            source=args.source,
            normalize=args.normalize,
            dimgc_key=args.dimgc_key,
            kspace_key=args.kspace_key,
            dmap_key=args.dmap_key,
            prefix=args.prefix,
            overwrite=args.overwrite,
        )
        saved = np.load(output_path, mmap_mode="r")
        print(
            f"{input_path.name} -> {output_path} "
            f"shape={saved.shape} dtype={saved.dtype} abs_max={float(np.max(np.abs(saved))):.6g}"
        )


if __name__ == "__main__":
    main()
