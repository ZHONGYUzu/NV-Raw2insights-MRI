#!/usr/bin/env python3
"""Verify that FastMRIReader preprocessing reproduces reconstruction_rss geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.fft import fftshift, ifftn, ifftshift

from readers import FastMRIKeys, FastMRIReader


def rss_from_kspace(kspace: np.ndarray) -> np.ndarray:
    spatial_axes = (-2, -1)
    coil_images = fftshift(
        ifftn(ifftshift(kspace, axes=spatial_axes), axes=spatial_axes, norm="ortho"),
        axes=spatial_axes,
    )
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=2))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--nmse-threshold", type=float, default=1e-6)
    args = parser.parse_args()

    if args.max_cases <= 0:
        parser.error("--max-cases must be positive")
    paths = sorted(args.input_dir.glob("*.h5"))[: args.max_cases]
    if not paths:
        parser.error(f"No H5 files found under {args.input_dir}")

    reader = FastMRIReader()
    failures = []
    for path in paths:
        raw = reader.read(path)
        processed_kspace, metadata = reader.get_data(raw)
        reconstruction = rss_from_kspace(processed_kspace)
        target = np.asarray(raw[FastMRIKeys.RECON], dtype=np.float32)
        if reconstruction.shape != target.shape:
            raise ValueError(f"{path.name}: RSS {reconstruction.shape} != target {target.shape}")
        difference = reconstruction.astype(np.float64) - target.astype(np.float64)
        nmse = float(np.sum(difference**2) / max(np.sum(target.astype(np.float64) ** 2), 1e-12))
        max_abs_error = float(np.max(np.abs(difference)))
        source_shape = tuple(int(size) for size in metadata["source_shape"])
        processed_shape = tuple(int(size) for size in processed_kspace.shape)
        print(
            f"{path.name}: source={source_shape}, processed={processed_shape}, "
            f"target={target.shape}, zero_filled_nmse={nmse:.3e}, max_abs_error={max_abs_error:.3e}"
        )
        if not np.isfinite(nmse) or nmse > args.nmse_threshold:
            failures.append((path.name, nmse))

    if failures:
        raise RuntimeError(
            f"FastMRI reader geometry validation failed at threshold {args.nmse_threshold}: {failures}"
        )
    print(f"Validated fastMRI reader geometry for {len(paths)} case(s).")


if __name__ == "__main__":
    main()
