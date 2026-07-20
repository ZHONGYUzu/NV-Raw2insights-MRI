#!/usr/bin/env python3
"""Verify that FastMRIReader preprocessing reproduces reconstruction_rss geometry."""

from __future__ import annotations

import argparse
from pathlib import Path


def rss_from_kspace(kspace):
    import numpy as np
    from scipy.fft import fftshift, ifftn, ifftshift

    spatial_axes = (-2, -1)
    coil_images = fftshift(
        ifftn(ifftshift(kspace, axes=spatial_axes), axes=spatial_axes, norm="ortho"),
        axes=spatial_axes,
    )
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=-3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--slice-index", type=int, help="Slice to check; default is the middle slice.")
    parser.add_argument("--nmse-threshold", type=float, default=1e-6)
    args = parser.parse_args()

    if args.max_cases <= 0:
        parser.error("--max-cases must be positive")
    paths = sorted(args.input_dir.glob("*.h5"))[: args.max_cases]
    if not paths:
        parser.error(f"No H5 files found under {args.input_dir}")

    print("Loading lightweight fastMRI preprocessing...", flush=True)
    import h5py
    import numpy as np

    from fastmri_preprocessing import crop_kspace_via_image_domain

    failures = []
    for path in paths:
        print(f"Checking one slice from {path.name}...", flush=True)
        with h5py.File(path, "r") as h5_file:
            kspace_dataset = h5_file["kspace"]
            target_dataset = h5_file["reconstruction_rss"]
            num_slices = int(kspace_dataset.shape[0])
            slice_index = num_slices // 2 if args.slice_index is None else args.slice_index
            if not 0 <= slice_index < num_slices:
                raise IndexError(f"{path.name}: slice {slice_index} is outside [0, {num_slices})")
            source_shape = tuple(int(size) for size in kspace_dataset.shape)
            source_kspace = np.asarray(kspace_dataset[slice_index])
            target = np.asarray(target_dataset[slice_index], dtype=np.float32)

        processed_kspace = crop_kspace_via_image_domain(source_kspace, target.shape[-2:])
        reconstruction = rss_from_kspace(processed_kspace)
        if reconstruction.shape != target.shape:
            raise ValueError(f"{path.name}: RSS {reconstruction.shape} != target {target.shape}")
        difference = reconstruction.astype(np.float64) - target.astype(np.float64)
        nmse = float(np.sum(difference**2) / max(np.sum(target.astype(np.float64) ** 2), 1e-12))
        max_abs_error = float(np.max(np.abs(difference)))
        processed_shape = tuple(int(size) for size in processed_kspace.shape)
        print(
            f"{path.name}: slice={slice_index}/{num_slices - 1}, source={source_shape}, "
            f"processed_slice={processed_shape}, "
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
