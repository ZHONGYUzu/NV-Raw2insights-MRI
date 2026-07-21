#!/usr/bin/env python3
"""Create zero-filled RSS reconstructions for a prepared fastMRI cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.io
from scipy.fft import fftshift, ifftn, ifftshift

from fastmri_preprocessing import crop_kspace_via_image_domain


def equispaced_mask(num_cols: int, center_fraction: float, acceleration: float, offset: int) -> np.ndarray:
    """Match the repository's official-style EquispacedKspaceMask calculation."""
    num_low_freqs = int(round(num_cols * center_fraction))
    if not 0 < num_low_freqs < num_cols:
        raise ValueError(f"Invalid low-frequency count {num_low_freqs} for {num_cols} columns")
    adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
        num_low_freqs * acceleration - num_cols
    )
    rounded_accel = round(adjusted_accel)
    if rounded_accel <= 0:
        raise ValueError(f"Invalid adjusted acceleration {adjusted_accel}")

    mask = np.zeros(num_cols, dtype=np.float32)
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad : pad + num_low_freqs] = 1.0
    samples = np.arange(int(offset) % rounded_accel, num_cols - 1, adjusted_accel)
    mask[np.around(samples).astype(np.uint64)] = 1.0
    return mask


def rss_from_kspace(kspace: np.ndarray) -> np.ndarray:
    spatial_axes = (-2, -1)
    coil_images = fftshift(
        ifftn(ifftshift(kspace, axes=spatial_axes), axes=spatial_axes, norm="ortho"),
        axes=spatial_axes,
    )
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=-3)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--acceleration", type=float, default=8.0)
    parser.add_argument("--center-fraction", type=float, default=0.04)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    cases = manifest.get("cases", [])
    if args.max_cases is not None:
        if args.max_cases <= 0:
            parser.error("--max-cases must be positive")
        cases = cases[: args.max_cases]
    if not cases:
        parser.error(f"No cases selected from {args.manifest}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for case in cases:
        source = Path(str(case["source"]))
        output_path = args.output_dir / f"{source.stem}.mat"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to replace {output_path}; pass --overwrite if intended")

        with h5py.File(source, "r", swmr=True) as h5_file:
            kspace_dataset = h5_file["kspace"]
            target_dataset = h5_file["reconstruction_rss"]
            target_spatial_shape = tuple(int(size) for size in target_dataset.shape[-2:])
            num_slices = int(kspace_dataset.shape[0])
            num_cols = target_spatial_shape[-1]
            mask = equispaced_mask(num_cols, args.center_fraction, args.acceleration, args.offset)
            reconstructions = []
            for slice_index in range(num_slices):
                source_kspace = np.asarray(kspace_dataset[slice_index])
                processed_kspace = crop_kspace_via_image_domain(source_kspace, target_spatial_shape)
                masked_kspace = processed_kspace * mask[None, None, :]
                reconstructions.append(rss_from_kspace(masked_kspace))

        reconstruction = np.stack(reconstructions)
        repo_layout = reconstruction.transpose(2, 1, 0)[..., None]
        scipy.io.savemat(output_path, {"img4ranking": repo_layout}, do_compression=True)
        sampled_lines = int(mask.sum())
        effective_acceleration = float(mask.size / sampled_lines)
        records.append(
            {
                "case": source.stem,
                "source": str(source.resolve()),
                "shape": list(reconstruction.shape),
                "sampled_lines": sampled_lines,
                "num_columns": int(mask.size),
                "effective_acceleration": effective_acceleration,
            }
        )
        print(
            f"{source.name}: zero-filled RSS {reconstruction.shape}, "
            f"sampled={sampled_lines}/{mask.size}, effective_acceleration={effective_acceleration:.4f}"
        )

    metadata = {
        "manifest": str(args.manifest.resolve()),
        "nominal_acceleration": args.acceleration,
        "center_fraction": args.center_fraction,
        "equispaced_offset": args.offset,
        "mask_axis": "last spatial k-space axis (fastMRI columns/W)",
        "cases": records,
    }
    (args.output_dir / "zero_filled_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(records)} zero-filled baseline case(s): {args.output_dir}")


if __name__ == "__main__":
    main()
