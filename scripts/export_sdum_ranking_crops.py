#!/usr/bin/env python3
"""LEGACY ONLY: export historical ranking crops to a separate directory; preserve full-size originals."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io

from run4ranking import run4Ranking


def load_reconstruction(path: Path, key: str) -> np.ndarray:
    values = scipy.io.loadmat(path)
    if key not in values:
        available = sorted(name for name in values if not name.startswith("__"))
        raise KeyError(f"{path}: missing {key!r}; available keys: {available}")
    reconstruction = np.asarray(values[key])
    if reconstruction.ndim != 4:
        raise ValueError(f"{path}:{key} must be 4D, got {reconstruction.shape}")
    if not np.isfinite(reconstruction).all():
        raise ValueError(f"{path}:{key} contains NaN or Inf")
    return reconstruction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--filetype", default="cine_lax")
    parser.add_argument("--key", default="img4ranking")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("Input and output directories must differ; preserve full-size reconstructions")

    inputs = sorted(args.input_dir.glob("*.mat"))
    if not inputs:
        parser.error(f"No MAT files found under {args.input_dir}")
    # Check all collisions before writing any crop, including symlink/hardlink aliases.
    for input_path in inputs:
        output_path = args.output_dir / input_path.name
        if output_path.resolve() == input_path.resolve() or (
            output_path.exists() and output_path.samefile(input_path)
        ):
            parser.error(f"Output aliases the original reconstruction: {output_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in inputs:
        output_path = args.output_dir / input_path.name
        if output_path.exists() and not args.overwrite:
            print(f"skip existing: {output_path}")
            continue
        full = load_reconstruction(input_path, args.key)
        ranking = run4Ranking(full, args.filetype, do_center_crop_infer=True)
        scipy.io.savemat(output_path, {args.key: ranking}, do_compression=True)
        print(f"{input_path.name}: full {full.shape} -> ranking {ranking.shape}")


if __name__ == "__main__":
    main()
