#!/usr/bin/env python3
"""Create CINE fine-tuning descriptors that point directly to raw H5 k-space.

This avoids copying large k-space arrays into the repository/work directory.  It
creates small fixed-mask MAT files from VISTA txt masks and JSON descriptors that
CMRxReconReader can load after raw-H5 support is enabled.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import h5py
import numpy as np
import scipy.io

SPLIT_DIR_NAMES = {
    "train": "train",
    "val": "val",
    "fixed_test_external_benchmark": "fixed_test",
}


def save_mat_atomic(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".mat", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        scipy.io.savemat(tmp_path, values, appendmat=False, do_compression=True)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def read_raw_shape(h5_path: Path, kspace_key: str) -> tuple[int, int, int, int, int]:
    with h5py.File(h5_path, "r") as h5_file:
        if kspace_key not in h5_file:
            raise KeyError(f"{h5_path}: missing key {kspace_key!r}; keys={list(h5_file.keys())}")
        shape = tuple(int(x) for x in h5_file[kspace_key].shape)
        dtype = h5_file[kspace_key].dtype
    if len(shape) != 5:
        raise ValueError(f"{h5_path}:{kspace_key} expected 5D (slice, coil, time, PE, FE), got {shape}")
    if not np.issubdtype(dtype, np.complexfloating):
        raise ValueError(f"{h5_path}:{kspace_key} expected complex dtype, got {dtype}")
    return shape


def load_vista_mask(mask_path: Path, frequency_size: int, acs_lines: int, delimiter: str | None) -> np.ndarray:
    mask_phase_time = np.loadtxt(mask_path, delimiter=delimiter, dtype=np.float32)
    if mask_phase_time.ndim != 2:
        raise ValueError(f"{mask_path}: expected 2D (phase,time), got {mask_phase_time.shape}")
    mask_time_phase = (mask_phase_time.T > 0).astype(np.float32)
    if not 0 <= acs_lines <= mask_time_phase.shape[1]:
        raise ValueError(f"acs_lines={acs_lines} incompatible with mask shape {mask_phase_time.shape}")
    if acs_lines:
        start = (mask_time_phase.shape[1] - acs_lines) // 2
        mask_time_phase[:, start:start + acs_lines] = 1.0
    return np.repeat(mask_time_phase[:, :, None], frequency_size, axis=2)


def find_mask(mask_dir: Path, phase_size: int, time_size: int, acceleration: int, seed: int) -> Path:
    path = mask_dir / f"mask_VISTA_{phase_size}x{time_size}_acc{acceleration}_{seed}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing VISTA mask {path}")
    return path


def write_descriptor(
    json_path: Path,
    h5_path: Path,
    mask_path: Path,
    case_id: str,
    split_name: str,
    kspace_shape: tuple[int, int, int, int, int],
    kspace_key: str,
    overwrite: bool,
) -> None:
    if json_path.exists() and not overwrite:
        raise FileExistsError(f"Descriptor exists: {json_path}; use --overwrite")
    payload = {
        "kspace": str(h5_path.resolve()),
        "kspace_key": kspace_key,
        "raw_h5_layout": "slice,coil,time,phase,frequency",
        "acquisition": "Cine",
        "mask": [str(mask_path.resolve())],
        "case_id": case_id,
        "split": split_name,
        "source_kspace_shape": list(kspace_shape),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--raw-h5-root", type=Path, default=Path("/mnt/qdata/rawdata/CINE/2D_h5_compressed"))
    parser.add_argument("--mask-dir", type=Path, default=Path("/home/students/studxusiy1/mr_recon/masks"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--kspace-key", default="kSpace")
    parser.add_argument("--acceleration", type=int, default=8)
    parser.add_argument("--mask-seed", type=int, default=8)
    parser.add_argument("--mask-type", default="ktRadial8")
    parser.add_argument("--acs-lines", type=int, default=20)
    parser.add_argument("--delimiter", default=",", help="Use whitespace for whitespace-delimited files")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    delimiter = None if args.delimiter == "whitespace" else args.delimiter
    split = json.loads(args.split.read_text())
    manifest = {
        "source_split": str(args.split),
        "raw_h5_root": str(args.raw_h5_root),
        "mask_dir": str(args.mask_dir),
        "output_root": str(args.output_root),
        "kspace_key": args.kspace_key,
        "acceleration": args.acceleration,
        "mask_seed": args.mask_seed,
        "mask_type": args.mask_type,
        "acs_lines": args.acs_lines,
        "splits": {},
    }

    for split_key, dir_name in SPLIT_DIR_NAMES.items():
        cases = split[split_key]
        manifest["splits"][dir_name] = []
        for case_id in cases:
            h5_path = args.raw_h5_root / f"{case_id}.h5"
            if not h5_path.is_file():
                raise FileNotFoundError(f"Missing raw H5 {h5_path}")
            kspace_shape = read_raw_shape(h5_path, args.kspace_key)
            num_slice, num_coil, num_time, num_phase, num_frequency = kspace_shape
            vista_mask = find_mask(args.mask_dir, num_phase, num_time, args.acceleration, args.mask_seed)
            mask = load_vista_mask(vista_mask, num_frequency, args.acs_lines, delimiter)
            if mask.shape != (num_time, num_phase, num_frequency):
                raise ValueError(f"{case_id}: mask shape {mask.shape} does not match expected {(num_time, num_phase, num_frequency)}")
            if not np.all(mask[:, num_phase // 2, num_frequency // 2] > 0):
                raise ValueError(f"{case_id}: ACS center check failed")

            mask_path = args.output_root / "MultiCoil" / "Cine" / "Mask_TaskR1" / f"{case_id}_mask_{args.mask_type}.mat"
            if mask_path.exists() and not args.overwrite:
                raise FileExistsError(f"Mask exists: {mask_path}; use --overwrite")
            save_mat_atomic(mask_path, {"mask": mask})

            json_path = args.output_root / dir_name / f"{case_id}_cine.json"
            write_descriptor(json_path, h5_path, mask_path, case_id, dir_name, kspace_shape, args.kspace_key, args.overwrite)
            manifest["splits"][dir_name].append({
                "case_id": case_id,
                "json": str(json_path),
                "raw_h5": str(h5_path),
                "mask": str(mask_path),
                "vista_mask": str(vista_mask),
                "kspace_shape": list(kspace_shape),
            })

    manifest_path = args.output_root / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest exists: {manifest_path}; use --overwrite")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("manifest", manifest_path)
    for split_name, records in manifest["splits"].items():
        print(split_name, len(records))


if __name__ == "__main__":
    main()
