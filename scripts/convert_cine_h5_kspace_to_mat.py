#!/usr/bin/env python3
"""Convert custom CINE H5 k-space into the CMRxRecon inference layout."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import scipy.io


def save_mat_atomic(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".mat", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        scipy.io.savemat(tmp_path, values, appendmat=False, do_compression=True)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_case_json(json_path: Path, kspace_path: Path, overwrite: bool) -> None:
    descriptor = {"kspace": str(kspace_path.resolve()), "mask": []}
    if json_path.exists():
        with json_path.open() as f:
            existing = json.load(f)
        descriptor["mask"] = existing.get("mask", [])
        if existing.get("kspace") not in (None, descriptor["kspace"]) and not overwrite:
            raise FileExistsError(f"Refusing to replace k-space path in {json_path}; use --overwrite")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(descriptor, f, indent=2)
        f.write("\n")


def convert_file(input_path: Path, output_root: Path, dataset_key: str, overwrite: bool) -> tuple[Path, Path]:
    case_id = input_path.stem
    output_path = output_root / "MultiCoil" / "Cine" / "UnderSample_TaskR1" / f"{case_id}_kspace_full.mat"
    json_path = output_root / "json_input" / f"{case_id}.json"

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}; use --overwrite")

    with h5py.File(input_path, "r") as h5_file:
        if dataset_key not in h5_file:
            raise KeyError(f"{input_path} has no dataset {dataset_key!r}; keys: {list(h5_file.keys())}")
        source = h5_file[dataset_key]
        if source.ndim != 5:
            raise ValueError(
                f"{input_path}:{dataset_key} must have shape (slice, coil, time, PE, FE), got {source.shape}"
            )
        kspace = np.asarray(source, dtype=np.complex64)

    # Logical reader order is (time, slice, coil, PE, FE). CMRxReconReader
    # reverses scipy-loaded complex MAT axes, so store the reverse on disk.
    logical = np.transpose(kspace, (2, 0, 1, 3, 4))
    mat_order = logical.transpose()
    save_mat_atomic(output_path, {"kspace_full": mat_order})
    write_case_json(json_path, output_path, overwrite)
    return output_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/CustomCINEDataR1"),
        help="Converted dataset root (default: dataset/CustomCINEDataR1)",
    )
    parser.add_argument("--glob", default="*.h5", help="Input filename glob (default: *.h5)")
    parser.add_argument("--dataset-key", default="kSpace")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inputs = sorted(args.input_h5_dir.glob(args.glob))
    if not inputs:
        parser.error(f"No files matched {args.input_h5_dir / args.glob}")

    for input_path in inputs:
        output_path, json_path = convert_file(
            input_path, args.output_root, args.dataset_key, args.overwrite
        )
        print(f"{input_path.name} -> {output_path}")
        print(f"descriptor -> {json_path}")


if __name__ == "__main__":
    main()
