#!/usr/bin/env python3
"""Extract H5 coil sensitivity maps and attach them to existing CINE descriptors."""

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


def update_descriptor(json_path: Path, smap_path: Path) -> None:
    if not json_path.is_file():
        raise FileNotFoundError(f"Existing case descriptor not found: {json_path}")
    with json_path.open() as descriptor_file:
        descriptor = json.load(descriptor_file)
    if not descriptor.get("kspace"):
        raise ValueError(f"{json_path}: missing kspace path")
    if not descriptor.get("mask"):
        raise ValueError(f"{json_path}: mask list is empty")
    descriptor["sensitivity_maps"] = str(smap_path.resolve())
    with tempfile.NamedTemporaryFile(
        mode="w", dir=json_path.parent, suffix=".json", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(descriptor, tmp, indent=2)
        tmp.write("\n")
    os.replace(tmp_path, json_path)


def convert_case(
    h5_path: Path,
    output_root: Path,
    smap_key: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    case_id = h5_path.stem
    smap_path = (
        output_root
        / "MultiCoil"
        / "Cine"
        / "SensitivityMap_TaskR1"
        / f"{case_id}_sensitivity_maps.mat"
    )
    json_path = output_root / "json_input" / f"{case_id}.json"

    if smap_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {smap_path}; use --overwrite")

    with h5py.File(h5_path, "r") as h5_file:
        if smap_key not in h5_file:
            raise KeyError(f"{h5_path} has no dataset {smap_key!r}; keys: {list(h5_file.keys())}")
        source = h5_file[smap_key]
        if source.ndim != 5:
            raise ValueError(
                f"{h5_path}:{smap_key} must have shape (slice, coil, time, PE, FE), "
                f"got {source.shape}"
            )
        smap = np.asarray(source, dtype=np.complex64)

    # Logical reader order is (time, slice, coil, PE, FE). CMRxReconReader
    # reverses scipy-loaded complex MAT axes, so store the reverse on disk.
    smap_logical = np.transpose(smap, (2, 0, 1, 3, 4))
    save_mat_atomic(smap_path, {"sensitivity_maps": smap_logical.transpose()})
    update_descriptor(json_path, smap_path)
    return smap_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--glob", default="*.h5", help="Input filename glob (default: *.h5)")
    parser.add_argument("--smap-key", default="dMap")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inputs = sorted(args.input_h5_dir.glob(args.glob))
    if not inputs:
        parser.error(f"No files matched {args.input_h5_dir / args.glob}")

    for h5_path in inputs:
        smap_path, json_path = convert_case(
            h5_path,
            args.output_root,
            args.smap_key,
            args.overwrite,
        )
        print(f"{h5_path.name}:{args.smap_key} -> {smap_path}")
        print(f"descriptor -> {json_path}")


if __name__ == "__main__":
    main()
