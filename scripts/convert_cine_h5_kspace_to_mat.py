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


def write_case_json(json_path: Path, kspace_path: Path, smap_path: Path | None, overwrite: bool) -> None:
    descriptor = {"kspace": str(kspace_path.resolve()), "mask": []}
    if json_path.exists():
        with json_path.open() as f:
            existing = json.load(f)
        descriptor["mask"] = existing.get("mask", [])
        if existing.get("sensitivity_maps"):
            descriptor["sensitivity_maps"] = existing["sensitivity_maps"]
        if existing.get("kspace") not in (None, descriptor["kspace"]) and not overwrite:
            raise FileExistsError(f"Refusing to replace k-space path in {json_path}; use --overwrite")
        if (
            smap_path is not None
            and existing.get("sensitivity_maps") not in (None, str(smap_path.resolve()))
            and not overwrite
        ):
            raise FileExistsError(f"Refusing to replace sensitivity-map path in {json_path}; use --overwrite")
    if smap_path is not None:
        descriptor["sensitivity_maps"] = str(smap_path.resolve())
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(descriptor, f, indent=2)
        f.write("\n")


def convert_file(
    input_path: Path,
    output_root: Path,
    dataset_key: str,
    smap_key: str | None,
    kspace_subdir: str,
    smap_subdir: str,
    overwrite: bool,
) -> tuple[Path, Path, Path | None]:
    case_id = input_path.stem
    output_path = output_root / "MultiCoil" / "Cine" / kspace_subdir / f"{case_id}_kspace_full.mat"
    smap_path = (
        output_root / "MultiCoil" / "Cine" / smap_subdir / f"{case_id}_sensitivity_maps.mat"
        if smap_key
        else None
    )
    json_path = output_root / "json_input" / f"{case_id}.json"

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}; use --overwrite")
    if smap_path is not None and smap_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {smap_path}; use --overwrite")

    with h5py.File(input_path, "r") as h5_file:
        if dataset_key not in h5_file:
            raise KeyError(f"{input_path} has no dataset {dataset_key!r}; keys: {list(h5_file.keys())}")
        source = h5_file[dataset_key]
        if source.ndim != 5:
            raise ValueError(
                f"{input_path}:{dataset_key} must have shape (slice, coil, time, PE, FE), got {source.shape}"
            )
        kspace = np.asarray(source, dtype=np.complex64)
        smap = None
        if smap_key:
            if smap_key not in h5_file:
                raise KeyError(f"{input_path} has no dataset {smap_key!r}; keys: {list(h5_file.keys())}")
            smap_source = h5_file[smap_key]
            if smap_source.ndim != 5:
                raise ValueError(
                    f"{input_path}:{smap_key} must have shape (slice, coil, time, PE, FE), got {smap_source.shape}"
                )
            smap = np.asarray(smap_source, dtype=np.complex64)

    # Logical reader order is (time, slice, coil, PE, FE). CMRxReconReader
    # reverses scipy-loaded complex MAT axes, so store the reverse on disk.
    logical = np.transpose(kspace, (2, 0, 1, 3, 4))
    mat_order = logical.transpose()
    save_mat_atomic(output_path, {"kspace_full": mat_order})
    if smap is not None:
        smap_logical = np.transpose(smap, (2, 0, 1, 3, 4))
        save_mat_atomic(smap_path, {"sensitivity_maps": smap_logical.transpose()})
    write_case_json(json_path, output_path, smap_path, overwrite)
    return output_path, json_path, smap_path


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
    parser.add_argument(
        "--smap-key",
        default=None,
        help="Optional H5 sensitivity-map key to convert, for example dMap.",
    )
    parser.add_argument(
        "--kspace-subdir",
        default="UnderSample_TaskR1",
        help="Subdirectory under MultiCoil/Cine for converted k-space.",
    )
    parser.add_argument(
        "--smap-subdir",
        default="SensitivityMap_TaskR1",
        help="Subdirectory under MultiCoil/Cine for converted sensitivity maps.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inputs = sorted(args.input_h5_dir.glob(args.glob))
    if not inputs:
        parser.error(f"No files matched {args.input_h5_dir / args.glob}")

    for input_path in inputs:
        output_path, json_path, smap_path = convert_file(
            input_path,
            args.output_root,
            args.dataset_key,
            args.smap_key,
            args.kspace_subdir,
            args.smap_subdir,
            args.overwrite,
        )
        print(f"{input_path.name} -> {output_path}")
        if smap_path is not None:
            print(f"{input_path.name}:{args.smap_key} -> {smap_path}")
        print(f"descriptor -> {json_path}")


if __name__ == "__main__":
    main()
