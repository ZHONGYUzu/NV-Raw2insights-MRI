#!/usr/bin/env python3
"""Validate converted custom CINE cases before expensive model inference."""

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import scipy.io


def read_mat(path: Path) -> dict:
    try:
        with h5py.File(path, "r", swmr=True) as mat_file:
            return {key: mat_file[key][()] for key in mat_file}
    except (OSError, ValueError):
        return {key: value for key, value in scipy.io.loadmat(path).items() if not key.startswith("__")}


def logical_kspace(values: dict, path: Path) -> np.ndarray:
    key = next((candidate for candidate in ("kus", "kspace_full", "kspace") if candidate in values), None)
    if key is None:
        raise ValueError(f"{path}: expected one of kus, kspace_full, or kspace")
    value = values[key]
    if np.issubdtype(value.dtype, np.complexfloating):
        shape = (1,) * (5 - value.ndim) + value.shape[::-1]
        return value.transpose().reshape(shape)
    if value.dtype.fields and {"real", "imag"}.issubset(value.dtype.fields):
        result = value["real"] + 1j * value["imag"]
        return result.reshape((1,) * (5 - result.ndim) + result.shape)
    raise ValueError(f"{path}:{key} is not complex data (dtype={value.dtype})")


def logical_mask(values: dict, path: Path) -> np.ndarray:
    if "mask" not in values:
        raise ValueError(f"{path}: missing mask key")
    mask = np.asarray(values["mask"])
    if not np.all(np.isin(mask, (0, 1))):
        raise ValueError(f"{path}: mask contains values other than 0 and 1")
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=(0, 1))
    elif mask.ndim == 3:
        mask = np.expand_dims(mask, axis=(1, 2))
    else:
        raise ValueError(f"{path}: mask must be 2D or 3D before reader expansion, got {mask.shape}")
    return (mask > 0).astype(np.float32)


def validate_case(json_path: Path) -> str:
    with json_path.open() as descriptor_file:
        descriptor = json.load(descriptor_file)
    kspace_path = Path(descriptor["kspace"])
    mask_paths = [Path(path) for path in descriptor["mask"]]
    if not kspace_path.is_file():
        raise FileNotFoundError(f"{json_path}: missing k-space file {kspace_path}")
    if not mask_paths:
        raise ValueError(f"{json_path}: mask list is empty")
    if not re.search(r"(?:^|[/\\])MultiCoil[/\\]Cine(?:[/\\])", str(kspace_path), flags=re.I):
        raise ValueError(f"{json_path}: k-space path must contain MultiCoil/Cine for acquisition parsing")

    kspace = logical_kspace(read_mat(kspace_path), kspace_path)
    if kspace.ndim != 5:
        raise ValueError(f"{kspace_path}: logical k-space must be 5D, got {kspace.shape}")
    num_time, _, _, num_phase, num_frequency = kspace.shape

    for mask_path in mask_paths:
        if not mask_path.is_file():
            raise FileNotFoundError(f"{json_path}: missing mask file {mask_path}")
        match = re.search(r"_mask_([^/\\]+)\.mat$", mask_path.name)
        if not match or not re.search(r"\d+$", match.group(1)):
            raise ValueError(f"{mask_path}: filename must end like _mask_ktRadial8.mat")
        mask = logical_mask(read_mat(mask_path), mask_path)
        if mask.shape[-2:] != (num_phase, num_frequency):
            raise ValueError(
                f"{mask_path}: spatial shape {mask.shape[-2:]} does not match k-space {(num_phase, num_frequency)}"
            )
        if mask.shape[0] not in (1, num_time):
            raise ValueError(f"{mask_path}: time size {mask.shape[0]} does not match k-space time size {num_time}")
        center = mask[..., num_phase // 2, num_frequency // 2]
        if not np.all(center > 0):
            raise ValueError(f"{mask_path}: center is not positive for every frame (ACS check would fail)")

    return f"{json_path.name}: k-space {tuple(kspace.shape)}, masks {len(mask_paths)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        default=Path("dataset/CustomCINEDataR1/json_input"),
        help="Directory containing case JSON descriptors (default: dataset/CustomCINEDataR1/json_input)",
    )
    args = parser.parse_args()
    descriptors = sorted(args.input_path.glob("*.json"))
    if not descriptors:
        parser.error(f"No JSON descriptors found in {args.input_path}")

    failures = []
    for descriptor in descriptors:
        try:
            print(f"OK: {validate_case(descriptor)}")
        except Exception as error:
            failures.append((descriptor, error))
            print(f"ERROR: {descriptor.name}: {error}")
    if failures:
        raise SystemExit(f"Validation failed for {len(failures)} of {len(descriptors)} cases")
    print(f"Validated {len(descriptors)} case(s).")


if __name__ == "__main__":
    main()
