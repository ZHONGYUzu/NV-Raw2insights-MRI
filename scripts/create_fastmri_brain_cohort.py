#!/usr/bin/env python3
"""Create a reproducible fastMRI brain cohort using symlinks to read-only H5 files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np


SUPPORTED_ACQUISITIONS = {"AXT1", "AXT1PRE", "AXT1POST", "AXT2", "AXFLAIR"}


def decode_attribute(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode()
    return str(value)


def inspect_case(path: Path) -> dict[str, object]:
    with h5py.File(path, "r", swmr=True) as h5_file:
        missing = [key for key in ("kspace", "reconstruction_rss") if key not in h5_file]
        if missing:
            raise KeyError(f"{path}: missing keys {missing}; keys={list(h5_file.keys())}")
        kspace = h5_file["kspace"]
        target = h5_file["reconstruction_rss"]
        acquisition = decode_attribute(h5_file.attrs.get("acquisition", ""))
        patient_id = decode_attribute(h5_file.attrs.get("patient_id", ""))
        kspace_shape = tuple(int(size) for size in kspace.shape)
        target_shape = tuple(int(size) for size in target.shape)
        kspace_dtype = kspace.dtype
        target_dtype = target.dtype
    if len(kspace_shape) != 4:
        raise ValueError(f"{path}: kspace must be (slice, coil, height, width), got {kspace_shape}")
    if not np.issubdtype(kspace_dtype, np.complexfloating):
        raise TypeError(f"{path}: kspace must be complex, got {kspace_dtype}")
    if len(target_shape) != 3 or target_shape[0] != kspace_shape[0]:
        raise ValueError(f"{path}: reconstruction_rss {target_shape} is incompatible with kspace {kspace_shape}")
    if acquisition not in SUPPORTED_ACQUISITIONS:
        raise ValueError(f"{path}: unsupported acquisition {acquisition!r}")
    return {
        "filename": path.name,
        "source": str(path.resolve()),
        "acquisition": acquisition,
        "patient_id": patient_id,
        "kspace_shape": list(kspace_shape),
        "kspace_dtype": str(kspace_dtype),
        "target_shape": list(target_shape),
        "target_dtype": str(target_dtype),
    }


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".json", mode="w", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(value, temporary, indent=2)
        temporary.write("\n")
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def stable_rank(seed: int, filename: str) -> str:
    return hashlib.sha256(f"{seed}:{filename}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument(
        "--acquisition",
        nargs="+",
        default=["AXT2"],
        choices=sorted(SUPPORTED_ACQUISITIONS),
        help="Eligible raw fastMRI acquisition labels (default: AXT2).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_cases <= 0:
        parser.error("--num-cases must be positive")
    sources = sorted(args.source_dir.glob("*.h5"))
    if not sources:
        parser.error(f"No H5 files found under {args.source_dir}")

    eligible = []
    failures = []
    for source in sources:
        try:
            record = inspect_case(source)
            if record["acquisition"] in args.acquisition:
                eligible.append(record)
        except Exception as error:
            failures.append({"file": str(source), "error": str(error)})
    if args.num_cases > len(eligible):
        parser.error(
            f"Requested {args.num_cases} cases but only {len(eligible)} valid cases matched "
            f"acquisitions {args.acquisition}"
        )

    selected = sorted(eligible, key=lambda record: stable_rank(args.seed, str(record["filename"])))[: args.num_cases]
    selected = sorted(selected, key=lambda record: str(record["filename"]))
    input_dir = args.output_root / "h5_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for existing in input_dir.glob("*.h5"):
            existing.unlink()

    for record in selected:
        source = Path(str(record["source"]))
        link = input_dir / str(record["filename"])
        if link.exists() or link.is_symlink():
            if not args.overwrite:
                raise FileExistsError(f"{link} already exists; use --overwrite")
            link.unlink()
        link.symlink_to(source)
        record["input_link"] = str(link.absolute())
        print(
            f"{record['filename']}: acquisition={record['acquisition']}, "
            f"kspace={tuple(record['kspace_shape'])}, target={tuple(record['target_shape'])}"
        )

    manifest = {
        "source_dir": str(args.source_dir.resolve()),
        "input_dir": str(input_dir.resolve()),
        "selection_method": "sha256(seed:filename), lowest hashes",
        "seed": args.seed,
        "acquisitions": args.acquisition,
        "num_source_files": len(sources),
        "num_eligible_files": len(eligible),
        "num_selected_cases": len(selected),
        "selected_files": [str(record["filename"]) for record in selected],
        "cases": selected,
        "inspection_failures": failures,
    }
    manifest_path = args.output_root / "cohort_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(f"Selected files: {', '.join(manifest['selected_files'])}")
    print(f"Manifest: {manifest_path}")
    print(f"H5 input: {input_dir}")


if __name__ == "__main__":
    main()
