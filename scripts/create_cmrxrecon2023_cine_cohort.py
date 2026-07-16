#!/usr/bin/env python3
"""Create reproducible CMRxRecon 2023 CINE descriptors from native MAT files."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import tempfile
from pathlib import Path

import h5py
import numpy as np
import scipy.io


def save_mat_atomic(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".mat", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        scipy.io.savemat(temporary_path, values, appendmat=False, do_compression=True)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


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


def inspect_kspace(path: Path, key: str) -> tuple[int, ...]:
    with h5py.File(path, "r", swmr=True) as mat_file:
        if key not in mat_file:
            raise KeyError(f"{path}: missing {key!r}; keys={list(mat_file.keys())}")
        dataset = mat_file[key]
        shape = tuple(int(size) for size in dataset.shape)
        dtype = dataset.dtype
    if len(shape) != 5:
        raise ValueError(f"{path}:{key} must be (time, slice, coil, PE, FE), got {shape}")
    if not dtype.fields or not {"real", "imag"}.issubset(dtype.fields):
        raise TypeError(f"{path}:{key} must use compound complex dtype, got {dtype}")
    return shape


def load_mask(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, "r", swmr=True) as mat_file:
        if key not in mat_file:
            raise KeyError(f"{path}: missing {key!r}; keys={list(mat_file.keys())}")
        mask = np.asarray(mat_file[key], dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"{path}:{key} must be a 2D PE/FE mask, got {mask.shape}")
    if not np.isfinite(mask).all() or not np.all(np.isin(mask, (0, 1))):
        raise ValueError(f"{path}:{key} must be finite and binary")
    return mask


def force_acs(mask: np.ndarray, lines: int) -> np.ndarray:
    if not 0 <= lines <= mask.shape[0]:
        raise ValueError(f"--force-acs-lines must be between 0 and PE={mask.shape[0]}")
    result = mask.copy()
    if lines:
        start = (result.shape[0] - lines) // 2
        result[start : start + lines, :] = 1
    return result


def central_fully_sampled_width(mask: np.ndarray) -> int:
    sampled_rows = np.all(mask > 0, axis=1)
    center = mask.shape[0] // 2
    if not sampled_rows[center]:
        return 0
    start = center
    stop = center + 1
    while start > 0 and sampled_rows[start - 1]:
        start -= 1
    while stop < sampled_rows.size and sampled_rows[stop]:
        stop += 1
    return stop - start


def available_cases(raw_root: Path, view: str, acceleration_folder: str) -> list[dict[str, object]]:
    full_root = raw_root / "FullSample"
    mask_root = raw_root / acceleration_folder
    cases = []
    for subject_dir in sorted(full_root.glob("P[0-9][0-9][0-9]")):
        subject = subject_dir.name
        kspace_path = subject_dir / f"{view}.mat"
        mask_path = mask_root / subject / f"{view}_mask.mat"
        if kspace_path.is_file() and mask_path.is_file():
            cases.append({"subject": subject, "kspace": kspace_path, "mask": mask_path})
    return cases


def parse_subjects(value: str | None) -> list[str] | None:
    if value is None:
        return None
    subjects = [item.strip().upper() for item in value.split(",") if item.strip()]
    for subject in subjects:
        if not re.fullmatch(r"P\d{3}", subject):
            raise ValueError(f"Invalid subject {subject!r}; expected P001-style identifiers")
    if len(set(subjects)) != len(subjects):
        raise ValueError("--subjects contains duplicates")
    return subjects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "/mnt/qdata/rawdata/CMRxRecon/CMRxRecon_2023_training_init/"
            "ChallengeData/MultiCoil/CINE/TrainingSet/FullSample"
        ),
        help="FullSample directory containing P001-style subject folders.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--view", choices=("cine_lax", "cine_sax"), default="cine_lax")
    parser.add_argument("--acceleration-folder", default="AccFactor08")
    parser.add_argument("--mask-key", default="mask08")
    parser.add_argument("--mask-type", default="ktRadial8")
    parser.add_argument("--kspace-key", default="kspace_full")
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--subjects",
        help="Optional comma-separated P001-style list; bypasses random selection.",
    )
    parser.add_argument(
        "--force-acs-lines",
        type=int,
        default=0,
        help="Force this many central PE lines; 0 preserves official masks unchanged.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_cases <= 0:
        parser.error("--num-cases must be positive")
    if not re.fullmatch(r"[A-Za-z_]+\d+", args.mask_type):
        parser.error("--mask-type must end in the nominal acceleration, such as ktRadial8")

    # Accept either .../TrainingSet/FullSample or .../TrainingSet for convenience.
    full_root = args.raw_root / "FullSample" if (args.raw_root / "FullSample").is_dir() else args.raw_root
    if full_root.name != "FullSample" or not full_root.is_dir():
        parser.error(f"Could not locate a FullSample directory from {args.raw_root}")
    training_root = full_root.parent
    candidates = available_cases(training_root, args.view, args.acceleration_folder)
    if not candidates:
        parser.error(
            f"No paired {args.view} k-space/masks found under {full_root} and "
            f"{training_root / args.acceleration_folder}"
        )

    requested = parse_subjects(args.subjects)
    by_subject = {str(case["subject"]): case for case in candidates}
    if requested is not None:
        missing = [subject for subject in requested if subject not in by_subject]
        if missing:
            parser.error(f"Requested subjects are missing paired files: {', '.join(missing)}")
        selected = [by_subject[subject] for subject in requested]
    else:
        if args.num_cases > len(candidates):
            parser.error(f"Requested {args.num_cases} cases but only {len(candidates)} paired cases exist")
        selected = random.Random(args.seed).sample(candidates, args.num_cases)
    selected = sorted(selected, key=lambda case: str(case["subject"]))

    mask_dir = args.output_root / "MultiCoil" / "CINE" / "Mask_TaskR1"
    json_dir = args.output_root / "json_input"
    records = []
    for case in selected:
        subject = str(case["subject"])
        kspace_path = Path(case["kspace"]).resolve()
        source_mask_path = Path(case["mask"]).resolve()
        case_id = f"{subject}_{args.view}"
        output_mask = mask_dir / f"{case_id}_mask_{args.mask_type}.mat"
        output_json = json_dir / f"{case_id}.json"
        if not args.overwrite and (output_mask.exists() or output_json.exists()):
            raise FileExistsError(f"Derived files already exist for {case_id}; use --overwrite")

        kspace_shape = inspect_kspace(kspace_path, args.kspace_key)
        mask = load_mask(source_mask_path, args.mask_key)
        if mask.shape != kspace_shape[-2:]:
            raise ValueError(f"{case_id}: mask {mask.shape} does not match k-space PE/FE {kspace_shape[-2:]}")
        mask = force_acs(mask, args.force_acs_lines)
        if mask[mask.shape[0] // 2, mask.shape[1] // 2] <= 0:
            raise ValueError(f"{case_id}: mask center is not sampled; use --force-acs-lines if intentional")

        save_mat_atomic(output_mask, {"mask": mask.astype(np.float32, copy=False)})
        descriptor = {"kspace": str(kspace_path), "mask": [str(output_mask.resolve())]}
        write_json_atomic(output_json, descriptor)
        sampled = float(mask.sum())
        record = {
            "case_id": case_id,
            "subject": subject,
            "view": args.view,
            "kspace": str(kspace_path),
            "source_mask": str(source_mask_path),
            "derived_mask": str(output_mask.resolve()),
            "descriptor": str(output_json.resolve()),
            "kspace_shape": list(kspace_shape),
            "mask_shape": list(mask.shape),
            "sampled_pe_lines": int(mask.sum() / mask.shape[1]),
            "central_fully_sampled_width": central_fully_sampled_width(mask),
            "effective_acceleration": float(mask.size / sampled),
        }
        records.append(record)
        print(
            f"{case_id}: k-space {kspace_shape}, mask {mask.shape}, "
            f"ACS width {record['central_fully_sampled_width']}, "
            f"effective R {record['effective_acceleration']:.4f}"
        )

    manifest = {
        "raw_fullsample_root": str(full_root.resolve()),
        "acceleration_folder": args.acceleration_folder,
        "view": args.view,
        "nominal_mask_type": args.mask_type,
        "random_seed": None if requested is not None else args.seed,
        "selection_mode": "explicit" if requested is not None else "random",
        "force_acs_lines": args.force_acs_lines,
        "num_available_paired_cases": len(candidates),
        "num_selected_cases": len(records),
        "selected_subjects": [record["subject"] for record in records],
        "cases": records,
    }
    manifest_path = args.output_root / "cohort_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(f"Selected subjects: {', '.join(manifest['selected_subjects'])}")
    print(f"Manifest: {manifest_path}")
    print(f"Descriptors: {json_dir}")


if __name__ == "__main__":
    main()
