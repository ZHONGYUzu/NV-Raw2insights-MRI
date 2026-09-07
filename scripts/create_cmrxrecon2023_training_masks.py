#!/usr/bin/env python3
"""Create reproducible CMRxRecon2023 runs with NV training-style masks."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import scipy.io

from mri_data.ktSampling import kt_gaussian_sampling, kt_radial_sampling, uniform_sampling


FAMILIES = ("Uniform", "ktGaussian", "ktRadial")


def cohort_seeds(cases: list[dict], base_seed: int, seed_manifest: Path | None = None) -> dict[str, int]:
    """Preserve historical full-cohort seeds; filter cases only after assignment.

    Supply a previous generated manifest to preserve the same masks even when
    reordering the source cohort. Without it, source order is part of the protocol.
    """
    ids = [str(case["case_id"]) for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Source manifest contains duplicate case IDs")
    if seed_manifest is not None:
        saved = json.loads(seed_manifest.read_text())
        if saved.get("mask_family") != "ktGaussian" or saved.get("base_seed") != base_seed:
            raise ValueError("Seed manifest must be ktGaussian with the requested base seed")
        rows = saved["cases"]
        mapping = {str(row["case_id"]): int(row["gaussian_seed"]) for row in rows}
        if len(mapping) != len(rows) or not set(ids) <= mapping.keys():
            raise ValueError("Seed manifest has duplicate IDs or does not cover the source cohort")
        return mapping
    rng = np.random.default_rng(base_seed)
    return {case_id: int(rng.integers(1, 1001)) for case_id in ids}


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


def save_mat_atomic(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".mat", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        scipy.io.savemat(temporary_path, values, appendmat=False, do_compression=True)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def inspect_kspace(path: Path, key: str) -> tuple[int, int, int, int, int]:
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
    return shape  # type: ignore[return-value]


def generate_mask(
    family: str,
    nx: int,
    ny: int,
    nt: int,
    acs_lines: int,
    acceleration: int,
    gaussian_seed: int,
) -> np.ndarray:
    if family == "Uniform":
        generated = uniform_sampling(nx, ny, nt, acs_lines, acceleration)
    elif family == "ktGaussian":
        generated = kt_gaussian_sampling(
            nx,
            ny,
            nt,
            acs_lines,
            acceleration,
            alpha=0.2,
            seed=gaussian_seed,
        )
    elif family == "ktRadial":
        generated = kt_radial_sampling(
            nx,
            ny,
            nt,
            acs_lines,
            acceleration * 0.6,
            angle4next=137.5,
            cropcorner=True,
        )
    else:  # pragma: no cover - argparse constrains this value
        raise ValueError(f"Unsupported mask family: {family}")

    expected = (nx, ny, nt)
    if generated.shape != expected:
        raise ValueError(f"Generator returned {generated.shape}, expected {expected}")
    if not np.isfinite(generated).all():
        raise ValueError("Generated mask contains NaN or Inf")
    mask = (generated > 0).astype(np.float32).transpose(2, 1, 0)  # (time, PE, FE)
    if not np.all(np.isin(mask, (0, 1))):
        raise ValueError("Generated mask is not binary")
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("runs/sdum-002/cohort_manifest.json"),
        help="Existing cohort manifest whose cases and full k-space paths are reused.",
    )
    parser.add_argument("--output-root", type=Path, required=True, help="New run root, e.g. runs/sdum-008.")
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--acceleration", type=int, choices=(8, 16, 24), default=8)
    parser.add_argument("--acs-lines", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260823, help="Base seed for reproducible Gaussian masks.")
    parser.add_argument("--seed-manifest", type=Path, help="Previous ktGaussian manifest fixing case-to-seed mapping across cohort reordering.")
    parser.add_argument("--kspace-key", default="kspace_full")
    parser.add_argument("--case-id", help="Optionally generate only one named case for a debug check.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.acs_lines < 0:
        parser.error("--acs-lines must be non-negative")
    with args.source_manifest.open() as manifest_file:
        source_manifest = json.load(manifest_file)
    cases = source_manifest.get("cases", [])
    if not cases:
        parser.error(f"No cases found in {args.source_manifest}")
    if args.seed_manifest and args.family != "ktGaussian":
        parser.error("--seed-manifest is only meaningful for ktGaussian")
    seeds = cohort_seeds(cases, args.seed, args.seed_manifest)
    if args.case_id:
        cases = [case for case in cases if str(case.get("case_id")) == args.case_id]
        if not cases:
            parser.error(f"Case {args.case_id!r} was not found in {args.source_manifest}")

    mask_alias = f"{args.family}{args.acceleration}"
    mask_dir = args.output_root / "masks"
    json_dir = args.output_root / "json_input"
    output_manifest_path = args.output_root / "cohort_manifest.json"
    if output_manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_manifest_path} already exists; use --overwrite to replace derived files")

    if json_dir.exists():
        extra_ids = {path.stem for path in json_dir.glob("*.json")} - {str(case["case_id"]) for case in cases}
        if extra_ids:
            raise ValueError(f"Output contains descriptors outside the requested cohort: {sorted(extra_ids)}; use a new output root")
    records: list[dict[str, object]] = []
    for case in cases:
        case_id = str(case["case_id"])
        kspace_path = Path(case["kspace"]).resolve()
        nt, nz, nc, ny, nx = inspect_kspace(kspace_path, args.kspace_key)
        if args.acs_lines > min(nx, ny):
            raise ValueError(f"{case_id}: ACS {args.acs_lines} exceeds spatial size {(ny, nx)}")
        gaussian_seed = seeds[case_id]
        mask = generate_mask(
            args.family,
            nx,
            ny,
            nt,
            args.acs_lines,
            args.acceleration,
            gaussian_seed,
        )
        expected_mask_shape = (nt, ny, nx)
        if mask.shape != expected_mask_shape:
            raise ValueError(f"{case_id}: mask {mask.shape} does not match {expected_mask_shape}")
        center = mask[:, ny // 2, nx // 2]
        if not np.all(center > 0):
            raise ValueError(f"{case_id}: center is not sampled in every frame")

        mask_path = mask_dir / f"{case_id}_mask_{mask_alias}.mat"
        descriptor_path = json_dir / f"{case_id}.json"
        if not args.overwrite and (mask_path.exists() or descriptor_path.exists()):
            raise FileExistsError(f"Derived files already exist for {case_id}; use --overwrite")
        save_mat_atomic(mask_path, {"mask": mask})
        descriptor = {"kspace": str(kspace_path), "mask": [str(mask_path.resolve())]}
        write_json_atomic(descriptor_path, descriptor)

        per_frame_r = np.array([ny * nx / frame.sum() for frame in mask], dtype=np.float64)
        record = {
            "case_id": case_id,
            "kspace": str(kspace_path),
            "kspace_shape": [nt, nz, nc, ny, nx],
            "mask": str(mask_path.resolve()),
            "descriptor": str(descriptor_path.resolve()),
            "mask_shape": list(mask.shape),
            "mask_family": args.family,
            "mask_alias": mask_alias,
            "nominal_acceleration": args.acceleration,
            "acs_lines": args.acs_lines,
            "gaussian_seed": gaussian_seed if args.family == "ktGaussian" else None,
            "effective_acceleration_mean": float(per_frame_r.mean()),
            "effective_acceleration_min": float(per_frame_r.min()),
            "effective_acceleration_max": float(per_frame_r.max()),
        }
        records.append(record)
        print(
            f"{case_id}: k-space {(nt, nz, nc, ny, nx)}, mask {mask.shape}, "
            f"effective R mean/min/max "
            f"{per_frame_r.mean():.4f}/{per_frame_r.min():.4f}/{per_frame_r.max():.4f}"
        )

    manifest = {
        "source_manifest": str(args.source_manifest.resolve()),
        "generator": "scripts/mri_data/ktSampling.py",
        "mask_family": args.family,
        "mask_alias": mask_alias,
        "nominal_acceleration": args.acceleration,
        "acs_lines": args.acs_lines,
        "base_seed": args.seed,
        "seed_manifest": str(args.seed_manifest.resolve()) if args.seed_manifest else None,
        "seed_assignment": "saved manifest" if args.seed_manifest else "full source cohort order before case filtering",
        "case_filter": args.case_id,
        "num_cases": len(records),
        "cases": records,
    }
    write_json_atomic(output_manifest_path, manifest)
    print(f"Created {len(records)} cases under {args.output_root}")
    print(f"Manifest: {output_manifest_path}")
    print(f"Descriptors: {json_dir}")


if __name__ == "__main__":
    main()
