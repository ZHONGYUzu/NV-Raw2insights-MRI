#!/usr/bin/env python3
"""Prepare the sdum-019 in-house CINE rotation-stability pilot.

The source H5 files remain read-only. Fully sampled coil images are obtained
with a centered inverse FFT, rotated in the PE/FE plane, and transformed back
to fully sampled k-space. The existing active-benchmark acc8 masks are reused
through symlinks, so inference still applies undersampling internally.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import scipy.io
from scipy.ndimage import rotate as scipy_rotate


DEFAULT_CONDITIONS = (
    ("rot_000", 0.0),
    ("rot_m010", -10.0),
    ("rot_p010", 10.0),
    ("rot_180", 180.0),
)


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    shifted = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(shifted, axes=(-2, -1), norm="ortho")
    return np.fft.fftshift(image, axes=(-2, -1))


def fft2c(image: np.ndarray) -> np.ndarray:
    shifted = np.fft.ifftshift(image, axes=(-2, -1))
    kspace = np.fft.fft2(shifted, axes=(-2, -1), norm="ortho")
    return np.fft.fftshift(kspace, axes=(-2, -1))


def rotate_real(array: np.ndarray, angle: float, order: int) -> np.ndarray:
    normalized = angle % 360.0
    if np.isclose(normalized, 0.0):
        return np.array(array, copy=True)
    if np.isclose(normalized, 180.0):
        return np.ascontiguousarray(np.rot90(array, k=2, axes=(-2, -1)))
    return scipy_rotate(
        array,
        angle=angle,
        axes=(-2, -1),
        reshape=False,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=order > 1,
    )


def rotate_complex(array: np.ndarray, angle: float, order: int) -> np.ndarray:
    real = rotate_real(array.real, angle, order)
    imag = rotate_real(array.imag, angle, order)
    return (real + 1j * imag).astype(np.complex64, copy=False)


def parse_condition(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("condition must be LABEL=ANGLE, e.g. rot_m010=-10")
    label, angle_text = value.split("=", 1)
    label = label.strip()
    if not label or "/" in label or "\\" in label:
        raise argparse.ArgumentTypeError(f"invalid condition label: {label!r}")
    try:
        angle = float(angle_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid rotation angle: {angle_text!r}") from error
    if not np.isfinite(angle):
        raise argparse.ArgumentTypeError("rotation angle must be finite")
    return label, angle


def save_mat_atomic(path: Path, values: dict[str, np.ndarray], compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".mat", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        scipy.io.savemat(
            tmp_path,
            values,
            appendmat=False,
            do_compression=compress,
            oned_as="row",
        )
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_json_atomic(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".json", mode="w", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(values, tmp, indent=2)
        tmp.write("\n")
    try:
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".txt", mode="w", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(value)
    try:
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    error_energy = 0.0
    reference_energy = 0.0
    for index in range(left.shape[0]):
        difference = left[index].astype(np.complex128) - right[index].astype(np.complex128)
        reference = right[index].astype(np.complex128)
        error_energy += float(np.vdot(difference.ravel(), difference.ravel()).real)
        reference_energy += float(np.vdot(reference.ravel(), reference.ravel()).real)
    return float(np.sqrt(error_energy / reference_energy)) if reference_energy > 0 else float("nan")


def energy_ratio(value: np.ndarray, reference: np.ndarray) -> float:
    value_energy = 0.0
    reference_energy = 0.0
    for index in range(value.shape[0]):
        current = value[index].astype(np.complex128)
        baseline = reference[index].astype(np.complex128)
        value_energy += float(np.vdot(current.ravel(), current.ravel()).real)
        reference_energy += float(np.vdot(baseline.ravel(), baseline.ravel()).real)
    return value_energy / reference_energy if reference_energy > 0 else float("nan")


def ensure_mask_link(target: Path, source: Path, resume: bool) -> None:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if resume and target.is_symlink() and target.resolve() == source:
            return
        raise FileExistsError(f"Refusing to replace existing mask path: {target}")
    target.symlink_to(source)


def file_identity(path: Path, checksum: bool = False) -> dict:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    identity = {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if checksum:
        identity["sha256"] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return identity


def preparation_config(args, conditions: list[tuple[str, float]]) -> dict:
    """Fingerprint large raw inputs by stat; hash only small descriptors/masks."""
    sources = []
    for case_id in args.cases:
        descriptor_path = args.source_json_root / f"{case_id}.json"
        descriptor = json.loads(descriptor_path.read_text())
        masks = descriptor.get("mask", [])
        if len(masks) != 1:
            raise ValueError(f"{descriptor_path}: expected exactly one source mask")
        sources.append({"case_id": case_id,
                        "h5": file_identity(args.raw_h5_root / f"{case_id}.h5"),
                        "descriptor": file_identity(descriptor_path, checksum=True),
                        "mask": file_identity(Path(masks[0]), checksum=True)})
    return {"schema_version": 1, "experiment_id": args.experiment_id,
            "cohort": args.cases, "conditions": [list(item) for item in conditions],
            "interpolation_order": args.interpolation_order,
            "compress_kspace": args.compress_kspace, "sources": sources}


def check_preparation_config(run_root: Path, config: dict, resume: bool) -> None:
    path = run_root / "preparation_config.json"
    if path.exists():
        if not resume or json.loads(path.read_text()) != config:
            raise ValueError("Existing preparation parameters/sources differ, or --resume was not requested; use a new run root")
    elif run_root.exists() and any((run_root / name).exists() for name in (
        "conditions", "manifest.json", "cases.txt", "config.yaml"
    )):
        raise ValueError("Legacy/partial run has no verifiable preparation config; preserve it and use a new run root")


def load_prepared_record(paths: dict[str, Path], angle: float, order: int, source_mask: Path) -> dict:
    record = json.loads(paths["record"].read_text())
    if record["angle_degrees"] != angle or record["interpolation_order"] != order:
        raise ValueError("Prepared record does not match requested angle/interpolation")
    if not paths["mask"].is_symlink() or paths["mask"].resolve() != source_mask.resolve():
        raise ValueError("Prepared mask link no longer matches the original source")
    for key in ("kspace", "descriptor", "ground_truth", "valid_roi"):
        if file_identity(paths[key]) != record["output_identities"][key]:
            raise ValueError(f"Prepared output changed: {paths[key]}")
    return record


def output_paths(run_root: Path, label: str, case_id: str) -> dict[str, Path]:
    condition_root = run_root / "conditions" / label
    return {
        "root": condition_root,
        "kspace": condition_root
        / "MultiCoil"
        / "Cine"
        / "UnderSample_TaskR1"
        / f"{case_id}_kspace_full.mat",
        "mask": condition_root
        / "MultiCoil"
        / "Cine"
        / "Mask_TaskR1"
        / f"{case_id}_mask_ktRadial8.mat",
        "descriptor": condition_root / "json_input" / f"{case_id}.json",
        "ground_truth": condition_root / "ground_truth" / f"{case_id}.mat",
        "valid_roi": condition_root / "valid_roi" / f"{case_id}.mat",
        "record": condition_root / "preparation_records" / f"{case_id}.json",
    }


def prepare_case(
    case_id: str,
    raw_h5_root: Path,
    source_descriptor_root: Path,
    run_root: Path,
    conditions: list[tuple[str, float]],
    interpolation_order: int,
    compress_mat: bool,
    resume: bool,
) -> list[dict]:
    h5_path = raw_h5_root / f"{case_id}.h5"
    source_json = source_descriptor_root / f"{case_id}.json"
    if not h5_path.is_file() or not source_json.is_file():
        raise FileNotFoundError(f"{case_id}: missing {h5_path} or {source_json}")
    with source_json.open() as source_file:
        source_descriptor = json.load(source_file)
    source_masks = source_descriptor.get("mask", [])
    if len(source_masks) != 1:
        raise ValueError(f"{source_json}: expected exactly one fixed acc8 mask")
    source_mask = Path(source_masks[0])

    print(f"{case_id}: loading read-only source {h5_path}", flush=True)
    with h5py.File(h5_path, "r") as h5_file:
        if "kSpace" not in h5_file or "dImgC" not in h5_file:
            raise KeyError(f"{h5_path}: expected kSpace and dImgC; keys={list(h5_file.keys())}")
        source_kspace = np.asarray(h5_file["kSpace"], dtype=np.complex64)
        dimgc = np.asarray(h5_file["dImgC"], dtype=np.complex64)
    if source_kspace.ndim != 5:
        raise ValueError(f"{h5_path}: kSpace must be 5D, got {source_kspace.shape}")
    if dimgc.ndim != 5 or dimgc.shape[1] != 1:
        raise ValueError(f"{h5_path}: dImgC must be (slice,1,time,PE,FE), got {dimgc.shape}")
    if source_kspace.shape[0] != dimgc.shape[0] or source_kspace.shape[2:] != dimgc.shape[2:]:
        raise ValueError(f"{h5_path}: kSpace {source_kspace.shape} and dImgC {dimgc.shape} disagree")

    gt_complex = dimgc[:, 0]
    normalization_scale = float(np.max(np.abs(gt_complex)))
    if not np.isfinite(normalization_scale) or normalization_scale <= 0:
        raise ValueError(f"{h5_path}: invalid dImgC max magnitude {normalization_scale}")
    gt_complex = (gt_complex / normalization_scale).astype(np.complex64, copy=False)

    print(f"{case_id}: centered IFFT of kSpace {source_kspace.shape}", flush=True)
    coil_images = ifft2c(source_kspace).astype(np.complex64, copy=False)
    records = []
    for label, angle in conditions:
        paths = output_paths(run_root, label, case_id)
        expected = (paths["kspace"], paths["descriptor"], paths["ground_truth"], paths["valid_roi"])
        if resume and all(path.is_file() for path in expected) and paths["mask"].is_file() and paths["record"].is_file():
            record = load_prepared_record(paths, angle, interpolation_order, source_mask)
            print(f"{case_id} {label}: complete outputs already exist; skipping", flush=True)
            records.append(record)
            continue
        collisions = [path for path in expected if path.exists()]
        if collisions or paths["record"].exists() or paths["mask"].exists() or paths["mask"].is_symlink():
            raise FileExistsError(f"{case_id} {label}: partial/existing outputs: {collisions}")

        print(f"{case_id} {label}: rotating complex coil images by {angle:g} degrees", flush=True)
        rotated_images = rotate_complex(coil_images, angle, interpolation_order)
        rotated_kspace = fft2c(rotated_images).astype(np.complex64, copy=False)
        rotated_gt = rotate_complex(gt_complex, angle, interpolation_order)
        roi_pe_fe = rotate_real(
            np.ones(source_kspace.shape[-2:], dtype=np.float32), angle, order=0
        ) > 0.5

        logical_kspace = np.transpose(rotated_kspace, (2, 0, 1, 3, 4))
        save_mat_atomic(
            paths["kspace"], {"kspace_full": logical_kspace.transpose()}, compress_mat
        )
        gt_mat = np.abs(rotated_gt).astype(np.float32).transpose(3, 2, 0, 1)
        save_mat_atomic(paths["ground_truth"], {"gt": gt_mat}, compress=True)
        save_mat_atomic(
            paths["valid_roi"], {"valid_roi": roi_pe_fe.T.astype(np.uint8)}, compress=True
        )
        ensure_mask_link(paths["mask"], source_mask, resume=False)
        write_json_atomic(
            paths["descriptor"],
            {
                "kspace": str(paths["kspace"].resolve()),
                "mask": [str(paths["mask"].absolute())],
            },
        )

        identity_error = relative_l2(rotated_kspace, source_kspace) if np.isclose(angle % 360, 0) else None
        record = {
            "case_id": case_id,
            "condition": label,
            "angle_degrees": angle,
            "status": "created",
            "source_h5": str(h5_path),
            "source_kspace_shape": list(source_kspace.shape),
            "logical_kspace_shape": list(logical_kspace.shape),
            "normalization": "GT only: dImgC original-case max magnitude before rotation; coil images retain source scale",
            "normalization_scale": normalization_scale,
            "interpolation_order": interpolation_order,
            "boundary_mode": "constant_zero",
            "reshape": False,
            "valid_roi_fraction": float(np.mean(roi_pe_fe)),
            "kspace_energy_ratio_to_unrotated": energy_ratio(rotated_kspace, source_kspace),
            "identity_kspace_relative_l2": identity_error,
            "kspace": str(paths["kspace"].resolve()),
            "kspace_size_bytes": paths["kspace"].stat().st_size,
            "mask": str(paths["mask"].absolute()),
            "source_mask": str(source_mask.resolve()),
            "descriptor": str(paths["descriptor"].resolve()),
            "ground_truth": str(paths["ground_truth"].resolve()),
            "valid_roi": str(paths["valid_roi"].resolve()),
        }
        record["output_identities"] = {
            key: file_identity(paths[key]) for key in ("kspace", "descriptor", "ground_truth", "valid_roi")
        }
        write_json_atomic(paths["record"], record)
        records.append(record)
        print(
            f"{case_id} {label}: logical={logical_kspace.shape} "
            f"ROI={record['valid_roi_fraction']:.4f} energy={record['kspace_energy_ratio_to_unrotated']:.6f}",
            flush=True,
        )
        del rotated_images, rotated_kspace, rotated_gt, logical_kspace, gt_mat, roi_pe_fe
        gc.collect()

    del source_kspace, dimgc, gt_complex, coil_images
    gc.collect()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/sdum-019"))
    parser.add_argument("--experiment-id", default="sdum-019")
    parser.add_argument(
        "--raw-h5-root",
        type=Path,
        default=Path("/mnt/qdata/rawdata/CINE/2D_h5_compressed"),
    )
    parser.add_argument(
        "--source-json-root", type=Path, default=Path("dataset/CINE_test_acc8/json_input")
    )
    parser.add_argument("--cases", nargs="+", default=["Sub0014", "Sub0047"])
    parser.add_argument(
        "--condition",
        action="append",
        type=parse_condition,
        dest="conditions",
        help="Repeat LABEL=ANGLE; defaults to rot_000, rot_m010, rot_p010, rot_180.",
    )
    parser.add_argument("--interpolation-order", type=int, choices=(0, 1, 3), default=1)
    parser.add_argument("--compress-kspace", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    conditions = args.conditions or list(DEFAULT_CONDITIONS)
    labels = [label for label, _ in conditions]
    if len(labels) != len(set(labels)):
        parser.error("condition labels must be unique")
    if not args.experiment_id or args.run_root.name != args.experiment_id:
        parser.error("--run-root basename must equal --experiment-id")
    if len(args.cases) != len(set(args.cases)):
        parser.error("--cases must be unique")

    if any(value in (".", "..") or "/" in value or "\\" in value for value in [*labels, *args.cases]):
        parser.error("Case IDs and condition labels must be plain directory/file names")
    config = preparation_config(args, conditions)
    check_preparation_config(args.run_root, config, args.resume)
    # Validate all reusable records before rewriting even the cohort text file.
    for source in config["sources"]:
        for label, angle in conditions:
            paths = output_paths(args.run_root, label, source["case_id"])
            outputs = [paths[key] for key in ("kspace", "mask", "descriptor", "ground_truth", "valid_roi", "record")]
            if any(path.exists() or path.is_symlink() for path in outputs):
                if not args.resume or not all(path.is_file() for path in outputs):
                    raise FileExistsError(f"Partial/existing outputs for {source['case_id']} {label}; preserve and inspect them")
                load_prepared_record(paths, angle, args.interpolation_order, Path(source["mask"]["path"]))

    args.run_root.mkdir(parents=True, exist_ok=True)
    if not (args.run_root / "preparation_config.json").exists():
        write_json_atomic(args.run_root / "preparation_config.json", config)
    write_text_atomic(args.run_root / "cases.txt", "\n".join(args.cases) + "\n")
    records = []
    for case_id in args.cases:
        records.extend(
            prepare_case(
                case_id=case_id,
                raw_h5_root=args.raw_h5_root,
                source_descriptor_root=args.source_json_root,
                run_root=args.run_root,
                conditions=conditions,
                interpolation_order=args.interpolation_order,
                compress_mat=args.compress_kspace,
                resume=args.resume,
            )
        )

    previous_manifest_path = args.run_root / "manifest.json"
    previous_manifest = json.loads(previous_manifest_path.read_text()) if previous_manifest_path.exists() else {}
    manifest = {
        "experiment_id": args.experiment_id,
        "created_utc": previous_manifest.get("created_utc", datetime.now(timezone.utc).isoformat()),
        "purpose": "two-case acc8 in-plane rotation stability pilot",
        "model": "NV-Raw2Insights-MRI base",
        "cohort": args.cases,
        "conditions": [{"label": label, "angle_degrees": angle} for label, angle in conditions],
        "source_h5_root": str(args.raw_h5_root),
        "source_json_root": str(args.source_json_root.resolve()),
        "mask": "active CINE_test_acc8 fixed VISTA seed 8 mask, reused per case",
        "sensitivity_maps": "internal estimation from the masked rotated k-space ACS region",
        "rotation_domain": "complex coil image after centered IFFT of full k-space",
        "interpolation_order": args.interpolation_order,
        "boundary_mode": "constant_zero",
        "reshape": False,
        "kspace_mat_compression": args.compress_kspace,
        "records": records,
    }
    write_json_atomic(args.run_root / "manifest.json", manifest)
    write_text_atomic(
        args.run_root / "config.yaml",
        "\n".join(
            [
                f"experiment_id: {args.experiment_id}",
                "dataset: in-house CINE active benchmark",
                "stage: rotation_stability_pilot",
                "model:",
                "  variant: base",
                "  checkpoint: nvidia/NV-Raw2Insights-MRI",
                "sampling:",
                "  mask: VISTA",
                "  acceleration: 8",
                "  seed: 8",
                "  forced_acs_lines: 20",
                "rotation:",
                "  domain: complex_multicoil_image",
                f"  interpolation_order: {args.interpolation_order}",
                "  reshape: false",
                "  boundary_mode: constant_zero",
                "conditions:",
                *[f"  {label}: {angle:g}" for label, angle in conditions],
                "cases:",
                *[f"  - {case_id}" for case_id in args.cases],
                "",
            ]
        ),
    )
    write_text_atomic(
        args.run_root / "README.md",
        "\n".join(
            [
                f"# {args.experiment_id}: CINE rotation stability pilot",
                "",
                "Two active-benchmark cases are reconstructed at acc8 after a global",
                "in-plane rotation is applied to the fully sampled complex coil images.",
                "The raw H5 files and source VISTA masks are read-only.",
                "",
                "Conditions: " + ", ".join(f"{label}={angle:g} deg" for label, angle in conditions),
                "Cases: " + ", ".join(args.cases),
                "",
                "Each condition contains its derived k-space, mask symlink, JSON input,",
                "rotated dImgC ground truth, valid ROI, and inference output. Aggregate",
                "results are written under metrics/ and figures/.",
                "",
            ]
        ),
    )
    print(f"Prepared {len(records)} case-condition records in {args.run_root}", flush=True)


if __name__ == "__main__":
    main()
