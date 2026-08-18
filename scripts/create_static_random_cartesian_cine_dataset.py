#!/usr/bin/env python3
"""Create fixed static-random Cartesian masks for a CINE OOD experiment.

The same phase-encoding lines are used for every cardiac frame.  External
sensitivity maps are required, so a filled ACS region is optional rather than
needed for sensitivity-map estimation.  The model-conditioning alias remains
one of the pretrained model's known mask classes (Uniform by default).
"""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import scipy.io

from validate_cine_inference_data import logical_mask, read_mat


def save_mat_atomic(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".mat", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        scipy.io.savemat(tmp_path, values, appendmat=False, do_compression=True)
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


def source_mask_shape(descriptor: dict, descriptor_path: Path) -> tuple[int, int, int]:
    mask_paths = descriptor.get("mask", [])
    if not mask_paths:
        raise ValueError(f"{descriptor_path}: source descriptor has no mask for shape discovery")
    mask_path = Path(mask_paths[0])
    if not mask_path.is_file():
        raise FileNotFoundError(f"{descriptor_path}: missing source mask {mask_path}")
    mask = logical_mask(read_mat(mask_path), mask_path)
    return mask.shape[0], mask.shape[-2], mask.shape[-1]


def choose_phase_lines(
    num_phase: int,
    acceleration: int,
    seed: int,
    acs_lines: int,
) -> np.ndarray:
    sampled_lines = int(np.rint(num_phase / acceleration))
    if sampled_lines <= 0:
        raise ValueError(f"PE={num_phase}, R={acceleration} gives no sampled lines")
    if not 0 <= acs_lines <= sampled_lines:
        raise ValueError(
            f"--acs-lines={acs_lines} must be between 0 and the {sampled_lines}-line sampling budget"
        )

    selected = np.zeros(num_phase, dtype=bool)
    if acs_lines:
        acs_start = (num_phase - acs_lines) // 2
        selected[acs_start : acs_start + acs_lines] = True

    remaining = sampled_lines - int(selected.sum())
    if remaining:
        candidates = np.flatnonzero(~selected)
        rng = np.random.default_rng(np.random.SeedSequence([seed, num_phase, acceleration]))
        selected[rng.choice(candidates, size=remaining, replace=False)] = True
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("dataset/h5_converted"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/EXP009_CINE_static_random_cartesian_acc8_seed9009"),
    )
    parser.add_argument("--case-glob", default="Sub*.json")
    parser.add_argument("--acceleration", type=int, default=8)
    parser.add_argument("--seed", type=int, default=9009)
    parser.add_argument(
        "--acs-lines",
        type=int,
        default=0,
        help="Reserve this many central PE lines inside the fixed sampling budget (default: 0).",
    )
    parser.add_argument(
        "--conditioning-family",
        choices=("Uniform", "ktGaussian", "ktRadial"),
        default="Uniform",
        help="Pretrained model mask-conditioning alias; not the physical mask family.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.acceleration <= 0:
        parser.error("--acceleration must be positive")
    source_json_dir = args.source_root / "json_input"
    descriptors = sorted(source_json_dir.glob(args.case_glob))
    if not descriptors:
        parser.error(f"No descriptors matched {source_json_dir / args.case_glob}")

    alias = f"{args.conditioning_family}{args.acceleration}"
    if not re.fullmatch(r"(?:Uniform|ktGaussian|ktRadial)\d+", alias):
        raise ValueError(f"Invalid model-conditioning alias: {alias}")

    target_mask_dir = args.output_root / "MultiCoil" / "Cine" / "Mask_EXP009"
    target_json_dir = args.output_root / "json_input"
    selections: dict[tuple[int, int], np.ndarray] = {}
    manifest_cases = []

    for source_json in descriptors:
        with source_json.open() as descriptor_file:
            source_descriptor = json.load(descriptor_file)
        case_id = source_json.stem
        kspace_path = Path(source_descriptor["kspace"])
        smap_value = (
            source_descriptor.get("sensitivity_maps")
            or source_descriptor.get("smap")
            or source_descriptor.get("dMap")
        )
        if not kspace_path.is_file():
            raise FileNotFoundError(f"{source_json}: missing k-space {kspace_path}")
        if not smap_value:
            raise ValueError(f"{source_json}: EXP009 requires external sensitivity maps")
        smap_path = Path(smap_value[0] if isinstance(smap_value, list) else smap_value)
        if not smap_path.is_file():
            raise FileNotFoundError(f"{source_json}: missing sensitivity maps {smap_path}")

        num_time, num_phase, num_frequency = source_mask_shape(source_descriptor, source_json)
        selection_key = (num_phase, num_frequency)
        if selection_key not in selections:
            selections[selection_key] = choose_phase_lines(
                num_phase, args.acceleration, args.seed, args.acs_lines
            )
        selected = selections[selection_key]

        mask_time_phase = np.repeat(selected[None, :], num_time, axis=0)
        mask = np.repeat(mask_time_phase[:, :, None], num_frequency, axis=2).astype(np.float32)
        mask_path = target_mask_dir / f"{case_id}_mask_{alias}.mat"
        json_path = target_json_dir / f"{case_id}.json"
        if (mask_path.exists() or json_path.exists()) and not args.overwrite:
            raise FileExistsError(f"Output exists for {case_id}; use --overwrite to replace it")

        save_mat_atomic(mask_path, {"mask": mask})
        write_json_atomic(
            json_path,
            {
                "kspace": str(kspace_path),
                "mask": [str(mask_path.resolve())],
                "sensitivity_maps": str(smap_path),
            },
        )

        sampled_lines = int(selected.sum())
        center_sampled = bool(selected[num_phase // 2])
        manifest_cases.append(
            {
                "case_id": case_id,
                "time": num_time,
                "phase": num_phase,
                "frequency": num_frequency,
                "sampled_phase_lines_per_frame": sampled_lines,
                "effective_acceleration": num_phase / sampled_lines,
                "center_line_sampled": center_sampled,
                "sampled_phase_indices": np.flatnonzero(selected).tolist(),
            }
        )
        print(
            f"{case_id}: mask={tuple(mask.shape)} lines={sampled_lines}/{num_phase} "
            f"R_eff={num_phase / sampled_lines:.6f} center={center_sampled} alias={alias}"
        )

    write_json_atomic(
        args.output_root / "experiment_manifest.json",
        {
            "experiment": "EXP009",
            "physical_mask": "static random Cartesian PE lines; identical across cardiac frames",
            "conditioning_alias": alias,
            "nominal_acceleration": args.acceleration,
            "seed": args.seed,
            "forced_acs_lines": args.acs_lines,
            "sensitivity_maps": "external H5 dMap inherited from source descriptors",
            "source_root": str(args.source_root),
            "cases": manifest_cases,
        },
    )
    print(f"Created {len(manifest_cases)} EXP009 descriptor(s) under {target_json_dir}")


if __name__ == "__main__":
    main()
