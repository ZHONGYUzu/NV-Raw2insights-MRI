#!/usr/bin/env python3
"""Prepare the sdum-007 static-random Cartesian CINE OOD run."""

import argparse
import json
import os
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


def save_json_atomic(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".json", mode="w", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(values, tmp, indent=2)
        tmp.write("\n")
    try:
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_cases(path: Path) -> list[str]:
    cases = [line.strip() for line in path.read_text().splitlines()]
    cases = [case for case in cases if case and not case.startswith("#")]
    if not cases or len(cases) != len(set(cases)):
        raise ValueError(f"{path}: cases must be nonempty and unique")
    return cases


def source_geometry(descriptor: dict, descriptor_path: Path) -> tuple[int, int, int, str]:
    source_masks = descriptor.get("mask", [])
    if not source_masks:
        raise ValueError(f"{descriptor_path}: source descriptor has no mask")
    source_mask = Path(source_masks[0])
    if not source_mask.is_file():
        raise FileNotFoundError(source_mask)
    mask = logical_mask(read_mat(source_mask), source_mask)
    return mask.shape[0], mask.shape[-2], mask.shape[-1], str(source_mask)


def select_lines(num_phase: int, acceleration: int, seed: int, forced_acs_lines: int) -> np.ndarray:
    line_budget = int(np.rint(num_phase / acceleration))
    if not 0 <= forced_acs_lines <= line_budget:
        raise ValueError(f"forced ACS {forced_acs_lines} does not fit {line_budget}-line budget")
    selected = np.zeros(num_phase, dtype=bool)
    if forced_acs_lines:
        start = (num_phase - forced_acs_lines) // 2
        selected[start : start + forced_acs_lines] = True
    remaining = line_budget - int(selected.sum())
    if remaining:
        candidates = np.flatnonzero(~selected)
        rng = np.random.default_rng(np.random.SeedSequence([seed, num_phase, acceleration]))
        selected[rng.choice(candidates, size=remaining, replace=False)] = True
    return selected


def centered_width(selected: np.ndarray) -> int:
    center = len(selected) // 2
    if not selected[center]:
        return 0
    left = right = center
    while left > 0 and selected[left - 1]:
        left -= 1
    while right + 1 < len(selected) and selected[right + 1]:
        right += 1
    return right - left + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("dataset/h5_converted"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/sdum-007"))
    parser.add_argument("--cases", type=Path, default=Path("runs/sdum-007/cases.txt"))
    parser.add_argument("--acceleration", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--forced-acs-lines", type=int, default=0)
    parser.add_argument("--conditioning-alias", default="Uniform8")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.acceleration <= 0:
        parser.error("--acceleration must be positive")
    if args.conditioning_alias != f"Uniform{args.acceleration}":
        parser.error("sdum-007 expects the matching Uniform acceleration alias")

    cases = load_cases(args.cases)
    source_json_dir = args.source_root / "json_input"
    mask_dir = args.run_root / "MultiCoil" / "CINE" / "Mask_TaskR1"
    json_dir = args.run_root / "json_input"
    for directory in (mask_dir, json_dir, args.run_root / "logs", args.run_root / "debug_output", args.run_root / "output"):
        directory.mkdir(parents=True, exist_ok=True)

    selections: dict[int, np.ndarray] = {}
    manifest_cases = []
    for case_id in cases:
        source_json = source_json_dir / f"{case_id}.json"
        if not source_json.is_file():
            raise FileNotFoundError(source_json)
        with source_json.open() as source_file:
            source = json.load(source_file)
        kspace = Path(source["kspace"])
        smap_value = source.get("sensitivity_maps") or source.get("smap") or source.get("dMap")
        if not smap_value:
            raise ValueError(f"{source_json}: external dMap is required")
        smap = Path(smap_value[0] if isinstance(smap_value, list) else smap_value)
        if not kspace.is_file() or not smap.is_file():
            raise FileNotFoundError(f"{case_id}: missing k-space or dMap")

        num_time, num_phase, num_frequency, source_mask = source_geometry(source, source_json)
        if num_phase not in selections:
            selections[num_phase] = select_lines(num_phase, args.acceleration, args.seed, args.forced_acs_lines)
        selected = selections[num_phase]
        mask_tp = np.repeat(selected[None, :], num_time, axis=0)
        mask = np.repeat(mask_tp[:, :, None], num_frequency, axis=2).astype(np.float32)
        target_mask = mask_dir / f"{case_id}_mask_{args.conditioning_alias}.mat"
        target_json = json_dir / f"{case_id}.json"
        if (target_mask.exists() or target_json.exists()) and not args.overwrite:
            raise FileExistsError(f"Outputs exist for {case_id}; pass --overwrite")
        save_mat_atomic(target_mask, {"mask": mask})
        save_json_atomic(target_json, {
            "kspace": str(kspace),
            "mask": [str(target_mask.resolve())],
            "sensitivity_maps": str(smap),
        })
        sampled = int(selected.sum())
        record = {
            "case_id": case_id,
            "source_descriptor": str(source_json.resolve()),
            "kspace": str(kspace),
            "sensitivity_maps": str(smap),
            "source_mask": source_mask,
            "derived_mask": str(target_mask.resolve()),
            "descriptor": str(target_json.resolve()),
            "time": num_time,
            "phase": num_phase,
            "frequency": num_frequency,
            "sampled_phase_lines_per_frame": sampled,
            "effective_acceleration": num_phase / sampled,
            "center_line_sampled": bool(selected[num_phase // 2]),
            "centered_sampled_width": centered_width(selected),
            "sampled_phase_indices": np.flatnonzero(selected).tolist(),
        }
        manifest_cases.append(record)
        print(f"{case_id}: T/PE/FE={num_time}/{num_phase}/{num_frequency} lines={sampled} R_eff={num_phase / sampled:.6f} center_width={record['centered_sampled_width']}")

    manifest = {
        "experiment_id": "sdum-007",
        "model": "SDUM / NV-Raw2Insights Base",
        "dataset": "in-house CINE active 10-case benchmark",
        "source_root": str(args.source_root.resolve()),
        "physical_mask": "static random Cartesian PE lines, identical across all cardiac frames",
        "runtime_mask_alias": args.conditioning_alias,
        "nominal_acceleration": args.acceleration,
        "random_seed": args.seed,
        "forced_acs_lines": args.forced_acs_lines,
        "sensitivity_maps": "external H5 dMap inherited from source descriptors",
        "number_of_cases": len(manifest_cases),
        "cases": manifest_cases,
    }
    save_json_atomic(args.run_root / "cohort_manifest.json", manifest)
    print(f"Prepared {len(manifest_cases)} cases in {args.run_root}")


if __name__ == "__main__":
    main()
