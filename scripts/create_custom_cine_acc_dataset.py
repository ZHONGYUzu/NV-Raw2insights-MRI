#!/usr/bin/env python3
"""Create a custom CINE dataset variant for a different fixed-mask acceleration."""

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

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


def choose_single_mask(mask_dir: Path, pattern: str) -> Path:
    matches = sorted(mask_dir.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one mask matching {mask_dir / pattern}, found {len(matches)}")
    return matches[0]


def make_kspace_link_or_copy(source: Path, target: Path, mode: str, overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not overwrite:
            return
        target.unlink()
    if mode == "symlink":
        target.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, target)
    else:
        raise ValueError(f"Unsupported k-space mode: {mode}")


def convert_mask(mask_txt: Path, output_path: Path, frequency_size: int, acs_lines: int, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return
    mask_phase_time = np.loadtxt(mask_txt, delimiter=",", dtype=np.float32)
    if mask_phase_time.ndim != 2:
        raise ValueError(f"Expected 2D phase-time mask, got {mask_phase_time.shape}")
    mask_time_phase = (mask_phase_time.T > 0).astype(np.float32)
    if not 0 <= acs_lines <= mask_time_phase.shape[1]:
        raise ValueError(f"--acs-lines must be between 0 and {mask_time_phase.shape[1]}")
    if acs_lines:
        start = (mask_time_phase.shape[1] - acs_lines) // 2
        mask_time_phase[:, start : start + acs_lines] = 1.0
    mask = np.repeat(mask_time_phase[:, :, None], frequency_size, axis=2)
    save_mat_atomic(output_path, {"mask": mask})


def read_source_smap(source_root: Path, case_id: str) -> str | None:
    source_json = source_root / "json_input" / f"{case_id}.json"
    if not source_json.exists():
        return None
    with source_json.open() as f:
        descriptor = json.load(f)
    return descriptor.get("sensitivity_maps") or descriptor.get("smap") or descriptor.get("dMap")


def write_json(json_path: Path, kspace_path: Path, mask_path: Path, smap_path: str | None = None) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = {"kspace": str(kspace_path), "mask": [str(mask_path)]}
    if smap_path:
        descriptor["sensitivity_maps"] = smap_path
    with json_path.open("w") as f:
        json.dump(descriptor, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("dataset/CustomCINEDataR1"))
    parser.add_argument("--output-root", type=Path, default=Path("dataset/CustomCINEDataR2"))
    parser.add_argument("--input-mask-dir", type=Path, required=True)
    parser.add_argument("--mask-glob", default="mask_VISTA_132x25_acc16_8.txt")
    parser.add_argument("--mask-type", default="ktRadial16")
    parser.add_argument("--case-glob", default="Sub000[1-5]_kspace_full.mat")
    parser.add_argument("--frequency-size", type=int, default=176)
    parser.add_argument("--acs-lines", type=int, default=20)
    parser.add_argument(
        "--no-force-acs",
        action="store_true",
        help="Do not force central ACS phase lines to 1; equivalent to --acs-lines 0.",
    )
    parser.add_argument("--kspace-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z_]+\d+", args.mask_type):
        raise ValueError("--mask-type must end in the intended acceleration, for example ktRadial16")
    mask_txt = choose_single_mask(args.input_mask_dir, args.mask_glob)
    acs_lines = 0 if args.no_force_acs else args.acs_lines

    source_kspace_dir = args.source_root / "MultiCoil" / "Cine" / "UnderSample_TaskR1"
    source_kspaces = sorted(source_kspace_dir.glob(args.case_glob))
    if not source_kspaces:
        raise FileNotFoundError(f"No source k-space files matched {source_kspace_dir / args.case_glob}")

    target_kspace_dir = args.output_root / "MultiCoil" / "Cine" / "UnderSample_TaskR1"
    target_mask_dir = args.output_root / "MultiCoil" / "Cine" / "Mask_TaskR1"
    target_json_dir = args.output_root / "json_input"

    for source_kspace in source_kspaces:
        case_id = source_kspace.name.replace("_kspace_full.mat", "")
        target_kspace = target_kspace_dir / source_kspace.name
        target_mask = target_mask_dir / f"{case_id}_mask_{args.mask_type}.mat"
        target_json = target_json_dir / f"{case_id}.json"
        smap_path = read_source_smap(args.source_root, case_id)

        make_kspace_link_or_copy(source_kspace, target_kspace, args.kspace_mode, args.overwrite)
        convert_mask(mask_txt, target_mask, args.frequency_size, acs_lines, args.overwrite)
        write_json(target_json, target_kspace, target_mask, smap_path)
        print(f"{case_id}: kspace={target_kspace} mask={target_mask} json={target_json}")


if __name__ == "__main__":
    main()
