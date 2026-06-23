#!/usr/bin/env python3
"""Convert a VISTA phase-time text mask into a fixed CMRxRecon mask."""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import scipy.io


def infer_acceleration(path: Path) -> int:
    match = re.search(r"(?:^|_)acc(\d+)(?:_|\.)", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot infer acceleration from {path.name}; pass --mask-type")
    return int(match.group(1))


def save_mat_atomic(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".mat", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        scipy.io.savemat(tmp_path, values, appendmat=False, do_compression=True)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def update_case_json(json_path: Path, kspace_path: Path, mask_path: Path) -> None:
    if json_path.exists():
        with json_path.open() as f:
            descriptor = json.load(f)
    else:
        descriptor = {"kspace": str(kspace_path.resolve()), "mask": []}

    descriptor["kspace"] = descriptor.get("kspace") or str(kspace_path.resolve())
    masks = [str(Path(path).resolve()) for path in descriptor.get("mask", [])]
    resolved_mask = str(mask_path.resolve())
    if resolved_mask not in masks:
        masks.append(resolved_mask)
    descriptor["mask"] = masks

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(descriptor, f, indent=2)
        f.write("\n")


def convert_mask(
    input_path: Path,
    output_root: Path,
    case_id: str,
    frequency_size: int,
    acs_lines: int,
    mask_type: str | None,
    delimiter: str | None,
    overwrite: bool,
) -> tuple[Path, Path]:
    if frequency_size <= 0:
        raise ValueError("--frequency-size must be positive")
    if mask_type is None:
        mask_type = f"ktRadial{infer_acceleration(input_path)}"
    if not re.fullmatch(r"[A-Za-z_]+\d+", mask_type):
        raise ValueError("--mask-type must end in the intended numeric acceleration, for example ktRadial8")

    mask_phase_time = np.loadtxt(input_path, delimiter=delimiter, dtype=np.float32)
    if mask_phase_time.ndim != 2:
        raise ValueError(f"Expected a 2D (phase, time) mask, got {mask_phase_time.shape}")
    mask_time_phase = (mask_phase_time.T > 0).astype(np.float32)
    if not 0 <= acs_lines <= mask_time_phase.shape[1]:
        raise ValueError(f"--acs-lines must be between 0 and {mask_time_phase.shape[1]}")
    if acs_lines:
        start = (mask_time_phase.shape[1] - acs_lines) // 2
        mask_time_phase[:, start : start + acs_lines] = 1.0
    mask = np.repeat(mask_time_phase[:, :, None], frequency_size, axis=2)

    kspace_path = (
        output_root / "MultiCoil" / "Cine" / "UnderSample_TaskR1" / f"{case_id}_kspace_full.mat"
    )
    if not kspace_path.exists():
        raise FileNotFoundError(f"Convert k-space first; expected {kspace_path}")
    output_path = output_root / "MultiCoil" / "Cine" / "Mask_TaskR1" / f"{case_id}_mask_{mask_type}.mat"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}; use --overwrite")
    save_mat_atomic(output_path, {"mask": mask})

    json_path = output_root / "json_input" / f"{case_id}.json"
    update_case_json(json_path, kspace_path, output_path)
    return output_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mask-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/CustomCINEDataR1"),
        help="Converted dataset root (default: dataset/CustomCINEDataR1)",
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--frequency-size", type=int, required=True)
    parser.add_argument("--glob", default="*.txt", help="Must match exactly one mask")
    parser.add_argument("--acs-lines", type=int, default=20)
    parser.add_argument(
        "--no-force-acs",
        action="store_true",
        help="Do not force central ACS phase lines to 1; equivalent to --acs-lines 0.",
    )
    parser.add_argument("--mask-type", default=None, help="Model-known alias such as ktRadial8")
    parser.add_argument("--delimiter", default=",", help="Use 'whitespace' for whitespace-delimited files")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inputs = sorted(args.input_mask_dir.glob(args.glob))
    if len(inputs) != 1:
        parser.error(f"Expected exactly one mask matching {args.input_mask_dir / args.glob}, found {len(inputs)}")
    delimiter = None if args.delimiter == "whitespace" else args.delimiter
    acs_lines = 0 if args.no_force_acs else args.acs_lines
    output_path, json_path = convert_mask(
        inputs[0],
        args.output_root,
        args.case_id,
        args.frequency_size,
        acs_lines,
        args.mask_type,
        delimiter,
        args.overwrite,
    )
    print(f"{inputs[0].name} -> {output_path}")
    print(f"descriptor -> {json_path}")


if __name__ == "__main__":
    main()
