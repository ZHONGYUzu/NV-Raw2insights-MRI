#!/usr/bin/env python3
"""Create one-case CINE inference probes for unsupported acceleration labels.

The generated masks are deterministic phase-time masks for checking whether the
reader/config/model path accepts acceleration labels such as 1, 2, or 3. They
are not intended to replace the project VISTA masks for final experiments.
"""

import argparse
import json
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


def logical_kspace_shape(values: dict, path: Path) -> tuple[int, int, int, int, int]:
    key = next((candidate for candidate in ("kus", "kspace_full", "kspace") if candidate in values), None)
    if key is None:
        raise ValueError(f"{path}: expected one of kus, kspace_full, or kspace")
    value = values[key]
    if np.issubdtype(value.dtype, np.complexfloating):
        shape = (1,) * (5 - value.ndim) + value.shape[::-1]
    elif value.dtype.fields and {"real", "imag"}.issubset(value.dtype.fields):
        shape = (1,) * (5 - value.ndim) + value.shape
    else:
        raise ValueError(f"{path}:{key} is not complex data (dtype={value.dtype})")
    if len(shape) != 5:
        raise ValueError(f"{path}:{key} expected logical 5D shape, got {shape}")
    return tuple(int(size) for size in shape)


def make_phase_time_mask(num_time: int, num_phase: int, acceleration: int, acs_lines: int) -> np.ndarray:
    if acceleration < 1:
        raise ValueError("Acceleration must be >= 1")
    if not 0 <= acs_lines <= num_phase:
        raise ValueError(f"ACS lines must be between 0 and {num_phase}")

    if acceleration == 1:
        phase = np.ones(num_phase, dtype=np.float32)
    else:
        target_lines = max(acs_lines, int(round(num_phase / acceleration)))
        phase = np.zeros(num_phase, dtype=np.float32)

        acs_start = (num_phase - acs_lines) // 2
        acs_stop = acs_start + acs_lines
        phase[acs_start:acs_stop] = 1.0

        outer = np.concatenate((np.arange(0, acs_start), np.arange(acs_stop, num_phase)))
        extra_count = max(target_lines - acs_lines, 0)
        if extra_count:
            outer_indices = np.linspace(0, len(outer) - 1, extra_count, dtype=int)
            phase[outer[outer_indices]] = 1.0

    return np.repeat(phase[None, :], num_time, axis=0)


def write_json(path: Path, kspace_path: Path, mask_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as descriptor_file:
        json.dump({"kspace": str(kspace_path), "mask": [str(mask_path)]}, descriptor_file, indent=2)
        descriptor_file.write("\n")


def write_config(base_config: Path, path: Path, acceleration: int) -> None:
    with base_config.open() as config_file:
        config = json.load(config_file)

    accelerations = [8.0, 16.0, 24.0]
    if float(acceleration) not in accelerations:
        accelerations.append(float(acceleration))

    config["fixed_mask_types"] = [f"mask_ktRadial{acceleration}"]
    config["accelerations"] = accelerations
    config["center_fractions"] = [0.0] * len(accelerations)
    config["num_workers"] = 0
    config["exp"] = f"acc_rate_probe_acc{acceleration}"

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as config_file:
        json.dump(config, config_file, indent=4)
        config_file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, default=Path("dataset/CustomCINEDataR1/json_input/Sub0001.json"))
    parser.add_argument("--base-config", type=Path, default=Path("configs/nv_raw2insights_mri_base.json"))
    parser.add_argument("--output-root", type=Path, default=Path("dataset/AccRateProbe"))
    parser.add_argument("--config-output-root", type=Path, default=Path("configs/acc_rate_probe"))
    parser.add_argument("--accelerations", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--acs-lines", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with args.source_json.open() as descriptor_file:
        source_descriptor = json.load(descriptor_file)
    kspace_path = Path(source_descriptor["kspace"])
    if not kspace_path.is_file():
        raise FileNotFoundError(f"Missing source k-space: {kspace_path}")

    num_time, _, _, num_phase, num_frequency = logical_kspace_shape(read_mat(kspace_path), kspace_path)
    case_id = args.source_json.stem

    for acceleration in args.accelerations:
        probe_root = args.output_root / f"acc{acceleration}"
        mask_path = (
            probe_root
            / "MultiCoil"
            / "Cine"
            / "Mask_TaskR1"
            / f"{case_id}_mask_ktRadial{acceleration}.mat"
        )
        json_path = probe_root / "json_input" / f"{case_id}.json"
        config_path = args.config_output_root / f"nv_raw2insights_mri_base_acc{acceleration}.json"

        if mask_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {mask_path}; use --overwrite")

        mask_time_phase = make_phase_time_mask(num_time, num_phase, acceleration, args.acs_lines)
        mask = np.repeat(mask_time_phase[:, :, None], num_frequency, axis=2)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(mask_path, {"mask": mask}, appendmat=False, do_compression=True)

        write_json(json_path, kspace_path, mask_path)
        write_config(args.base_config, config_path, acceleration)
        sampled = float(mask_time_phase.mean())
        effective_acc = 1.0 / sampled if sampled else float("inf")
        print(
            f"acc{acceleration}: json={json_path} config={config_path} "
            f"mask_shape={mask.shape} effective_acc={effective_acc:.3f}"
        )


if __name__ == "__main__":
    main()
