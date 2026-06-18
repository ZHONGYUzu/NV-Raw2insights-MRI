#!/usr/bin/env python3
"""Report effective acceleration rates for CINE masks."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    import scipy.io
except ImportError:
    scipy = None


def read_mat(path: Path) -> dict:
    if h5py is not None:
        try:
            with h5py.File(path, "r", swmr=True) as mat_file:
                return {key: mat_file[key][()] for key in mat_file}
        except (OSError, ValueError):
            pass
    if scipy is None:
        raise ImportError(f"Reading non-HDF5 MAT files requires scipy: {path}")
    return {key: value for key, value in scipy.io.loadmat(path).items() if not key.startswith("__")}


def load_mat_mask(path: Path) -> np.ndarray:
    values = read_mat(path)
    if "mask" not in values:
        raise ValueError(f"{path}: missing mask key")
    mask = np.asarray(values["mask"])
    if mask.ndim == 2:
        return (mask > 0)
    if mask.ndim == 3:
        return (mask > 0)
    if mask.ndim == 5:
        return (mask > 0)
    raise ValueError(f"{path}: expected 2D, 3D, or expanded 5D mask, got {mask.shape}")


def load_txt_mask(path: Path, delimiter: str | None, acs_lines: int, frequency_size: int | None) -> np.ndarray:
    mask_phase_time = np.loadtxt(path, delimiter=delimiter, dtype=np.float32)
    if mask_phase_time.ndim != 2:
        raise ValueError(f"{path}: expected 2D VISTA mask shaped (phase, time), got {mask_phase_time.shape}")

    mask_time_phase = (mask_phase_time.T > 0)
    if not 0 <= acs_lines <= mask_time_phase.shape[1]:
        raise ValueError(f"--acs-lines must be between 0 and {mask_time_phase.shape[1]}")
    if acs_lines:
        start = (mask_time_phase.shape[1] - acs_lines) // 2
        mask_time_phase[:, start : start + acs_lines] = True
    if frequency_size is None:
        return mask_time_phase
    if frequency_size <= 0:
        raise ValueError("--frequency-size must be positive")
    return np.repeat(mask_time_phase[:, :, None], frequency_size, axis=2)


def collapse_to_time_phase(mask: np.ndarray) -> np.ndarray:
    """Return a logical (time, phase) mask when frequency/coil axes are repeated."""
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3:
        return mask.any(axis=-1)
    if mask.ndim == 5:
        return mask.any(axis=(1, 2, 4))
    raise ValueError(f"Cannot collapse mask with shape {mask.shape}")


def summarize(path: Path, mask: np.ndarray) -> dict:
    mask_time_phase = collapse_to_time_phase(mask)
    sampled_per_frame = mask_time_phase.sum(axis=1)
    per_frame_acc = mask_time_phase.shape[1] / sampled_per_frame
    total_samples = mask_time_phase.size
    sampled_samples = int(mask_time_phase.sum())
    return {
        "path": str(path),
        "shape": "x".join(str(size) for size in mask.shape),
        "time": mask_time_phase.shape[0],
        "phase": mask_time_phase.shape[1],
        "total_phase_time_samples": total_samples,
        "sampled_phase_time_samples": sampled_samples,
        "effective_acceleration": total_samples / sampled_samples,
        "mean_frame_acceleration": float(per_frame_acc.mean()),
        "min_frame_acceleration": float(per_frame_acc.min()),
        "max_frame_acceleration": float(per_frame_acc.max()),
        "min_sampled_lines_per_frame": int(sampled_per_frame.min()),
        "max_sampled_lines_per_frame": int(sampled_per_frame.max()),
    }


def iter_input_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    descriptors = sorted(input_path.glob("*.json"))
    if descriptors:
        mask_paths = []
        for descriptor_path in descriptors:
            with descriptor_path.open() as descriptor_file:
                descriptor = json.load(descriptor_file)
            mask_paths.extend(Path(path) for path in descriptor.get("mask", []))
        return mask_paths
    return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in {".mat", ".txt"})


def print_table(rows: list[dict]) -> None:
    columns = [
        "path",
        "shape",
        "sampled_phase_time_samples",
        "effective_acceleration",
        "mean_frame_acceleration",
        "min_frame_acceleration",
        "max_frame_acceleration",
    ]
    print("\t".join(columns))
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        print("\t".join(values))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        default=Path("dataset/CustomCINEDataR1/json_input"),
        help="Mask .mat/.txt file, json_input directory, or directory containing masks",
    )
    parser.add_argument("--acs-lines", type=int, default=20, help="ACS lines to force for VISTA .txt inputs")
    parser.add_argument(
        "--frequency-size",
        type=int,
        default=None,
        help="Optionally expand VISTA .txt masks to (time, phase, frequency)",
    )
    parser.add_argument("--delimiter", default=",", help="Use 'whitespace' for whitespace-delimited .txt masks")
    parser.add_argument("-o", "--output-csv", type=Path, default=None)
    args = parser.parse_args()

    delimiter = None if args.delimiter == "whitespace" else args.delimiter
    rows = []
    for input_path in iter_input_paths(args.input_path):
        suffix = input_path.suffix.lower()
        if suffix == ".mat":
            mask = load_mat_mask(input_path)
        elif suffix == ".txt":
            mask = load_txt_mask(input_path, delimiter, args.acs_lines, args.frequency_size)
        else:
            continue
        rows.append(summarize(input_path, mask))

    if not rows:
        parser.error(f"No .mat or .txt masks found from {args.input_path}")
    print_table(rows)
    if args.output_csv:
        write_csv(args.output_csv, rows)
        print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
