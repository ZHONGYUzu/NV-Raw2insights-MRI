#!/usr/bin/env python3
"""Plot fixed CINE masks referenced by inference JSON descriptors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def parse_label_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Expected LABEL=PATH, for example acc8=dataset/CINE1acr8/json_input"
        )
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Label cannot be empty")
    return label, Path(path)


def parse_label_text(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=TEXT, for example acc8=mask_ktRadialVISTA8")
    label, text = value.split("=", 1)
    label = label.strip()
    text = text.strip()
    if not label or not text:
        raise argparse.ArgumentTypeError("LABEL and TEXT cannot be empty")
    return label, text


def read_mat_values(path: Path) -> dict:
    try:
        import h5py

        with h5py.File(path, "r", swmr=True) as mat_file:
            return {key: mat_file[key][()] for key in mat_file}
    except (ImportError, OSError, ValueError):
        import scipy.io

        return {key: value for key, value in scipy.io.loadmat(path).items() if not key.startswith("__")}


def logical_mask_time_phase(values: dict, path: Path) -> np.ndarray:
    if "mask" not in values:
        raise ValueError(f"{path}: missing MAT key 'mask'")
    mask = np.asarray(values["mask"])
    if not np.all(np.isin(mask, (0, 1))):
        raise ValueError(f"{path}: mask contains values other than 0 and 1")
    if mask.ndim == 2:
        mask_time_phase = mask
    elif mask.ndim == 3:
        mask_time_phase = mask[:, :, 0]
        if not np.all(mask == mask[:, :, :1]):
            raise ValueError(f"{path}: expected mask to be repeated along frequency, got shape {mask.shape}")
    elif mask.ndim == 5:
        mask_time_phase = mask[:, 0, 0, :, 0]
    else:
        raise ValueError(f"{path}: expected 2D, 3D, or expanded 5D mask, got shape {mask.shape}")
    return (mask_time_phase > 0).astype(np.float32)


def collect_cases(inputs: Sequence[Tuple[str, Path]]) -> List[str]:
    case_sets = []
    for label, json_dir in inputs:
        if not json_dir.is_dir():
            raise FileNotFoundError(f"{label}: JSON directory does not exist: {json_dir}")
        cases = {path.stem for path in json_dir.glob("Sub*.json")}
        if not cases:
            raise FileNotFoundError(f"{label}: no Sub*.json descriptors found in {json_dir}")
        case_sets.append(cases)
    return sorted(set.intersection(*case_sets))


def load_case_mask(json_dir: Path, case_id: str, fixed_mask_type: Optional[str]) -> Tuple[Path, np.ndarray]:
    json_path = json_dir / f"{case_id}.json"
    if not json_path.is_file():
        raise FileNotFoundError(json_path)
    with json_path.open() as descriptor_file:
        descriptor = json.load(descriptor_file)
    mask_paths = [Path(path) for path in descriptor.get("mask", [])]
    if fixed_mask_type:
        mask_paths = [path for path in mask_paths if fixed_mask_type in str(path)]
    if not mask_paths:
        raise ValueError(f"{json_path}: no mask path matched {fixed_mask_type or '<any>'}")
    mask_path = mask_paths[0]
    if not mask_path.is_file():
        raise FileNotFoundError(f"{json_path}: mask file does not exist: {mask_path}")
    return mask_path, logical_mask_time_phase(read_mat_values(mask_path), mask_path)


def mask_stats(mask_time_phase: np.ndarray) -> Dict[str, float]:
    sampled = mask_time_phase.sum(axis=1)
    phase = mask_time_phase.shape[1]
    center_index = phase // 2
    return {
        "effective_acc": float(mask_time_phase.size / mask_time_phase.sum()),
        "min_frame_acc": float(phase / sampled.max()),
        "max_frame_acc": float(phase / sampled.min()),
        "min_lines": int(sampled.min()),
        "max_lines": int(sampled.max()),
        "center_frames": int(mask_time_phase[:, center_index].sum()),
    }


def save_case_plot(
    case_id: str,
    inputs: Sequence[Tuple[str, Path]],
    output_dir: Path,
    fixed_mask_types: Dict[str, str],
) -> Path:
    import matplotlib.pyplot as plt

    masks = []
    stats = []
    mask_paths = []
    for label, json_dir in inputs:
        mask_path, mask = load_case_mask(json_dir, case_id, fixed_mask_types.get(label))
        masks.append(mask)
        stats.append(mask_stats(mask))
        mask_paths.append(mask_path)

    num_cols = len(inputs)
    fig, axes = plt.subplots(2, num_cols, figsize=(4.5 * num_cols, 5.5), squeeze=False)
    for col, ((label, _), mask, stat, mask_path) in enumerate(zip(inputs, masks, stats, mask_paths)):
        title = (
            f"{label}\n"
            f"eff {stat['effective_acc']:.2f}x | frame {stat['min_frame_acc']:.2f}-{stat['max_frame_acc']:.2f}x\n"
            f"lines {stat['min_lines']:.0f}-{stat['max_lines']:.0f} | center {stat['center_frames']:.0f}/{mask.shape[0]}"
        )
        axes[0, col].imshow(mask, cmap="gray", origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
        axes[0, col].set_title(title, fontsize=10)
        axes[0, col].set_xlabel("phase")
        axes[0, col].set_ylabel("time")

        sampled = mask.sum(axis=1)
        axes[1, col].plot(np.arange(mask.shape[0]), sampled, marker="o", linewidth=1.5)
        axes[1, col].set_ylim(0, mask.shape[1])
        axes[1, col].set_xlabel("time")
        axes[1, col].set_ylabel("sampled PE lines")
        axes[1, col].grid(alpha=0.25)
        axes[1, col].text(
            0.5,
            -0.32,
            mask_path.name,
            transform=axes[1, col].transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )

    fig.suptitle(f"{case_id}: masks referenced by inference JSON", fontsize=13)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case_id}_inference_masks.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        action="append",
        type=parse_label_path,
        default=None,
        help="Inference JSON directory as LABEL=PATH. Repeat once per acceleration.",
    )
    parser.add_argument("--case", action="append", default=None, help="Case id to plot, for example Sub0007.")
    parser.add_argument("--max-cases", type=int, default=1, help="Limit plotted cases when --case is omitted.")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("output/CINE1_mask_plots"))
    parser.add_argument(
        "--fixed-mask-types",
        action="append",
        type=parse_label_text,
        default=None,
        help="Optional LABEL=substring filter matching inference --fixed-mask-types.",
    )
    args = parser.parse_args()

    inputs = args.input_json or [
        ("acc2", Path("dataset/CINE1acr2/json_input")),
        ("acc4", Path("dataset/CINE1acr4/json_input")),
        ("acc8", Path("dataset/CINE1acr8/json_input")),
        ("acc16", Path("dataset/CINE1acr16/json_input")),
    ]
    fixed_mask_types = dict(args.fixed_mask_types or [])
    if not fixed_mask_types:
        fixed_mask_types = {f"acc{acc}": f"mask_ktRadialVISTA{acc}" for acc in (2, 4, 8, 16)}

    cases = collect_cases(inputs)
    if args.case:
        requested = set(args.case)
        missing = sorted(requested - set(cases))
        if missing:
            parser.error(f"Requested cases missing from at least one JSON directory: {', '.join(missing)}")
        cases = [case for case in cases if case in requested]
    else:
        cases = cases[: args.max_cases]

    for case_id in cases:
        output_path = save_case_plot(case_id, inputs, args.output_dir, fixed_mask_types)
        print(output_path)


if __name__ == "__main__":
    main()
