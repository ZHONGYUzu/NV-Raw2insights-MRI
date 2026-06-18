#!/usr/bin/env python3
"""Plot input/reconstruction/GT/error grids across CINE acceleration settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from inspect_cine_recon_quality import choose_indices, load_gt, read_mat_key


def parse_acc_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Acceleration input must be LABEL=PATH, for example acc8=output/CustomCINEOutputR1/val_img4ranking"
        )
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Acceleration label cannot be empty")
    return label, Path(path)


def parse_label_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Input JSON argument must be LABEL=PATH, for example acc8=dataset/CustomCINEDataR1/json_input"
        )
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Input JSON label cannot be empty")
    return label, Path(path)


def infer_json_dir(pred_dir: Path) -> Optional[Path]:
    """Infer the custom CINE JSON folder from the established output layout."""
    for parent in [pred_dir, *pred_dir.parents]:
        name = parent.name
        if name.startswith("CustomCINEOutput"):
            suffix = name.removeprefix("CustomCINEOutput")
            candidate = parent.parent.parent / "dataset" / f"CustomCINEData{suffix}" / "json_input"
            if candidate.is_dir():
                return candidate
            cwd_candidate = Path("dataset") / f"CustomCINEData{suffix}" / "json_input"
            if cwd_candidate.is_dir():
                return cwd_candidate
    return None


def resolve_json_inputs(
    acc_inputs: Sequence[Tuple[str, Path]],
    explicit_inputs: Optional[Sequence[Tuple[str, Path]]],
) -> Dict[str, Path]:
    explicit = dict(explicit_inputs or [])
    acc_labels = [label for label, _ in acc_inputs]
    extra = sorted(set(explicit) - set(acc_labels))
    if extra:
        raise ValueError(f"--input-json labels without matching --acc: {', '.join(extra)}")

    resolved: Dict[str, Path] = {}
    missing = []
    for label, pred_dir in acc_inputs:
        json_dir = explicit.get(label) or infer_json_dir(pred_dir)
        if json_dir is None:
            missing.append(label)
            continue
        if not json_dir.is_dir():
            raise FileNotFoundError(f"{label}: input JSON directory does not exist: {json_dir}")
        resolved[label] = json_dir

    if missing:
        raise ValueError(
            "Missing input JSON directories for labels "
            + ", ".join(missing)
            + ". Pass --input-json LABEL=PATH for each missing label."
        )
    return resolved


def collect_cases(acc_inputs: Sequence[Tuple[str, Path]], case_glob: str) -> List[str]:
    case_sets = []
    for label, pred_dir in acc_inputs:
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"{label}: prediction directory does not exist: {pred_dir}")
        cases = {path.stem for path in pred_dir.glob(case_glob)}
        if not cases:
            raise FileNotFoundError(f"{label}: no files matching {case_glob!r} in {pred_dir}")
        case_sets.append(cases)
    return sorted(set.intersection(*case_sets))


def frame_metrics(pred_frame: np.ndarray, gt_frame: np.ndarray) -> Tuple[float, float]:
    import numpy as np

    diff = pred_frame - gt_frame
    denom = np.linalg.norm(gt_frame)
    nrmse = float(np.linalg.norm(diff) / denom) if denom > 0 else float("nan")
    mse = float(np.mean(diff**2))
    data_range = float(gt_frame.max() - gt_frame.min())
    psnr = float("nan") if mse <= 0 or data_range <= 0 else 20.0 * np.log10(data_range / np.sqrt(mse))
    return psnr, nrmse


def load_prediction(path: Path, key: str) -> np.ndarray:
    import numpy as np

    pred = read_mat_key(path, key).squeeze().astype(np.float32)
    if pred.ndim != 4:
        raise ValueError(f"{path}: expected 4D prediction, got shape {pred.shape}")
    if not np.isfinite(pred).all():
        raise ValueError(f"{path}: prediction contains NaN or Inf")
    return np.abs(pred)


def read_mat_values(path: Path) -> dict:
    try:
        import h5py

        with h5py.File(path, "r", swmr=True) as mat_file:
            return {key: mat_file[key][()] for key in mat_file}
    except (ImportError, OSError, ValueError):
        import scipy.io

        return {key: value for key, value in scipy.io.loadmat(path).items() if not key.startswith("__")}


def logical_kspace(values: dict, path: Path) -> np.ndarray:
    import numpy as np

    key = next((candidate for candidate in ("kus", "kspace_full", "kspace") if candidate in values), None)
    if key is None:
        raise ValueError(f"{path}: expected one of kus, kspace_full, or kspace")
    value = np.asarray(values[key])
    if np.issubdtype(value.dtype, np.complexfloating):
        shape = (1,) * (5 - value.ndim) + value.shape[::-1]
        return value.transpose().reshape(shape)
    if value.dtype.fields and {"real", "imag"}.issubset(value.dtype.fields):
        result = value["real"] + 1j * value["imag"]
        return result.reshape((1,) * (5 - result.ndim) + result.shape)
    raise ValueError(f"{path}:{key} is not complex data (dtype={value.dtype})")


def logical_mask(values: dict, path: Path) -> np.ndarray:
    import numpy as np

    if "mask" not in values:
        raise ValueError(f"{path}: missing mask key")
    mask = np.asarray(values["mask"])
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=(0, 1))
    elif mask.ndim == 3:
        mask = np.expand_dims(mask, axis=(1, 2))
    else:
        raise ValueError(f"{path}: mask must be 2D or 3D before expansion, got {mask.shape}")
    return (mask > 0).astype(np.float32)


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    import numpy as np

    shifted = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(shifted, axes=(-2, -1), norm="ortho")
    return np.fft.fftshift(image, axes=(-2, -1))


def load_zero_filled_input(json_path: Path) -> np.ndarray:
    import numpy as np

    with json_path.open() as descriptor_file:
        descriptor = json.load(descriptor_file)
    kspace_path = Path(descriptor["kspace"])
    mask_paths = [Path(path) for path in descriptor.get("mask", [])]
    if not mask_paths:
        raise ValueError(f"{json_path}: mask list is empty")

    kspace = logical_kspace(read_mat_values(kspace_path), kspace_path)  # (time, slice, coil, phase, frequency)
    mask = logical_mask(read_mat_values(mask_paths[0]), mask_paths[0])  # (time, 1, 1, phase, frequency)
    if mask.shape[0] == 1 and kspace.shape[0] != 1:
        mask = np.repeat(mask, kspace.shape[0], axis=0)
    if mask.shape[0] != kspace.shape[0] or mask.shape[-2:] != kspace.shape[-2:]:
        raise ValueError(f"{json_path}: mask shape {mask.shape} does not match k-space shape {kspace.shape}")

    image = ifft2c(kspace * mask)
    rss = np.sqrt(np.sum(np.abs(image) ** 2, axis=2)).astype(np.float32)  # (time, slice, phase, frequency)
    return rss.transpose(3, 2, 1, 0)  # (frequency, phase, slice, time)


def save_case_grid(
    case_id: str,
    acc_inputs: Sequence[Tuple[str, Path]],
    json_inputs: Dict[str, Path],
    gt_root: Path,
    output_dir: Path,
    key: str,
    percentile: float,
    error_percentile: float,
    slice_index: Optional[int],
    time_index: Optional[int],
) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    gt_path = gt_root / f"norm_img_{case_id}.npy"
    if not gt_path.is_file():
        raise FileNotFoundError(f"{case_id}: missing ground truth {gt_path}")

    gt = load_gt(gt_path)
    selected_slice, selected_time = choose_indices(gt.shape, slice_index, time_index)
    gt_frame = np.abs(gt[:, :, selected_slice, selected_time])

    pred_frames = []
    input_frames = []
    error_frames = []
    titles = []
    for label, pred_dir in acc_inputs:
        pred_path = pred_dir / f"{case_id}.mat"
        json_path = json_inputs[label] / f"{case_id}.json"
        if not pred_path.is_file():
            raise FileNotFoundError(f"{label}: missing prediction {pred_path}")
        if not json_path.is_file():
            raise FileNotFoundError(f"{label}: missing input descriptor {json_path}")
        input_image = load_zero_filled_input(json_path)
        pred = load_prediction(pred_path, key)
        if pred.shape != gt.shape:
            raise ValueError(f"{label} {case_id}: pred shape {pred.shape} != GT shape {gt.shape}")
        if input_image.shape != gt.shape:
            raise ValueError(f"{label} {case_id}: input shape {input_image.shape} != GT shape {gt.shape}")
        input_frame = input_image[:, :, selected_slice, selected_time]
        pred_frame = pred[:, :, selected_slice, selected_time]
        error_frame = np.abs(pred_frame - gt_frame)
        psnr, nrmse = frame_metrics(pred_frame, gt_frame)
        input_frames.append(input_frame)
        pred_frames.append(pred_frame)
        error_frames.append(error_frame)
        titles.append(f"{label}\nPSNR {psnr:.2f} dB | NRMSE {nrmse:.4f}")

    image_vmax = np.percentile(np.stack([*input_frames, *pred_frames, gt_frame]), percentile)
    if not np.isfinite(image_vmax) or image_vmax <= 0:
        image_vmax = None
    error_vmax = np.percentile(np.stack(error_frames), error_percentile)
    if not np.isfinite(error_vmax) or error_vmax <= 0:
        error_vmax = None

    num_cols = len(acc_inputs)
    fig, axes = plt.subplots(4, num_cols, figsize=(4.0 * num_cols, 12.0), squeeze=False)
    for col, title in enumerate(titles):
        panels = [
            ("Input", input_frames[col], "gray", image_vmax),
            ("Recon", pred_frames[col], "gray", image_vmax),
            ("GT", gt_frame, "gray", image_vmax),
            ("Abs error", error_frames[col], "magma", error_vmax),
        ]
        axes[0, col].set_title(title)
        for row, (row_label, image, cmap, vmax) in enumerate(panels):
            axis = axes[row, col]
            im = axis.imshow(image.T, cmap=cmap, origin="lower", vmax=vmax)
            axis.axis("off")
            fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
            if col == 0:
                axis.text(
                    -0.08,
                    0.5,
                    row_label,
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    clip_on=False,
                )

    fig.suptitle(f"{case_id}: slice={selected_slice}, time={selected_time}", fontsize=14)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case_id}_slice{selected_slice:02d}_time{selected_time:02d}_acc_compare.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acc",
        action="append",
        type=parse_acc_arg,
        required=True,
        help="Acceleration label and prediction directory as LABEL=PATH. Repeat for each column.",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("/home/students/studxuzho1/dataset_v0/norm_img"),
        help="Directory containing norm_img_<case>.npy ground truth files.",
    )
    parser.add_argument(
        "--input-json",
        action="append",
        type=parse_label_path,
        default=None,
        help=(
            "Matching input JSON directory as LABEL=PATH. Repeat for each acceleration. "
            "If omitted, CustomCINEOutputR* paths are mapped to dataset/CustomCINEDataR*/json_input when present."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output/acc_comparison_pngs"),
        help="Directory for saved comparison PNGs.",
    )
    parser.add_argument("--case-glob", default="Sub*.mat", help="Prediction case glob used in every acc directory.")
    parser.add_argument("--key", default="img4ranking", help="MAT key containing the reconstruction.")
    parser.add_argument("--percentile", type=float, default=99.5, help="Display percentile for GT/recon rows.")
    parser.add_argument("--error-percentile", type=float, default=99.0, help="Display percentile for error row.")
    parser.add_argument("--slice-index", type=int, default=None, help="Slice to visualize; default is center slice.")
    parser.add_argument("--time-index", type=int, default=None, help="Time frame to visualize; default is center frame.")
    args = parser.parse_args()

    try:
        json_inputs = resolve_json_inputs(args.acc, args.input_json)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    cases = collect_cases(args.acc, args.case_glob)
    if not cases:
        parser.error("No shared cases found across acceleration directories")

    for case_id in cases:
        output_path = save_case_grid(
            case_id=case_id,
            acc_inputs=args.acc,
            json_inputs=json_inputs,
            gt_root=args.gt_root,
            output_dir=args.output_dir,
            key=args.key,
            percentile=args.percentile,
            error_percentile=args.error_percentile,
            slice_index=args.slice_index,
            time_index=args.time_index,
        )
        print(output_path)


if __name__ == "__main__":
    main()
