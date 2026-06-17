#!/usr/bin/env python3
"""Plot GT/reconstruction/error grids across CINE acceleration settings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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


def save_case_grid(
    case_id: str,
    acc_inputs: Sequence[Tuple[str, Path]],
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
    error_frames = []
    titles = []
    for label, pred_dir in acc_inputs:
        pred_path = pred_dir / f"{case_id}.mat"
        if not pred_path.is_file():
            raise FileNotFoundError(f"{label}: missing prediction {pred_path}")
        pred = load_prediction(pred_path, key)
        if pred.shape != gt.shape:
            raise ValueError(f"{label} {case_id}: pred shape {pred.shape} != GT shape {gt.shape}")
        pred_frame = pred[:, :, selected_slice, selected_time]
        error_frame = np.abs(pred_frame - gt_frame)
        psnr, nrmse = frame_metrics(pred_frame, gt_frame)
        pred_frames.append(pred_frame)
        error_frames.append(error_frame)
        titles.append(f"{label}\nPSNR {psnr:.2f} dB | NRMSE {nrmse:.4f}")

    image_vmax = np.percentile(gt_frame, percentile)
    if not np.isfinite(image_vmax) or image_vmax <= 0:
        image_vmax = None
    error_vmax = np.percentile(np.stack(error_frames), error_percentile)
    if not np.isfinite(error_vmax) or error_vmax <= 0:
        error_vmax = None

    num_cols = len(acc_inputs)
    fig, axes = plt.subplots(3, num_cols, figsize=(4.0 * num_cols, 9.0), squeeze=False)
    for col, title in enumerate(titles):
        panels = [
            ("GT", gt_frame, "gray", image_vmax),
            ("Recon", pred_frames[col], "gray", image_vmax),
            ("Abs error", error_frames[col], "magma", error_vmax),
        ]
        axes[0, col].set_title(title)
        for row, (row_label, image, cmap, vmax) in enumerate(panels):
            axis = axes[row, col]
            axis.imshow(image.T, cmap=cmap, origin="lower", vmax=vmax)
            axis.axis("off")
            if col == 0:
                axis.set_ylabel(row_label, rotation=0, ha="right", va="center", labelpad=42, fontsize=12)

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

    cases = collect_cases(args.acc, args.case_glob)
    if not cases:
        parser.error("No shared cases found across acceleration directories")

    for case_id in cases:
        output_path = save_case_grid(
            case_id=case_id,
            acc_inputs=args.acc,
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
