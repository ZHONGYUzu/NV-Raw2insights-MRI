#!/usr/bin/env python3
"""Plot cross-acceleration CINE comparisons using MAT ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def parse_acc_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Acceleration input must be LABEL=PATH, for example acc8=output/CINE1acr8/val_img4ranking"
        )
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Acceleration label cannot be empty")
    return label, Path(path)


def read_mat_key(path: Path, key: str):
    try:
        import h5py

        with h5py.File(path, "r", swmr=True) as mat_file:
            if key not in mat_file:
                raise KeyError(f"{path}: missing MAT key {key!r}; available keys: {list(mat_file.keys())}")
            value = mat_file[key][()]
        return value.transpose()
    except (ImportError, OSError, ValueError):
        import scipy.io

        values = scipy.io.loadmat(path)
        if key not in values:
            keys = [candidate for candidate in values if not candidate.startswith("__")]
            raise KeyError(f"{path}: missing MAT key {key!r}; available keys: {keys}")
        return values[key]


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


def choose_indices(shape: Sequence[int], slice_index: Optional[int], time_index: Optional[int]) -> Tuple[int, int]:
    if len(shape) != 4:
        raise ValueError(f"Expected 4D array, got shape {shape}")
    num_slices = shape[2]
    num_times = shape[3]
    selected_slice = num_slices // 2 if slice_index is None else slice_index
    selected_time = num_times // 2 if time_index is None else time_index
    if not 0 <= selected_slice < num_slices:
        raise ValueError(f"Slice index {selected_slice} is outside [0, {num_slices - 1}]")
    if not 0 <= selected_time < num_times:
        raise ValueError(f"Time index {selected_time} is outside [0, {num_times - 1}]")
    return selected_slice, selected_time


def load_array(path: Path, key: str, label: str):
    import numpy as np

    array = np.asarray(read_mat_key(path, key)).squeeze().astype(np.float32)
    if array.ndim != 4:
        raise ValueError(f"{label}: expected 4D array at {path}:{key}, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label}: {path}:{key} contains NaN or Inf")
    return np.abs(array)


def frame_metrics(pred_frame, gt_frame) -> Tuple[float, float]:
    import numpy as np

    diff = pred_frame - gt_frame
    denom = np.linalg.norm(gt_frame)
    nrmse = float(np.linalg.norm(diff) / denom) if denom > 0 else float("nan")
    mse = float(np.mean(diff**2))
    data_range = float(gt_frame.max() - gt_frame.min())
    psnr = float("nan") if mse <= 0 or data_range <= 0 else 20.0 * np.log10(data_range / np.sqrt(mse))
    return psnr, nrmse


def save_case_grid(
    case_id: str,
    acc_inputs: Sequence[Tuple[str, Path]],
    gt_root: Path,
    output_dir: Path,
    pred_key: str,
    gt_key: str,
    percentile: float,
    error_percentile: float,
    zoom_error_percentile: float,
    slice_index: Optional[int],
    time_index: Optional[int],
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    gt_path = gt_root / f"{case_id}.mat"
    if not gt_path.is_file():
        raise FileNotFoundError(f"{case_id}: missing ground truth {gt_path}")

    gt = load_array(gt_path, gt_key, f"{case_id} ground truth")
    selected_slice, selected_time = choose_indices(gt.shape, slice_index, time_index)
    gt_frame = gt[:, :, selected_slice, selected_time]

    pred_frames = []
    error_frames = []
    titles = []
    for label, pred_dir in acc_inputs:
        pred_path = pred_dir / f"{case_id}.mat"
        if not pred_path.is_file():
            raise FileNotFoundError(f"{label}: missing prediction {pred_path}")
        pred = load_array(pred_path, pred_key, f"{label} {case_id} prediction")
        if pred.shape != gt.shape:
            raise ValueError(f"{label} {case_id}: pred shape {pred.shape} != GT shape {gt.shape}")
        pred_frame = pred[:, :, selected_slice, selected_time]
        diff_frame = pred_frame - gt_frame
        error_frame = np.abs(diff_frame)
        psnr, nrmse = frame_metrics(pred_frame, gt_frame)
        pred_frames.append(pred_frame)
        error_frames.append(error_frame)
        titles.append(f"{label}\nPSNR {psnr:.2f} dB | NRMSE {nrmse:.4f}")

    image_vmax = np.percentile(np.stack([gt_frame, *pred_frames]), percentile)
    if not np.isfinite(image_vmax) or image_vmax <= 0:
        image_vmax = None
    error_vmax = np.percentile(np.stack(error_frames), error_percentile)
    if not np.isfinite(error_vmax) or error_vmax <= 0:
        error_vmax = None
    zoom_error_vmax = np.percentile(np.stack(error_frames), zoom_error_percentile)
    if not np.isfinite(zoom_error_vmax) or zoom_error_vmax <= 0:
        zoom_error_vmax = error_vmax

    num_cols = len(acc_inputs) + 1
    fig, axes = plt.subplots(3, num_cols, figsize=(4.0 * num_cols, 10.0), squeeze=False)

    axes[0, 0].imshow(gt_frame.T, cmap="gray", origin="lower", vmax=image_vmax)
    axes[0, 0].set_title("GT")
    axes[0, 0].axis("off")
    for row in (1, 2):
        axes[row, 0].axis("off")

    for col, (title, pred_frame, error_frame) in enumerate(
        zip(titles, pred_frames, error_frames),
        start=1,
    ):
        panels = [
            (0, pred_frame, "gray", None, image_vmax, title),
            (1, error_frame, "magma", None, error_vmax, "Abs error"),
            (2, error_frame, "magma", None, zoom_error_vmax, "Abs error zoom"),
        ]
        for row, image, cmap, vmin, vmax, title_text in panels:
            axis = axes[row, col]
            im = axis.imshow(image.T, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            axis.set_title(title_text)
            axis.axis("off")
            fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(f"{case_id}: slice={selected_slice}, time={selected_time}", fontsize=14)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case_id}_slice{selected_slice:02d}_time{selected_time:02d}.png"
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
        help="Acceleration label and prediction directory as LABEL=PATH. Repeat for each acceleration.",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("/home/students/studxuzho1/NV-Raw2insights-MRI/dataset/GT_from_kspace_dMap"),
        help="Directory containing <case>.mat ground truth files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output/CINE1_acc_comparison_mat_gt"),
        help="Directory for saved comparison PNGs.",
    )
    parser.add_argument("--case-glob", default="Sub*.mat", help="Prediction case glob used in every acc directory.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional limit after sorting shared cases.")
    parser.add_argument("--case", action="append", default=None, help="Explicit case id to plot. Repeat as needed.")
    parser.add_argument("--key", default="img4ranking", help="MAT key containing the reconstruction.")
    parser.add_argument("--gt-key", default="gt", help="MAT key containing the ground truth.")
    parser.add_argument("--percentile", type=float, default=99.5, help="Display percentile for GT/recon panels.")
    parser.add_argument("--error-percentile", type=float, default=99.0, help="Display percentile for absolute error.")
    parser.add_argument(
        "--zoom-error-percentile",
        type=float,
        default=95.0,
        help="Display percentile for the third-row high-contrast absolute error map.",
    )
    parser.add_argument("--slice-index", type=int, default=None, help="Slice to visualize; default is center slice.")
    parser.add_argument("--time-index", type=int, default=None, help="Time frame to visualize; default is center frame.")
    args = parser.parse_args()

    cases = collect_cases(args.acc, args.case_glob)
    if args.case:
        requested = set(args.case)
        missing = sorted(requested - set(cases))
        if missing:
            parser.error(f"Requested cases missing from at least one acceleration directory: {', '.join(missing)}")
        cases = [case for case in cases if case in requested]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        parser.error("No shared cases found across acceleration directories")

    for case_id in cases:
        output_path = save_case_grid(
            case_id=case_id,
            acc_inputs=args.acc,
            gt_root=args.gt_root,
            output_dir=args.output_dir,
            pred_key=args.key,
            gt_key=args.gt_key,
            percentile=args.percentile,
            error_percentile=args.error_percentile,
            zoom_error_percentile=args.zoom_error_percentile,
            slice_index=args.slice_index,
            time_index=args.time_index,
        )
        print(output_path)


if __name__ == "__main__":
    main()
