#!/usr/bin/env python3
"""Create qualitative CINE input/reconstruction/GT/error figures.

The script compares one or more acceleration outputs for selected subjects and
can either use full-size reconstructions or apply the historical run4Ranking
crop rule to input, prediction, and ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io
from inspect_cine_recon_quality import choose_indices, load_gt, read_mat_key
from run4ranking import run4Ranking

try:
    import h5py
except ImportError:
    h5py = None


def parse_label_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH, for example acc8=output/CustomCINEOutputR1/val_img4ranking")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Label cannot be empty")
    return label, Path(path)


def read_mat_values(path: Path) -> dict:
    if h5py is not None:
        try:
            with h5py.File(path, "r", swmr=True) as mat_file:
                return {key: mat_file[key][()] for key in mat_file}
        except (OSError, ValueError):
            pass
    return {key: value for key, value in scipy.io.loadmat(path).items() if not key.startswith("__")}


def logical_kspace(values: dict, path: Path) -> np.ndarray:
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
    shifted = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(shifted, axes=(-2, -1), norm="ortho")
    return np.fft.fftshift(image, axes=(-2, -1))


def load_zero_filled_input(json_path: Path) -> np.ndarray:
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


def load_prediction(path: Path, key: str) -> np.ndarray:
    pred = read_mat_key(path, key).squeeze().astype(np.float32)
    if pred.ndim != 4:
        raise ValueError(f"{path}: expected 4D prediction, got {pred.shape}")
    if not np.isfinite(pred).all():
        raise ValueError(f"{path}: prediction contains NaN or Inf")
    return np.abs(pred)


def apply_original_crop(image: np.ndarray, filetype: str) -> np.ndarray:
    if image.ndim != 4:
        raise ValueError(f"Expected 4D image before crop, got {image.shape}")
    return run4Ranking(np.abs(image), filetype, do_center_crop_infer=True)


def align_for_mode(
    input_image: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    mode: str,
    filetype: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "full":
        if pred.shape != gt.shape:
            raise ValueError(f"full mode requires pred shape {pred.shape} to match GT {gt.shape}")
        if input_image.shape != gt.shape:
            raise ValueError(f"input shape {input_image.shape} does not match GT {gt.shape}")
        return input_image, pred, gt

    cropped_input = apply_original_crop(input_image, filetype)
    cropped_gt = apply_original_crop(gt, filetype)
    if pred.shape == gt.shape:
        pred = apply_original_crop(pred, filetype)
    elif pred.shape != cropped_gt.shape:
        raise ValueError(
            f"original-crop mode expected prediction shape {gt.shape} or {cropped_gt.shape}, got {pred.shape}"
        )
    return cropped_input, pred, cropped_gt


def collect_cases(acc_inputs: Sequence[Tuple[str, Path]], selected_cases: Optional[Sequence[str]], case_glob: str) -> List[str]:
    case_sets = []
    for label, pred_dir in acc_inputs:
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"{label}: prediction directory does not exist: {pred_dir}")
        cases = {path.stem for path in pred_dir.glob(case_glob)}
        if not cases:
            raise FileNotFoundError(f"{label}: no files matching {case_glob!r} in {pred_dir}")
        case_sets.append(cases)
    cases = sorted(set.intersection(*case_sets))
    if selected_cases:
        requested = list(selected_cases)
        missing = sorted(set(requested) - set(cases))
        if missing:
            raise FileNotFoundError(f"Requested cases missing from at least one acceleration output: {', '.join(missing)}")
        cases = requested
    return cases


def image_limits(images: Sequence[np.ndarray], percentile: float) -> Optional[float]:
    vmax = np.percentile(np.concatenate([np.ravel(np.abs(image)) for image in images]), percentile)
    return float(vmax) if np.isfinite(vmax) and vmax > 0 else None


def save_frame_grid(
    case_id: str,
    arrays_by_label: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
    slice_index: int,
    time_index: int,
    percentile: float,
    error_percentile: float,
    mode: str,
) -> None:
    import matplotlib.pyplot as plt

    labels = list(arrays_by_label)
    gt_frames = [arrays_by_label[label][2][:, :, slice_index, time_index] for label in labels]
    display_vmax = image_limits(gt_frames, percentile)
    error_frames = [
        np.abs(arrays_by_label[label][1][:, :, slice_index, time_index] - arrays_by_label[label][2][:, :, slice_index, time_index])
        for label in labels
    ]
    error_vmax = image_limits(error_frames, error_percentile)

    fig, axes = plt.subplots(4, len(labels), figsize=(4.0 * len(labels), 12.0), squeeze=False)
    for col, label in enumerate(labels):
        input_image, pred, gt = arrays_by_label[label]
        panels = [
            ("Input", input_image[:, :, slice_index, time_index], "gray", display_vmax),
            ("Recon", pred[:, :, slice_index, time_index], "gray", display_vmax),
            ("GT", gt[:, :, slice_index, time_index], "gray", display_vmax),
            ("Abs error", error_frames[col], "magma", error_vmax),
        ]
        axes[0, col].set_title(label)
        for row, (row_label, image, cmap, vmax) in enumerate(panels):
            axis = axes[row, col]
            im = axis.imshow(np.abs(image).T, cmap=cmap, origin="lower", vmax=vmax)
            axis.axis("off")
            fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
            if col == 0:
                axis.set_ylabel(row_label, rotation=0, ha="right", va="center", labelpad=48, fontsize=12)
    fig.suptitle(f"{case_id} {mode}: slice={slice_index}, time={time_index}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_yt_grid(
    case_id: str,
    arrays_by_label: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
    slice_index: int,
    x_index: int,
    percentile: float,
    error_percentile: float,
    mode: str,
) -> None:
    import matplotlib.pyplot as plt

    labels = list(arrays_by_label)
    gt_planes = [arrays_by_label[label][2][x_index, :, slice_index, :] for label in labels]
    display_vmax = image_limits(gt_planes, percentile)
    error_planes = [
        np.abs(arrays_by_label[label][1][x_index, :, slice_index, :] - arrays_by_label[label][2][x_index, :, slice_index, :])
        for label in labels
    ]
    error_vmax = image_limits(error_planes, error_percentile)

    fig, axes = plt.subplots(4, len(labels), figsize=(4.0 * len(labels), 10.0), squeeze=False)
    for col, label in enumerate(labels):
        input_image, pred, gt = arrays_by_label[label]
        panels = [
            ("Input", input_image[x_index, :, slice_index, :], "gray", display_vmax),
            ("Recon", pred[x_index, :, slice_index, :], "gray", display_vmax),
            ("GT", gt[x_index, :, slice_index, :], "gray", display_vmax),
            ("Abs error", error_planes[col], "magma", error_vmax),
        ]
        axes[0, col].set_title(label)
        for row, (row_label, plane, cmap, vmax) in enumerate(panels):
            axis = axes[row, col]
            im = axis.imshow(np.abs(plane), cmap=cmap, origin="lower", aspect="auto", vmax=vmax)
            axis.set_xlabel("time")
            if col == 0:
                axis.set_ylabel(f"{row_label}\nphase")
            else:
                axis.set_ylabel("")
            fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"{case_id} {mode}: y-t plane, x={x_index}, slice={slice_index}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def save_time_gif(
    case_id: str,
    arrays_by_label: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
    slice_index: int,
    percentile: float,
    error_percentile: float,
    fps: float,
    mode: str,
) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    labels = list(arrays_by_label)
    num_times = next(iter(arrays_by_label.values()))[1].shape[3]
    gt_frames = [
        arrays_by_label[label][2][:, :, slice_index, time_index]
        for label in labels
        for time_index in range(num_times)
    ]
    display_vmax = image_limits(gt_frames, percentile)
    error_frames = [
        np.abs(
            arrays_by_label[label][1][:, :, slice_index, time_index]
            - arrays_by_label[label][2][:, :, slice_index, time_index]
        )
        for label in labels
        for time_index in range(num_times)
    ]
    error_vmax = image_limits(error_frames, error_percentile)

    frames = []
    for time_index in range(num_times):
        fig, axes = plt.subplots(4, len(labels), figsize=(4.0 * len(labels), 12.0), squeeze=False)
        for col, label in enumerate(labels):
            input_image, pred, gt = arrays_by_label[label]
            error = np.abs(pred[:, :, slice_index, time_index] - gt[:, :, slice_index, time_index])
            panels = [
                ("Input", input_image[:, :, slice_index, time_index], "gray", display_vmax),
                ("Recon", pred[:, :, slice_index, time_index], "gray", display_vmax),
                ("GT", gt[:, :, slice_index, time_index], "gray", display_vmax),
                ("Abs error", error, "magma", error_vmax),
            ]
            axes[0, col].set_title(label)
            for row, (row_label, image, cmap, vmax) in enumerate(panels):
                axis = axes[row, col]
                axis.imshow(np.abs(image).T, cmap=cmap, origin="lower", vmax=vmax)
                axis.axis("off")
                if col == 0:
                    axis.set_ylabel(row_label, rotation=0, ha="right", va="center", labelpad=48, fontsize=12)
        fig.suptitle(f"{case_id} {mode}: slice={slice_index}, time={time_index}")
        fig.tight_layout()
        frames.append(Image.fromarray(figure_to_rgb(fig)))
        plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / fps))
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def process_case(
    case_id: str,
    acc_inputs: Sequence[Tuple[str, Path]],
    json_inputs: Dict[str, Path],
    gt_root: Path,
    output_dir: Path,
    key: str,
    mode: str,
    filetype: str,
    percentile: float,
    error_percentile: float,
    slice_index: Optional[int],
    time_index: Optional[int],
    x_index: Optional[int],
    make_gif: bool,
    fps: float,
) -> List[Path]:
    gt_path = gt_root / f"norm_img_{case_id}.npy"
    if not gt_path.is_file():
        raise FileNotFoundError(f"{case_id}: missing ground truth {gt_path}")
    gt_full = load_gt(gt_path)

    arrays_by_label: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, pred_dir in acc_inputs:
        if label not in json_inputs:
            raise ValueError(f"Missing --input-json for label {label}")
        pred_path = pred_dir / f"{case_id}.mat"
        json_path = json_inputs[label] / f"{case_id}.json"
        if not pred_path.is_file():
            raise FileNotFoundError(f"{label} {case_id}: missing prediction {pred_path}")
        if not json_path.is_file():
            raise FileNotFoundError(f"{label} {case_id}: missing descriptor {json_path}")
        input_full = load_zero_filled_input(json_path)
        pred = load_prediction(pred_path, key)
        arrays_by_label[label] = align_for_mode(input_full, pred, gt_full, mode, filetype)

    reference_shape = next(iter(arrays_by_label.values()))[1].shape
    selected_slice, selected_time = choose_indices(reference_shape, slice_index, time_index)
    if x_index is None:
        selected_x = reference_shape[0] // 2
    else:
        selected_x = x_index
    if not 0 <= selected_x < reference_shape[0]:
        raise ValueError(f"x index {selected_x} outside [0, {reference_shape[0] - 1}] for {case_id} {mode}")

    saved = []
    case_dir = output_dir / mode / case_id
    frame_path = case_dir / f"{case_id}_{mode}_slice{selected_slice:02d}_time{selected_time:02d}_frame.png"
    yt_path = case_dir / f"{case_id}_{mode}_slice{selected_slice:02d}_x{selected_x:03d}_yt.png"
    save_frame_grid(
        case_id, arrays_by_label, frame_path, selected_slice, selected_time, percentile, error_percentile, mode
    )
    save_yt_grid(case_id, arrays_by_label, yt_path, selected_slice, selected_x, percentile, error_percentile, mode)
    saved.extend([frame_path, yt_path])
    if make_gif:
        gif_path = case_dir / f"{case_id}_{mode}_slice{selected_slice:02d}_allframes.gif"
        save_time_gif(case_id, arrays_by_label, gif_path, selected_slice, percentile, error_percentile, fps, mode)
        saved.append(gif_path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acc",
        action="append",
        type=parse_label_path,
        required=True,
        help="Acceleration label and prediction directory as LABEL=PATH. Repeat for each acceleration.",
    )
    parser.add_argument(
        "--input-json",
        action="append",
        type=parse_label_path,
        required=True,
        help="Matching input JSON directory as LABEL=PATH. Labels must match --acc labels.",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("/home/students/studxuzho1/dataset_v0/norm_img"),
        help="Directory containing norm_img_<case>.npy ground truth files.",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("output/cine_qualitative"))
    parser.add_argument("--case", action="append", dest="cases", help="Case id to process, for example Sub0001.")
    parser.add_argument("--case-glob", default="Sub*.mat", help="Prediction case glob used in every acc directory.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional limit after sorting shared cases.")
    parser.add_argument(
        "--mode",
        action="append",
        choices=["full", "original-crop"],
        default=None,
        help="Output mode. Repeat to create both. Default: full.",
    )
    parser.add_argument("--key", default="img4ranking", help="MAT key containing the reconstruction.")
    parser.add_argument("--filetype", default="cine", help="File type passed to old run4Ranking crop logic.")
    parser.add_argument("--percentile", type=float, default=99.5, help="Display percentile for input/recon/GT.")
    parser.add_argument("--error-percentile", type=float, default=99.0, help="Display percentile for error.")
    parser.add_argument("--slice-index", type=int, default=None, help="Slice to visualize after mode alignment.")
    parser.add_argument("--time-index", type=int, default=None, help="Time frame to visualize after mode alignment.")
    parser.add_argument("--x-index", type=int, default=None, help="Frequency/x index for y-t plane after mode alignment.")
    parser.add_argument("--no-gif", action="store_true", help="Skip animated GIF generation.")
    parser.add_argument("--fps", type=float, default=5.0, help="GIF playback speed.")
    args = parser.parse_args()

    json_inputs = dict(args.input_json)
    acc_labels = [label for label, _ in args.acc]
    missing_json = sorted(set(acc_labels) - set(json_inputs))
    extra_json = sorted(set(json_inputs) - set(acc_labels))
    if missing_json:
        parser.error(f"Missing --input-json labels: {', '.join(missing_json)}")
    if extra_json:
        parser.error(f"--input-json labels without matching --acc: {', '.join(extra_json)}")

    cases = collect_cases(args.acc, args.cases, args.case_glob)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        parser.error("No shared cases found across acceleration directories")

    modes = args.mode or ["full"]
    for mode in modes:
        for case_id in cases:
            saved = process_case(
                case_id=case_id,
                acc_inputs=args.acc,
                json_inputs=json_inputs,
                gt_root=args.gt_root,
                output_dir=args.output_dir,
                key=args.key,
                mode=mode,
                filetype=args.filetype,
                percentile=args.percentile,
                error_percentile=args.error_percentile,
                slice_index=args.slice_index,
                time_index=args.time_index,
                x_index=args.x_index,
                make_gif=not args.no_gif,
                fps=args.fps,
            )
            for path in saved:
                print(path)


if __name__ == "__main__":
    main()
