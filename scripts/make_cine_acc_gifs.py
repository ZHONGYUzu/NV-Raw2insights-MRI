#!/usr/bin/env python3
"""Create per-case CINE GIFs for one or more acceleration outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple


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
    import numpy as np

    try:
        import h5py

        with h5py.File(path, "r", swmr=True) as mat_file:
            if key not in mat_file:
                raise KeyError(f"{path}: missing key {key!r}; keys={list(mat_file.keys())}")
            return np.asarray(mat_file[key][()])
    except (ImportError, OSError):
        import scipy.io

        data = scipy.io.loadmat(path)
        if key not in data:
            keys = [candidate for candidate in data if not candidate.startswith("__")]
            raise KeyError(f"{path}: missing key {key!r}; keys={keys}")
        return np.asarray(data[key])


def normalize_to_uint8(frame, vmin: float, vmax: float):
    import numpy as np

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros(frame.shape, dtype=np.uint8)
    frame = np.clip((frame - vmin) / (vmax - vmin), 0.0, 1.0)
    return (frame * 255.0).round().astype(np.uint8)


def colorize_gray(frame_uint8):
    from PIL import Image

    return Image.fromarray(frame_uint8, mode="L").convert("RGB")


def colorize_magma(frame_uint8):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    rgba = plt.get_cmap("magma")(frame_uint8.astype(np.float32) / 255.0)
    rgb = (rgba[:, :, :3] * 255.0).round().astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def label_image(image, text: str):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 18), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    return image


def load_prediction(path: Path, key: str):
    import numpy as np

    pred = read_mat_key(path, key).squeeze()
    if pred.ndim != 4:
        raise ValueError(f"{path}: expected 4D array (frequency, phase, slice, time), got {pred.shape}")
    if not np.isfinite(pred).all():
        raise ValueError(f"{path}: prediction contains NaN or Inf")
    return np.abs(pred).astype(np.float32)


def load_gt(gt_root: Path, case_id: str, key: str):
    import numpy as np

    gt_path = gt_root / f"{case_id}.mat"
    if not gt_path.is_file():
        raise FileNotFoundError(f"{case_id}: missing GT file {gt_path}")
    gt = read_mat_key(gt_path, key).squeeze()
    if gt.ndim != 4:
        raise ValueError(f"{gt_path}: expected 4D array (frequency, phase, slice, time), got {gt.shape}")
    if not np.isfinite(gt).all():
        raise ValueError(f"{gt_path}: GT contains NaN or Inf")
    return np.abs(gt).astype(np.float32)


def select_slice(shape: Sequence[int], slice_index: int | None, slice_number: int | None) -> int:
    num_slices = shape[2]
    if slice_index is not None and slice_number is not None:
        raise ValueError("Use only one of --slice-index or --slice-number")
    if slice_number is not None:
        if not 1 <= slice_number <= num_slices:
            raise ValueError(f"slice number {slice_number} outside [1, {num_slices}]")
        return slice_number - 1
    if slice_index is None:
        return num_slices // 2
    if not 0 <= slice_index < num_slices:
        raise ValueError(f"slice index {slice_index} outside [0, {num_slices - 1}]")
    return slice_index


def make_recon_frames(cine, label: str, percentile: float, per_frame_normalize: bool, transpose_display: bool):
    import numpy as np

    if transpose_display:
        cine = np.transpose(cine, (1, 0, 2))
    if per_frame_normalize:
        limits = [
            (float(frame.min()), float(np.percentile(frame, percentile)))
            for frame in (cine[:, :, time_index] for time_index in range(cine.shape[2]))
        ]
    else:
        limits = [(float(cine.min()), float(np.percentile(cine, percentile)))] * cine.shape[2]

    frames = []
    for time_index in range(cine.shape[2]):
        frame = normalize_to_uint8(cine[:, :, time_index], *limits[time_index])
        image = colorize_gray(frame)
        frames.append(label_image(image, f"{label} t={time_index:02d}"))
    return frames


def make_comparison_frames(
    pred_cine,
    gt_cine,
    label: str,
    percentile: float,
    error_percentile: float,
    per_frame_normalize: bool,
    transpose_display: bool,
):
    import numpy as np
    from PIL import Image

    if pred_cine.shape != gt_cine.shape:
        raise ValueError(f"{label}: prediction cine shape {pred_cine.shape} != GT cine shape {gt_cine.shape}")
    if transpose_display:
        pred_cine = np.transpose(pred_cine, (1, 0, 2))
        gt_cine = np.transpose(gt_cine, (1, 0, 2))
    err_cine = np.abs(pred_cine - gt_cine)

    if per_frame_normalize:
        image_limits = []
        error_limits = []
        for time_index in range(pred_cine.shape[2]):
            image_stack = np.stack([gt_cine[:, :, time_index], pred_cine[:, :, time_index]])
            image_limits.append((0.0, float(np.percentile(image_stack, percentile))))
            error_limits.append((0.0, float(np.percentile(err_cine[:, :, time_index], error_percentile))))
    else:
        image_limits = [(0.0, float(np.percentile(np.stack([gt_cine, pred_cine]), percentile)))] * pred_cine.shape[2]
        error_limits = [(0.0, float(np.percentile(err_cine, error_percentile)))] * pred_cine.shape[2]

    frames = []
    for time_index in range(pred_cine.shape[2]):
        vmin, vmax = image_limits[time_index]
        emin, emax = error_limits[time_index]
        gt_image = colorize_gray(normalize_to_uint8(gt_cine[:, :, time_index], vmin, vmax))
        pred_image = colorize_gray(normalize_to_uint8(pred_cine[:, :, time_index], vmin, vmax))
        err_image = colorize_magma(normalize_to_uint8(err_cine[:, :, time_index], emin, emax))
        panels = [
            label_image(gt_image, f"GT t={time_index:02d}"),
            label_image(pred_image, f"{label} t={time_index:02d}"),
            label_image(err_image, "Abs error"),
        ]
        frame = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height))
        x = 0
        for panel in panels:
            frame.paste(panel, (x, 0))
            x += panel.width
        frames.append(frame)
    return frames


def save_gif(frames, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / fps))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(output_path)


def collect_cases(acc_inputs: Sequence[Tuple[str, Path]], case_glob: str, max_cases: int | None) -> list[str]:
    case_sets = []
    for label, pred_dir in acc_inputs:
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"{label}: prediction directory does not exist: {pred_dir}")
        cases = {path.stem for path in pred_dir.glob(case_glob)}
        if not cases:
            raise FileNotFoundError(f"{label}: no files matching {case_glob!r} in {pred_dir}")
        case_sets.append(cases)
    cases = sorted(set.intersection(*case_sets))
    return cases[:max_cases] if max_cases is not None else cases


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
        default=None,
        help="Optional MAT GT root. If provided, GIFs are GT | Recon | Abs error panels.",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("output/CINE1_gifs"))
    parser.add_argument("--case-glob", default="Sub*.mat", help="Prediction case glob used in every acc directory.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional limit after sorting shared cases.")
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Zero-based slice index to animate. Default: center slice.",
    )
    parser.add_argument(
        "--slice-number",
        type=int,
        default=None,
        help="One-based slice number to animate. For the sixth slice, use --slice-number 6.",
    )
    parser.add_argument("--key", default="img4ranking", help="MAT key containing the reconstruction.")
    parser.add_argument("--gt-key", default="gt", help="MAT key containing the ground truth.")
    parser.add_argument("--fps", type=float, default=5.0, help="GIF playback speed.")
    parser.add_argument("--percentile", type=float, default=99.5, help="Display percentile for GT/recon scaling.")
    parser.add_argument("--error-percentile", type=float, default=99.0, help="Display percentile for error scaling.")
    parser.add_argument(
        "--per-frame-normalize",
        action="store_true",
        help="Normalize each time frame independently instead of using one scale for the whole GIF.",
    )
    parser.add_argument(
        "--transpose-display",
        action="store_true",
        default=True,
        help="Transpose each frame for display. Enabled by default to match comparison PNG orientation.",
    )
    parser.add_argument(
        "--no-transpose-display",
        action="store_false",
        dest="transpose_display",
        help="Do not transpose frames before GIF rendering.",
    )
    args = parser.parse_args()

    cases = collect_cases(args.acc, args.case_glob, args.max_cases)
    if not cases:
        parser.error("No shared cases found across acceleration directories")

    for label, pred_dir in args.acc:
        for case_id in cases:
            pred_path = pred_dir / f"{case_id}.mat"
            pred = load_prediction(pred_path, args.key)
            slice_index = select_slice(pred.shape, args.slice_index, args.slice_number)
            slice_number = slice_index + 1
            pred_cine = pred[:, :, slice_index, :]
            if args.gt_root is None:
                frames = make_recon_frames(
                    pred_cine,
                    label=label,
                    percentile=args.percentile,
                    per_frame_normalize=args.per_frame_normalize,
                    transpose_display=args.transpose_display,
                )
                output_path = args.output_dir / label / f"{case_id}_{label}_slice{slice_number:02d}_recon.gif"
            else:
                gt = load_gt(args.gt_root, case_id, args.gt_key)
                if pred.shape != gt.shape:
                    raise ValueError(f"{label} {case_id}: pred shape {pred.shape} != GT shape {gt.shape}")
                gt_cine = gt[:, :, slice_index, :]
                frames = make_comparison_frames(
                    pred_cine,
                    gt_cine,
                    label=label,
                    percentile=args.percentile,
                    error_percentile=args.error_percentile,
                    per_frame_normalize=args.per_frame_normalize,
                    transpose_display=args.transpose_display,
                )
                output_path = args.output_dir / label / f"{case_id}_{label}_slice{slice_number:02d}_gt_recon_error.gif"
            save_gif(frames, output_path, args.fps)


if __name__ == "__main__":
    main()
