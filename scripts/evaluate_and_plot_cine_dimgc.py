#!/usr/bin/env python3
"""Evaluate CINE reconstructions against H5 dImgC and plot input/GT/recon rows."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    import scipy.io
except ImportError:
    scipy = None

try:
    from skimage.metrics import structural_similarity
except ImportError:
    structural_similarity = None


def require_scipy_io():
    if scipy is None:
        raise ImportError("scipy is required to load MATLAB .mat files")
    return scipy.io


def require_structural_similarity():
    if structural_similarity is None:
        raise ImportError("scikit-image is required to compute SSIM")
    return structural_similarity


def parse_label_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Expected LABEL=PATH, for example acc8=output/CustomCINEOutputR1/val_img4ranking"
        )
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Label cannot be empty")
    return label, Path(path)


def natural_sort_key(value: str) -> tuple:
    """Sort labels containing numbers numerically, e.g. acc8 before acc16."""
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


def read_mat_values(path: Path) -> dict:
    if h5py is not None:
        try:
            with h5py.File(path, "r", swmr=True) as mat_file:
                return {key: mat_file[key][()] for key in mat_file}
        except (OSError, ValueError):
            pass
    scipy_io = require_scipy_io()
    return {key: value for key, value in scipy_io.loadmat(path).items() if not key.startswith("__")}


def read_mat_key(path: Path, key: str) -> np.ndarray:
    if h5py is not None:
        try:
            with h5py.File(path, "r", swmr=True) as mat_file:
                if key not in mat_file:
                    raise KeyError(f"{path}: missing key {key!r}; available keys={list(mat_file.keys())}")
                return np.asarray(mat_file[key][()]).transpose()
        except (OSError, ValueError):
            pass
    scipy_io = require_scipy_io()
    values = scipy_io.loadmat(path)
    if key not in values:
        keys = [candidate for candidate in values if not candidate.startswith("__")]
        raise KeyError(f"{path}: missing key {key!r}; available keys={keys}")
    return np.asarray(values[key])


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

    kspace = logical_kspace(read_mat_values(kspace_path), kspace_path)
    mask = logical_mask(read_mat_values(mask_paths[0]), mask_paths[0])
    if mask.shape[0] == 1 and kspace.shape[0] != 1:
        mask = np.repeat(mask, kspace.shape[0], axis=0)
    if mask.shape[0] != kspace.shape[0] or mask.shape[-2:] != kspace.shape[-2:]:
        raise ValueError(f"{json_path}: mask shape {mask.shape} does not match k-space shape {kspace.shape}")

    image = ifft2c(kspace * mask)
    rss = np.sqrt(np.sum(np.abs(image) ** 2, axis=2)).astype(np.float32)
    return rss.transpose(3, 2, 1, 0)


def load_dimgc_gt(h5_root: Path, case_id: str, key: str) -> np.ndarray:
    if h5py is None:
        raise ImportError("h5py is required to load dImgC ground truth from H5 files")
    h5_path = h5_root / f"{case_id}.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(f"{case_id}: missing source H5 {h5_path}")
    with h5py.File(h5_path, "r") as h5_file:
        if key not in h5_file:
            raise KeyError(f"{h5_path}: missing key {key!r}; keys={list(h5_file.keys())}")
        dimgc = np.asarray(h5_file[key])
    if dimgc.ndim != 5 or dimgc.shape[1] != 1:
        raise ValueError(f"{h5_path}:{key} expected shape (slice, 1, time, phase, frequency), got {dimgc.shape}")
    return np.abs(dimgc[:, 0]).astype(np.float32).transpose(3, 2, 0, 1)


def load_prediction(path: Path, key: str) -> np.ndarray:
    pred = np.asarray(read_mat_key(path, key)).squeeze().astype(np.float32)
    if pred.ndim != 4:
        raise ValueError(f"{path}: expected 4D prediction, got {pred.shape}")
    if not np.isfinite(pred).all():
        raise ValueError(f"{path}: prediction contains NaN or Inf")
    return np.abs(pred)


def normalize_volume(image: np.ndarray, mode: str) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if mode == "none":
        return image
    if mode == "max":
        scale = float(np.max(np.abs(image)))
    elif mode.startswith("p"):
        scale = float(np.percentile(np.abs(image), float(mode[1:])))
    else:
        raise ValueError(f"Unknown normalization mode {mode!r}; use none, max, or p99.5")
    if not np.isfinite(scale) or scale <= 0:
        return image
    return image / scale


def frame_metrics(pred_frame: np.ndarray, gt_frame: np.ndarray) -> Dict[str, float]:
    ssim_fn = require_structural_similarity()
    diff = pred_frame.astype(np.float32) - gt_frame.astype(np.float32)
    squared_error = float(np.sum(diff**2))
    gt_squared = float(np.sum(gt_frame.astype(np.float32) ** 2))
    mse = float(np.mean(diff**2))
    nrmse = float(np.sqrt(squared_error) / np.sqrt(gt_squared)) if gt_squared > 0 else float("nan")
    data_range = float(gt_frame.max() - gt_frame.min())
    psnr = float("nan") if mse <= 0 or data_range <= 0 else float(20.0 * np.log10(data_range / np.sqrt(mse)))
    ssim = float("nan") if data_range <= 0 else float(ssim_fn(gt_frame, pred_frame, data_range=data_range))
    return {"nrmse": nrmse, "psnr": psnr, "ssim": ssim}


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
        missing = sorted(set(selected_cases) - set(cases))
        if missing:
            raise FileNotFoundError(f"Requested cases missing from at least one prediction directory: {missing}")
        return list(selected_cases)
    return cases


def choose_indices(shape: Tuple[int, int, int, int], slice_index: Optional[int], time_index: Optional[int]) -> Tuple[int, int]:
    _, _, num_slices, num_times = shape
    selected_slice = num_slices // 2 if slice_index is None else slice_index
    selected_time = num_times // 2 if time_index is None else time_index
    if not 0 <= selected_slice < num_slices:
        raise ValueError(f"slice index {selected_slice} outside [0, {num_slices - 1}]")
    if not 0 <= selected_time < num_times:
        raise ValueError(f"time index {selected_time} outside [0, {num_times - 1}]")
    return selected_slice, selected_time


def image_vmax(images: Sequence[np.ndarray], percentile: float) -> Optional[float]:
    values = np.concatenate([np.ravel(np.abs(image)) for image in images])
    vmax = float(np.percentile(values, percentile))
    return vmax if np.isfinite(vmax) and vmax > 0 else None


def save_recon_error_plot(
    case_id: str,
    arrays_by_label: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
    slice_index: int,
    time_index: int,
    percentile: float,
    dpi: int,
    vmax: Optional[float] = None,
    error_vmax: Optional[float] = None,
) -> None:
    import matplotlib.pyplot as plt

    labels = list(arrays_by_label)
    display_frames = []
    error_frames = []
    for input_image, gt, pred in arrays_by_label.values():
        display_frames.extend(
            [
                input_image[:, :, slice_index, time_index],
                gt[:, :, slice_index, time_index],
                pred[:, :, slice_index, time_index],
            ]
        )
        error_frames.append(np.abs(pred[:, :, slice_index, time_index] - gt[:, :, slice_index, time_index]))
    if vmax is None:
        vmax = image_vmax(display_frames, percentile)
    if error_vmax is None:
        error_vmax = image_vmax(error_frames, percentile)

    fig, axes = plt.subplots(4, len(labels), figsize=(4.0 * len(labels), 12.0), squeeze=False)
    for col, label in enumerate(labels):
        input_image, gt, pred = arrays_by_label[label]
        error = np.abs(pred[:, :, slice_index, time_index] - gt[:, :, slice_index, time_index])
        panels = [
            ("Input", input_image[:, :, slice_index, time_index], "gray", vmax),
            ("Target (dImgC)", gt[:, :, slice_index, time_index], "gray", vmax),
            ("Reconstruction", pred[:, :, slice_index, time_index], "gray", vmax),
            ("Absolute error", error, "magma", error_vmax),
        ]
        axes[0, col].set_title(label)
        for row, (row_label, image, cmap, panel_vmax) in enumerate(panels):
            axis = axes[row, col]
            im = axis.imshow(np.abs(image).T, cmap=cmap, origin="lower", vmax=panel_vmax)
            axis.axis("off")
            fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
            if col == 0:
                axis.text(
                    -0.12,
                    0.5,
                    row_label,
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    rotation=90,
                    clip_on=False,
                )
    fig.suptitle(f"{case_id}: slice={slice_index}, time={time_index}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def selected_frame_panels(
    arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    slice_index: int,
    time_index: int,
) -> List[Tuple[str, str, np.ndarray, str]]:
    input_image, gt, pred = arrays
    input_frame = input_image[:, :, slice_index, time_index]
    target_frame = gt[:, :, slice_index, time_index]
    reconstruction_frame = pred[:, :, slice_index, time_index]
    error_frame = np.abs(reconstruction_frame - target_frame)
    return [
        ("input", "Input", input_frame, "gray"),
        ("target", "Target (dImgC)", target_frame, "gray"),
        ("reconstruction", "Reconstruction", reconstruction_frame, "gray"),
        ("error_map", "Absolute error", error_frame, "magma"),
    ]


def shared_frame_limits(
    arrays_by_label: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    slice_index: int,
    time_index: int,
    percentile: float,
) -> Tuple[Optional[float], Optional[float]]:
    display_frames = []
    error_frames = []
    for arrays in arrays_by_label.values():
        panels = selected_frame_panels(arrays, slice_index, time_index)
        display_frames.extend(panel[2] for panel in panels[:3])
        error_frames.append(panels[3][2])
    return image_vmax(display_frames, percentile), image_vmax(error_frames, percentile)


def save_individual_panel(
    image: np.ndarray,
    output_path: Path,
    title: str,
    cmap: str,
    vmax: Optional[float],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(4.5, 4.2))
    displayed = axis.imshow(np.abs(image).T, cmap=cmap, origin="lower", vmin=0, vmax=vmax)
    axis.set_title(title)
    axis.axis("off")
    fig.colorbar(displayed, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_case_frame_outputs(
    case_id: str,
    arrays_by_label: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
    slice_index: int,
    time_index: int,
    percentile: float,
    dpi: int,
    combine_accelerations: bool,
) -> List[Path]:
    saved_paths: List[Path] = []
    frame_dir = output_dir / "plots" / case_id / f"slice{slice_index:02d}_time{time_index:02d}"
    vmax, error_vmax = shared_frame_limits(arrays_by_label, slice_index, time_index, percentile)

    for label, arrays in arrays_by_label.items():
        label_dir = frame_dir / label
        for slug, panel_title, image, cmap in selected_frame_panels(arrays, slice_index, time_index):
            panel_path = label_dir / f"{slug}.png"
            panel_vmax = error_vmax if slug == "error_map" else vmax
            save_individual_panel(
                image,
                panel_path,
                f"{case_id} {label} — {panel_title}",
                cmap,
                panel_vmax,
                dpi,
            )
            saved_paths.append(panel_path)

        four_rows_path = label_dir / "four_rows.png"
        save_recon_error_plot(
            case_id,
            {label: arrays},
            four_rows_path,
            slice_index,
            time_index,
            percentile,
            dpi,
            vmax=vmax,
            error_vmax=error_vmax,
        )
        saved_paths.append(four_rows_path)

    if combine_accelerations:
        labels = list(arrays_by_label)
        combined_path = frame_dir / "combined" / f"{'_'.join(labels)}_four_rows.png"
        save_recon_error_plot(
            case_id,
            arrays_by_label,
            combined_path,
            slice_index,
            time_index,
            percentile,
            dpi,
            vmax=vmax,
            error_vmax=error_vmax,
        )
        saved_paths.append(combined_path)

    return saved_paths


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: List[Dict[str, object]], metrics: Sequence[str]) -> List[Dict[str, object]]:
    summary = []
    for label in sorted({str(row["acceleration"]) for row in rows}, key=natural_sort_key):
        label_rows = [row for row in rows if row["acceleration"] == label]
        item: Dict[str, object] = {
            "acceleration": label,
            "num_frames": len(label_rows),
            "num_cases": len({row["case"] for row in label_rows}),
        }
        for metric in metrics:
            values = np.array([float(row[metric]) for row in label_rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            item[f"{metric}_mean"] = float(np.mean(values)) if values.size else float("nan")
            item[f"{metric}_std"] = float(np.std(values)) if values.size else float("nan")
            item[f"{metric}_median"] = float(np.median(values)) if values.size else float("nan")
            item[f"{metric}_q25"] = float(np.percentile(values, 25)) if values.size else float("nan")
            item[f"{metric}_q75"] = float(np.percentile(values, 75)) if values.size else float("nan")
        summary.append(item)
    return summary


def metric_values_by_label(rows: List[Dict[str, object]], labels: Sequence[str], metric: str) -> List[np.ndarray]:
    values_by_label = []
    for label in labels:
        values = np.array(
            [float(row[metric]) for row in rows if row["acceleration"] == label],
            dtype=np.float64,
        )
        values_by_label.append(values[np.isfinite(values)])
    return values_by_label


def save_distribution_plot(
    rows: List[Dict[str, object]],
    metrics: Sequence[str],
    output_path: Path,
    kind: str,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    labels = sorted({str(row["acceleration"]) for row in rows}, key=natural_sort_key)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.5), squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        values = metric_values_by_label(rows, labels, metric)
        if kind == "violin":
            axis.violinplot(values, showmeans=True, showmedians=True)
        else:
            axis.boxplot(values, showmeans=True, showfliers=False)
        axis.set_title(metric.upper())
        axis.set_xticks(range(1, len(labels) + 1))
        axis.set_xticklabels(labels)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def evaluate_and_plot_case(
    case_id: str,
    acc_inputs: Sequence[Tuple[str, Path]],
    json_inputs: Dict[str, Path],
    h5_root: Path,
    output_dir: Path,
    key: str,
    gt_key: str,
    normalize: str,
    percentile: float,
    dpi: int,
    combine_accelerations: bool,
    slice_index: Optional[int],
    time_index: Optional[int],
    generate_case_plots: bool,
) -> List[Dict[str, object]]:
    gt = normalize_volume(load_dimgc_gt(h5_root, case_id, gt_key), normalize)
    rows: List[Dict[str, object]] = []
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

        pred = normalize_volume(load_prediction(pred_path, key), normalize)
        if pred.shape != gt.shape:
            raise ValueError(f"{label} {case_id}: pred shape {pred.shape} != dImgC GT shape {gt.shape}")

        if generate_case_plots:
            input_image = normalize_volume(load_zero_filled_input(json_path), normalize)
            if input_image.shape != gt.shape:
                raise ValueError(f"{label} {case_id}: input shape {input_image.shape} != dImgC GT shape {gt.shape}")
            arrays_by_label[label] = (input_image, gt, pred)
        _, _, num_slices, num_times = pred.shape
        for frame_slice in range(num_slices):
            for frame_time in range(num_times):
                metrics = frame_metrics(pred[:, :, frame_slice, frame_time], gt[:, :, frame_slice, frame_time])
                rows.append(
                    {
                        "acceleration": label,
                        "case": case_id,
                        "slice": frame_slice,
                        "time": frame_time,
                        "shape": "x".join(str(dim) for dim in pred.shape),
                        "normalization": normalize,
                        **metrics,
                    }
                )

    if generate_case_plots:
        selected_slice, selected_time = choose_indices(gt.shape, slice_index, time_index)
        saved_paths = save_case_frame_outputs(
            case_id,
            arrays_by_label,
            output_dir,
            selected_slice,
            selected_time,
            percentile,
            dpi,
            combine_accelerations,
        )
        for path in saved_paths:
            print(f"{case_id}: saved {path}")
    return rows


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
        "--h5-root",
        type=Path,
        default=Path("/mnt/qdata/rawdata/CINE/2D_h5_compressed"),
        help="Directory containing source <case>.h5 files with dImgC.",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("output/cine_dimgc_eval"))
    parser.add_argument("--case", action="append", dest="cases", help="Case id to process, for example Sub0001.")
    parser.add_argument(
        "--plot-case",
        action="append",
        dest="plot_cases",
        help=(
            "Case id for which to save the full input/target/reconstruction/error PNG set. "
            "Repeat as needed. By default, plots are saved for every evaluated case. "
            "All evaluated cases always remain in the CSV files and metric distribution plots."
        ),
    )
    parser.add_argument("--case-glob", default="Sub*.mat", help="Prediction case glob used in every acc directory.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional limit after sorting shared cases.")
    parser.add_argument("--key", default="img4ranking", help="MAT key containing the reconstruction.")
    parser.add_argument("--gt-key", default="dImgC", help="H5 key containing the dImgC ground truth.")
    parser.add_argument(
        "--normalize",
        default="p99.5",
        help="Apply the same per-volume normalization to input, dImgC GT, and recon before metrics/plotting: none, max, or p99.5.",
    )
    parser.add_argument("--percentile", type=float, default=99.5, help="Display percentile for the plot.")
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution in dots per inch (default: 200).")
    parser.add_argument("--slice-index", type=int, default=None, help="Slice to visualize.")
    parser.add_argument("--time-index", type=int, default=None, help="Time frame to visualize.")
    parser.add_argument(
        "--combine-accelerations",
        action="store_true",
        help="Also save one four-row figure with all supplied acceleration rates as columns.",
    )
    parser.add_argument(
        "--skip-metric-plots",
        action="store_true",
        help="Only write CSV files and reconstruction plots; do not generate metric box/violin plots.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["nrmse", "psnr", "ssim"],
        choices=["nrmse", "psnr", "ssim"],
        help="Metrics to summarize.",
    )
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
    plot_cases = set(cases if args.plot_cases is None else args.plot_cases)
    unknown_plot_cases = sorted(plot_cases - set(cases))
    if unknown_plot_cases:
        parser.error(
            "--plot-case values must also be evaluated cases: "
            + ", ".join(unknown_plot_cases)
        )

    rows: List[Dict[str, object]] = []
    for case_id in cases:
        rows.extend(
            evaluate_and_plot_case(
                case_id=case_id,
                acc_inputs=args.acc,
                json_inputs=json_inputs,
                h5_root=args.h5_root,
                output_dir=args.output_dir,
                key=args.key,
                gt_key=args.gt_key,
                normalize=args.normalize,
                percentile=args.percentile,
                dpi=args.dpi,
                combine_accelerations=args.combine_accelerations,
                slice_index=args.slice_index,
                time_index=args.time_index,
                generate_case_plots=case_id in plot_cases,
            )
        )

    frame_csv = args.output_dir / "frame_metrics.csv"
    summary_csv = args.output_dir / "summary_metrics.csv"
    boxplot_path = args.output_dir / "metrics_boxplot.png"
    violin_path = args.output_dir / "metrics_violin.png"
    write_csv(rows, frame_csv)
    write_csv(summarize_rows(rows, args.metrics), summary_csv)
    if not args.skip_metric_plots:
        save_distribution_plot(rows, args.metrics, boxplot_path, kind="box", dpi=args.dpi)
        save_distribution_plot(rows, args.metrics, violin_path, kind="violin", dpi=args.dpi)
    print(f"Evaluated cases: {', '.join(cases)}")
    print(f"Cases with full image plots: {', '.join(case for case in cases if case in plot_cases)}")
    print(f"Total frame rows: {len(rows)}")
    print(f"Saved frame metrics: {frame_csv}")
    print(f"Saved summary metrics: {summary_csv}")
    if not args.skip_metric_plots:
        print(f"Saved box plot: {boxplot_path}")
        print(f"Saved violin plot: {violin_path}")


if __name__ == "__main__":
    main()
