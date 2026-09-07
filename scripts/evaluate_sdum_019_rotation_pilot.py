#!/usr/bin/env python3
"""Evaluate absolute quality and rotation consistency for sdum-019."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io
from scipy.ndimage import binary_erosion, rotate as scipy_rotate
from skimage.metrics import structural_similarity


METRICS = ("psnr", "ssim", "nrmse", "nmse", "mse", "mae")


def load_mat(path: Path, key: str) -> np.ndarray:
    values = scipy.io.loadmat(path)
    if key not in values:
        available = [candidate for candidate in values if not candidate.startswith("__")]
        raise KeyError(f"{path}: missing {key!r}; keys={available}")
    value = np.asarray(values[key])
    if not np.isfinite(value).all():
        raise ValueError(f"{path}:{key} contains NaN or Inf")
    return value


def rotate_saved_layout_back(array: np.ndarray, input_angle: float, order: int) -> np.ndarray:
    """Undo input-domain rotation after PE/FE has been transposed to FE/PE."""
    normalized = input_angle % 360.0
    if np.isclose(normalized, 0.0):
        return np.array(array, copy=True)
    if np.isclose(normalized, 180.0):
        return np.ascontiguousarray(np.rot90(array, k=2, axes=(0, 1)))
    # Transposing PE/FE to FE/PE reverses the displayed rotation sign, so the
    # original positive angle is the inverse transform in saved layout.
    return scipy_rotate(
        array,
        angle=input_angle,
        axes=(0, 1),
        reshape=False,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=order > 1,
    )


def masked_metrics(pred: np.ndarray, gt: np.ndarray, roi: np.ndarray) -> dict[str, float]:
    mask = np.asarray(roi, dtype=bool)
    if pred.shape != gt.shape or pred.shape != mask.shape:
        raise ValueError(f"metric shape mismatch: pred={pred.shape}, gt={gt.shape}, roi={mask.shape}")
    if int(mask.sum()) < 64:
        raise ValueError(f"valid ROI is too small: {int(mask.sum())} pixels")
    pred_values = pred[mask].astype(np.float64)
    gt_values = gt[mask].astype(np.float64)
    difference = pred_values - gt_values
    squared_error = float(np.dot(difference, difference))
    gt_squared = float(np.dot(gt_values, gt_values))
    mse = float(np.mean(difference**2))
    mae = float(np.mean(np.abs(difference)))
    data_range = float(gt_values.max() - gt_values.min())
    psnr = (float("nan") if data_range <= 0 else
            float("inf") if mse == 0 else
            float(20 * np.log10(data_range / np.sqrt(mse))))
    nrmse = float(np.sqrt(squared_error / gt_squared)) if gt_squared > 0 else float("nan")
    nmse = float(squared_error / gt_squared) if gt_squared > 0 else float("nan")
    if data_range <= 0:
        ssim = float("nan")
    else:
        _, ssim_map = structural_similarity(
            gt.astype(np.float32),
            pred.astype(np.float32),
            data_range=data_range,
            full=True,
        )
        interior = binary_erosion(mask, iterations=3, border_value=0)
        if not np.any(interior):
            interior = mask
        ssim = float(np.mean(ssim_map[interior]))
    return {"psnr": psnr, "ssim": ssim, "nrmse": nrmse, "nmse": nmse, "mse": mse, "mae": mae}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: list[float]) -> float:
    """Ignore undefined NaNs but preserve perfect-match positive-infinite PSNR."""
    array = np.asarray(values, dtype=np.float64)
    array = array[~np.isnan(array)]
    return float(np.mean(array)) if array.size else float("nan")


def finite_median(values: list[float]) -> float:
    """Ignore undefined NaNs but preserve perfect-match positive-infinite PSNR."""
    array = np.asarray(values, dtype=np.float64)
    array = array[~np.isnan(array)]
    return float(np.median(array)) if array.size else float("nan")


def load_condition(run_root: Path, label: str, case_id: str, prediction_key: str = "img4ranking") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = run_root / "conditions" / label
    pred = load_mat(root / "output" / "val_img4ranking" / f"{case_id}.mat", prediction_key).astype(np.float32)
    gt = load_mat(root / "ground_truth" / f"{case_id}.mat", "gt").astype(np.float32)
    roi = load_mat(root / "valid_roi" / f"{case_id}.mat", "valid_roi").astype(bool)
    if pred.ndim != 4 or gt.ndim != 4 or pred.shape != gt.shape:
        raise ValueError(f"{label} {case_id}: prediction {pred.shape} and GT {gt.shape} must match in 4D")
    if roi.shape != pred.shape[:2]:
        raise ValueError(f"{label} {case_id}: ROI {roi.shape} != spatial shape {pred.shape[:2]}")
    return np.abs(pred), np.abs(gt), roi


def make_comparison_figure(
    run_root: Path,
    case_id: str,
    conditions: list[dict],
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    import matplotlib.pyplot as plt

    labels = [condition["label"] for condition in conditions]
    first_pred = arrays[labels[0]][0]
    slice_index = first_pred.shape[2] // 2
    time_index = first_pred.shape[3] // 2
    gt_frames = [arrays[label][1][:, :, slice_index, time_index] for label in labels]
    pred_frames = [arrays[label][0][:, :, slice_index, time_index] for label in labels]
    vmax = float(np.percentile(np.concatenate([frame.ravel() for frame in gt_frames]), 99.5))
    errors = [np.abs(pred - gt) for pred, gt in zip(pred_frames, gt_frames)]
    error_vmax = float(np.percentile(np.concatenate([frame.ravel() for frame in errors]), 99.5))
    fig, axes = plt.subplots(3, len(labels), figsize=(3.2 * len(labels), 8.8), squeeze=False)
    for column, (label, gt, pred, error) in enumerate(zip(labels, gt_frames, pred_frames, errors)):
        for row, (image, title, limit, cmap) in enumerate(
            ((gt, "GT", vmax, "gray"), (pred, "Recon", vmax, "gray"), (error, "|Error|", error_vmax, "magma"))
        ):
            axes[row, column].imshow(image.T, origin="lower", cmap=cmap, vmin=0, vmax=limit)
            axes[row, column].set_title(f"{label} {title}")
            axes[row, column].axis("off")
    fig.suptitle(f"{case_id}, slice {slice_index}, frame {time_index}")
    fig.tight_layout()
    output = run_root / "figures" / "comparison_grids" / f"{case_id}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_rotation_curves(run_root: Path, subject_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    conditions = sorted(
        {(str(row["condition"]), float(row["angle_degrees"])) for row in subject_rows},
        key=lambda item: item[1],
    )
    labels = [item[0] for item in conditions]
    x = np.arange(len(labels))
    cases = sorted({str(row["case"]) for row in subject_rows})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), squeeze=False)
    for axis, metric in zip(axes[0], ("ssim", "psnr", "nrmse")):
        case_values = []
        for case_id in cases:
            by_label = {
                str(row["condition"]): float(row[f"{metric}_mean"])
                for row in subject_rows
                if row["case"] == case_id
            }
            values = [by_label[label] for label in labels]
            case_values.append(values)
            axis.plot(x, values, marker="o", alpha=0.55, label=case_id)
        axis.plot(x, np.mean(case_values, axis=0), marker="o", color="black", linewidth=2.5, label="mean")
        axis.set_title(metric.upper())
        axis.set_xticks(x, labels, rotation=25)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    output = run_root / "figures" / "rotation_curves.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/sdum-019"))
    parser.add_argument("--prediction-key", default="img4ranking")
    parser.add_argument("--inverse-interpolation-order", type=int, choices=(0, 1, 3), default=1)
    args = parser.parse_args()

    manifest_path = args.run_root / "manifest.json"
    with manifest_path.open() as manifest_file:
        manifest = json.load(manifest_file)
    conditions = manifest["conditions"]
    cases = manifest["cohort"]
    baseline_labels = [item["label"] for item in conditions if np.isclose(float(item["angle_degrees"]) % 360, 0)]
    if len(baseline_labels) != 1:
        raise ValueError(f"Expected one zero-degree condition, found {baseline_labels}")
    baseline_label = baseline_labels[0]

    frame_rows: list[dict] = []
    arrays_by_case: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for case_id in cases:
        arrays = {
            condition["label"]: load_condition(args.run_root, condition["label"], case_id, args.prediction_key)
            for condition in conditions
        }
        arrays_by_case[case_id] = arrays
        baseline_pred, baseline_gt, baseline_roi = arrays[baseline_label]
        for condition in conditions:
            label = condition["label"]
            angle = float(condition["angle_degrees"])
            pred, gt, roi = arrays[label]
            aligned_pred = rotate_saved_layout_back(pred, angle, args.inverse_interpolation_order)
            aligned_gt = rotate_saved_layout_back(gt, angle, args.inverse_interpolation_order)
            aligned_roi = rotate_saved_layout_back(roi.astype(np.float32), angle, order=0) > 0.5
            consistency_roi = baseline_roi & aligned_roi
            for slice_index in range(pred.shape[2]):
                for time_index in range(pred.shape[3]):
                    quality = masked_metrics(
                        pred[:, :, slice_index, time_index],
                        gt[:, :, slice_index, time_index],
                        roi,
                    )
                    consistency = masked_metrics(
                        aligned_pred[:, :, slice_index, time_index],
                        baseline_pred[:, :, slice_index, time_index],
                        consistency_roi,
                    )
                    floor = masked_metrics(
                        aligned_gt[:, :, slice_index, time_index],
                        baseline_gt[:, :, slice_index, time_index],
                        consistency_roi,
                    )
                    frame_rows.append(
                        {
                            "experiment_id": manifest["experiment_id"],
                            "condition": label,
                            "angle_degrees": angle,
                            "case": case_id,
                            "slice": slice_index,
                            "time": time_index,
                            "valid_roi_fraction": float(np.mean(roi)),
                            **quality,
                            **{f"consistency_{key}": value for key, value in consistency.items()},
                            **{f"floor_{key}": value for key, value in floor.items()},
                        }
                    )
            print(f"{case_id} {label}: {pred.shape[2] * pred.shape[3]} frames", flush=True)
        make_comparison_figure(args.run_root, case_id, conditions, arrays)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in frame_rows:
        grouped[(str(row["condition"]), str(row["case"]))].append(row)
    subject_rows = []
    for (label, case_id), rows in grouped.items():
        angle = float(rows[0]["angle_degrees"])
        item = {
            "experiment_id": manifest["experiment_id"],
            "condition": label,
            "angle_degrees": angle,
            "case": case_id,
            "num_frames": len(rows),
        }
        for prefix in ("", "consistency_", "floor_"):
            for metric in METRICS:
                key = f"{prefix}{metric}"
                item[f"{key}_mean"] = finite_mean([float(row[key]) for row in rows])
                item[f"{key}_median"] = finite_median([float(row[key]) for row in rows])
        subject_rows.append(item)

    baseline_by_case = {row["case"]: row for row in subject_rows if row["condition"] == baseline_label}
    for row in subject_rows:
        baseline = baseline_by_case[row["case"]]
        row["psnr_drop_from_zero"] = float(baseline["psnr_mean"]) - float(row["psnr_mean"])
        row["ssim_drop_from_zero"] = float(baseline["ssim_mean"]) - float(row["ssim_mean"])
        baseline_nrmse = float(baseline["nrmse_mean"])
        row["nrmse_ratio_to_zero"] = float(row["nrmse_mean"]) / baseline_nrmse if baseline_nrmse > 0 else float("nan")

    summary_rows = []
    for condition in conditions:
        label = condition["label"]
        rows = [row for row in subject_rows if row["condition"] == label]
        item = {
            "experiment_id": manifest["experiment_id"],
            "condition": label,
            "angle_degrees": float(condition["angle_degrees"]),
            "num_cases": len(rows),
        }
        summary_keys = [f"{prefix}{metric}_mean" for prefix in ("", "consistency_", "floor_") for metric in METRICS]
        summary_keys += ["psnr_drop_from_zero", "ssim_drop_from_zero", "nrmse_ratio_to_zero"]
        for key in summary_keys:
            item[f"{key}_across_subject_mean"] = finite_mean([float(row[key]) for row in rows])
            item[f"{key}_across_subject_median"] = finite_median([float(row[key]) for row in rows])
        summary_rows.append(item)

    metrics_root = args.run_root / "metrics"
    write_csv(metrics_root / "frame_metrics.csv", frame_rows)
    write_csv(metrics_root / "subject_metrics.csv", subject_rows)
    write_csv(metrics_root / "summary_metrics.csv", summary_rows)
    make_rotation_curves(args.run_root, subject_rows)
    print(f"Evaluation written to {metrics_root}", flush=True)


if __name__ == "__main__":
    main()
