#!/usr/bin/env python3
"""LEGACY ONLY: paired ranking-crop evaluation; not the active full-image CINE protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from evaluate_cmrxrecon2023_fullsample import metrics, summarize_frames
from run4ranking import run4Ranking


METRIC_NAMES = ("psnr", "ssim", "nrmse", "nmse", "mse", "mae")


def load_mat_array(path: Path, key: str) -> np.ndarray:
    values = scipy.io.loadmat(path)
    if key not in values:
        available = sorted(name for name in values if not name.startswith("__"))
        raise KeyError(f"{path}: missing {key!r}; available keys: {available}")
    array = np.abs(np.asarray(values[key])).astype(np.float32)
    if array.ndim != 4:
        raise ValueError(f"{path}:{key} must be 4D, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}:{key} contains NaN or Inf")
    return array


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in METRIC_NAMES:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        result[name] = float(np.nanmean(values))
    return result


def save_case_figure(
    reference: np.ndarray,
    predictions: dict[str, np.ndarray],
    path: Path,
    slice_index: int = 1,
    time_index: int = 1,
) -> None:
    slice_index = min(slice_index, reference.shape[2] - 1)
    time_index = min(time_index, reference.shape[3] - 1)
    ref = reference[:, :, slice_index, time_index]
    panels: list[tuple[str, np.ndarray, str, float]] = []
    image_values = [ref, *(value[:, :, slice_index, time_index] for value in predictions.values())]
    vmax = float(np.percentile(np.stack(image_values), 99.5))
    panels.append(("FullSample RSS GT", ref, "gray", vmax))
    for label, prediction in predictions.items():
        pred = prediction[:, :, slice_index, time_index]
        panels.append((label, pred, "gray", vmax))
        error = np.abs(pred - ref)
        panels.append((f"|{label} - GT|", error, "magma", float(np.percentile(error, 99.5))))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), constrained_layout=True)
    for axis, (title, image, cmap, limit) in zip(np.atleast_1d(axes), panels):
        rendered = axis.imshow(image, cmap=cmap, vmin=0, vmax=max(limit, np.finfo(float).eps))
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"slice={slice_index}, time={time_index}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metric_boxplot(rows: list[dict[str, Any]], run_labels: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for axis, name in zip(axes, ("psnr", "ssim", "nrmse", "nmse")):
        groups = [
            np.asarray([float(row[name]) for row in rows if row["run"] == label], dtype=np.float64)
            for label in run_labels
        ]
        axis.boxplot(groups, tick_labels=run_labels, showmeans=True, showfliers=False)
        axis.set_title(name.upper())
        axis.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Run must use LABEL=/path/to/predictions")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("Run must use LABEL=/path/to/predictions")
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--filetype", default="cine_lax")
    args = parser.parse_args()

    if len(args.run) != 2:
        parser.error("Exactly two --run arguments are required for paired evaluation")
    runs = dict(args.run)
    if len(runs) != 2:
        parser.error("Run labels must be unique")

    manifest = json.loads(args.manifest.read_text())
    cases = manifest.get("cases", [])
    if not cases:
        parser.error(f"No cases found in {args.manifest}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking_gt_dir = args.output_dir / "ground_truth"
    figure_dir = args.output_dir / "comparison_figures"
    ranking_gt_dir.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []

    for case in cases:
        case_id = str(case["case_id"])
        full_gt_path = args.ground_truth_dir / f"{case_id}_fullsample_rss.mat"
        full_reference = load_mat_array(full_gt_path, "gt")
        reference = run4Ranking(full_reference, args.filetype, do_center_crop_infer=True)
        scipy.io.savemat(
            ranking_gt_dir / f"{case_id}.mat",
            {"img4ranking": reference},
            do_compression=True,
        )

        predictions: dict[str, np.ndarray] = {}
        for label, prediction_dir in runs.items():
            prediction = load_mat_array(prediction_dir / f"{case_id}.mat", "img4ranking")
            if prediction.shape != reference.shape:
                raise ValueError(
                    f"{case_id} {label}: prediction {prediction.shape} != ranking GT {reference.shape}"
                )
            predictions[label] = prediction
            current_case_rows: list[dict[str, Any]] = []
            for slice_index in range(reference.shape[2]):
                for time_index in range(reference.shape[3]):
                    row: dict[str, Any] = {
                        "run": label,
                        "case": case_id,
                        "subject": str(case["subject"]),
                        "slice": slice_index,
                        "time": time_index,
                        "shape": "x".join(str(dim) for dim in reference.shape),
                    }
                    row.update(
                        metrics(
                            reference[:, :, slice_index, time_index],
                            prediction[:, :, slice_index, time_index],
                        )
                    )
                    current_case_rows.append(row)
            frame_rows.extend(current_case_rows)
            case_rows.append(
                {
                    "run": label,
                    "case": case_id,
                    "subject": str(case["subject"]),
                    "num_frames": len(current_case_rows),
                    **mean_metrics(current_case_rows),
                }
            )

        save_case_figure(reference, predictions, figure_dir / f"{case_id}_slice01_time01.png")
        print(f"{case_id}: ranking shape={reference.shape}")

    labels = list(runs)
    paired_rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        by_label = {row["run"]: row for row in case_rows if row["case"] == case_id}
        paired: dict[str, Any] = {
            "case": case_id,
            "subject": str(case["subject"]),
            "comparison": f"{labels[1]} minus {labels[0]}",
        }
        for name in METRIC_NAMES:
            paired[f"{labels[0]}_{name}"] = by_label[labels[0]][name]
            paired[f"{labels[1]}_{name}"] = by_label[labels[1]][name]
            paired[f"delta_{name}"] = float(by_label[labels[1]][name]) - float(by_label[labels[0]][name])
        paired_rows.append(paired)

    write_csv(frame_rows, args.output_dir / "frame_metrics.csv")
    write_csv(case_rows, args.output_dir / "case_metrics.csv")
    write_csv(paired_rows, args.output_dir / "paired_case_deltas.csv")

    summary_rows: list[dict[str, Any]] = []
    summary_json: dict[str, Any] = {}
    for label in labels:
        selected = [row for row in frame_rows if row["run"] == label]
        summary = summarize_frames(selected)
        summary_json[label] = summary
        summary_rows.append(
            {
                "run": label,
                "num_cases": len(cases),
                "num_frames": len(selected),
                **{f"{name}_mean": summary[name]["mean"] for name in METRIC_NAMES},
                **{f"{name}_std": summary[name]["std"] for name in METRIC_NAMES},
            }
        )
    write_csv(summary_rows, args.output_dir / "summary_metrics.csv")
    save_metric_boxplot(frame_rows, labels, args.output_dir / "metrics_boxplot.png")

    provenance = {
        "manifest": str(args.manifest.resolve()),
        "ground_truth_dir": str(args.ground_truth_dir.resolve()),
        "runs": {label: str(path.resolve()) for label, path in runs.items()},
        "filetype": args.filetype,
        "ranking_crop": "run4Ranking(..., do_center_crop_infer=True)",
        "num_cases": len(cases),
        "frames_per_run": len(frame_rows) // len(labels),
        "summary": summary_json,
        "paired_delta_definition": f"{labels[1]} minus {labels[0]}; positive favors the second run for PSNR/SSIM, negative favors it for error metrics.",
    }
    (args.output_dir / "evaluation_config.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=True) + "\n"
    )
    print(f"Evaluated {len(cases)} paired cases, {len(frame_rows) // len(labels)} frames per run")
    print(f"Saved results: {args.output_dir}")


if __name__ == "__main__":
    main()
