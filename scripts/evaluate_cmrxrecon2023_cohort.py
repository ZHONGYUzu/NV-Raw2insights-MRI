#!/usr/bin/env python3
"""Evaluate a prepared CMRxRecon 2023 cohort against full-sampled RSS references."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np
import scipy.io

from evaluate_cmrxrecon2023_fullsample import (
    array_stats,
    fit_global_scale,
    fullsample_rss,
    load_compound_kspace,
    load_prediction,
    metrics,
    save_comparison,
    save_metric_plot,
    summarize_frames,
)


def write_csv(rows: list[Dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No metric rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Cohort manifest created by the cohort builder.")
    parser.add_argument("--prediction-dir", type=Path, required=True, help="Directory containing case MAT predictions.")
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--recon-key", default="img4ranking")
    parser.add_argument("--kspace-key", default="kspace_full")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    cases = manifest.get("cases", [])
    if not cases:
        parser.error(f"No cases found in {args.manifest}")

    ground_truth_dir = args.output_dir / "ground_truth"
    comparison_dir = args.output_dir / "comparisons"
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    frame_rows: list[Dict[str, object]] = []
    case_summaries: list[Dict[str, object]] = []
    for case in cases:
        case_id = str(case["case_id"])
        prediction_path = args.prediction_dir / f"{case_id}.mat"
        kspace_path = Path(str(case["kspace"]))
        if not prediction_path.is_file():
            raise FileNotFoundError(f"{case_id}: missing prediction {prediction_path}")
        if not kspace_path.is_file():
            raise FileNotFoundError(f"{case_id}: missing full k-space {kspace_path}")

        prediction = load_prediction(prediction_path, args.recon_key)
        reference = fullsample_rss(load_compound_kspace(kspace_path, args.kspace_key))
        if prediction.shape != reference.shape:
            raise ValueError(f"{case_id}: prediction {prediction.shape} != reference {reference.shape}")

        gt_path = ground_truth_dir / f"{case_id}_fullsample_rss.mat"
        scipy.io.savemat(gt_path, {"gt": reference}, do_compression=True)

        case_rows: list[Dict[str, object]] = []
        for slice_index in range(reference.shape[2]):
            for time_index in range(reference.shape[3]):
                row: Dict[str, object] = {
                    "case": case_id,
                    "subject": str(case["subject"]),
                    "view": str(case["view"]),
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
                case_rows.append(row)
        frame_rows.extend(case_rows)

        scale, scaled_prediction = fit_global_scale(reference, prediction)
        case_summary = {
            "case": case_id,
            "subject": str(case["subject"]),
            "shape": list(reference.shape),
            "num_frames": len(case_rows),
            "reference": array_stats(reference),
            "prediction": array_stats(prediction),
            "direct_volume_metrics": metrics(reference, prediction),
            "frame_metrics_summary": summarize_frames(case_rows),
            "diagnostic_global_scale": scale,
            "diagnostic_scale_fitted_volume_metrics": metrics(reference, scaled_prediction),
        }
        case_summaries.append(case_summary)

        middle_slice = reference.shape[2] // 2
        middle_time = reference.shape[3] // 2
        save_comparison(
            reference,
            prediction,
            comparison_dir / f"{case_id}.png",
            middle_slice,
            middle_time,
        )
        print(
            f"{case_id}: shape={reference.shape}, frames={len(case_rows)}, "
            f"PSNR={case_summary['direct_volume_metrics']['psnr']:.4f}, "
            f"SSIM={case_summary['direct_volume_metrics']['ssim']:.6f}, "
            f"NMSE={case_summary['direct_volume_metrics']['nmse']:.6g}"
        )

    write_csv(frame_rows, args.output_dir / "frame_metrics.csv")
    save_metric_plot(frame_rows, args.output_dir / "frame_metrics_boxplot.png")
    summary = {
        "manifest": str(args.manifest.resolve()),
        "prediction_dir": str(args.prediction_dir.resolve()),
        "num_cases": len(case_summaries),
        "num_frames": len(frame_rows),
        "selected_subjects": manifest.get("selected_subjects", []),
        "aggregate_frame_metrics": summarize_frames(frame_rows),
        "cases": case_summaries,
        "notes": {
            "reference": "Centered orthonormal IFFT of full k-space followed by coil RSS.",
            "primary_metrics": "Direct-scale frame and volume metrics.",
            "scale_fit": "Diagnostic only; not the primary quantitative result.",
            "layout": "frequency, phase, slice, time",
        },
    }
    summary_path = args.output_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print(f"Evaluated {len(case_summaries)} cases and {len(frame_rows)} frames")
    print(f"Saved results: {args.output_dir}")


if __name__ == "__main__":
    main()
