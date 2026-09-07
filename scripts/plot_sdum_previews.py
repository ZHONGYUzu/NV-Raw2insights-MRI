#!/usr/bin/env python3
"""Create one Input / Recon / Error preview for each completed SDUM run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_cine_qualitative_results import (
    ifft2c,
    load_ground_truth,
    load_prediction,
    logical_kspace,
    logical_mask,
    read_mat_values,
)
from run4ranking import run4Ranking


RUN_CASES = {
    "sdum-001": "P026_cine_lax",
    "sdum-002": "P006_cine_lax",
    "sdum-003": "P006_cine_lax",
    "sdum-004": "P006_cine_lax",
    "sdum-005": "P006_cine_lax",
}


def display_limits(reference: np.ndarray, error: np.ndarray) -> tuple[float, float]:
    vmax = float(np.percentile(reference, 99.5))
    error_vmax = float(np.percentile(error, 99.5))
    return max(vmax, np.finfo(np.float32).eps), max(error_vmax, np.finfo(np.float32).eps)


def match_prediction_layout(image: np.ndarray, prediction: np.ndarray, layout: str) -> np.ndarray:
    """Infer crop from actual shapes, never from the legacy directory name."""
    if layout == "full" or (layout == "auto" and image.shape == prediction.shape):
        result = image
    else:
        result = run4Ranking(image, "cine", do_center_crop_infer=True)
    if result.shape != prediction.shape:
        raise ValueError(f"{layout} layout: reference {result.shape} != prediction {prediction.shape}")
    return result


def load_zero_filled_input(json_path: Path) -> np.ndarray:
    descriptor = json.loads(json_path.read_text())
    kspace_path = Path(descriptor["kspace"])
    mask_path = Path(descriptor["mask"][0])
    kspace = logical_kspace(read_mat_values(kspace_path), kspace_path)
    mask = logical_mask(read_mat_values(mask_path), mask_path)
    while mask.ndim < kspace.ndim:
        mask = np.expand_dims(mask, axis=1)
    if mask.shape[0] == 1 and kspace.shape[0] != 1:
        mask = np.repeat(mask, kspace.shape[0], axis=0)
    if mask.shape[0] != kspace.shape[0] or mask.shape[-2:] != kspace.shape[-2:]:
        raise ValueError(f"{json_path}: mask {mask.shape} does not match k-space {kspace.shape}")
    coil_images = ifft2c(kspace * mask)
    rss = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=2)).astype(np.float32)
    return rss.transpose(3, 2, 1, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--prediction-layout", choices=("auto", "full", "ranking"), default="auto")
    args = parser.parse_args()

    shared_gt = args.runs_root / "sdum-002" / "ground_truth"
    for run_name, case_id in RUN_CASES.items():
        run_root = args.runs_root / run_name
        json_path = run_root / "json_input" / f"{case_id}.json"
        full_dir = run_root / "output" / "val_img_full"
        cropped_dir = run_root / "output" / "val_img4ranking"
        use_full = (full_dir / f"{case_id}.mat").is_file()
        prediction_path = (full_dir if use_full else cropped_dir) / f"{case_id}.mat"

        input_image = load_zero_filled_input(json_path)
        reconstruction = load_prediction(prediction_path, "img4ranking")
        full_shape = input_image.shape
        input_image = match_prediction_layout(input_image, reconstruction, args.prediction_layout)

        if run_name == "sdum-001":
            reference = input_image
            error = np.abs(reconstruction - input_image)
            error_title = "|Recon - Input|\n(GT unavailable)"
        else:
            ground_truth = load_ground_truth(shared_gt, f"{case_id}_fullsample_rss", "gt")
            ground_truth = match_prediction_layout(ground_truth, reconstruction, args.prediction_layout)
            reference = ground_truth
            error = np.abs(reconstruction - ground_truth)
            error_title = "|Recon - GT|"

        if input_image.shape != reconstruction.shape or reference.shape != reconstruction.shape:
            raise ValueError(
                f"{run_name}: input={input_image.shape}, recon={reconstruction.shape}, "
                f"reference={reference.shape}"
            )

        slice_index = reconstruction.shape[2] // 2
        time_index = reconstruction.shape[3] // 2
        input_frame = input_image[:, :, slice_index, time_index]
        recon_frame = reconstruction[:, :, slice_index, time_index]
        error_frame = error[:, :, slice_index, time_index]
        vmax, error_vmax = display_limits(reference[:, :, slice_index, time_index], error_frame)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        panels = (
            ("Input (zero-filled RSS)", input_frame, "gray", vmax),
            ("Recon", recon_frame, "gray", vmax),
            (error_title, error_frame, "magma", error_vmax),
        )
        for axis, (title, image, cmap, panel_vmax) in zip(axes, panels):
            shown = axis.imshow(image.T, cmap=cmap, origin="lower", vmin=0, vmax=panel_vmax)
            axis.set_title(title)
            axis.axis("off")
            fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
        mode = "full-size" if reconstruction.shape == full_shape else "ranking-crop"
        fig.suptitle(
            f"{run_name} | {case_id} | {mode} | slice={slice_index}, time={time_index}",
            fontsize=13,
        )

        preview_dir = run_root / "output" / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        output_path = preview_dir / f"{run_name}_{case_id}_input_recon_error.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(output_path)


if __name__ == "__main__":
    main()
