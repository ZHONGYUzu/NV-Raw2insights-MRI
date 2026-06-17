#!/usr/bin/env python3
"""Create an animated GIF over all time frames for one CINE slice."""

from __future__ import annotations

import argparse
from pathlib import Path


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


def make_gif(
    mat_path: Path,
    output_path: Path,
    key: str,
    slice_index: int,
    fps: float,
    percentile: float,
    per_frame_normalize: bool,
    transpose_display: bool,
) -> None:
    import numpy as np
    from PIL import Image

    img = read_mat_key(mat_path, key).squeeze()
    if img.ndim != 4:
        raise ValueError(f"{mat_path}: expected 4D array (frequency, phase, slice, time), got {img.shape}")
    if not 0 <= slice_index < img.shape[2]:
        raise ValueError(f"slice index {slice_index} outside [0, {img.shape[2] - 1}]")

    cine = np.abs(img[:, :, slice_index, :]).astype(np.float32)
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
        frames.append(Image.fromarray(frame, mode="L"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / fps))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved GIF: {output_path}")
    print(f"frames={len(frames)} slice={slice_index} fps={fps:g} shape={img.shape}")


def default_output_path(mat_path: Path, output_dir: Path, slice_index: int) -> Path:
    return output_dir / f"{mat_path.stem}_slice{slice_index:02d}_allframes.gif"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_path", type=Path, help="Prediction MAT file containing img4ranking.")
    parser.add_argument("--slice-index", type=int, required=True, help="Slice index to animate.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the GIF. Defaults to <MAT parent>/../gifs when possible.",
    )
    parser.add_argument("--output-path", type=Path, default=None, help="Explicit GIF output path.")
    parser.add_argument("--key", default="img4ranking", help="MAT key to animate.")
    parser.add_argument("--fps", type=float, default=5.0, help="GIF playback speed.")
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="Upper percentile used for display scaling.",
    )
    parser.add_argument(
        "--per-frame-normalize",
        action="store_true",
        help="Normalize each time frame independently instead of using one scale for the whole GIF.",
    )
    parser.add_argument(
        "--transpose-display",
        action="store_true",
        help="Transpose each frame for display, useful if matching GT comparison PNG orientation.",
    )
    args = parser.parse_args()

    if args.output_path is not None:
        output_path = args.output_path
    else:
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = args.mat_path.parent.parent / "gifs"
        output_path = default_output_path(args.mat_path, output_dir, args.slice_index)

    make_gif(
        mat_path=args.mat_path,
        output_path=output_path,
        key=args.key,
        slice_index=args.slice_index,
        fps=args.fps,
        percentile=args.percentile,
        per_frame_normalize=args.per_frame_normalize,
        transpose_display=args.transpose_display,
    )


if __name__ == "__main__":
    main()
