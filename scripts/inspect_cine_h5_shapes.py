#!/usr/bin/env python3
"""Inspect selected custom CINE H5 shapes before conversion or inference."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

import h5py


def read_subjects(path: Path) -> list[str]:
    subjects = []
    with path.open() as file:
        for line in file:
            value = line.strip()
            if value.startswith("Sub") and value[3:].isdigit():
                subjects.append(value)
    if not subjects:
        raise ValueError(f"No subject IDs found in {path}")
    return subjects


def shape_or_missing(h5_file: h5py.File, key: str) -> tuple[int, ...] | None:
    if key not in h5_file:
        return None
    return tuple(int(dim) for dim in h5_file[key].shape)


def format_shape(shape: tuple[int, ...] | None) -> str:
    return "missing" if shape is None else "x".join(str(dim) for dim in shape)


def inspect_case(input_h5_dir: Path, subject: str) -> dict[str, object]:
    h5_path = input_h5_dir / f"{subject}.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    with h5py.File(h5_path, "r") as h5_file:
        dimgc = shape_or_missing(h5_file, "dImgC")
        dmap = shape_or_missing(h5_file, "dMap")
        kspace = shape_or_missing(h5_file, "kSpace")

    if kspace is None or len(kspace) != 5:
        logical = None
    else:
        logical = (kspace[2], kspace[0], kspace[1], kspace[3], kspace[4])

    return {
        "case": subject,
        "h5": str(h5_path),
        "dImgC": dimgc,
        "dMap": dmap,
        "kSpace": kspace,
        "logical_kspace": logical,
    }


def print_table(rows: Iterable[dict[str, object]]) -> None:
    header = [
        "case",
        "dImgC(slice,cha,time,PE,FE)",
        "dMap(slice,coil,time,PE,FE)",
        "kSpace(slice,coil,time,PE,FE)",
        "logical kspace(time,slice,coil,PE,FE)",
        "PE",
        "FE",
        "time",
        "slices",
        "coils",
    ]
    print(",".join(header))
    for row in rows:
        kspace = row["kSpace"]
        logical = row["logical_kspace"]
        if isinstance(kspace, tuple) and len(kspace) == 5:
            slices, coils, time, phase, frequency = kspace
            extras = [phase, frequency, time, slices, coils]
        else:
            extras = ["", "", "", "", ""]
        values = [
            row["case"],
            format_shape(row["dImgC"]),
            format_shape(row["dMap"]),
            format_shape(row["kSpace"]),
            format_shape(logical),
            *[str(value) for value in extras],
        ]
        print(",".join(values))


def print_summary(rows: list[dict[str, object]]) -> None:
    kspace_shapes = Counter(row["kSpace"] for row in rows)
    logical_shapes = Counter(row["logical_kspace"] for row in rows)
    spatial_shapes = Counter(
        (row["kSpace"][3], row["kSpace"][4])
        for row in rows
        if isinstance(row["kSpace"], tuple) and len(row["kSpace"]) == 5
    )
    print()
    print("Summary")
    print("kSpace shapes:")
    for shape, count in sorted(kspace_shapes.items(), key=lambda item: str(item[0])):
        print(f"  {format_shape(shape)}: {count}")
    print("logical kspace shapes:")
    for shape, count in sorted(logical_shapes.items(), key=lambda item: str(item[0])):
        print(f"  {format_shape(shape)}: {count}")
    print("mask spatial requirements (PE, FE):")
    for shape, count in sorted(spatial_shapes.items()):
        print(f"  {shape[0]}x{shape[1]}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5-dir",
        type=Path,
        default=Path("/mnt/qdata/rawdata/CINE/2D_h5_compressed"),
        help="Directory containing source Sub*.h5 files.",
    )
    parser.add_argument(
        "--subjects-file",
        type=Path,
        default=Path("docs/test_subjects.txt"),
        help="Text file containing selected subject IDs.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Explicit subject IDs. Overrides --subjects-file.",
    )
    args = parser.parse_args()

    subjects = args.subjects if args.subjects else read_subjects(args.subjects_file)
    rows = [inspect_case(args.input_h5_dir, subject) for subject in subjects]
    print_table(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
