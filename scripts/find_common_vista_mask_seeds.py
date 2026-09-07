#!/usr/bin/env python3
"""Find VISTA mask seeds shared by every requested PE and acceleration."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_int_csv(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mask_dir", type=Path)
    parser.add_argument("--pe", type=parse_int_csv, default=parse_int_csv("156,162,174,180"))
    parser.add_argument("--acc", type=parse_int_csv, default=parse_int_csv("8,16,24"))
    args = parser.parse_args()

    if not args.mask_dir.is_dir():
        parser.error(f"mask directory does not exist: {args.mask_dir}")

    seed_sets: dict[tuple[int, int], set[int]] = {}
    for pe in args.pe:
        for acc in args.acc:
            pattern = f"mask_VISTA_{pe}x25_acc{acc}_*.txt"
            regex = re.compile(rf"^mask_VISTA_{pe}x25_acc{acc}_(\d+)\.txt$")
            seeds = {
                int(match.group(1))
                for path in args.mask_dir.glob(pattern)
                if (match := regex.match(path.name)) is not None
            }
            seed_sets[(pe, acc)] = seeds
            print(f"PE={pe} acc={acc}: {','.join(map(str, sorted(seeds))) or 'NONE'}")

    common = set.intersection(*seed_sets.values()) if seed_sets else set()
    print(f"COMMON_SEEDS={','.join(map(str, sorted(common)))}")
    if not common:
        raise SystemExit("No VISTA seed is shared by every requested PE/acceleration combination")


if __name__ == "__main__":
    main()
