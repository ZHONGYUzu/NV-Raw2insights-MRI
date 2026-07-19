#!/usr/bin/env python3
"""Validate a prepared fastMRI H5 cohort before expensive inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from create_fastmri_brain_cohort import inspect_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    expected = sorted(str(name) for name in manifest.get("selected_files", []))
    actual = sorted(path.name for path in args.input_dir.glob("*.h5"))
    if not expected:
        parser.error(f"No selected files recorded in {args.manifest}")
    if actual != expected:
        raise ValueError(f"Prepared H5 files differ from manifest: expected={expected}, actual={actual}")

    for filename in actual:
        path = args.input_dir / filename
        record = inspect_case(path)
        print(
            f"OK: {filename}: acquisition={record['acquisition']}, "
            f"kspace={tuple(record['kspace_shape'])}, target={tuple(record['target_shape'])}"
        )
    print(f"Validated {len(actual)} fastMRI case(s).")


if __name__ == "__main__":
    main()
