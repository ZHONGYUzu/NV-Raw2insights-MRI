#!/usr/bin/env python3
"""Prepare a reproducible CMRxRecon2024 subset from byte-split ZIP parts.

The source archive is treated as read-only. Only selected members are extracted
to an external derived-data root, while JSON descriptors and experiment
metadata are written below the requested run directory.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


PART_RE = re.compile(r"-part-(\d+)$")
DATA_RE = re.compile(
    r"^(?P<prefix>.+/MultiCoil/(?P<acquisition>[^/]+)/ValidationSet/)"
    r"UnderSample_Task1/(?P<subject>P\d+)/"
    r"(?P<basename>.+)_kus_Uniform8\.mat$"
)


class SparseSplitZip(io.RawIOBase):
    """Seekable view over fixed-size ZIP chunks; unavailable chunks read as zeroes."""

    def __init__(self, root: Path, basename: str, chunk_size: int, last_index: int):
        self.root = root
        self.basename = basename
        self.chunk_size = chunk_size
        self.last_index = last_index
        self.parts: dict[int, Path] = {}
        for path in root.glob(f"{basename}-part-*"):
            match = PART_RE.search(path.name)
            if match:
                self.parts[int(match.group(1))] = path
        last_path = self.parts.get(last_index)
        if last_path is None:
            raise FileNotFoundError(f"Required final ZIP part {last_index} is missing")
        self.total_size = last_index * chunk_size + last_path.stat().st_size
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.total_size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = position
        return position

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.total_size - self.position
        size = min(size, self.total_size - self.position)
        if size <= 0:
            return b""
        blocks = []
        remaining = size
        while remaining:
            index, offset = divmod(self.position, self.chunk_size)
            take = min(remaining, self.chunk_size - offset)
            path = self.parts.get(index)
            if path is None:
                blocks.append(b"\0" * take)
            else:
                with path.open("rb") as stream:
                    stream.seek(offset)
                    block = stream.read(take)
                if len(block) != take:
                    block += b"\0" * (take - len(block))
                blocks.append(block)
            self.position += take
            remaining -= take
        return b"".join(blocks)


@dataclass(frozen=True)
class Candidate:
    case_id: str
    subject: str
    acquisition: str
    basename: str
    data_info: zipfile.ZipInfo
    mask_info: zipfile.ZipInfo


def covered_parts(info: zipfile.ZipInfo, chunk_size: int) -> set[int]:
    # Include a conservative local-header guard at the end of the payload.
    start = info.header_offset
    end = info.header_offset + info.compress_size + 65536
    return set(range(start // chunk_size, end // chunk_size + 1))


def find_candidates(
    archive: zipfile.ZipFile,
    present_parts: set[int],
    chunk_size: int,
    allowed_acquisitions: set[str],
) -> list[Candidate]:
    infos = {info.filename: info for info in archive.infolist()}
    candidates = []
    for name, data_info in infos.items():
        match = DATA_RE.match(name)
        if not match:
            continue
        acquisition = match.group("acquisition")
        if acquisition not in allowed_acquisitions:
            continue
        subject = match.group("subject")
        basename = match.group("basename")
        mask_name = (
            f"{match.group('prefix')}Mask_Task1/{subject}/"
            f"{basename}_mask_Uniform8.mat"
        )
        mask_info = infos.get(mask_name)
        if mask_info is None:
            continue
        required = covered_parts(data_info, chunk_size) | covered_parts(mask_info, chunk_size)
        if not required <= present_parts:
            continue
        case_id = f"{subject}_{basename}"
        candidates.append(Candidate(case_id, subject, acquisition, basename, data_info, mask_info))
    return sorted(candidates, key=lambda item: (item.subject, item.acquisition, item.basename))


def select_unique_subjects(candidates: list[Candidate], count: int, seed: int) -> list[Candidate]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.subject].append(candidate)
    if len(grouped) < count:
        raise RuntimeError(f"Only {len(grouped)} eligible subjects are available; requested {count}")
    rng = random.Random(seed)
    subjects = sorted(grouped)
    rng.shuffle(subjects)
    selected = [rng.choice(grouped[subject]) for subject in subjects[:count]]
    return sorted(selected, key=lambda item: item.case_id)


def destination_for(info: zipfile.ZipInfo, derived_root: Path) -> Path:
    marker = "/MultiCoil/"
    if marker not in info.filename:
        raise ValueError(f"Unexpected archive member: {info.filename}")
    relative = Path("MultiCoil") / info.filename.split(marker, 1)[1]
    return derived_root / relative


def extract_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path):
    if destination.exists():
        if destination.stat().st_size != info.file_size:
            raise FileExistsError(
                f"Existing file has unexpected size: {destination} "
                f"({destination.stat().st_size} != {info.file_size})"
            )
        print(f"reuse {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Remove or inspect stale partial file first: {temporary}")
    try:
        with archive.open(info, "r") as source, temporary.open("xb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024**2)
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    print(f"extracted {destination}")


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_run_metadata(
    run_dir: Path,
    repo_root: Path,
    derived_root: Path,
    part_root: Path,
    selected: list[Candidate],
    seed: int,
    environment_path: Path,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    json_dir = run_dir / "json_input"
    json_dir.mkdir(parents=True, exist_ok=True)

    for candidate in selected:
        data_path = destination_for(candidate.data_info, derived_root).resolve()
        mask_path = destination_for(candidate.mask_info, derived_root).resolve()
        descriptor = {"kspace": str(data_path), "mask": [str(mask_path)]}
        (json_dir / f"{candidate.case_id}.json").write_text(
            json.dumps(descriptor, indent=2) + "\n", encoding="utf-8"
        )

    (run_dir / "cases.txt").write_text(
        "".join(f"{candidate.case_id}\n" for candidate in selected), encoding="utf-8"
    )

    config = {
        "experiment_id": "sdum-001",
        "model": "SDUM",
        "model_variant": "nv_raw2insights_mri_base",
        "checkpoint": "auto",
        "inference_config": "configs/nv_raw2insights_mri_base.json",
        "dataset": "CMRxRecon2024",
        "source_archive_parts": str(part_root.resolve()),
        "source_data_policy": "read_only",
        "derived_data_root": str(derived_root.resolve()),
        "split": "ValidationSet",
        "task": "Task1",
        "sampling": "Uniform Cartesian",
        "mask_type": "Uniform8",
        "nominal_acceleration": 8,
        "case_unit": "one acquisition from each unique subject",
        "number_of_cases": len(selected),
        "random_seed": seed,
        "selection_population": "complete members in available ZIP parts and release-config-supported acquisitions",
        "allowed_acquisitions": ["BlackBlood", "Cine", "Flow2d", "Mapping"],
        "excluded_acquisitions": ["Aorta", "Tagging"],
        "environment": str(environment_path),
        "repository": str(repo_root.resolve()),
        "git_commit": git_commit(repo_root),
        "input_descriptors": str(json_dir.resolve()),
        "output": str((run_dir / "output").resolve()),
        "selected_cases": [
            {
                "case_id": candidate.case_id,
                "subject": candidate.subject,
                "acquisition": candidate.acquisition,
                "kspace_member": candidate.data_info.filename,
                "mask_member": candidate.mask_info.filename,
            }
            for candidate in selected
        ],
    }
    # JSON is valid YAML 1.2 and avoids adding a PyYAML runtime dependency.
    (run_dir / "config.yaml").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part-root", required=True, type=Path)
    parser.add_argument("--derived-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--count", default=15, type=int)
    parser.add_argument("--seed", default=20260810, type=int)
    parser.add_argument(
        "--allowed-acquisitions",
        nargs="+",
        default=["BlackBlood", "Cine", "Flow2d", "Mapping"],
    )
    parser.add_argument("--basename", default="ChallengeData.zip")
    parser.add_argument("--chunk-size", default=4 * 1024**3, type=int)
    parser.add_argument("--last-index", default=209, type=int)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()

    sparse = SparseSplitZip(args.part_root, args.basename, args.chunk_size, args.last_index)
    with zipfile.ZipFile(sparse) as archive:
        candidates = find_candidates(
            archive,
            set(sparse.parts),
            args.chunk_size,
            set(args.allowed_acquisitions),
        )
        selected = select_unique_subjects(candidates, args.count, args.seed)
        print(f"eligible_cases={len(candidates)} eligible_subjects={len({x.subject for x in candidates})}")
        print("eligible_acquisitions=" + json.dumps(Counter(x.acquisition for x in candidates), sort_keys=True))
        print("selected_cases:")
        for candidate in selected:
            print(f"  {candidate.case_id}\t{candidate.acquisition}\t{candidate.subject}")
        if args.extract:
            for candidate in selected:
                extract_member(archive, candidate.data_info, destination_for(candidate.data_info, args.derived_root))
                extract_member(archive, candidate.mask_info, destination_for(candidate.mask_info, args.derived_root))

    if args.extract:
        write_run_metadata(
            args.run_dir,
            args.repo_root,
            args.derived_root,
            args.part_root,
            selected,
            args.seed,
            args.environment,
        )
        print(f"wrote run metadata to {args.run_dir}")
    else:
        print("dry run only; pass --extract to write selected data and run metadata")


if __name__ == "__main__":
    main()
