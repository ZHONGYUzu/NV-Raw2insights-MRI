"""Small synthetic regression tests; no server data, model, or GPU needed.

Run: python -m unittest discover -s tests -p 'test_code_consolidation.py' -v
Dependencies: numpy scipy h5py scikit-image matplotlib.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import create_cmrxrecon2023_training_masks as masks
import evaluate_sdum_019_rotation_pilot as evaluate
import prepare_sdum_019_rotation_pilot as prepare
import plot_sdum_previews as preview
from run4ranking import run4Ranking


class ConsolidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def cli(self, script, *args, ok=True):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
            capture_output=True, text=True,
            env={**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def rotation_fixture(self):
        raw = self.root / "raw"
        descriptors = self.root / "json"
        raw.mkdir()
        descriptors.mkdir()
        rng = np.random.default_rng(4)
        shape = (1, 2, 2, 12, 16)
        kspace = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
        gt = (rng.normal(size=(1, 1, 2, 12, 16)) + 2j).astype(np.complex64)
        with h5py.File(raw / "Toy.h5", "w") as file:
            file["kSpace"] = kspace
            file["dImgC"] = gt
        mask = self.root / "Toy_mask_ktRadial8.mat"
        scipy.io.savemat(mask, {"mask": np.ones((2, 12, 16), dtype=np.float32)})
        (descriptors / "Toy.json").write_text(json.dumps({"mask": [str(mask)]}))
        run = self.root / "test-run"
        args = ["--run-root", run, "--experiment-id", "test-run", "--raw-h5-root", raw,
                "--source-json-root", descriptors, "--cases", "Toy", "--condition", "zero=0",
                "--condition", "half=180"]
        return run, args, kspace, gt

    def test_rotation_resume_preserves_records_and_raw_inputs(self):
        run, args, kspace, gt = self.rotation_fixture()
        before = (self.root / "raw" / "Toy.h5").read_bytes()
        self.cli("prepare_sdum_019_rotation_pilot.py", *args)
        original_manifest = (run / "manifest.json").read_bytes()
        self.cli("prepare_sdum_019_rotation_pilot.py", *args, "--resume")
        self.assertEqual(original_manifest, (run / "manifest.json").read_bytes())
        self.assertEqual(before, (self.root / "raw" / "Toy.h5").read_bytes())
        paths = prepare.output_paths(run, "zero", "Toy")
        recovered = scipy.io.loadmat(paths["kspace"])["kspace_full"].transpose()
        np.testing.assert_allclose(recovered, kspace.transpose(2, 0, 1, 3, 4), atol=2e-6)
        saved_gt = scipy.io.loadmat(paths["ground_truth"])["gt"]
        np.testing.assert_allclose(saved_gt, np.abs(gt[:, 0] / np.abs(gt).max()).transpose(3, 2, 0, 1), atol=1e-6)

    def test_rotation_resume_rejects_parameter_and_source_changes_before_writes(self):
        run, args, _, _ = self.rotation_fixture()
        self.cli("prepare_sdum_019_rotation_pilot.py", *args)
        original = {p: p.read_bytes() for p in (run / "cases.txt", run / "manifest.json")}
        changed = ["half=90" if value == "half=180" else value for value in args]
        self.cli("prepare_sdum_019_rotation_pilot.py", *changed, "--resume", ok=False)
        self.cli("prepare_sdum_019_rotation_pilot.py", *args, "--resume", "--interpolation-order", "3", ok=False)
        (self.root / "json" / "Toy.json").write_text((self.root / "json" / "Toy.json").read_text() + "\n")
        self.cli("prepare_sdum_019_rotation_pilot.py", *args, "--resume", ok=False)
        for path, value in original.items():
            self.assertEqual(path.read_bytes(), value)

    def test_rotation_rejects_changed_output_and_legacy_resume(self):
        run, args, _, _ = self.rotation_fixture()
        self.cli("prepare_sdum_019_rotation_pilot.py", *args)
        paths = prepare.output_paths(run, "zero", "Toy")
        paths["ground_truth"].write_bytes(b"damaged")
        self.cli("prepare_sdum_019_rotation_pilot.py", *args, "--resume", ok=False)
        (run / "preparation_config.json").unlink()
        result = self.cli("prepare_sdum_019_rotation_pilot.py", *args, "--resume", ok=False)
        self.assertIn("Legacy/partial", result.stderr)

    def test_psnr_exact_match_and_infinite_aggregation(self):
        gt = np.arange(192, dtype=np.float32).reshape(12, 16)
        result = evaluate.masked_metrics(gt, gt, np.ones_like(gt, dtype=bool))
        self.assertEqual(result["psnr"], float("inf"))
        self.assertEqual(result["nmse"], 0)
        self.assertAlmostEqual(result["ssim"], 1)
        self.assertEqual(evaluate.finite_mean([float("inf"), float("nan")]), float("inf"))
        self.assertEqual(evaluate.finite_median([float("inf"), float("nan")]), float("inf"))

    def test_custom_prediction_key_and_single_slice(self):
        root = self.root / "conditions" / "zero"
        array = np.arange(384, dtype=np.float32).reshape(16, 12, 1, 2)
        for folder, key, value in (("output/val_img4ranking", "custom", array),
                                   ("ground_truth", "gt", array),
                                   ("valid_roi", "valid_roi", np.ones((16, 12)))):
            target = root / folder
            target.mkdir(parents=True)
            scipy.io.savemat(target / "Toy.mat", {key: value})
        pred, gt, _ = evaluate.load_condition(self.root, "zero", "Toy", "custom")
        self.assertEqual(pred.shape, (16, 12, 1, 2))
        np.testing.assert_array_equal(pred, gt)

    def test_inverse_rotation_saved_layout(self):
        grid = np.zeros((24, 32), dtype=np.float32)
        grid[7:12, 13:20] = 1
        for angle in (0, 180):
            transformed = prepare.rotate_real(grid, angle, 1).T
            np.testing.assert_array_equal(evaluate.rotate_saved_layout_back(transformed, angle, 1), grid.T)
        # A non-right angle verifies the PE/FE transposition sign convention.
        transformed = prepare.rotate_real(grid, 10, 1).T
        restored = evaluate.rotate_saved_layout_back(transformed, 10, 1)
        self.assertLess(np.mean((restored - grid.T) ** 2), np.mean((transformed - grid.T) ** 2))

    def test_seed_full_cohort_compatibility_and_saved_reordering(self):
        cases = [{"case_id": name} for name in ("A", "B", "C")]
        rng = np.random.default_rng(20260823)
        expected = {case["case_id"]: int(rng.integers(1, 1001)) for case in cases}
        self.assertEqual(masks.cohort_seeds(cases, 20260823), expected)
        saved = self.root / "saved.json"
        saved.write_text(json.dumps({"mask_family": "ktGaussian", "base_seed": 20260823,
                                     "cases": [{"case_id": name, "gaussian_seed": value} for name, value in expected.items()]}))
        self.assertEqual(masks.cohort_seeds(cases[::-1], 20260823, saved), expected)

    def test_gaussian_debug_mask_matches_full_cohort(self):
        compound = np.zeros((3, 1, 2, 32, 40), dtype=[("real", "f4"), ("imag", "f4")])
        kspace = self.root / "kspace.mat"
        with h5py.File(kspace, "w") as file:
            file["kspace_full"] = compound
        manifest = self.root / "source.json"
        manifest.write_text(json.dumps({"cases": [{"case_id": name, "kspace": str(kspace)} for name in ("A", "B")]}))
        common = ["--source-manifest", manifest, "--family", "ktGaussian", "--acs-lines", "4"]
        self.cli("create_cmrxrecon2023_training_masks.py", *common, "--output-root", self.root / "full")
        self.cli("create_cmrxrecon2023_training_masks.py", *common, "--output-root", self.root / "debug", "--case-id", "B")
        full = scipy.io.loadmat(self.root / "full/masks/B_mask_ktGaussian8.mat")["mask"]
        debug = scipy.io.loadmat(self.root / "debug/masks/B_mask_ktGaussian8.mat")["mask"]
        np.testing.assert_array_equal(full, debug)
        self.cli("create_cmrxrecon2023_training_masks.py", *common, "--output-root", self.root / "full", "--case-id", "B", "--overwrite", ok=False)

    def test_preview_handles_full_and_legacy_shapes(self):
        array = np.ones((36, 32, 4, 5), dtype=np.float32)
        np.testing.assert_array_equal(preview.match_prediction_layout(array, array, "auto"), array)
        cropped = run4Ranking(array, "cine", True)
        np.testing.assert_array_equal(preview.match_prediction_layout(array, cropped, "auto"), cropped)
        with self.assertRaises(ValueError):
            preview.match_prediction_layout(array, cropped, "full")

    def test_crop_export_refuses_same_directory_and_aliases(self):
        source = self.root / "source"
        source.mkdir()
        original = source / "Toy.mat"
        scipy.io.savemat(original, {"img4ranking": np.ones((36, 32, 1, 3), dtype=np.float32)})
        initial = original.read_bytes()
        self.cli("export_sdum_ranking_crops.py", source, source, "--overwrite", ok=False)
        target = self.root / "target"
        target.mkdir()
        (target / "Toy.mat").symlink_to(original)
        self.cli("export_sdum_ranking_crops.py", source, target, "--overwrite", ok=False)
        (target / "Toy.mat").unlink()
        os.link(original, target / "Toy.mat")
        self.cli("export_sdum_ranking_crops.py", source, target, "--overwrite", ok=False)
        self.assertEqual(initial, original.read_bytes())
        self.cli("export_sdum_ranking_crops.py", source, self.root / "cropped")
        self.assertEqual(scipy.io.loadmat(self.root / "cropped/Toy.mat")["img4ranking"].shape, (12, 16, 1, 3))


if __name__ == "__main__":
    unittest.main()
