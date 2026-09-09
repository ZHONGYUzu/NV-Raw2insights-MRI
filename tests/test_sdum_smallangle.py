"""Small synthetic tests; no real MRI files, network, or model execution."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from sdum_smallangle import prepare, trajectory, mix_lines, metrics, evaluate, fingerprint
from prepare_sdum_019_rotation_pilot import fft2c, ifft2c
from validate_cine_inference_data import logical_kspace, read_mat, validate_case


class SmallAngleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / 'raw'; source.mkdir()
        masks = self.root / 'masks'; masks.mkdir()
        rng = np.random.default_rng(9)
        # Two slices, three complex coils, two frames, non-square spatial shape.
        coils = (rng.normal(size=(2, 3, 2, 24, 26)) + 1j * rng.normal(size=(2, 3, 2, 24, 26))).astype(np.complex64)
        self.kspace = fft2c(coils).astype(np.complex64)
        with h5py.File(source / 'SubTest.h5', 'w') as f:
            f['kSpace'] = self.kspace
            # Deliberately incompatible diagnostics: these must not be used.
            f['dImgC'] = np.zeros((1, 1, 1, 1, 1))
            f['dMap'] = np.zeros((1, 1, 1, 1, 1))
        self.raw_hash = fingerprint(source / 'SubTest.h5')
        rawmask = np.zeros((24, 2)); rawmask[::3] = 1
        np.savetxt(masks / 'mask_VISTA_24x2_acc8_8.txt', rawmask, delimiter=',')
        self.args = SimpleNamespace(root=self.root / 'run', source=source, masks=masks,
                                    case='SubTest', acc=8, angle=0, mode='pose', normalization='none')

    def read_k(self, out):
        return logical_kspace(read_mat(out['kspace']), out['kspace'])

    def test_zero_preserves_raw_kspace_and_full_dimensions(self):
        out = prepare(self.args)
        np.testing.assert_array_equal(self.read_k(out), self.kspace.transpose(2, 0, 1, 3, 4))
        gt = scipy.io.loadmat(out['gt'])['gt']
        expected = np.sqrt(np.sum(np.abs(ifft2c(self.kspace)) ** 2, axis=1)).transpose(3, 2, 0, 1)
        np.testing.assert_allclose(gt, expected, rtol=4e-7, atol=4e-7)
        self.assertEqual(gt.shape, (26, 24, 2, 2))
        self.assertIn('k-space (2, 2, 3, 24, 26)', validate_case(out['descriptor'], True))
        self.assertEqual(fingerprint(self.args.source / 'SubTest.h5'), self.raw_hash)
        prepare(self.args)

    def test_zero_motion_equals_fixed_zero(self):
        fixed = prepare(self.args)
        self.args.mode = 'motion'
        moving = prepare(self.args)
        np.testing.assert_array_equal(self.read_k(fixed), self.read_k(moving))
        np.testing.assert_allclose(scipy.io.loadmat(fixed['gt'])['gt'], scipy.io.loadmat(moving['gt'])['gt'], rtol=4e-7)

    def test_scaling_is_common_to_kspace_and_reference(self):
        raw = prepare(self.args)
        self.args.root = self.root / 'normalized'
        self.args.normalization = 'rss-max'
        normalized = prepare(self.args)
        record = json.loads(normalized['record'].read_text())
        scale = record['scale']
        np.testing.assert_allclose(self.read_k(normalized) * scale, self.read_k(raw), rtol=3e-7, atol=3e-7)
        np.testing.assert_allclose(scipy.io.loadmat(normalized['gt'])['gt'] * scale, scipy.io.loadmat(raw['gt'])['gt'], rtol=5e-7, atol=1e-6)
        self.assertAlmostEqual(record['metric_data_range'], 1)

    def test_acquired_line_assignment_and_acs(self):
        mask = np.zeros((2, 24, 26)); mask[:, ::2] = 1
        theta, events = trajectory(mask, 3)
        self.assertEqual(len(events), 24)
        for t in range(2):
            current = [e for e in events if e['frame'] == t]
            self.assertEqual([e['pe'] for e in current], list(range(0, 24, 2)))
            self.assertEqual([e['angle'] for e in current], [0] * 6 + [3] * 6)
        mixed = mix_lines(self.kspace, self.kspace * 2, theta)
        np.testing.assert_array_equal(mixed[:, :, :, 12::2], self.kspace[:, :, :, 12::2] * 2)
        np.testing.assert_array_equal(mixed[:, :, :, :12:2], self.kspace[:, :, :, :12:2])

    def test_nonzero_motion_uses_rotated_lines_and_unrotated_reference(self):
        fixed = prepare(self.args)
        self.args.angle = 3
        rotated = prepare(self.args)
        self.args.mode = 'motion'
        moving = prepare(self.args)
        mask = scipy.io.loadmat(moving['mask'])['mask']
        theta, _ = trajectory(mask, 3)
        k0 = self.read_k(fixed).transpose(1, 2, 0, 3, 4)
        ka = self.read_k(rotated).transpose(1, 2, 0, 3, 4)
        np.testing.assert_array_equal(self.read_k(moving), mix_lines(k0, ka, theta).transpose(2, 0, 1, 3, 4))
        np.testing.assert_allclose(scipy.io.loadmat(moving['gt'])['gt'], scipy.io.loadmat(fixed['gt'])['gt'], rtol=4e-7)
        self.assertEqual(json.loads(fixed['record'].read_text())['display_vmax'], json.loads(rotated['record'].read_text())['display_vmax'])

    def test_resume_refuses_missing_trajectory_and_parameter_change(self):
        out = prepare(self.args)
        self.args.normalization = 'rss-max'
        with self.assertRaises(ValueError): prepare(self.args)
        self.args.normalization = 'none'
        out['trajectory'].unlink()
        with self.assertRaises(ValueError): prepare(self.args)

    def test_metrics_do_not_hide_global_brightness_error(self):
        gt = np.arange(1, 17).reshape(4, 4).astype(float)
        roi = np.ones((4, 4), bool)
        value = metrics(gt * .5, gt, roi, 16)
        self.assertAlmostEqual(value['nmse'], .25)
        normalized = metrics(gt * .5 / 16, gt / 16, roi, 1)
        for key in value: self.assertAlmostEqual(value[key], normalized[key])

    def test_evaluation_requires_complete_pairs_and_renders_all_frames(self):
        out = prepare(self.args)
        args = SimpleNamespace(root=self.args.root, acc=8, cases=['SubTest'], conditions=['rot_p000'],
                               review_id='test', error_fraction=.2, accept_rss_reference=True)
        with self.assertRaises((FileNotFoundError, OSError)): evaluate(args)
        pred_dir = out['condition'] / 'output/val_img4ranking'; pred_dir.mkdir(parents=True)
        gt = scipy.io.loadmat(out['gt'])['gt']
        scipy.io.savemat(pred_dir / 'SubTest.mat', {'img4ranking': gt * .9})
        review = evaluate(args)
        self.assertEqual(len(list((review / 'figures').rglob('*.png'))), 2)
        self.assertEqual(json.loads((review / 'complete.json').read_text())['frame_scope_rows'], 8)
        windows = json.loads((review / 'display_settings.json').read_text())['windows'][0]
        self.assertEqual(windows['vmax'], json.loads(out['record'].read_text())['display_vmax'])
        with self.assertRaises(FileExistsError): evaluate(args)


if __name__ == '__main__': unittest.main()
