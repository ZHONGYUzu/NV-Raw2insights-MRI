#!/usr/bin/env python3
"""SDUM-022/023: complex coil-image rotation and synthetic acquired-line motion.

The sensitivity pattern embedded in each coil image rotates too. This is NOT
patient motion relative to fixed scanner coils. Original H5/TXT files are read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
import scipy.io

from prepare_sdum_019_rotation_pilot import (
    fft2c, ifft2c, rotate_complex, rotate_real, save_mat_atomic, write_json_atomic,
)

VERSION = 2
METRICS = ('psnr', 'nrmse', 'nmse')


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def rss(coils):
    """Coil axis is axis 0 for one slice: (coil,time,PE,FE)."""
    return np.sqrt(np.sum(np.abs(coils).astype(np.float64) ** 2, axis=0))


def trajectory(mask, angle):
    """Increasing acquired-PE order per frame, with one midpoint step.

    Unacquired lines have zero angle placeholders, but are not acquisition events.
    """
    theta = np.zeros(mask.shape[:2], np.float32)
    events = []
    start = (mask.shape[1] - 20) // 2
    for t in range(mask.shape[0]):
        lines = np.flatnonzero(mask[t, :, 0])
        split = len(lines) // 2
        for ordinal, pe in enumerate(lines):
            value = float(angle if ordinal >= split else 0)
            theta[t, pe] = value
            events.append({'frame': t, 'ordinal': ordinal, 'pe': int(pe),
                           'angle': value, 'acs': bool(start <= pe < start + 20)})
    return theta, events


def mix_lines(k0, ka, theta):
    """Inputs: (slice,coil,time,PE,FE); unacquired entries remain K0 placeholders."""
    return np.where(theta[None, None, :, :, None] != 0, ka, k0)


def paths(root, acc, label, case):
    condition = root / 'conditions' / f'acc{acc}' / label
    return {
        'condition': condition,
        'kspace': condition / 'MultiCoil/Cine/UnderSample_TaskR1' / f'{case}_kspace_full.mat',
        'mask': condition / 'MultiCoil/Cine/Mask_TaskR1' / f'{case}_mask_ktRadial{acc}.mat',
        'descriptor': condition / 'json_input' / f'{case}.json',
        'gt': condition / 'ground_truth' / f'{case}.mat',
        'roi': condition / 'valid_roi' / f'{case}.mat',
        'trajectory': condition / 'trajectories' / f'{case}.json',
        'record': condition / 'records' / f'{case}.json',
    }


def prepare(a):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.case):
        raise ValueError('Invalid case id')
    if a.mode not in ('pose', 'motion') or a.angle not in range(6) or a.acc not in (8, 16, 24):
        raise ValueError('Invalid mode, angle or acceleration')
    if a.normalization not in ('none', 'rss-max'):
        raise ValueError('Choose explicit normalization: none or rss-max')
    source = (a.source / f'{a.case}.h5').resolve(strict=True)
    with h5py.File(source, 'r') as f:
        shape = f['kSpace'].shape
        if len(shape) != 5 or not np.issubdtype(f['kSpace'].dtype, np.complexfloating):
            raise ValueError('Expected complex kSpace (slice,coil,time,PE,FE)')
    nz, nc, nt, pe, fe = shape
    if min(shape) < 1 or pe < 20:
        raise ValueError('Invalid dimensions or insufficient PE for 20 ACS lines')
    txt = a.masks / f'mask_VISTA_{pe}x{nt}_acc{a.acc}_8.txt'
    raw = np.loadtxt(txt, delimiter=',').astype(np.float32)
    if raw.shape != (pe, nt) or not np.all(np.isin(raw, (0, 1))):
        raise ValueError('Invalid source mask')
    mask = raw.T.copy()
    start = (pe - 20) // 2
    mask[:, start:start + 20] = 1
    mask = np.repeat(mask[:, :, None], fe, axis=2)
    theta, events = trajectory(mask, a.angle)
    label = f'rot_p{int(a.angle):03d}' if a.mode == 'pose' else f'step_p{int(a.angle):03d}_t050'
    root = a.root.resolve()
    out = paths(root, a.acc, label, a.case)
    if root == source.parent or source.parent in root.parents or root == a.masks.resolve() or a.masks.resolve() in root.parents:
        raise ValueError('Output must be outside source data and mask directories')
    identity = {'schema_version': VERSION, 'source': file_identity(source),
                'mask_sha256': fingerprint(txt), 'mask_source': str(txt.resolve()),
                'case': a.case, 'mode': a.mode, 'angle': a.angle, 'acc': a.acc,
                'normalization': a.normalization, 'reference': 'full_coil_RSS',
                'shape': list(shape), 'acs_lines': 20, 'mask_seed': 8,
                'generator_sha256': fingerprint(Path(__file__)),
                'rotation_helper_sha256': fingerprint(Path(__file__).with_name('prepare_sdum_019_rotation_pilot.py'))}
    required = ['kspace', 'mask', 'descriptor', 'gt', 'roi', 'trajectory']
    if out['record'].exists():
        record = json.loads(out['record'].read_text())
        if record['identity'] != identity:
            raise ValueError('Existing preparation differs; use a new run root')
        for key in required:
            if not out[key].is_file() or fingerprint(out[key]) != record['sha256'][key]:
                raise ValueError(f'Resume file missing or changed: {key}')
        print('Verified resume', a.case, label, flush=True)
        return out
    if any(out[key].exists() for key in required):
        raise FileExistsError('Partial preparation exists; preserve it and use another run root')
    # First pass: one original full-case RSS reference fixes scale and display.
    reference = np.empty((nz, nt, pe, fe), np.float64)
    with h5py.File(source, 'r') as f:
        for z in range(nz):
            k = f['kSpace'][z].astype(np.complex64)
            if not np.isfinite(k).all():
                raise ValueError('Nonfinite kSpace')
            reference[z] = rss(ifft2c(k))
    original_max = float(reference.max())
    if not np.isfinite(original_max) or original_max <= 0:
        raise ValueError('Zero/nonfinite original RSS reference')
    scale = original_max if a.normalization == 'rss-max' else 1.0
    display_max = float(np.percentile(reference, 99.5) / scale)
    metric_range = original_max / scale
    if display_max <= 0:
        raise ValueError('No positive percentile display window')
    output = np.empty(shape, np.complex64)
    gt = np.empty_like(reference, dtype=np.float32)
    roundtrip_errors = []
    with h5py.File(source, 'r') as f:
        for z in range(nz):
            k0 = f['kSpace'][z].astype(np.complex64) / scale
            coils = ifft2c(k0).astype(np.complex64)
            roundtrip = fft2c(coils)
            norm = np.linalg.norm(k0.ravel())
            err = float(np.linalg.norm((roundtrip - k0).ravel()) / norm) if norm else 0.0
            roundtrip_errors.append(err)
            if err > 1e-5:
                raise ValueError('FFT roundtrip gate failed')
            rotated = rotate_complex(coils, a.angle, 1)
            # Preserve 0-degree source exactly after the declared scale conversion.
            ka = k0 if a.angle == 0 else fft2c(rotated).astype(np.complex64)
            output[z] = ka if a.mode == 'pose' else np.where(theta[None, :, :, None] != 0, ka, k0)
            gt[z] = rss(rotated) if a.mode == 'pose' else reference[z] / scale
    roi = np.ones((pe, fe), bool)
    for angle in range(6):
        roi &= rotate_real(np.ones((pe, fe), np.float32), angle, 1) > .999
    logical = output.transpose(2, 0, 1, 3, 4)
    save_mat_atomic(out['kspace'], {'kspace_full': logical.transpose()}, False)
    save_mat_atomic(out['mask'], {'mask': mask}, True)
    save_mat_atomic(out['gt'], {'gt': gt.transpose(3, 2, 0, 1)}, True)
    save_mat_atomic(out['roi'], {'valid_roi': roi.T.astype(np.uint8)}, True)
    write_json_atomic(out['descriptor'], {'kspace': str(out['kspace']), 'mask': [str(out['mask'])]})
    write_json_atomic(out['trajectory'], {'mode': a.mode, 'events': events if a.mode == 'motion' else [],
                                         'description': 'increasing acquired PE per frame; midpoint step; all slices share angles'})
    record = {'identity': identity, 'scale': scale, 'original_rss_max': original_max,
              'metric_data_range': metric_range, 'display_vmin': 0, 'display_vmax': display_max,
              'fft_roundtrip_relative_l2': roundtrip_errors,
              'effective_acceleration': float(mask.size / mask.sum()),
              'data_semantics': 'full consistent k-space' if a.mode == 'pose' else 'motion carrier; unacquired K0 entries ignored; never a GT',
              'sha256': {key: fingerprint(out[key]) for key in required}}
    write_json_atomic(out['record'], record)
    print('Prepared', a.case, label, 'scale=', scale, 'vmax=', display_max, flush=True)
    return out


def metrics(pred, gt, roi, data_range):
    x = pred[roi].astype(np.float64)
    y = gt[roi].astype(np.float64)
    if not x.size or data_range <= 0:
        raise ValueError('Empty ROI or invalid metric range')
    mse = float(np.mean((x - y) ** 2))
    energy = float(np.mean(y * y))
    return {'psnr': float(20 * np.log10(data_range / np.sqrt(mse))) if mse else float('inf'),
            'nrmse': float(np.sqrt(mse / energy)) if energy else float('nan'),
            'nmse': mse / energy if energy else float('nan')}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(a):
    """Evaluate only explicitly listed case/condition pairs; refuse incomplete runs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not np.isfinite(a.error_fraction) or a.error_fraction <= 0:
        raise ValueError('error-fraction must be positive')
    if not a.accept_rss_reference:
        raise ValueError('Review the zero-degree model/RSS comparison before accepting RSS metrics')
    if not a.cases or len(set(a.cases)) != len(a.cases) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', case) for case in a.cases):
        raise ValueError('Provide unique valid case ids')
    if not a.conditions or len(set(a.conditions)) != len(a.conditions):
        raise ValueError('Provide unique conditions')
    destination = a.root / 'reviews' / a.review_id
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.review_id):
        raise ValueError('Invalid review id')
    if destination.exists():
        raise FileExistsError('Use a new review id; existing reviews are immutable')
    # Validate completeness and shared scale before producing any review files.
    entries = []
    case_identities = {}
    for case in a.cases:
        for label in a.conditions:
            if not re.fullmatch(r'(rot_p00[0-5]|step_p00[0-5]_t050)', label):
                raise ValueError('Invalid condition label')
            out = paths(a.root.resolve(), a.acc, label, case)
            record = json.loads(out['record'].read_text())
            shared = {k: record[k] for k in ('scale', 'original_rss_max', 'display_vmax', 'metric_data_range')}
            shared['source'] = record['identity']['source']
            if case in case_identities and case_identities[case] != shared:
                raise ValueError('Conditions have different scales or source identities')
            case_identities[case] = shared
            for key in ('gt', 'roi'):
                if fingerprint(out[key]) != record['sha256'][key]:
                    raise ValueError(f'Prepared {key} changed')
            predpath = out['condition'] / 'output/val_img4ranking' / f'{case}.mat'
            pred = scipy.io.loadmat(predpath)['img4ranking']
            gt = scipy.io.loadmat(out['gt'])['gt']
            if pred.shape != gt.shape or pred.ndim != 4 or not np.isfinite(pred).all() or not np.isfinite(gt).all():
                raise ValueError(f'Invalid prediction/reference: {predpath}')
            entries.append((case, label, out, record, fingerprint(predpath)))
    destination.mkdir(parents=True)
    rows, subjects, windows = [], [], []
    for case, label, out, record, pred_hash in entries:
        pred = scipy.io.loadmat(out['condition'] / 'output/val_img4ranking' / f'{case}.mat')['img4ranking']
        gt = scipy.io.loadmat(out['gt'])['gt']
        roi = scipy.io.loadmat(out['roi'])['valid_roi'].astype(bool)
        if roi.shape != gt.shape[:2]:
            raise ValueError('ROI shape mismatch')
        vmax = record['display_vmax']
        error_max = a.error_fraction * record['metric_data_range']
        windows.append({'case': case, 'condition': label, 'vmin': 0, 'vmax': vmax,
                        'error_vmax': error_max, 'prediction_sha256': pred_hash,
                        'gt_saturated_fraction': float(np.mean(gt > vmax)),
                        'recon_saturated_fraction': float(np.mean(pred > vmax)),
                        'error_saturated_fraction': float(np.mean(np.abs(pred - gt) > error_max))})
        for scope, region in [('full', np.ones(gt.shape[:2], bool)), ('common_support', roi)]:
            current = []
            for z in range(gt.shape[2]):
                for t in range(gt.shape[3]):
                    value = metrics(pred[:, :, z, t], gt[:, :, z, t], region, record['metric_data_range'])
                    current.append(value)
                    rows.append(dict(acc=a.acc, condition=label, case=case, scope=scope, slice=z, frame=t, **value))
            subjects.append(dict(acc=a.acc, condition=label, case=case, scope=scope,
                                 **{k: float(np.mean([r[k] for r in current])) for k in METRICS}))
        dest = destination / 'figures' / case / label
        dest.mkdir(parents=True)
        # One slice per grid; every time frame included, no ranking crop.
        for z in range(gt.shape[2]):
            nt = gt.shape[3]
            for first in range(0, nt, 5):
                times = list(range(first, min(first + 5, nt)))
                fig, axes = plt.subplots(3, len(times), figsize=(3 * len(times), 8), squeeze=False, layout='constrained')
                for col, t in enumerate(times):
                    images = [gt[:, :, z, t].T, pred[:, :, z, t].T, np.abs(pred[:, :, z, t] - gt[:, :, z, t]).T]
                    for row, (image, name, limit, cmap) in enumerate(zip(images, ['RSS reference', 'Recon', 'Absolute error'], [vmax, vmax, error_max], ['gray', 'gray', 'magma'])):
                        im = axes[row, col].imshow(image, origin='lower', cmap=cmap, vmin=0, vmax=limit, interpolation='nearest', aspect='equal')
                        axes[row, col].set_title(f'{name}, frame {t}', fontsize=9)
                        axes[row, col].axis('off')
                        if col == len(times) - 1:
                            fig.colorbar(im, ax=axes[row, :].tolist(), shrink=.7)
                fig.suptitle(f'{case} acc{a.acc} {label}, slice index {z}; no independent rescaling')
                fig.savefig(dest / f'slice_{z:03d}_frames_{times[0]:03d}-{times[-1]:03d}.png', dpi=140)
                plt.close(fig)
    write_csv(destination / 'frame_metrics.csv', rows)
    write_csv(destination / 'subject_metrics.csv', subjects)
    summary = []
    for label in a.conditions:
        for scope in ('full', 'common_support'):
            selected = [r for r in subjects if r['condition'] == label and r['scope'] == scope]
            summary.append(dict(acc=a.acc, condition=label, scope=scope, cases=len(selected),
                                **{k: float(np.mean([r[k] for r in selected])) for k in METRICS}))
    write_csv(destination / 'summary_metrics.csv', summary)
    write_json_atomic(destination / 'display_settings.json', {'error_fraction': a.error_fraction, 'windows': windows})
    write_json_atomic(destination / 'complete.json', {'cases': a.cases, 'conditions': a.conditions,
                                                     'reference': 'RSS explicitly accepted for this review', 'frame_scope_rows': len(rows)})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--masks', type=Path, required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--acc', type=int, choices=[8, 16, 24], default=8)
    p.add_argument('--angle', type=int, choices=range(6), required=True)
    p.add_argument('--mode', choices=['pose', 'motion'], required=True)
    p.add_argument('--normalization', choices=['none', 'rss-max'], required=True)
    e = sub.add_parser('evaluate')
    e.add_argument('--root', type=Path, required=True)
    e.add_argument('--acc', type=int, choices=[8, 16, 24], required=True)
    e.add_argument('--cases', nargs='+', required=True)
    e.add_argument('--conditions', nargs='+', required=True)
    e.add_argument('--review-id', required=True)
    e.add_argument('--error-fraction', type=float, required=True, help='Error display maximum as fraction of original reference range; choose once across runs')
    e.add_argument('--accept-rss-reference', action='store_true', help='Explicitly acknowledge completed zero-degree output/reference review')
    a = parser.parse_args()
    prepare(a) if a.action == 'prepare' else evaluate(a)


if __name__ == '__main__':
    main()
