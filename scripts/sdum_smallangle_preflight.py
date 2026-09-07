"""Read-only source audit for SDUM small-angle fixed-scanner-coil simulation."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from prepare_sdum_019_rotation_pilot import fft2c, ifft2c, write_json_atomic


def audit(path):
    rows = []
    with h5py.File(path, 'r') as f:
        shapes = {key: list(f[key].shape) for key in ('kSpace', 'dMap', 'dImgC')}
        for z in range(shapes['kSpace'][0]):
            k = f['kSpace'][z].astype(np.complex64)
            coil = ifft2c(k)
            maps = f['dMap'][z].astype(np.complex64)
            obj = f['dImgC'][z].astype(np.complex64)
            synth = maps * obj
            alpha = np.vdot(synth, coil) / np.vdot(synth, synth)
            err = np.linalg.norm(coil - alpha * synth) / np.linalg.norm(coil)
            power = np.sum(np.abs(maps)**2, axis=0, keepdims=True)
            sense = np.sum(np.conj(maps)*coil, axis=0, keepdims=True) / np.maximum(power, 1e-12)
            projected = maps*sense
            residual = np.linalg.norm(coil-projected)/np.linalg.norm(coil)
            rows.append(dict(slice=z, dimgc_fit_relative_l2=float(err),
                             scale_real=float(alpha.real), scale_imag=float(alpha.imag),
                             sense_projection_relative_l2=float(residual),
                             map_power_min=float(power.min()), map_power_max=float(power.max()),
                             fft_roundtrip_relative_l2=float(np.linalg.norm(fft2c(coil)-k)/np.linalg.norm(k))))
            print(path.stem, rows[-1], flush=True)
    return dict(case=path.stem, shapes=shapes, slices=rows)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', nargs='+', default=['Sub0014','Sub0047'])
    a=p.parse_args()
    results=[audit(a.source / (case+'.h5')) for case in a.cases]
    write_json_atomic(a.output, {'status':'audited_requires_review', 'cases':results})
