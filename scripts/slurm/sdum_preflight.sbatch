#!/bin/bash
#SBATCH --job-name=sdum022-023-qc
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/students/studxuzho1/NV-Raw2insights-MRI-experiments/shared/sdum-022-023/preflight-%j.out
#SBATCH --error=/home/students/studxuzho1/NV-Raw2insights-MRI-experiments/shared/sdum-022-023/preflight-%j.err
set -euo pipefail
cd /home/students/studxuzho1/NV-Raw2insights-MRI-cine
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
PY=/home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python
ROOT=/home/students/studxuzho1/NV-Raw2insights-MRI-experiments/shared/sdum-022-023
"$PY" scripts/sdum_detect_resources.py -o "$ROOT/resources.json"
"$PY" scripts/sdum_smallangle_preflight.py --source /mnt/qdata/rawdata/CINE/2D_h5_compressed --output "$ROOT/forward_model_qc.json"
