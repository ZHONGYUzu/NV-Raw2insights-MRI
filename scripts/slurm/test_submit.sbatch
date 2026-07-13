#!/bin/bash

#SBATCH --job-name=slurm_smoke_test
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=slurm_smoke_test_%j.out
#SBATCH --error=slurm_smoke_test_%j.err

set -euo pipefail

echo "SLURM submission works."
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Started: $(date --iso-8601=seconds)"

python --version
python -c 'print("Python execution works.")'

echo "Finished: $(date --iso-8601=seconds)"
