#!/bin/bash
#SBATCH --job-name=sdum-smallangle
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# CPU preparation by default. For ACTION=infer, explicitly submit --gres=gpu:1.
# Supply absolute --output/--error paths under your existing run logs directory.
set -euo pipefail
: "${REPO_ROOT:?Set the CINE code worktree}"
: "${RUN_ROOT:?Set sdum-022 or sdum-023 run_001/pilot}"
: "${EXPECTED_COMMIT:?Set the reviewed commit SHA}"
: "${MODE:?Set pose or motion}"
: "${NORMALIZATION:?Set none or rss-max consistently across both experiments}"
: "${ACTION:?Set prepare or infer}"
: "${ANGLES:?Set explicit angles, e.g. 0 or 0 1 3 5}"
PYTHON_BIN=${PYTHON_BIN:-/home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python}
CASE_IDS=${CASE_IDS:-Sub0014 Sub0047}
ACC=8
cd "$REPO_ROOT"
[[ $(git rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || { echo 'Commit mismatch'; exit 1; }
[[ -z $(git status --porcelain --untracked-files=normal) ]] || { echo 'Code worktree must be clean'; exit 1; }
[[ "$MODE" == pose || "$MODE" == motion ]] || exit 1
[[ "$ACTION" == prepare || "$ACTION" == infer ]] || exit 1
if [[ "$ACTION" == prepare ]]; then
  : "${SOURCE_H5_ROOT:?Set read-only raw CINE H5 root}"
  : "${SOURCE_MASK_ROOT:?Set read-only VISTA mask root}"
else
  : "${MODEL_CKPT:?Set the reviewed Base checkpoint file}"
  [[ -f "$MODEL_CKPT" ]] || exit 1
  [[ -n ${CUDA_VISIBLE_DEVICES:-} ]] || { echo 'Submit inference with --gres=gpu:1'; exit 1; }
fi
mkdir -p "$RUN_ROOT/provenance"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/sdum-smallangle-mpl-${SLURM_JOB_ID}"
"$PYTHON_BIN" scripts/sdum_detect_resources.py -o "$RUN_ROOT/provenance/resources-${SLURM_JOB_ID}.json"
git rev-parse HEAD > "$RUN_ROOT/provenance/commit-${SLURM_JOB_ID}.txt"
printf 'mode=%s\naction=%s\nangles=%s\ncases=%s\nnormalization=%s\n' "$MODE" "$ACTION" "$ANGLES" "$CASE_IDS" "$NORMALIZATION" > "$RUN_ROOT/provenance/parameters-${SLURM_JOB_ID}.txt"
if [[ "$ACTION" == infer ]]; then
  sha256sum "$MODEL_CKPT" > "$RUN_ROOT/provenance/checkpoint-${SLURM_JOB_ID}.sha256"
fi
for angle in $ANGLES; do
  [[ "$angle" =~ ^[0-5]$ ]] || { echo 'Angles must be integers 0..5'; exit 1; }
  if [[ "$MODE" == pose ]]; then
    printf -v label 'rot_p%03d' "$angle"
  else
    printf -v label 'step_p%03d_t050' "$angle"
  fi
  for case_id in $CASE_IDS; do
    if [[ "$ACTION" == prepare ]]; then
      "$PYTHON_BIN" scripts/sdum_smallangle.py prepare --root "$RUN_ROOT" \
        --source "$SOURCE_H5_ROOT" --masks "$SOURCE_MASK_ROOT" --case "$case_id" \
        --acc "$ACC" --angle "$angle" --mode "$MODE" --normalization "$NORMALIZATION"
    fi
  done
  condition="$RUN_ROOT/conditions/acc${ACC}/$label"
  "$PYTHON_BIN" scripts/validate_cine_inference_data.py "$condition/json_input"
  if [[ "$ACTION" == infer ]]; then
    # Prevent silently mixing cases or normalization modes from stale inputs.
    "$PYTHON_BIN" - "$condition" "$NORMALIZATION" "$CASE_IDS" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'scripts'))
from sdum_smallangle import fingerprint, paths
root=Path(sys.argv[1]); expected=set(sys.argv[3].split())
assert {p.stem for p in (root/'json_input').glob('*.json')} == expected
for case in expected:
    record=json.loads((root/'records'/f'{case}.json').read_text())
    assert record['identity']['schema_version']==2
    assert record['identity']['normalization']==sys.argv[2]
    artifacts=paths(root.parents[2], 8, root.name, case)
    for key, digest in record['sha256'].items():
        assert fingerprint(artifacts[key]) == digest, f'Changed artifact: {case} {key}'
# Existing predictions require a separate reviewed resume policy; never mix checkpoints.
assert not any((root/'output/val_img4ranking').glob('*.mat')), 'Use a new run root for a rerun'
PY
    "$PYTHON_BIN" scripts/inference.py -c configs/nv_raw2insights_mri_base.json \
      -m "$MODEL_CKPT" -i "$condition/json_input" -o "$condition/output" \
      --fixed-mask-types mask_ktRadial8 --num-workers 0 --profile-timing
  fi
done
