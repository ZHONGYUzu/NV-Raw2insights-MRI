#!/bin/bash

# sdum-019: two-case, four-condition in-plane rotation stability pilot.
# Historical 0/-10/+10/180-degree protocol, not the future 0-5-degree motion study.
# New preparation code refuses legacy runs without preparation_config.json.
# Preserve old results; do not rerun this fixed-path wrapper on an old run.

#SBATCH --job-name=sdum019-rotation
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/students/studxuzho1/NV-Raw2insights-MRI/runs/sdum-019/logs/pipeline-%j.out
#SBATCH --error=/home/students/studxuzho1/NV-Raw2insights-MRI/runs/sdum-019/logs/pipeline-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=/home/students/studxuzho1/hf_cache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

REPO_ROOT=${REPO_ROOT:-/home/students/studxuzho1/NV-Raw2insights-MRI}
PYTHON_BIN=${PYTHON_BIN:-/home/students/studxuzho1/envs/nv-raw2insights-mri/bin/python}
RUN_ID=sdum-019
RUN_DIR="$REPO_ROOT/runs/$RUN_ID"
CONFIG="$REPO_ROOT/configs/nv_raw2insights_mri_base.json"
EXPECTED_CASES=2
CONDITIONS=(rot_000 rot_m010 rot_p010 rot_180)
START_TIME=$(date --iso-8601=seconds)

cd "$REPO_ROOT"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/metrics" "$RUN_DIR/figures"

record_slurm_state() {
    local state=$1
    local finish_time
    finish_time=$(date --iso-8601=seconds)
    "$PYTHON_BIN" -c 'import json, os, pathlib, sys; path=pathlib.Path(sys.argv[1]); json.dump({"experiment_id":"sdum-019","job_id":os.environ.get("SLURM_JOB_ID"),"array_task_id":os.environ.get("SLURM_ARRAY_TASK_ID"),"node":os.environ.get("SLURMD_NODENAME") or os.environ.get("HOSTNAME"),"partition":os.environ.get("SLURM_JOB_PARTITION"),"gpu_ids":os.environ.get("CUDA_VISIBLE_DEVICES"),"cpus":os.environ.get("SLURM_CPUS_PER_TASK"),"memory_per_node_mb":os.environ.get("SLURM_MEM_PER_NODE"),"start":sys.argv[2],"last_update":sys.argv[3],"state":sys.argv[4]}, path.open("w"), indent=2); path.open("a").write("\n")' "$RUN_DIR/slurm.json" "$START_TIME" "$finish_time" "$state"
}

on_exit() {
    local code=$?
    if [[ "$code" -eq 0 ]]; then
        record_slurm_state completed
    else
        record_slurm_state "failed_exit_${code}"
    fi
}
trap on_exit EXIT

record_slurm_state running
"$PYTHON_BIN" -c 'import json, pathlib, subprocess; root=pathlib.Path.cwd(); payload={"commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"branch":subprocess.check_output(["git","branch","--show-current"],text=True).strip(),"status":subprocess.check_output(["git","status","--porcelain"],text=True).splitlines()}; p=pathlib.Path("runs/sdum-019/git_state.json"); json.dump(payload,p.open("w"),indent=2); p.open("a").write("\n")'
{
    echo "timestamp=$START_TIME"
    echo "host=$(hostname)"
    echo "python=$($PYTHON_BIN -V 2>&1)"
    echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
    echo "slurm_partition=${SLURM_JOB_PARTITION:-unknown}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unknown}"
    "$PYTHON_BIN" -c 'import h5py,numpy,scipy,skimage,torch; print("numpy="+numpy.__version__); print("scipy="+scipy.__version__); print("h5py="+h5py.__version__); print("skimage="+skimage.__version__); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda)); print("cuda_available="+str(torch.cuda.is_available()))'
} > "$RUN_DIR/environment.txt"

echo "Experiment: $RUN_ID"
echo "Start: $START_TIME"
echo "Node: $(hostname)"
echo "Run root: $RUN_DIR"
echo "Cases: Sub0014 Sub0047"
echo "Conditions: ${CONDITIONS[*]}"
nvidia-smi

"$PYTHON_BIN" scripts/prepare_sdum_019_rotation_pilot.py \
    --run-root "$RUN_DIR" \
    --experiment-id "$RUN_ID" \
    --raw-h5-root /mnt/qdata/rawdata/CINE/2D_h5_compressed \
    --source-json-root dataset/CINE_test_acc8/json_input \
    --cases Sub0014 Sub0047 \
    --interpolation-order 1 \
    --resume

for condition in "${CONDITIONS[@]}"; do
    condition_root="$RUN_DIR/conditions/$condition"
    input_dir="$condition_root/json_input"
    output_dir="$condition_root/output"
    echo "Validating $condition"
    "$PYTHON_BIN" scripts/validate_cine_inference_data.py "$input_dir"
    echo "Running inference for $condition"
    "$PYTHON_BIN" scripts/inference.py \
        -c "$CONFIG" \
        -i "$input_dir" \
        -o "$output_dir" \
        --fixed-mask-types mask_ktRadial8 \
        --num-workers 0 \
        --profile-timing
    prediction_count=$(find "$output_dir/val_img4ranking" -maxdepth 1 -type f -name 'Sub*.mat' | wc -l)
    if [[ "$prediction_count" -ne "$EXPECTED_CASES" ]]; then
        echo "$condition: expected $EXPECTED_CASES predictions, found $prediction_count" >&2
        exit 1
    fi
done

"$PYTHON_BIN" scripts/evaluate_sdum_019_rotation_pilot.py --run-root "$RUN_DIR"

"$PYTHON_BIN" -c 'import json,pathlib; root=pathlib.Path("runs/sdum-019"); files=[]; patterns=("manifest.json","config.yaml","git_state.json","environment.txt","slurm.json","conditions/*/output/config.json","conditions/*/output/val_img4ranking/*.mat","conditions/*/ground_truth/*.mat","conditions/*/valid_roi/*.mat","metrics/*","figures/**/*.png"); seen=set(); [(seen.add(p),files.append({"path":str(p.relative_to(root)),"size_bytes":p.stat().st_size})) for pattern in patterns for p in root.glob(pattern) if p.is_file() and p not in seen]; out=root/"artifacts.json"; json.dump({"experiment_id":"sdum-019","files":files},out.open("w"),indent=2); out.open("a").write("\n")'

echo "Completed: $(date --iso-8601=seconds)"
