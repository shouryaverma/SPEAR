#!/bin/bash
# Build the inductor and triton compiler caches once so that experiment jobs start from a
# compiled state. run_experiments.sbatch copies this cache into a private directory per job
# when it exists, so jobs never share or write to it.
#
# Run once on an interactive GPU node, with no other compiled job running concurrently:
#     bash warm_inductor_cache.sh [data] [split] [fold]
# Defaults: norman additive 0. The compiled graphs depend on the model flags and batch size,
# not on the dataset contents, so one dataset is enough for jobs that use the same flags.
#
# Environment variables (all optional):
#     CONDA_ENV     conda environment name or prefix (default: the name in environment.yml;
#                   if neither is available, the already active Python environment is used)
#     CONDA_MODULE  environment module that provides conda, loaded only when conda is not on
#                   PATH (default: conda)
#     BATCH         training batch size (default: 48, matching run_experiments.sbatch)
#     WARM_CACHE    output cache directory (default: .inductor_warm)
#
# The log is written to <repo_root>/logs/warm_inductor_cache.log.

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_DIR" || exit 1
mkdir -p logs
exec > >(tee -a logs/warm_inductor_cache.log) 2>&1

DATA=${1:-norman}
SPLIT=${2:-additive}
FOLD=${3:-0}
BATCH=${BATCH:-48}

activate_env() {
  local name=${CONDA_ENV:-}
  if [ -z "$name" ] && [ -f environment.yml ]; then
    name=$(sed -n 's/^name:[[:space:]]*//p' environment.yml | head -n1)
  fi
  if ! command -v conda >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load "${CONDA_MODULE:-conda}" 2>/dev/null
  fi
  if [ -n "$name" ] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$name" || return 1
  fi
  return 0
}
activate_env || { echo "ABORT: could not activate the conda environment"; exit 1; }
set -eo pipefail

if [ ! -e data ]; then
  echo "ABORT: $REPO_DIR/data not found; link or copy the datasets there"
  exit 1
fi

export PYTHONPATH=.
export HDF5_USE_FILE_LOCKING=FALSE
export SPEAR_COMPILE=1

WARM_CACHE=${WARM_CACHE:-$REPO_DIR/.inductor_warm}
WARM_RESULT=$REPO_DIR/result/warm
WARM_TMP=$REPO_DIR/.cache/warm_tmp

rm -rf "${WARM_CACHE:?}" "$WARM_RESULT" "$WARM_TMP"
mkdir -p "$WARM_CACHE/inductor" "$WARM_CACHE/triton" "$WARM_TMP"
export TORCHINDUCTOR_CACHE_DIR="$WARM_CACHE/inductor"
export TRITON_CACHE_DIR="$WARM_CACHE/triton"
export TMPDIR="$WARM_TMP"
trap 'rm -rf "$WARM_RESULT" "$WARM_TMP"' EXIT

PERT_FN=crisper
case "$DATA" in combosciplex|parse_pbmc) PERT_FN=drug ;; esac

COMMON=(
  --data_name="$DATA" --model_type=origin --d_model=128 --lr=5e-5 --eta_min=1e-6
  --fusion_method=differential_perceiver --split_method="$SPLIT" --fold="$FOLD"
  --infer_top_gene=1000 --n_top_genes=5000 --topk=30 --neighbor_cap=128
  --perturbation_function="$PERT_FN" --noise_type=Gaussian --mode=predict_y
  --deg_epsilon=0.1 --sinkhorn_eps=0.1 --sinkhorn_iters=50
  --precision_epsilon=0.1 --n_slices=128 --sliced_top_frac=0.1 --sparse_lambda=0.1
  --use_negative_edge --use_repo --use_wire
)

SAMS_FLAGS=(
  --use_samsvae --samsvae_mask_prior=concrete --samsvae_latent_dim=64
  --samsvae_lambda_mask=1e-4 --samsvae_lr=1e-3 --samsvae_warmup_steps=500
)

run_warm() {   # steps print_every [extra flags]
  local steps=$1 print_every=$2
  shift 2
  python src/script/run.py "${COMMON[@]}" \
    --steps="$steps" --print_every="$print_every" --batch_size="$BATCH" \
    --result_path="$WARM_RESULT" --endpoint_loss=deg_sinkhorn --gamma=0.1 --seed=42 "$@"
}

# Two graph variants exist in the matrix: SAMS-VAE off and on. SAMS-VAE indexes genes by vocab
# token, so it is compiled only for gene-perturbation datasets.
echo "=== warming: baseline ($DATA $SPLIT fold $FOLD) ==="
run_warm 8 1000000
case "$DATA" in
  norman|replogle)
    echo "=== warming: samsvae ==="
    run_warm 8 1000000 "${SAMS_FLAGS[@]}"
    ;;
esac

# The evaluation graph runs at a different batch size; compile it so the first checkpoint
# evaluation of a job is not a stall.
echo "=== warming: evaluation graph ==="
run_warm 2 1

echo "warm cache ready: $(du -sh "$WARM_CACHE" | cut -f1) at $WARM_CACHE"
