#!/bin/bash
#SBATCH -A pccr
#SBATCH -N 1
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=14
#SBATCH --mail-user=verma198@purdue.edu
#SBATCH --mail-type=FAIL
#SBATCH --job-name=spear_exp
#SBATCH --array=0-16
#SBATCH --output=logs/%x_%A_%a.out

# Single entry point for every experiment. Submit from the repository root.
#
#   SAMS-VAE suite (tasks 0-16, the default array above):
#     cd /depot/natallah/data/shourya/SPEAR_CAUSAL/SPEAR && mkdir -p logs
#     JID=$(sbatch --parsable run_experiments.sbatch)
#
#   Analysis after the SAMS-VAE suite finishes (writes logs/samsvae_analysis_all.log):
#     sbatch --dependency=afterok:$JID --export=ALL,MODE=analyze --array=0 run_experiments.sbatch
#
#   Everything, including the endpoint-loss matrix on all four datasets (tasks 0-106):
#     sbatch --array=0-106 run_experiments.sbatch
#   Only the matrix:  sbatch --array=17-106 run_experiments.sbatch
#   Job count:        bash run_experiments.sbatch count
#
# Task table (index -> job) is built below, in order. A finished run is skipped, an
# interrupted or preempted run resumes from its latest checkpoint, and a run that nears the
# 24 h limit requeues itself.
#
#   0-4    base                                  norman additive fold 1, seeds 42-46
#   5-9    samsvae concrete L64 lambda 1e-4      seeds 42-46
#   10-12  samsvae l1 L64 lambda 1e-4            seeds 42-44
#   13-16  samsvae concrete: lambda 1e-5, lambda 1e-3, L32, L128   seed 42
#   17-106 endpoint-loss matrix (7 losses x 3 seeds x 4 datasets) plus SAMS-VAE arms on
#          norman and replogle, 600k steps, legacy result layout

REPO_DIR=${REPO_DIR:-/depot/natallah/data/shourya/SPEAR_CAUSAL/SPEAR}
CONDA_ENV=${CONDA_ENV:-/depot/natallah/data/shourya/scDFM_env}
GAMMA=0.1

SAMS_ROOT=./result/samsvae_norman_additive_f1
SAMS_STEPS=600000
MATRIX_STEPS=600000

# ---------------------------------------------------------------------------------------
# Job table. Fields: tag | result_path | data | split | fold | steps | loss | seed | extra
# ---------------------------------------------------------------------------------------
JOBS=()
add_job() { JOBS+=("$1|$2|$3|$4|$5|$6|$7|$8|${9:-}"); }

sams_args() {   # prior latent_dim lambda
  echo "--use_samsvae --samsvae_mask_prior=$1 --samsvae_latent_dim=$2 --samsvae_lambda_mask=$3 --samsvae_lr=1e-3 --samsvae_warmup_steps=500"
}

for seed in 42 43 44 45 46; do
  add_job "base_s${seed}" "${SAMS_ROOT}/base_s${seed}" norman additive 1 $SAMS_STEPS deg_sinkhorn_sparse $seed ""
done
for seed in 42 43 44 45 46; do
  add_job "samsvae_concrete_L64_lm1e-4_s${seed}" "${SAMS_ROOT}/samsvae_concrete_L64_lm1e-4_s${seed}" \
    norman additive 1 $SAMS_STEPS deg_sinkhorn_sparse $seed "$(sams_args concrete 64 1e-4)"
done
for seed in 42 43 44; do
  add_job "samsvae_l1_L64_lm1e-4_s${seed}" "${SAMS_ROOT}/samsvae_l1_L64_lm1e-4_s${seed}" \
    norman additive 1 $SAMS_STEPS deg_sinkhorn_sparse $seed "$(sams_args l1 64 1e-4)"
done
add_job "samsvae_concrete_L64_lm1e-5_s42" "${SAMS_ROOT}/samsvae_concrete_L64_lm1e-5_s42" \
  norman additive 1 $SAMS_STEPS deg_sinkhorn_sparse 42 "$(sams_args concrete 64 1e-5)"
add_job "samsvae_concrete_L64_lm1e-3_s42" "${SAMS_ROOT}/samsvae_concrete_L64_lm1e-3_s42" \
  norman additive 1 $SAMS_STEPS deg_sinkhorn_sparse 42 "$(sams_args concrete 64 1e-3)"
add_job "samsvae_concrete_L32_lm1e-4_s42" "${SAMS_ROOT}/samsvae_concrete_L32_lm1e-4_s42" \
  norman additive 1 $SAMS_STEPS deg_sinkhorn_sparse 42 "$(sams_args concrete 32 1e-4)"
add_job "samsvae_concrete_L128_lm1e-4_s42" "${SAMS_ROOT}/samsvae_concrete_L128_lm1e-4_s42" \
  norman additive 1 $SAMS_STEPS deg_sinkhorn_sparse 42 "$(sams_args concrete 128 1e-4)"

MATRIX_LOSSES=(mmd deg_mmd sinkhorn deg_sinkhorn max_sliced deg_max_sliced deg_sinkhorn_sparse)
MATRIX_SEEDS=(42 43 44)
for spec in norman:additive replogle:additive combosciplex:additive parse_pbmc:state_pbmc; do
  data=${spec%%:*}
  split=${spec##*:}
  root="./result/causal_matrix_${data}_${split}_f0"
  for loss in "${MATRIX_LOSSES[@]}"; do
    for seed in "${MATRIX_SEEDS[@]}"; do
      add_job "off_${loss}_s${seed}" "$root" "$data" "$split" 0 $MATRIX_STEPS "$loss" "$seed" ""
    done
  done
  # SAMS-VAE indexes genes by vocab token, so it only applies to gene-perturbation datasets.
  case "$data" in
    norman|replogle)
      for seed in "${MATRIX_SEEDS[@]}"; do
        add_job "causal_deg_sinkhorn_sparse_s${seed}" "${root}_samsvae" "$data" "$split" 0 $MATRIX_STEPS \
          deg_sinkhorn_sparse "$seed" "$(sams_args concrete 64 1e-4)"
      done
      ;;
  esac
done

for seed in 42 43 44; do
  add_job "noendpoint_s${seed}" "${SAMS_ROOT}/noendpoint_s${seed}" norman additive 1 $SAMS_STEPS none $seed ""
done

if [ "${1:-}" = "count" ]; then
  echo "${#JOBS[@]}"
  exit 0
fi

# ---------------------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------------------
cd "$REPO_DIR" || exit 1
mkdir -p logs
module load conda
conda activate "$CONDA_ENV"
set -uo pipefail

export PYTHONPATH=.
export HDF5_USE_FILE_LOCKING=FALSE
export SPEAR_COMPILE=${SPEAR_COMPILE:-1}

# Per-job compiler caches and temp files, kept out of /tmp. A cache shared by concurrent jobs
# corrupts (interleaved autotune JSON), so every job owns its directory.
CACHE_DIR="${REPO_DIR}/.cache/job_${SLURM_JOB_ID:-local}"
mkdir -p "$CACHE_DIR/inductor" "$CACHE_DIR/triton" "$CACHE_DIR/tmp"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/inductor"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export TMPDIR="$CACHE_DIR/tmp"
trap 'rm -rf "$CACHE_DIR"' EXIT

if [ "${MODE:-train}" = "analyze" ]; then
  python src/script/samsvae_analysis.py all
  exit $?
fi

TASK=${SLURM_ARRAY_TASK_ID:-0}
if [ "$TASK" -ge "${#JOBS[@]}" ]; then
  echo "ABORT: task $TASK is outside the job table (${#JOBS[@]} jobs)"
  exit 1
fi
IFS='|' read -r TAG RESULT_PATH DATA SPLIT FOLD STEPS LOSS SEED EXTRA <<< "${JOBS[$TASK]}"

common_args() {   # data split fold steps
  local data=$1 split=$2 fold=$3 steps=$4 pert_fn=crisper
  case "$data" in combosciplex|parse_pbmc) pert_fn=drug ;; esac
  echo "--data_name=$data --batch_size=48 --model_type=origin --d_model=128 --lr=5e-5 --eta_min=1e-6 \
--steps=$steps --print_every=10000 --fusion_method=differential_perceiver --split_method=$split --fold=$fold \
--infer_top_gene=1000 --n_top_genes=5000 --topk=30 --neighbor_cap=128 \
--perturbation_function=$pert_fn --noise_type=Gaussian --mode=predict_y \
--deg_epsilon=0.1 --sinkhorn_eps=0.1 --sinkhorn_iters=50 \
--precision_epsilon=0.1 --n_slices=128 --sliced_top_frac=0.1 --sparse_lambda=0.1 \
--use_negative_edge --use_repo --use_wire"
}

# The mask and wire files are created on first use. Concurrent tasks would write them at the
# same time, so tasks take a directory lock (mkdir is atomic on any filesystem) and the first
# one builds the files with a zero-step run.
prepare_data() {   # data split fold
  local data=$1 split=$2 fold=$3
  local mask="data/${data}/mask_fold_${fold}topk_30${split}_negative_edge.pt"
  local wire="${mask%.pt}_wire_eigvecs.pt"
  local lock="logs/.prep_${data}_${split}_f${fold}.lock"
  until mkdir "$lock" 2>/dev/null; do
    find "$lock" -maxdepth 0 -mmin +120 -exec rmdir {} \; 2>/dev/null
    sleep 30
  done
  if [ ! -f "$mask" ] || [ ! -f "$wire" ]; then
    echo "[prep] building mask and wire files for $data $split fold $fold"
    python src/script/run.py $(common_args "$data" "$split" "$fold" 0) \
      --endpoint_loss=mmd --gamma="$GAMMA" --seed=0 --result_path="./result/prep_${data}_${split}_f${fold}"
    if [ $? -ne 0 ]; then
      rmdir "$lock"
      return 1
    fi
  fi
  rmdir "$lock"
  [ -f "$mask" ] && [ -f "$wire" ]
}

echo "[cfg] task=$TASK tag=$TAG data=$DATA split=$SPLIT fold=$FOLD steps=$STEPS loss=$LOSS seed=$SEED"
echo "[cfg] result_path=$RESULT_PATH"
echo "[cfg] extra=$EXTRA"
echo "[cfg] compile=$SPEAR_COMPILE cache=$CACHE_DIR"

# Per-tag result directories (SAMS-VAE suite) can be checked for a finished run directly. The
# matrix suite shares one result root per dataset, so run.py itself decides there whether
# there is anything left to do from the checkpoints of that run.
if [[ "$RESULT_PATH" == */"$TAG" ]] && \
   find "$RESULT_PATH" -path "*iteration_${STEPS}/results.csv" 2>/dev/null | grep -q .; then
  echo "[skip] $TAG already finished (iteration_${STEPS} evaluated)"
  exit 0
fi

prepare_data "$DATA" "$SPLIT" "$FOLD" || { echo "ABORT: mask or wire file unavailable for $DATA $SPLIT fold $FOLD"; exit 1; }

starts=$(date +"%s")
start=$(date +"%r, %m-%d-%Y")

PY_PID=""
on_time_limit() {
  echo "[time] time limit approaching; stopping and requeueing (the run resumes from its latest checkpoint)"
  if [ -n "$PY_PID" ]; then
    kill -TERM "$PY_PID" 2>/dev/null
    wait "$PY_PID" 2>/dev/null
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap on_time_limit USR1

python src/script/run.py $(common_args "$DATA" "$SPLIT" "$FOLD" "$STEPS") \
  --endpoint_loss="$LOSS" --gamma="$GAMMA" --seed="$SEED" --result_path="$RESULT_PATH" $EXTRA &
PY_PID=$!
wait "$PY_PID"
STATUS=$?

ends=$(date +"%s")
end=$(date +"%r, %m-%d-%Y")
diff=$(($ends-$starts))
hours=$(($diff / 3600)); dif=$(($diff % 3600))
minutes=$(($dif / 60)); seconds=$(($dif % 60))
printf "\n\t===========Time Stamp===========\n"
printf "\tStart\t:$start\n\tEnd\t:$end\n\tTime\t:%02d:%02d:%02d\n" "$hours" "$minutes" "$seconds"
printf "\t================================\n\n"
sacct --jobs="${SLURM_JOB_ID}" --format=jobid,jobname,qos,nnodes,ncpu,maxrss,cputime,avecpu,elapsed
exit $STATUS
