#!/bin/bash
#SBATCH -A pccr
#SBATCH -N 1
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=14
#SBATCH --mail-user=verma198@purdue.edu
#SBATCH --mail-type=FAIL
#SBATCH --job-name=spear_causal
#SBATCH --output=logs/%x_%j.out

starts=$(date +"%s")
start=$(date +"%r, %m-%d-%Y")

module load conda
source /depot/natallah/data/shourya/SPEAR_CAUSAL/SPEAR/exp_env.sh
conda activate "$CONDA_ENV"
export PYTHONPATH=./
export HDF5_USE_FILE_LOCKING=FALSE
export SPEAR_COMPILE=1

# Node-local inductor/triton cache, one per job. A cache dir shared across
# concurrent jobs corrupts (interleaved autotune best_config JSON), which kills
# every job with JSONDecodeError. Each job autotunes independently on its own node.
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/inductor_${SLURM_JOB_ID}
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton_${SLURM_JOB_ID}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
trap 'rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"' EXIT

cd "$REPO_DIR"

echo "[cfg] DATA=$DATA SPLIT=$SPLIT FOLD=$FOLD TOPK=$TOPK NGENES=$NGENES"
echo "[cfg] MASK=$MASK_PATH"
echo "[cfg] WIRE=$WIRE_PATH"
echo "[cfg] CACHE=$TORCHINDUCTOR_CACHE_DIR"
echo "[cfg] ARGS=$@"

python src/script/run.py $COMMON "$@"

ends=$(date +"%s")
end=$(date +"%r, %m-%d-%Y")
diff=$(($ends-$starts))
hours=$(($diff / 3600)); dif=$(($diff % 3600))
minutes=$(($dif / 60)); seconds=$(($dif % 60))
printf "\n\t===========Time Stamp===========\n"
printf "\tStart\t:$start\n\tEnd\t:$end\n\tTime\t:%02d:%02d:%02d\n" "$hours" "$minutes" "$seconds"
printf "\t================================\n\n"
sacct --jobs=$SLURM_JOBID --format=jobid,jobname,qos,nnodes,ncpu,maxrss,cputime,avecpu,elapsed