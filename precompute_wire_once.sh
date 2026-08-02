#!/bin/bash
#SBATCH -A pccr
#SBATCH -N 1
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -t 02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=14
#SBATCH --job-name=wire_pre
#SBATCH --output=logs/%x_%j.out

module load conda
source /depot/natallah/data/shourya/SPEAR_CAUSAL/SPEAR/exp_env.sh
conda activate "$CONDA_ENV"
export PYTHONPATH=./
cd "$REPO_DIR"

echo "[wire] DATA=$DATA SPLIT=$SPLIT FOLD=$FOLD TOPK=$TOPK NEG_EDGE=$NEG_EDGE"
echo "[wire] out -> $WIRE_PATH"

python src/data_process/precompute_wire.py \
  --data_path_root ./data \
  --data_name "$DATA" \
  --out_path "$WIRE_PATH" \
  --n_top_genes "$NGENES" \
  --infer_top_gene "$INFER_TOP" \
  --d_model "$DMODEL" \
  --nhead "$NHEAD" \
  --eigvec_dim "$EIGVEC_DIM" \
  --split_method "$SPLIT" \
  --fold "$FOLD" \
  --topk "$TOPK" \
  $NEG_FLAG

ls -la "$WIRE_PATH"