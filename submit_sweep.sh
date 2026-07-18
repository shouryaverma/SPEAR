#!/bin/bash
set -e
mkdir -p logs

# --- reference baselines (empirical IPM, raw gene space) ---
sbatch --job-name=e1_mmd \
      train.sbatch --endpoint_loss=mmd            --gamma=0.1
sbatch --job-name=e2_sinkhorn \
      train.sbatch --endpoint_loss=sinkhorn       --gamma=0.1

# --- anchor: reweighting + OT ---
sbatch --job-name=e3_degsink \
      train.sbatch --endpoint_loss=deg_sinkhorn   --gamma=0.1

# --- axis 1: metric (projection OT), DEG off / on ---
sbatch --job-name=e4_maxsliced \
      train.sbatch --endpoint_loss=max_sliced     --gamma=0.1 \
      --n_slices=128 --sliced_top_frac=0.1
sbatch --job-name=e5_deg_maxsliced \
      train.sbatch --endpoint_loss=deg_max_sliced --gamma=0.1 \
      --n_slices=128 --sliced_top_frac=0.1

# --- axis 2: reweighting (precision vs DEG), same OT geometry ---
sbatch --job-name=e6_precisionsink \
      train.sbatch --endpoint_loss=precision_sinkhorn --gamma=0.1 \
      --precision_epsilon=0.1

# --- axis 3: structure (delta sparsity on top of DEG-Sinkhorn) ---
sbatch --job-name=e7_degsink_sparse \
      train.sbatch --endpoint_loss=deg_sinkhorn_sparse --gamma=0.1 \
      --sparse_lambda=0.1

# --- velocity gate ablation: gate OFF on anchor and on structure arm ---
sbatch --job-name=e8_degsink_nogate \
      train.sbatch --endpoint_loss=deg_sinkhorn        --gamma=0.1 \
      --no-use-velocity-gate
sbatch --job-name=e9_degsink_sparse_nogate \
      train.sbatch --endpoint_loss=deg_sinkhorn_sparse --gamma=0.1 \
      --sparse_lambda=0.1 --no-use-velocity-gate