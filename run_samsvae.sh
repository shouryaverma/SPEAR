#!/bin/bash
# SAMS-VAE from-scratch training launcher (norman/additive/fold1, full config).
# Comparators (byte-identical code path, already trained @50k) live in
# ../SPEAR-main/result/scratch2/{base,iplc,grn}_s{42,43,...}. Matched compare point = 50k.
# Usage: ./run_samsvae.sh <gpu> <seed> <mask_prior> <latent_dim> <tag>
set -e
cd /home/wangmengbo/p1/SPEAR/SPEAR_samsvae
export PYTHONPATH=. HDF5_USE_FILE_LOCKING=FALSE SPEAR_COMPILE=0

GPU=${1:-0}; SEED=${2:-42}; PRIOR=${3:-l1}; LDIM=${4:-64}; TAG=${5:-samsvae_${PRIOR}_L${LDIM}_s${SEED}}

COMMON="--data_name=norman --batch_size=48 --model_type=origin --d_model=128 --eta_min=1e-6 \
  --fusion_method=differential_perceiver --split_method=additive --fold=1 \
  --infer_top_gene=1000 --n_top_genes=5000 --topk=30 \
  --perturbation_function=crisper --noise_type=Gaussian --mode=predict_y \
  --deg_epsilon=0.1 --sinkhorn_eps=0.1 --sinkhorn_iters=50 \
  --use_negative_edge --use_repo --use_wire \
  --endpoint_loss=deg_sinkhorn_sparse --gamma=0.1 --sparse_lambda=0.1"

LOG=/home/wangmengbo/p1/SPEAR/${TAG}.log
echo "[launch] GPU=$GPU seed=$SEED prior=$PRIOR ldim=$LDIM tag=$TAG -> $LOG"
nohup env CUDA_VISIBLE_DEVICES=$GPU ~/p1/SPEAR/venv/bin/python src/script/run.py $COMMON \
  --use_scm --scm_head_type=samsvae --samsvae_mask_prior=$PRIOR --samsvae_latent_dim=$LDIM \
  --scm_lr=1e-3 --scm_warmup_steps=500 --scm_lambda_mask=1e-4 \
  --lr=5e-5 --steps=60000 --print_every=10000 \
  --result_path=./result/scratch2/${TAG} --seed=$SEED \
  > "$LOG" 2>&1 &
echo "[launch] pid=$! (nohup, detached)"
