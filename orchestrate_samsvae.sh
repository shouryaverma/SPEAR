#!/bin/bash
# Autonomous SAMS-VAE experiment orchestrator. Owns the full matrix across 2 GPUs,
# 2 concurrent runs per GPU (measured: ~3.1 it/s, ~37% util, 9GB -> 2/GPU ~= 2x, clean).
# Resumable: skips any run whose 50k results.csv already exists. Main (concrete x5)
# is front-loaded so a mid-way OD recycle still yields the headline result.
# Launch: cd SPEAR_samsvae && nohup ./orchestrate_samsvae.sh > /home/wangmengbo/p1/SPEAR/orch_samsvae.log 2>&1 &
set -u
cd /home/wangmengbo/p1/SPEAR/SPEAR_samsvae
export PYTHONPATH=. HDF5_USE_FILE_LOCKING=FALSE SPEAR_COMPILE=0
K=2   # concurrent runs per GPU

COMMON="--data_name=norman --batch_size=48 --model_type=origin --d_model=128 --eta_min=1e-6 \
  --fusion_method=differential_perceiver --split_method=additive --fold=1 \
  --infer_top_gene=1000 --n_top_genes=5000 --topk=30 \
  --perturbation_function=crisper --noise_type=Gaussian --mode=predict_y \
  --deg_epsilon=0.1 --sinkhorn_eps=0.1 --sinkhorn_iters=50 \
  --use_negative_edge --use_repo --use_wire \
  --endpoint_loss=deg_sinkhorn_sparse --gamma=0.1 --sparse_lambda=0.1"

# spec = seed:prior:ldim:lambda   (concrete=gauge-robust main; l1=ablation)
# Front-loaded: concrete main seeds first, then prior/lambda/latent-dim ablations.
GPU0_JOBS=( 42:concrete:64:1e-4  44:concrete:64:1e-4  46:concrete:64:1e-4 \
            42:l1:64:1e-4        42:concrete:64:1e-5   42:concrete:32:1e-4 )
GPU1_JOBS=( 43:concrete:64:1e-4  45:concrete:64:1e-4   43:l1:64:1e-4 \
            44:l1:64:1e-4        42:concrete:64:1e-3    42:concrete:128:1e-4 )

run_job() {
  local gpu=$1 spec=$2
  IFS=: read -r seed prior ldim lam <<< "$spec"
  local tag="samsvae_${prior}_L${ldim}_lm${lam}_s${seed}"
  local out="./result/scratch2/${tag}"
  if find "$out" -path "*iteration_50000/results.csv" 2>/dev/null | grep -q .; then
    echo "[orch] $(date +%H:%M:%S) SKIP $tag (50k done)"; return 0
  fi
  local log="/home/wangmengbo/p1/SPEAR/${tag}.log"
  echo "[orch] $(date +%H:%M:%S) START gpu=$gpu $tag -> $log"
  CUDA_VISIBLE_DEVICES=$gpu ~/p1/SPEAR/venv/bin/python src/script/run.py $COMMON \
    --use_scm --scm_head_type=samsvae --samsvae_mask_prior=$prior --samsvae_latent_dim=$ldim \
    --scm_lr=1e-3 --scm_warmup_steps=500 --scm_lambda_mask=$lam \
    --lr=5e-5 --steps=60000 --print_every=10000 \
    --result_path="$out" --seed=$seed > "$log" 2>&1
  echo "[orch] $(date +%H:%M:%S) DONE  gpu=$gpu $tag (exit $?)"
}

run_gpu_queue() {
  local gpu=$1; shift
  local jobs=("$@") running=0
  for spec in "${jobs[@]}"; do
    run_job "$gpu" "$spec" &
    running=$((running+1))
    if (( running >= K )); then wait -n; running=$((running-1)); fi
  done
  wait
}

echo "[orch] $(date) launch: GPU0=${#GPU0_JOBS[@]} jobs, GPU1=${#GPU1_JOBS[@]} jobs, K=$K/gpu"
run_gpu_queue 0 "${GPU0_JOBS[@]}" &
run_gpu_queue 1 "${GPU1_JOBS[@]}" &
wait
echo "[orch] $(date) ALL DONE"
