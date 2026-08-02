#!/bin/bash
# Serially build the shared read-only inductor cache. Run ONCE on an interactive
# GPU node before submit_causal_matrix.sh. No other compiled job may run concurrently.
set -e
cd "$(dirname "$0")"
source ./exp_env.sh
module load conda 2>/dev/null
conda activate "$CONDA_ENV"
export PYTHONPATH=./
export HDF5_USE_FILE_LOCKING=FALSE
export SPEAR_COMPILE=1

rm -rf .inductor_warm .inductor_warm_triton
export TORCHINDUCTOR_CACHE_DIR=$REPO_DIR/.inductor_warm
export TRITON_CACHE_DIR=$REPO_DIR/.inductor_warm/triton

WARM=./result/warm
rm -rf "$WARM"

# Two graph variants exist in the matrix: samsvae off and on. Compile both.
for FLAGS in "" "--use_samsvae --samsvae_mask_prior=concrete --samsvae_latent_dim=64 --samsvae_lambda_mask=1e-4 --samsvae_lr=1e-3 --samsvae_warmup_steps=500"; do
  echo "=== warming: ${FLAGS:-baseline} ==="
  python src/script/run.py $COMMON \
    --steps=8 --print_every=1000000 --batch_size=$BATCH \
    --result_path=$WARM --endpoint_loss=deg_sinkhorn --gamma=0.1 --seed=42 $FLAGS
done

# Eval graph uses batch 64; compile it too so the first checkpoint eval is not a stall.
python src/script/run.py $COMMON \
  --steps=2 --print_every=1 --batch_size=$BATCH \
  --result_path=$WARM --endpoint_loss=deg_sinkhorn --gamma=0.1 --seed=42

chmod -R a-w .inductor_warm
rm -rf "$WARM"
echo "warm cache ready: $(du -sh .inductor_warm | cut -f1)"