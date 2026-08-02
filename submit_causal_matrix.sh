#!/bin/bash
mkdir -p logs
source "$(dirname "$0")/exp_env.sh"

cd "$REPO_DIR"

if [ ! -f "$MASK_PATH" ]; then
  echo "ABORT: mask not found: $MASK_PATH"; exit 1
fi
if [ ! -f "$WIRE_PATH" ]; then
  echo "ABORT: wire eigvecs not found: $WIRE_PATH -> sbatch precompute_wire_once.sh"; exit 1
fi

case "$DATA" in
  norman|dixit|replogle|parse_pbmc|combosciplex) ;;
  *) echo "ABORT: dataset '$DATA' is not validated for this matrix (see C1/C2)."; exit 1 ;;
esac

# Datasets that ignore split_method entirely: combosciplex uses a hardcoded
# test_conditions list, replogle builds its own held-out-gene split before the
# split_method dispatch. Any FOLD/SPLIT variation emits duplicate runs under
# distinct names.
case "$DATA" in
  combosciplex)
    if [ "$SPLIT" != "additive" ] || [ "$FOLD" != "0" ]; then
      echo "ABORT: combosciplex ignores split_method/fold; use SPLIT=additive FOLD=0."; exit 1
    fi
    ;;
  replogle)
    if [ "$SPLIT" != "additive" ]; then
      echo "ABORT: replogle builds its own unseen-gene split; use SPLIT=additive."; exit 1
    fi
    ;;
  parse_pbmc)
    if [ "$SPLIT" != "state_pbmc" ]; then
      echo "ABORT: parse_pbmc has no double perturbations; use SPLIT=state_pbmc."; exit 1
    fi
    ;;
  norman|dixit)
    case "$SPLIT" in
      additive|combinations|unseen) ;;
      *) echo "ABORT: split '$SPLIT' is not valid for $DATA."; exit 1 ;;
    esac
    ;;
esac

LOSSES=(mmd deg_mmd sinkhorn deg_sinkhorn max_sliced deg_max_sliced deg_sinkhorn_sparse)
SEEDS=(42 43 44)

# SAMS-VAE hard-anchors factor k to gene _pert_names[k], so every perturbation must resolve
# to a vocab gene token. Drug perturbations yield an empty samsvae_pert_vocab_ids and
# instantiate_model asserts. Gene-perturbation datasets only.
case "$DATA" in
  norman|replogle) CAUSAL_LOSSES=(deg_sinkhorn_sparse) ;;
  *)               CAUSAL_LOSSES=() ;;
esac

CAUSAL="--use_samsvae --samsvae_mask_prior=concrete --samsvae_latent_dim=64 \
--samsvae_lambda_mask=1e-4 --samsvae_lr=1e-3 --samsvae_warmup_steps=500"

TAG="${DATA}_${SPLIT}_f${FOLD}"
echo "[submit] $TAG -> $RESULT_ROOT (pert_fn=${PERT_FN}, causal arms: ${#CAUSAL_LOSSES[@]})"

n=0
for loss in "${LOSSES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    sbatch --job-name="off_${loss}_s${seed}_${TAG}" \
      train_causal.sbatch \
      --endpoint_loss=${loss} --gamma=0.1 --seed=${seed} && n=$((n+1))
  done
done

for loss in ${CAUSAL_LOSSES[@]+"${CAUSAL_LOSSES[@]}"}; do
  for seed in "${SEEDS[@]}"; do
    sbatch --job-name="causal_${loss}_s${seed}_${TAG}" \
      train_causal.sbatch \
      --endpoint_loss=${loss} --gamma=0.1 --seed=${seed} ${CAUSAL} && n=$((n+1))
  done
done

expected=$(( ${#LOSSES[@]} * ${#SEEDS[@]} + ${#CAUSAL_LOSSES[@]} * ${#SEEDS[@]} ))
echo "submitted $n / $expected jobs"