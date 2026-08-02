#!/bin/bash
# Single source of truth for the experiment matrix. Every other script sources this.
# Change DATA / SPLIT / FOLD / TOPK / NGENES here and all scripts follow.

export DATA=${DATA:-parse_pbmc}
export SPLIT=${SPLIT:-state_pbmc}
export FOLD=${FOLD:-0}
export TOPK=${TOPK:-30}
export NGENES=${NGENES:-5000}
export INFER_TOP=${INFER_TOP:-1000}
export DMODEL=${DMODEL:-128}
export NHEAD=${NHEAD:-8}
export EIGVEC_DIM=${EIGVEC_DIM:-32}
export NBR_CAP=${NBR_CAP:-128}
export BATCH=${BATCH:-48}
export STEPS=${STEPS:-300000}
export PRINT_EVERY=${PRINT_EVERY:-10000}
export LR=${LR:-5e-5}
export NEG_EDGE=${NEG_EDGE:-1}          # 1 = --use_negative_edge, 0 = off
export REPO_DIR=${REPO_DIR:-/depot/natallah/data/shourya/SPEAR_CAUSAL/SPEAR}
export CONDA_ENV=${CONDA_ENV:-/depot/natallah/data/shourya/scDFM_env}

# crisper routes perturbation_id through GeneEncoder, which requires every perturbation to
# be a gene token present in the vocab and to have a row in the co-expression neighbor
# table. Drug perturbations have neither, so they take the BatchLabelEncoder path instead.
case "$DATA" in
  combosciplex|parse_pbmc) PERT_FN=drug ;;
  *)                       PERT_FN=crisper ;;
esac
export PERT_FN

# Derived: must mirror data.py mask naming and run.py's wire_path exactly.
if [ "$NEG_EDGE" = "1" ]; then
  NEG_FLAG="--use_negative_edge"; NEG_SUFFIX="_negative_edge"
else
  NEG_FLAG=""; NEG_SUFFIX=""
fi
export NEG_FLAG
export MASK_PATH="data/${DATA}/mask_fold_${FOLD}topk_${TOPK}${SPLIT}${NEG_SUFFIX}.pt"
export WIRE_PATH="${MASK_PATH%.pt}_wire_eigvecs.pt"
export RESULT_ROOT="./result/causal_matrix_${DATA}_${SPLIT}_f${FOLD}"

export COMMON="--data_name=${DATA} --batch_size=${BATCH} --model_type=origin \
--d_model=${DMODEL} --lr=${LR} --eta_min=1e-6 --steps=${STEPS} --print_every=${PRINT_EVERY} \
--fusion_method=differential_perceiver --split_method=${SPLIT} --fold=${FOLD} \
--infer_top_gene=${INFER_TOP} --n_top_genes=${NGENES} --topk=${TOPK} \
--neighbor_cap=${NBR_CAP} \
--perturbation_function=${PERT_FN} --noise_type=Gaussian --mode=predict_y \
--deg_epsilon=0.1 --sinkhorn_eps=0.1 --sinkhorn_iters=50 \
--precision_epsilon=0.1 --n_slices=128 --sliced_top_frac=0.1 --sparse_lambda=0.1 \
${NEG_FLAG} --use_repo --use_wire \
--result_path=${RESULT_ROOT}"