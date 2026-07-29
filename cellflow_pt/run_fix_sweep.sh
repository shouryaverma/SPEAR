#!/bin/bash
# Overnight sweep of the gene-space FIX: z-score standardization (the missing step vs PCA),
# plus a bigger-model variant. Goal: recover gene-axis DE-LFC to >= PCA, so +SPE lands above.
# Packed (8-wide) on both GPUs; auto-collects per (recipe, split).
cd /home/wangmengbo/SPEAR/SPEAR-main
PY=/home/wangmengbo/SPEAR/venv/bin/python
export PYTHONPATH=. HDF5_USE_FILE_LOCKING=FALSE
L=/home/wangmengbo/SPEAR
MAX=8

declare -a TASKS
for split in additive unseen; do
  for s in 42 43 44; do
    for arm in cellflow_gene cellflow_spe; do
      # recipe zs: z-score, d128 ; recipe zsbig: z-score + d256/n6
      TASKS+=("$arm $s $split zs   128 4")
      TASKS+=("$arm $s $split zsbig 256 6")
    done
  done
done

echo "[fix $(date +%H:%M)] ${#TASKS[@]} tasks, MAX=$MAX"
i=0
for t in "${TASKS[@]}"; do
  set -- $t; arm=$1; seed=$2; split=$3; recipe=$4; dm=$5; nl=$6
  gpu=$((i % 2))
  while [ "$(jobs -rp | wc -l)" -ge "$MAX" ]; do sleep 15; done
  echo "[fix $(date +%H:%M)] gpu$gpu $recipe $arm $split s$seed (d$dm n$nl)"
  CUDA_VISIBLE_DEVICES=$gpu $PY cellflow_pt/cellflow_pt.py --arm $arm --seed $seed \
    --split_method $split --gene_norm zscore --d_model $dm --n_layers $nl \
    --steps 9000 --batch_size 128 --eval_every 3000 --n_eval_ctrl 128 \
    --num_workers 3 --eval_threads 6 \
    --out result_cf/${recipe}_${split}_${arm}_s${seed} \
    > "$L/${recipe}_${split}_${arm}_s${seed}.log" 2>&1 &
  i=$((i + 1)); sleep 4
done
wait
echo "[fix $(date +%H:%M)] all runs done -> collecting"
for recipe in zs zsbig; do
  for split in additive unseen; do
    $PY cellflow_pt/collect_results.py --tag ${recipe}_${split} --seeds 42 43 44 --no_spear \
        > "$L/TABLE_FIX_${recipe}_${split}.txt" 2>&1
    echo "DONE" >> "$L/TABLE_FIX_${recipe}_${split}.txt"
  done
done
echo "[fix $(date +%H:%M)] ALL_DONE"
