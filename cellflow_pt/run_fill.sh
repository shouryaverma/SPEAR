#!/bin/bash
# GPU0-only fill: re-run the native baselines (A2/A3) that the fix-sweep does NOT cover
# (vanilla PCA cellflow + cellflow_spear, both splits, all seeds). Pinned to GPU0 so we
# never touch GPU1's ~70G zsbig sweep. Own concurrency counter (MAX_FILL) on top of the
# 4 zs/d128 sweep jobs already on GPU0. nat_ tag -> collects into TABLE_{additive,unseen}.
cd /home/wangmengbo/SPEAR/SPEAR-main
PY=/home/wangmengbo/SPEAR/venv/bin/python
export PYTHONPATH=. HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=0
L=/home/wangmengbo/SPEAR
MAX_FILL=4       # jobs I add to GPU0 (on top of the sweep's 4 d128 jobs)

declare -a TASKS
# additive (batch 64, matches the handoff's leg-1/2 numbers) — all 4 arms x 3 seeds
for s in 42 43 44; do for a in cellflow cellflow_gene cellflow_spe cellflow_spear; do
  TASKS+=("$a $s additive 12000 64"); done; done
# unseen (batch 128) — all 4 arms x 3 seeds
for s in 42 43 44; do for a in cellflow cellflow_gene cellflow_spe cellflow_spear; do
  TASKS+=("$a $s unseen 12000 128"); done; done

echo "[fill $(date +%H:%M)] ${#TASKS[@]} native-baseline tasks on GPU0, MAX_FILL=$MAX_FILL"
for t in "${TASKS[@]}"; do
  set -- $t; arm=$1; seed=$2; split=$3; steps=$4; batch=$5
  out=result_cf/nat_${split}_${arm}_s${seed}
  # skip if already completed (has an eval agg) — makes the script resumable
  if ls ${out}/eval_*/agg.csv >/dev/null 2>&1; then
    echo "[fill $(date +%H:%M)] skip done $arm $split s$seed"; continue; fi
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_FILL" ]; do sleep 20; done
  echo "[fill $(date +%H:%M)] launch $arm $split s$seed (batch $batch)"
  $PY cellflow_pt/cellflow_pt.py --arm $arm --seed $seed \
    --split_method $split --steps $steps --batch_size $batch --eval_every 3000 \
    --n_eval_ctrl 128 --num_workers 2 --eval_threads 4 \
    --out $out > "$L/nat_${split}_${arm}_s${seed}.log" 2>&1 &
  sleep 5
done
wait
echo "[fill $(date +%H:%M)] all fill runs done -> collecting"
$PY cellflow_pt/collect_results.py --tag nat_additive --seeds 42 43 44 > "$L/TABLE_additive.txt" 2>&1
$PY cellflow_pt/collect_results.py --tag nat_unseen   --seeds 42 43 44 > "$L/TABLE_unseen.txt" 2>&1
echo "[fill $(date +%H:%M)] ALL_DONE"
