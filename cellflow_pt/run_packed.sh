#!/bin/bash
# High-throughput packed runner: fill BOTH GPUs by running many small jobs concurrently
# (round-robin GPU, no barriers). Tiny models (~5GB) -> ~4 jobs/GPU on 80GB, 44 cores.
# additive remaining (batch 64, consistent w/ done seeds) + all unseen (batch 128).
cd /home/wangmengbo/SPEAR/SPEAR-main
PY=/home/wangmengbo/SPEAR/venv/bin/python
export PYTHONPATH=. HDF5_USE_FILE_LOCKING=FALSE
L=/home/wangmengbo/SPEAR
MAX=8   # concurrent jobs total (~4 per GPU)

declare -a TASKS
# additive remaining (batch 64): spear_s43 (was interrupted) + all of seed 44
TASKS+=("cellflow_spear 43 additive 12000 64")
for a in cellflow_gene cellflow_spe cellflow_spear cellflow; do TASKS+=("$a 44 additive 12000 64"); done
# unseen: all arms x 3 seeds (batch 128 for better GPU util; consistent within unseen)
for s in 42 43 44; do for a in cellflow_gene cellflow_spe cellflow_spear cellflow; do
  TASKS+=("$a $s unseen 12000 128"); done; done

echo "[packed $(date +%H:%M)] ${#TASKS[@]} tasks, MAX=$MAX concurrent"
i=0
for t in "${TASKS[@]}"; do
  set -- $t; arm=$1; seed=$2; split=$3; steps=$4; batch=$5
  gpu=$((i % 2))
  while [ "$(jobs -rp | wc -l)" -ge "$MAX" ]; do sleep 15; done
  echo "[packed $(date +%H:%M)] launch gpu$gpu $arm $split s$seed (batch $batch)"
  CUDA_VISIBLE_DEVICES=$gpu $PY cellflow_pt/cellflow_pt.py --arm $arm --seed $seed \
    --split_method $split --steps $steps --batch_size $batch --eval_every 3000 \
    --n_eval_ctrl 128 --num_workers 3 --eval_threads 6 \
    --out result_cf/nat_${split}_${arm}_s${seed} > "$L/nat_${split}_${arm}_s${seed}.log" 2>&1 &
  i=$((i + 1)); sleep 4
done
wait
echo "[packed $(date +%H:%M)] all runs done -> collecting"
$PY cellflow_pt/collect_results.py --tag nat_additive --seeds 42 43 44 --no_spear > "$L/TABLE_additive.txt" 2>&1
echo "DONE_additive" >> "$L/TABLE_additive.txt"
$PY cellflow_pt/collect_results.py --tag nat_unseen --seeds 42 43 44 --no_spear > "$L/TABLE_unseen.txt" 2>&1
echo "DONE_unseen" >> "$L/TABLE_unseen.txt"
echo "[packed $(date +%H:%M)] ALL_DONE"
