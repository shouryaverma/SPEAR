#!/bin/bash
# Leg-2 best honest shot: on the UNSEEN (holdout-double) split -- where co-expression
# geometry should matter (paper's most-pronounced-gap split) -- compare
# gene / +SPE / +SPE+AR (+ vanilla PCA for context), 4 seeds, stabilized eval.
# Waits for the additive multiseed + finalize to finish first (free GPUs, no contention).
cd /home/wangmengbo/SPEAR/SPEAR-main
PY=/home/wangmengbo/SPEAR/venv/bin/python
export PYTHONPATH=. HDF5_USE_FILE_LOCKING=FALSE
L=/home/wangmengbo/SPEAR

echo "[unseen $(date +%H:%M)] waiting for additive ALL_DONE + finalize DONE ..."
while ! grep -q ALL_DONE "$L/ms_orchestrator.log" 2>/dev/null; do sleep 60; done
while ! grep -q DONE "$L/FINAL_TABLE.txt" 2>/dev/null; do sleep 30; done
echo "[unseen $(date +%H:%M)] GPUs free -> running unseen-split arms"

run(){ # gpu arm seed
  CUDA_VISIBLE_DEVICES=$1 $PY cellflow_pt/cellflow_pt.py --arm $2 --seed $3 \
    --split_method unseen --steps 12000 --eval_every 3000 --n_eval_ctrl 256 --batch_size 64 \
    --out result_cf/unseen_${2}_s$3 > "$L/ms_unseen_${2}_s$3.log" 2>&1
}
for s in 42 43 44 45; do
  echo "[unseen $(date +%H:%M)] seed $s: spe(GPU0)+spear(GPU1)"
  run 0 cellflow_spe   $s &
  run 1 cellflow_spear $s &
  wait
  echo "[unseen $(date +%H:%M)] seed $s: gene(GPU0)+vanilla(GPU1)"
  run 0 cellflow_gene  $s &
  run 1 cellflow       $s &
  wait
done
echo "[unseen $(date +%H:%M)] collecting -> FINAL_TABLE_UNSEEN.txt"
$PY cellflow_pt/collect_results.py --tag ms_unseen --seeds 42 43 44 45 --no_spear \
    > "$L/FINAL_TABLE_UNSEEN.txt" 2>&1
echo "DONE $(date +%H:%M)" >> "$L/FINAL_TABLE_UNSEEN.txt"
echo "[unseen $(date +%H:%M)] DONE"
