#!/bin/bash
# Multi-seed CellFlow arms with a stabilized eval (n_eval_ctrl=256, eval every 3k,
# stop at 15k to avoid the overfitting tail). gene & spe run in parallel (one per GPU);
# vanilla (fast, PCA) runs after. Produces per-seed eval curves for a paired comparison.
cd /home/wangmengbo/SPEAR/SPEAR-main
PY=/home/wangmengbo/SPEAR/venv/bin/python
export PYTHONPATH=. HDF5_USE_FILE_LOCKING=FALSE
LOGDIR=/home/wangmengbo/SPEAR
run(){ # gpu arm seed steps
  CUDA_VISIBLE_DEVICES=$1 $PY cellflow_pt/cellflow_pt.py --arm $2 --seed $3 \
    --steps $4 --eval_every 3000 --n_eval_ctrl 256 --batch_size 64 \
    --out result_cf/ms_${2}_s$3 > $LOGDIR/ms_${2}_s$3.log 2>&1
}
for s in 42 43 44; do
  echo "[$(date +%H:%M)] wave seed $s: gene(GPU0) + spe(GPU1)"
  run 0 cellflow_gene $s 15000 &
  run 1 cellflow_spe  $s 15000 &
  wait
done
echo "[$(date +%H:%M)] vanilla seeds (GPU0)"
for s in 42 43 44; do run 0 cellflow $s 15000; done
echo "ALL_DONE $(date +%H:%M)"
