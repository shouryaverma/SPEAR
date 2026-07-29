#!/bin/bash
# CLEAN authoritative comparison — all arms, DOUBLES-ONLY eval (SPEAR-matched, writes
# results.csv), fresh cln_ namespace (no contamination from the stopped SPEAR-agent runs
# or the old all-conditions-eval nat_ runs). Both GPUs, packed. Ordered by seed so seed 42
# (all arms x both splits) completes first for a full first read.
#   arms: PCA / gene / +SPE / +SPE+AR / AE(learned enc-dec) / AE+SPE(SPE in encoder)
cd /home/wangmengbo/SPEAR/SPEAR-main
PY=/home/wangmengbo/SPEAR/venv/bin/python
export PYTHONPATH=. HDF5_USE_FILE_LOCKING=FALSE
# CRITICAL: cap BLAS/OpenMP threads per job. The OT (pot.emd), sklearn PCA and cell_eval
# otherwise each spawn ~1 thread/core -> with many packed jobs loadavg explodes (200+ on
# 44 cores) and everything thrashes. 64x64 EMD is <10ms single-threaded, so 1 is plenty.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
L=/home/wangmengbo/SPEAR
MAX="${MAX:-8}"   # concurrent across 2 GPUs; CPU-bound (OT on CPU) so ~18 is the useful ceiling

ARMS=(cellflow cellflow_gene cellflow_spe cellflow_spear cellflow_ae cellflow_ae_spe)
declare -a TASKS
for s in 42 43 44; do
  for split in additive unseen; do
    batch=64; [ "$split" = unseen ] && batch=128
    for a in "${ARMS[@]}"; do TASKS+=("$a $s $split $batch"); done
  done
done

echo "[clean $(date +%H:%M)] ${#TASKS[@]} tasks (6 arms x 2 splits x 3 seeds), MAX=$MAX"
i=0
for t in "${TASKS[@]}"; do
  set -- $t; arm=$1; seed=$2; split=$3; batch=$4
  out=result_cf/cln_${split}_${arm}_s${seed}
  if ls ${out}/eval_*/results.csv >/dev/null 2>&1; then
    echo "[clean $(date +%H:%M)] skip done $arm $split s$seed"; continue; fi
  gpu=$((i % 2))
  while [ "$(jobs -rp | wc -l)" -ge "$MAX" ]; do sleep 20; done
  echo "[clean $(date +%H:%M)] gpu$gpu $arm $split s$seed (batch $batch)"
  CUDA_VISIBLE_DEVICES=$gpu $PY cellflow_pt/cellflow_pt.py --arm $arm --seed $seed \
    --split_method $split --steps 12000 --batch_size $batch --eval_every 3000 \
    --n_eval_ctrl 128 --num_workers 1 --eval_threads 2 --ae_steps 4000 --ae_latent 128 \
    --out $out > "$L/cln_${split}_${arm}_s${seed}.log" 2>&1 &
  i=$((i + 1)); sleep 5
done
wait
echo "[clean $(date +%H:%M)] all runs done -> collecting DOUBLES-ONLY tables"
$PY cellflow_pt/collect_results.py --tag cln_additive --seeds 42 43 44 > "$L/TABLE_additive.txt" 2>&1
$PY cellflow_pt/collect_results.py --tag cln_unseen   --seeds 42 43 44 > "$L/TABLE_unseen.txt" 2>&1
echo "[clean $(date +%H:%M)] ALL_DONE"
