#!/bin/bash
# Wait for the multi-seed orchestrator to finish, then run a clean SPEAR eval on a
# free GPU and collect the final comparison table. Fully autonomous.
cd /home/wangmengbo/SPEAR/SPEAR-main
PY=/home/wangmengbo/SPEAR/venv/bin/python
export PYTHONPATH=. HDF5_USE_FILE_LOCKING=FALSE
L=/home/wangmengbo/SPEAR

echo "[finalize $(date +%H:%M)] waiting for multiseed ALL_DONE ..."
while ! grep -q ALL_DONE "$L/ms_orchestrator.log" 2>/dev/null; do sleep 60; done
echo "[finalize $(date +%H:%M)] multiseed done -> SPEAR eval (GPU0, clean)"
CUDA_VISIBLE_DEVICES=0 $PY cellflow_pt/eval_spear.py --ckpt result/checkpoint.pt \
    --n_eval_ctrl 256 --out result_cf/spear_eval > "$L/spear_eval.log" 2>&1
echo "[finalize $(date +%H:%M)] collecting results -> FINAL_TABLE.txt"
$PY cellflow_pt/collect_results.py > "$L/FINAL_TABLE.txt" 2>&1
echo "[finalize $(date +%H:%M)] DONE" >> "$L/FINAL_TABLE.txt"
echo "[finalize $(date +%H:%M)] DONE -> $L/FINAL_TABLE.txt"
