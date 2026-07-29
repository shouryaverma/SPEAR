"""
Plumbing validation for CellFlow-PT:
  1. Load Norman via SPEAR's exact data pipeline (e7 config).
  2. Grab eval conditions + control + per-condition targets.
  3. Make a TRIVIAL prediction (identity: pred = control cells unchanged).
  4. Score with cell_eval (same call as SPEAR's test()).
This validates the data+eval reuse BEFORE writing the CellFlow model, and gives a
floor reference number (identity predictor) to sanity-check CellFlow against.

Run from SPEAR-main with PYTHONPATH=. :
  cd /home/wangmengbo/SPEAR/SPEAR-main
  PYTHONPATH=. HDF5_USE_FILE_LOCKING=FALSE /home/wangmengbo/SPEAR/venv/bin/python cellflow_pt/validate_plumbing.py
"""
import os, sys, time
import numpy as np
import pandas as pd
import anndata as ad
import torch

from src.data_process.data import Data
from cell_eval import MetricsEvaluator

t0 = time.time()
print("[1] building data (Norman, additive, fold1, e7 config) ...", flush=True)
dm = Data('./data')
dm.load_data('norman')
dm.process_data(
    n_top_genes=5000, infer_top_gene=1000,
    split_method='additive', fold=1,
    use_negative_edge=True, k=30,
    split_file='', train_frac=1.0,
)
train_sampler, test_dl, _ = dm.load_flow_data(batch_size=48)  # (train, TESTset, [])
print(f"    data ready in {time.time()-t0:.1f}s; adata {dm.adata.shape}", flush=True)

conds = list(test_dl._perturbation_covariates)
print(f"[2] eval conditions: {len(conds)} -> {conds[:5]}{'...' if len(conds)>5 else ''}", flush=True)

control_data = test_dl.get_control_data()
ctrl_X = control_data['src_cell_data'].cpu().numpy()   # (n_ctrl, G)
print(f"    control cells: {ctrl_X.shape}", flush=True)

rng = np.random.default_rng(0)
N = 128
all_pred, all_real = [ctrl_X], [ctrl_X]
name_pred = ['control'] * ctrl_X.shape[0]
name_real = ['control'] * ctrl_X.shape[0]

for c in conds:
    pdata = test_dl.get_perturbation_data(c)
    target = pdata['tgt_cell_data'].cpu().numpy()          # (n_tgt, G) real perturbed
    idx = rng.permutation(ctrl_X.shape[0])[:N]
    pred = ctrl_X[idx].copy()                              # TRIVIAL: identity (control unchanged)
    all_pred.append(pred);  name_pred += [c] * pred.shape[0]
    all_real.append(target); name_real += [c] * target.shape[0]

all_pred = np.concatenate(all_pred, 0)
all_real = np.concatenate(all_real, 0)
print(f"[3] pred {all_pred.shape} real {all_real.shape}", flush=True)

pred = ad.AnnData(X=all_pred, obs=pd.DataFrame({'perturbation': name_pred}))
real = ad.AnnData(X=all_real, obs=pd.DataFrame({'perturbation': name_real}))

outdir = 'cellflow_pt/_validate_out'
os.makedirs(outdir, exist_ok=True)
print("[4] running cell_eval MetricsEvaluator ...", flush=True)
ev = MetricsEvaluator(adata_pred=pred, adata_real=real,
                      control_pert='control', pert_col='perturbation',
                      num_threads=32, outdir=outdir)
results, agg = ev.compute()
agg.write_csv(os.path.join(outdir, 'agg_results.csv'))
print("[5] DONE. agg_results (identity floor):", flush=True)
adf = agg.to_pandas() if hasattr(agg, 'to_pandas') else pd.DataFrame(agg)
# print the metrics we care about
cols = [c for c in adf.columns]
print(adf.to_string(max_rows=60))
print(f"\n[total {time.time()-t0:.1f}s]", flush=True)
