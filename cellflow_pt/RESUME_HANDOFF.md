# CellFlow adaptation ablation — RESUME / HANDOFF

> SPEAR NeurIPS-2026 rebuttal. Goal: show (leg-1) **SPEAR beats CellFlow**, and (leg-2)
> **our encoders (SPE / SPE+AR) improve CellFlow**. All work lives in `~/SPEAR`
> (= symlink to `~/p1/SPEAR`; registered in `~/bin/.p1_projects` → restored by `p1-resume`).
> Last updated: 2026-07-25 (switching OD).

---

## TL;DR state

- **leg-1 (SPEAR vs CellFlow): DONE & defensible.** On Norman/additive, **doubles-only, repo-native eval**:
  | metric (doubles) | SPEAR (295k) | CellFlow (PCA) |
  |---|---|---|
  | **DE Spearman-LFC** | **0.852 ± 0.005** | 0.719 |
  | direction match | 0.884 | 0.894 |
  | discrimination cos | **0.929** | 0.877 |
  | pearson_δ | 0.747 | 0.853 |
  SPEAR wins the DE metrics (DE-LFC +0.13, discrimination +0.05); CellFlow wins pearson_δ
  (housekeeping-gene dominated — same argument the paper makes for scDFM). **Write leg-1.**
- **leg-2 (SPE improves CellFlow): NOT yet.**
  - additive: SPE ≈ gene (no help; additive is near-additive, structure redundant).
  - unseen: SPE weak-positive (DE-LFC +0.036, 3/4 seeds, **p=0.22, not significant**).
  - **Root problem:** the gene-axis CellFlow arms (needed so SPE can attach) UNDERPERFORM
    vanilla PCA CellFlow, for two reasons → being fixed (see below).

## The gene-space FIX investigation (IN PROGRESS — resume here)

Why gene-axis < PCA (diagnosed):
1. **No standardization.** Data is log1p+normalize_total but NOT z-scored; PCA implicitly
   centers → gene-axis MSE is dominated by high-expression housekeeping genes → DE genes
   under-fit → low DE-LFC. **[conditioning]**
2. **1000-dim flow is hard.** PCA runs the flow in 128-dim (easy); gene-axis runs it in
   full 1000-dim gene space (hard). PCA's 128→1000 decode is just **inverse-PCA (fixed
   linear)**. **[dimensionality]**

Deep-research (done, ranked fixes, saved reasoning): **(1) per-gene z-score on HVG + clip ±10**
[conditioning], **(2) zero-init delta head + control→pert bridge**, **(3) DE/variance-weighted
loss**, **(4) multi-sample mean for LFC** (largely have it: N=128 eval), **(5) OT coupling**
(have it), **(6) learned AE latent + NB decoder, STAGED training (pretrain AE→freeze→flow)**
= the user's "pool genes→128 latent, flow there, learned decode" idea = **highest ceiling**.
Honesty caveat from lit (Ahlmann-Eltze/Huber, Nat.Methods 2025): complex models often DON'T
beat a **mean/linear baseline** on DE — so ALSO benchmark vs "predict training mean".

**Currently running when we switched OD:** `run_fix_sweep.sh` — z-score (±bigger model d256/n6)
× {gene,spe} × {additive,unseen} × seeds{42,43,44}, 9000 steps, batch 128, packed 8-wide.
→ writes `~/SPEAR/TABLE_FIX_{zs,zsbig}_{additive,unseen}.txt`. **It will NOT have finished
before the OD switch — RELAUNCH on the new OD** (see below), or if the old OD is still alive,
`p1-resume` rsyncs whatever completed.

## Plan (do in order on the new OD)

1. Relaunch the fix sweep; read `TABLE_FIX_*.txt`. **Target: gene-axis DE-LFC ≥ PCA (additive
   0.719 / unseen 0.611); then +SPE should land ABOVE → leg-2 works.**
2. Also run a **mean-baseline** (predict training-mean perturbed profile) as the honesty floor.
3. If z-score(±big) insufficient → add **zero-init delta head + DE-weighted loss** (cheap; knobs
   to add in `cellflow_pt.py`).
4. If still insufficient → implement the **AE arm** (`--ae_latent 128`, arg stub already added):
   SPE gene-encoder → pool → 128 latent → flow → learned decode; **staged** (pretrain AE w/
   reconstruction, freeze, then flow); check AE's own reconstruction DE-LFC first.
5. Final: multi-seed + paired test (spe vs gene, spe vs PCA); assemble rebuttal tables.

---

## How to resume on a new OD

```bash
touch ~/.p1_gpu_od           # arm sentinel (once per new OD)
p1-resume                    # restores ~/SPEAR (+ LCFM data dep) from old OD / bkuptool
cd ~/SPEAR/SPEAR-main
# sanity: venv + data
~/SPEAR/venv/bin/python -c "import torch,cell_eval,ot,sklearn,torchdiffeq;print('ok',torch.cuda.is_available())"
ls -lL data/norman.h5ad      # symlink -> ~/p1/LCFM/data/norman/ (LCFM must also be restored)
# relaunch the fix sweep (both GPUs, packed):
nohup bash cellflow_pt/run_fix_sweep.sh > ~/SPEAR/fix_sweep.log 2>&1 &
```

Data dep: `data/norman.h5ad` → `~/p1/LCFM/data/norman/` (LCFM is also in `.p1_projects`).
Masks/splits/wire eigvecs are cached in `data/norman/` (mask_fold_1topk_30{additive,unseen}_*).

## Key files (all under `~/SPEAR/SPEAR-main/cellflow_pt/`)

- `cellflow_pt.py` — CellFlow-PT reimpl. Arms: `cellflow`(PCA-MLP), `cellflow_gene`(gene-axis,
  no SPE), `cellflow_spe`(+SPE), `cellflow_spear`(+SPE+AR). Knobs: `--gene_norm zscore`,
  `--d_model --n_layers`, `--ae_latent`(stub), `--num_workers --eval_threads`. Eval mirrors
  repo `_test_impl` (N=128, results.csv) + reports **doubles-only** via repo's `cls()`.
- `eval_spear.py` — evaluate the trained SPEAR (result/checkpoint.pt) with the REPO's native
  `run.test()`; `--eval_seeds` for mean±std (SPEAR generation is noisy). Gave the 0.852 leg-1.
- `collect_results.py` — `--tag <prefix> --seeds ...`; best-ckpt-by-de_direction_match, mean±std,
  paired tests (spe vs gene, spear vs gene).
- `run_fix_sweep.sh` — the z-score fix sweep (CURRENTLY the active experiment).
- `run_packed.sh` — high-throughput packed runner (8-wide, both GPUs) — template for filling GPUs.
- Baselines already collected: `TABLE_additive.txt` (leg-1/2 additive), `spear_native5.log`
  (SPEAR ×5 draws). Per-run logs: `~/SPEAR/nat_*.log`, `~/SPEAR/{zs,zsbig}_*.log`.

## NAMING CAVEAT (important)

In the CellFlow ablation tables, `spear` / `cellflow_spear` = **CellFlow + our SPE+AR encoders**
(a CellFlow variant), **NOT the real SPEAR model**. Real SPEAR = `eval_spear.py` on
`result/checkpoint.pt` (DE-LFC 0.852). Don't confuse the two.

## Env / hardware notes

- venv: `~/SPEAR/venv` (torch 2.7.1+cu126, cell_eval 0.5.42, POT `ot` 0.9.5, sklearn, torchdiffeq).
- CellFlow's real repo (JAX) can't run here (no flax/optax/ott/diffrax, CPU-only jaxlib, no net)
  → that's WHY we reimplemented it in PyTorch ("CellFlow (our re-implementation)").
- 2× A100-80GB. Gene-axis runs are ~compute-bound at batch≥128 (attention over 1000 genes);
  low GPU RAM use is normal (small model) — pack ~4 jobs/GPU to fill compute.
