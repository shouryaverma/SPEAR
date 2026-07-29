"""SAMS-VAE interpretability analysis (gauge-invariant, cross-seed, pooled-permutation).

The headline claim for this line is NOT performance (pure-additive head cannot beat a
strong baseline on the additive split) but that the learned sparse mechanism is a
STABLE, biologically-meaningful per-perturbation property. This script tests that with
the safeguards the preflight design critique demanded:

  * GAUGE-INVARIANT readout: use the mechanism `mask (.) shift_hat` (shift already
    unit-normed per row), never the raw mask (whose scale is a gauge).
  * LATENT-RELABEL-INVARIANT comparison: never compare mask columns across seeds
    (latent dims are anonymous / permuted). Compare perturbation x perturbation
    similarity matrices S_seed[i,j]=cos(mech_i,mech_j), which are invariant to any
    within-seed latent permutation.
  * STABILITY AS A HARD GATE: cross-seed Mantel correlation of the S matrices, with a
    permutation null. If structure is not reproducible, no alignment claim is allowed.
  * POOLED-ACROSS-SEEDS permutation p-value with a SINGLE pre-registered statistic
    (Mantel r), never single-seed cherry-picked p-values.
  * Two ground truths: TRRUST downstream Jaccard (low power, ~few edges) and DE-signature
    correlation (higher power at this data scale) as independent biological anchors.

Usage:
  python src/script/samsvae_interpret.py "result/scratch2/samsvae_concrete_L64_lm1e-4_s*" \
         [--ck 50000] [--prior concrete] [--nperm 2000]
"""
import sys, os, json, glob, argparse, itertools
import numpy as np, torch, scanpy as sc

VOCAB = "src/tokenizer/norman_5000_highly_vocab.json"
NORMAN = "/home/wangmengbo/p1/LCFM/data/norman/norman.h5ad"
TRRUST = "data/norman/grn_trrust_adj.pt"
_HC_BETA, _HC_GAMMA, _HC_ZETA = 2.0 / 3.0, -0.1, 1.1


# ---------- mechanism extraction (mirrors sams_vae.py, no model build needed) ----------
def mask_table(mask_logits, prior):
    if prior == 'l1':
        return torch.sigmoid(mask_logits)
    s_bar = torch.sigmoid(mask_logits) * (_HC_ZETA - _HC_GAMMA) + _HC_GAMMA
    return s_bar.clamp(0.0, 1.0)


def load_mech(ckpt, prior):
    """Returns (mech [n_pert, L], row_to_vocab [n_pert])."""
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)['model_state_dict']
    ml = sd['samsvae.mask_logits'].float()
    sh = sd['samsvae.shift'].float()
    r2v = sd['samsvae.row_to_vocab'].long()
    mask = mask_table(ml, prior)
    shat = torch.nn.functional.normalize(sh, dim=-1, eps=1e-8)
    return (mask * shat), r2v


# ---------- gauge-invariant similarity + Mantel ----------
def sim_matrix(mech):
    """perturbation x perturbation cosine similarity (invariant to latent relabel/scale)."""
    m = torch.nn.functional.normalize(mech, dim=-1, eps=1e-8)
    return (m @ m.T).numpy()


def _upper(v):
    n = v.shape[0]
    iu = np.triu_indices(n, k=1)
    return v[iu]


def _pearson_safe(a, b):
    """Pearson r that returns NaN (not warning-laden garbage) on a constant vector --
    a degenerate ground truth must NEVER masquerade as a significant result."""
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def mantel(A, B, nperm, rng):
    """Mantel: Pearson r between upper triangles of two n x n matrices; permutation
    null permutes the shared perturbation index of B (rows AND cols). Returns (r_obs, p).
    Degenerate (constant) triangle -> (nan, 1.0): honest null, never a spurious small p."""
    a = _upper(A)
    n = A.shape[0]
    r_obs = _pearson_safe(a, _upper(B))
    if not np.isfinite(r_obs):
        return np.nan, 1.0
    ge = 1
    for _ in range(nperm):
        p = rng.permutation(n)
        r = _pearson_safe(a, _upper(B[np.ix_(p, p)]))
        if np.isfinite(r) and r >= r_obs:      # NaN comparisons never count toward the null
            ge += 1
    return float(r_obs), ge / (nperm + 1)


def pooled_mantel(mats_S, T, nperm, rng):
    """Pooled statistic = mean over seeds of Mantel r(S_seed, T); pooled null permutes
    the perturbation index of T ONCE per draw (rows AND cols) and re-averages r across
    seeds. Degenerate ground truth / any non-finite seed r -> (nan, 1.0, r_seeds): honest
    null (guards the 'constant matrix -> corrcoef NaN -> p~=1/(nperm+1)' false positive)."""
    tU = _upper(T)
    ups = [_upper(S) for S in mats_S]
    r_seeds = [_pearson_safe(u, tU) for u in ups]
    if tU.std() == 0 or not all(np.isfinite(r) for r in r_seeds):
        return np.nan, 1.0, r_seeds
    r_obs = float(np.mean(r_seeds))
    n = T.shape[0]
    ge = 1
    for _ in range(nperm):
        p = rng.permutation(n)
        tp = _upper(T[np.ix_(p, p)])
        rs = [_pearson_safe(u, tp) for u in ups]
        if all(np.isfinite(r) for r in rs) and np.mean(rs) >= r_obs:
            ge += 1
    return r_obs, ge / (nperm + 1), r_seeds


# ---------- ground-truth matrices over the shared perturbation ordering ----------
def trrust_jaccard(genes, vocab, gb):
    ds = {g: set(gb[vocab[g]].nonzero().ravel().tolist()) for g in genes if g in vocab}
    n = len(genes)
    J = np.eye(n)
    have = np.array([1.0 if (g in ds and ds[g]) else 0.0 for g in genes])
    for i, j in itertools.combinations(range(n), 2):
        gi, gj = genes[i], genes[j]
        x, y = ds.get(gi, set()), ds.get(gj, set())
        J[i, j] = J[j, i] = (len(x & y) / len(x | y)) if (x or y) else 0.0
    return J, have


def de_signatures(genes, adata):
    """Per-perturbation single-gene DE vector (mean perturbed - mean control) over all
    genes; returns DE-correlation matrix + a mask of which perturbations had a single."""
    c = adata.obs['condition'].astype(str)
    ctrl_mask = np.isin(c.values, ['ctrl', 'control'])
    X = adata.X
    X = np.asarray(X.todense()) if hasattr(X, 'todense') else np.asarray(X)
    ctrl_mean = X[ctrl_mask].mean(0)
    de = {}
    for g in genes:
        m = (c.values == f"{g}+ctrl") | (c.values == f"ctrl+{g}")
        if m.sum() >= 5:
            de[g] = X[m].mean(0) - ctrl_mean
    n = len(genes)
    D = np.eye(n)
    have = np.array([1.0 if g in de else 0.0 for g in genes])
    for i, j in itertools.combinations(range(n), 2):
        gi, gj = genes[i], genes[j]
        if gi in de and gj in de:
            r = _pearson_safe(de[gi], de[gj])            # NaN-safe; 0 if a DE vec is constant
            D[i, j] = D[j, i] = (r if np.isfinite(r) else 0.0)
    return D, have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('glob')
    ap.add_argument('--ck', default='50000')
    ap.add_argument('--prior', default='concrete')
    ap.add_argument('--nperm', type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.RandomState(0)

    ckpts = sorted(set(glob.glob(f"{args.glob}/**/iteration_{args.ck}/checkpoint.pt", recursive=True)))
    assert ckpts, f"no checkpoints under {args.glob} @ iteration_{args.ck}"
    print(f"### SAMS-VAE interpretability @ iter {args.ck}, prior={args.prior}, {len(ckpts)} seeds ###")
    for c in ckpts:
        print("  -", c.split('result/')[-1])

    # extract mechanisms; verify a shared perturbation ordering across seeds
    mechs, r2vs = [], []
    for c in ckpts:
        m, r2v = load_mech(c, args.prior)
        mechs.append(m); r2vs.append(r2v)
    assert all(torch.equal(r2vs[0], r) for r in r2vs), "row_to_vocab differs across seeds!"
    r2v = r2vs[0]

    vocab = json.load(open(VOCAB))
    inv = {v: k for k, v in vocab.items()}
    genes_all = [inv[int(v)] for v in r2v.tolist()]
    n_all = len(genes_all)

    # ACTIVE-perturbation filter: concrete L0 can prune a perturbation's ENTIRE mask ->
    # zero mechanism -> undefined cosine, and a zero row pollutes the similarity matrix
    # (cos(0,.)=0 injects fake dissimilarity). Restrict all structure analysis to
    # perturbations with a nonzero mechanism in EVERY seed (a "no mechanism" perturbation
    # has nothing to compare). Report how many are dropped (no silent truncation).
    norms = torch.stack([m.norm(dim=-1) for m in mechs])          # (n_seed, n_pert)
    thr = float(norms.max()) * 0.01
    active = (norms > thr).all(dim=0)                             # active in ALL seeds
    idx_active = torch.where(active)[0]
    mechs = [m[idx_active] for m in mechs]
    genes = [genes_all[i] for i in idx_active.tolist()]
    n = len(genes)
    print(f"\n{n_all} perturbable genes; {n} ACTIVE (nonzero mechanism in all {len(mechs)} "
          f"seeds, thr={thr:.4g}). L={mechs[0].shape[1]}")
    if n < 8:
        print("    WARNING: <8 active perturbations -> structure analysis underpowered.")

    # [1] gauge-invariant sparsity per seed
    print("\n[1] Gauge-invariant mechanism sparsity (mech = mask (.) shift_hat):")
    for c, m in zip(ckpts, mechs):
        active = (m.abs() > (m.abs().max() * 0.1)).float().sum(1).mean()
        print(f"    {os.path.basename(os.path.dirname(os.path.dirname(c)))[:24]:24s} "
              f"mech_l1/pert={m.abs().sum(1).mean():.3f}  active_dims(>10%max)/pert={active:.2f}/{m.shape[1]}")

    # [2] cross-seed STABILITY GATE
    S = [sim_matrix(m) for m in mechs]
    print("\n[2] Cross-seed STABILITY (Mantel r of perturbation-similarity matrices; GATE):")
    if len(S) >= 2:
        rs = []
        for i, j in itertools.combinations(range(len(S)), 2):
            r, p = mantel(S[i], S[j], args.nperm, rng)
            if np.isfinite(r):
                rs.append(r)
            print(f"    seed{i} vs seed{j}: Mantel r={r:+.3f}  perm p={p:.4f}")
        mean_r = float(np.mean(rs)) if rs else float('nan')
        gate = np.isfinite(mean_r) and mean_r > 0.2
        print(f"    >>> mean cross-seed Mantel r = {mean_r:+.3f}  ->  STABILITY GATE "
              f"{'PASS' if gate else 'FAIL'} (threshold 0.2)")
    else:
        gate = False; print("    <2 seeds -> cannot assess stability")

    # [3] biological alignment (only if the stability gate passes) -- pooled across seeds,
    # restricted to perturbations that HAVE each ground truth (no 0-padding bias), with a
    # pre-registered PRIMARY test (DE, higher power at this scale) + Bonferroni across the
    # two ground truths (alpha=0.05/2=0.025). TRRUST is secondary/corroborative.
    print("\n[3] Biological alignment (pooled across seeds; PRIMARY=DE, secondary=TRRUST; "
          "Bonferroni alpha=0.025):")
    ALPHA = 0.025
    MIN_PERTS, MIN_PAIRS = 8, 20      # power floor: below this, honest-null (test not run)

    def align(name, T, have, primary):
        idx = np.where(have > 0)[0]
        npairs = len(idx) * (len(idx) - 1) // 2
        tag = 'PRIMARY  ' if primary else 'secondary'
        if len(idx) < MIN_PERTS or npairs < MIN_PAIRS:
            print(f"    [{tag}] {name}: only {len(idx)} perts / {npairs} pairs with ground "
                  f"truth -> UNDERPOWERED, test not run (honest null).")
            return
        Ssub = [s[np.ix_(idx, idx)] for s in S]
        Tsub = T[np.ix_(idx, idx)]
        r, p, rseeds = pooled_mantel(Ssub, Tsub, args.nperm, rng)
        if not np.isfinite(r):
            print(f"    [{tag}] {name}: n={len(idx)} but ground truth has no variance "
                  f"-> test not run (honest null).")
            return
        verdict = f"SIGNIFICANT (p<{ALPHA})" if p < ALPHA else "n.s."
        print(f"    [{tag}] {name}: n={len(idx)} pooled Mantel r={r:+.3f}  perm p={p:.4f}  "
              f"[{verdict}]  per-seed r={[round(x,3) for x in rseeds]}")

    if not gate:
        print("    SKIPPED: stability gate failed -> no reproducible per-perturbation "
              "structure to align (honest null).")
    else:
        # PRIMARY: DE-signature alignment (higher power)
        try:
            adata = sc.read_h5ad(NORMAN)
            # norman.h5ad X is already log-normalized (max ~6.4); only normalize+log if it
            # actually looks like raw integer counts (avoids the double-log distortion).
            _xm = adata.X[:200]
            _xm = np.asarray(_xm.todense()) if hasattr(_xm, 'todense') else np.asarray(_xm)
            if _xm.max() > 50 and np.allclose(_xm, np.round(_xm)):
                sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
            D, dhave = de_signatures(genes, adata)
            print(f"    DE: {int(dhave.sum())}/{n} perts have a single-gene condition")
            align("DE-signature", D, dhave, primary=True)
        except Exception as e:
            print(f"    DE alignment skipped: {e}")
        # SECONDARY: TRRUST-downstream Jaccard (low power, corroborative)
        Td = torch.load(TRRUST, map_location='cpu', weights_only=False)
        J, jhave = trrust_jaccard(genes, vocab, Td['binary'].float())
        print(f"    TRRUST: {int(jhave.sum())}/{n} perts have >=1 downstream; "
              f"nonzero Jaccard pairs={int((np.triu(J,1)>0).sum())}")
        align("TRRUST-Jaccard", J, jhave, primary=False)

    print("\n### done ###")


if __name__ == '__main__':
    main()
