"""Cross-prior consistency: does the SAMS-VAE mechanism structure learned under the l1
prior match the one learned under the concrete prior?

A high cross-prior Mantel r (close to the within-prior stability r) means the interpretable
per-perturbation structure is a property of the DATA + sparse-additive inductive bias, NOT
an artifact of the specific sparsity prior — the strongest robustness evidence for the
interpretability claim. Gauge-invariant (cosine on mask (.) shift_hat) and latent-relabel
invariant (perturbation x perturbation similarity), with a pooled permutation null.

Usage:
  python src/script/samsvae_cross_prior.py "<concrete_glob>" "<l1_glob>" [--ck 50000] [--nperm 2000]
"""
import argparse, glob, itertools
import numpy as np, torch
from src.script.samsvae_interpret import load_mech, sim_matrix, _upper, _pearson_safe


def load_group(g, ck, prior):
    cks = sorted(set(glob.glob(f"{g}/**/iteration_{ck}/checkpoint.pt", recursive=True)))
    assert cks, f"no checkpoints under {g} @ {ck}"
    mechs, r2vs = [], []
    for c in cks:
        m, r2v = load_mech(c, prior)
        mechs.append(m); r2vs.append(r2v)
    assert all(torch.equal(r2vs[0], r) for r in r2vs), "row_to_vocab differs within group"
    return mechs, r2vs[0], cks


def within(Su):
    rs = [_pearson_safe(Su[i], Su[j]) for i, j in itertools.combinations(range(len(Su)), 2)]
    return float(np.mean(rs)) if rs else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('concrete_glob'); ap.add_argument('l1_glob')
    ap.add_argument('--ck', default='50000'); ap.add_argument('--nperm', type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.RandomState(0)

    C, r2vC, ckC = load_group(a.concrete_glob, a.ck, 'concrete')
    L, r2vL, ckL = load_group(a.l1_glob, a.ck, 'l1')
    assert torch.equal(r2vC, r2vL), "row_to_vocab mismatch between priors"

    # active in ALL runs of BOTH priors (a prior-independent common support)
    allm = C + L
    norms = torch.stack([m.norm(dim=-1) for m in allm])
    thr = float(norms.max()) * 0.01
    active = (norms > thr).all(dim=0)
    idx = torch.where(active)[0]
    n = len(idx)
    print(f"### Cross-prior consistency: concrete x{len(C)} vs l1 x{len(L)} @ iter {a.ck} ###")
    print(f"active-in-ALL-runs perturbations: {n}/{len(r2vC)}")

    SC = [sim_matrix(m[idx]) for m in C]          # precompute once
    SL = [sim_matrix(m[idx]) for m in L]
    SCu = [_upper(s) for s in SC]
    SLu = [_upper(s) for s in SL]
    pairs = [(i, j) for i in range(len(SCu)) for j in range(len(SLu))]

    def cross_mean(cu, lu):
        return float(np.mean([_pearson_safe(cu[i], lu[j]) for (i, j) in pairs]))

    r_obs = cross_mean(SCu, SLu)
    ge = 1
    for _ in range(a.nperm):
        p = rng.permutation(n)
        SLp = [_upper(SL[j][np.ix_(p, p)]) for j in range(len(SL))]   # permute precomputed matrices
        if cross_mean(SCu, SLp) >= r_obs:
            ge += 1
    pval = ge / (a.nperm + 1)

    print(f"within-concrete mean Mantel r = {within(SCu):+.3f}")
    print(f"within-l1       mean Mantel r = {within(SLu):+.3f}")
    print(f">>> CROSS-PRIOR (concrete vs l1) pooled Mantel r = {r_obs:+.3f}  perm p={pval:.4f}")
    print("    (cross-prior r ~ within-prior r  =>  structure is prior-independent = robust)")


if __name__ == '__main__':
    main()
