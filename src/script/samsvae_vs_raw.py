"""Does the SAMS-VAE structured mechanism recover MORE biology than the raw SPEAR
perturbation representation? (The interpretability-increment test.)

Same gauge-invariant, subset-to-ground-truth, pooled-permutation protocol as
samsvae_interpret, applied to THREE perturbation representations, each turned into a
perturbation x perturbation cosine-similarity matrix and Mantel-correlated to TRRUST /
DE ground truth:
  (A) SAMS-VAE mechanism   mask (.) shift_hat                 -- OUR structure
  (B) raw gene-token embedding from the SAME samsvae models   encoder.embedding.weight[pert]
  (C) raw gene-token embedding from BASE models (no head)     -- original SPEAR

Higher Mantel r == the representation's perturbation structure better matches real biology.
The claim "our structure adds interpretable biology" requires A > B (and A vs C).

Usage: python src/script/samsvae_vs_raw.py [--ck 50000] [--nperm 2000]
"""
import argparse, glob, json
import numpy as np, torch, scanpy as sc
from src.script.samsvae_interpret import (load_mech, mask_table, sim_matrix, _upper,
                                          _pearson_safe, pooled_mantel, trrust_jaccard,
                                          de_signatures, VOCAB, NORMAN, TRRUST)

SAMS = "result/scratch2/samsvae_concrete_L64_lm1e-4_s4*"
BASE = "../SPEAR-main/result/scratch2/base_s*"


def emb_rows(ckpt, vocab_ids):
    """raw gene-token embedding for the perturbation genes: (n_pert, d)."""
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)['model_state_dict']
    return sd['encoder.embedding.weight'].float()[vocab_ids]


def align(name, S_list, GT, have, active, nperm, rng):
    idx = np.where((have > 0) & active)[0]
    if len(idx) < 8:
        print(f"      {name:22s}: n={len(idx)} too few -> skip"); return None
    Ssub = [s[np.ix_(idx, idx)] for s in S_list]
    r, p, _ = pooled_mantel(Ssub, GT[np.ix_(idx, idx)], nperm, rng)
    print(f"      {name:22s}: n={len(idx)}  pooled Mantel r={r:+.3f}  perm p={p:.4f}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ck', default='50000'); ap.add_argument('--nperm', type=int, default=2000)
    a = ap.parse_args(); rng = np.random.RandomState(0)

    sams_ck = sorted(set(glob.glob(f"{SAMS}/**/iteration_{a.ck}/checkpoint.pt", recursive=True)))
    base_ck = sorted(set(glob.glob(f"{BASE}/**/iteration_{a.ck}/checkpoint.pt", recursive=True)))
    assert sams_ck and base_ck, f"missing ckpts (sams={len(sams_ck)}, base={len(base_ck)})"

    # shared perturbation ordering from samsvae row_to_vocab
    mechs, r2vs = zip(*[load_mech(c, 'concrete') for c in sams_ck])
    r2v = r2vs[0]
    assert all(torch.equal(r2v, r) for r in r2vs)
    vocab = json.load(open(VOCAB)); inv = {v: k for k, v in vocab.items()}
    genes = [inv[int(v)] for v in r2v.tolist()]

    # representations
    A = list(mechs)                                          # (A) samsvae mechanism
    B = [emb_rows(c, r2v) for c in sams_ck]                  # (B) raw emb, samsvae models
    C = [emb_rows(c, r2v) for c in base_ck]                  # (C) raw emb, base models

    # active = nonzero mechanism in ALL samsvae seeds (same support the mask claim uses)
    norms = torch.stack([m.norm(dim=-1) for m in A])
    active = (norms > float(norms.max()) * 0.01).all(dim=0).numpy()

    SA = [sim_matrix(m) for m in A]
    SB = [sim_matrix(m) for m in B]
    SC = [sim_matrix(m) for m in C]

    print(f"### Interpretability increment: SAMS-VAE mechanism vs raw SPEAR embedding @ {a.ck} ###")
    print(f"samsvae seeds={len(A)}, base seeds={len(C)}; {int(active.sum())}/{len(genes)} active perts\n")

    # ground truths
    Td = torch.load(TRRUST, map_location='cpu', weights_only=False)
    J, jhave = trrust_jaccard(genes, vocab, Td['binary'].float())
    adata = sc.read_h5ad(NORMAN)
    _xm = adata.X[:200]; _xm = np.asarray(_xm.todense()) if hasattr(_xm, 'todense') else np.asarray(_xm)
    if _xm.max() > 50 and np.allclose(_xm, np.round(_xm)):
        sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    D, dhave = de_signatures(genes, adata)

    for gtname, GT, have in [("TRRUST-Jaccard", J, jhave), ("DE-signature", D, dhave)]:
        print(f"  [{gtname}]  ({int(((have>0)&active).sum())} perts with ground truth & active)")
        rA = align("(A) SAMS-VAE mechanism", SA, GT, have, active, a.nperm, rng)
        rB = align("(B) raw emb (samsvae)", SB, GT, have, active, a.nperm, rng)
        rC = align("(C) raw emb (base SPEAR)", SC, GT, have, active, a.nperm, rng)
        if rA is not None and rB is not None:
            verdict = "A>B: structure ADDS biology" if rA > rB + 0.02 else \
                      ("A~=B: no clear increment" if abs(rA - rB) <= 0.02 else "A<B: raw emb stronger")
            print(f"      -> vs same-model raw emb: {verdict} (rA-rB={rA-rB:+.3f})")
        print()


if __name__ == '__main__':
    main()
