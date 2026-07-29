"""
CellFlow-PT: a faithful PyTorch re-implementation of CellFlow's method
(Klein/Fleck et al., bioRxiv 2025.04.11.648220), run inside SPEAR's data+eval
harness so the numbers are directly comparable to SPEAR's tables.

Faithful to CellFlow's OTFlowMatching (per the spec extracted from its JAX code):
  * flow runs in a low-dim CELL-EMBEDDING space (PCA of gene expression)
  * source = control cells, target = perturbed cells
  * minibatch OT coupling (Sinkhorn / EMD, squared-euclidean cost) before the path
  * straight probability path: x_t = (1-t) x0 + t x1 ; target velocity u_t = x1 - x0
  * set-based condition encoder: set of perturbed genes -> CLS-token multi-head
    attention pooling -> Dense(condition_embedding_dim)
  * velocity field: sinusoidal time enc -> MLP ; x -> MLP ; concat[t,x,cond] ->
    decoder MLP -> Dense(data_dim) ; SiLU throughout
  * Adam(5e-5) ; deterministic condition mode ; predict = integrate ODE 0->1
  * decode back to genes via inverse-PCA for cell_eval scoring

Arms (see --arm):
  cellflow      : vanilla CellFlow (PCA space, MLP velocity field)               [no gene axis]
  cellflow_gene : CellFlow recipe but velocity field is a gene-axis transformer   [control: gene axis, NO SPE]
  cellflow_spe  : cellflow_gene + our Spectral Positional Encoding (WIRE)         [our module added]

Usage (from SPEAR-main, PYTHONPATH=.):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. HDF5_USE_FILE_LOCKING=FALSE \
    /home/wangmengbo/SPEAR/venv/bin/python cellflow_pt/cellflow_pt.py \
      --arm cellflow --data_name norman --split_method additive --fold 1 \
      --seed 42 --steps 20000 --pca_dim 128 --out result_cf/cf_norman_s42
"""
import os, sys, time, json, argparse, math
import numpy as np
import pandas as pd
import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F
import ot as pot
from sklearn.decomposition import PCA
import torchdiffeq

from src.data_process.data import Data, PerturbationDataset
from src.models.origin.blocks import WireRotaryEncoding, apply_rotary, RMSNorm, GeneREPO
from src.data_process.precompute_wire import compute_laplacian_eigvecs
from torch.utils.data import DataLoader
from cell_eval import MetricsEvaluator

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cls(p):
    """EXACT copy of eval_spear.py / night_scan.py: classify a perturbation label as
    ctrl / single / double, so we can report the SPEAR-LOCKED doubles-only metric."""
    p = str(p)
    if "+" not in p:
        return "ctrl"
    a, b = p.split("+", 1)
    return "ctrl" if (a == "control" and b == "control") else \
           ("single" if (a == "control" or b == "control") else "double")


# --------------------------------------------------------------------------- #
# Condition encoder: set of perturbed genes -> embedding (CellFlow set encoder)
# --------------------------------------------------------------------------- #
class SetConditionEncoder(nn.Module):
    """Embed each perturbed-gene id, mask padding/control, pool with a CLS-token
    multi-head attention (CellFlow's default TokenAttentionPooling), project to dim."""
    def __init__(self, num_perturbations, emb_dim=128, out_dim=256, n_heads=8, control_id=None):
        super().__init__()
        self.control_id = control_id
        self.embed = nn.Embedding(num_perturbations, emb_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.normal_(self.cls, std=0.02)
        self.attn = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)
        self.proj = nn.Linear(emb_dim, out_dim)

    def forward(self, cond_ids):                      # cond_ids: (B, S) long
        B, S = cond_ids.shape
        tok = self.embed(cond_ids)                    # (B, S, E)
        # padding mask: True where a slot is padding/control (to be ignored)
        pad = torch.zeros(B, S, dtype=torch.bool, device=cond_ids.device)
        if self.control_id is not None:
            pad = cond_ids.eq(self.control_id)
        cls = self.cls.expand(B, -1, -1)              # (B,1,E)
        seq = torch.cat([cls, tok], dim=1)            # (B, S+1, E)
        key_pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=cond_ids.device), pad], dim=1)
        # guard: if a row is all-padding, unmask its CLS-self so attention is defined
        allpad = key_pad.all(dim=1)
        key_pad[allpad, 0] = False
        out, _ = self.attn(cls, seq, seq, key_padding_mask=key_pad)   # query=CLS
        return self.proj(out[:, 0])                   # (B, out_dim)


def sinusoidal_time(t, dim=1024, max_period=10000.0):
    """CellFlow sinusoidal time encoder (note the t*max_period pre-scaling)."""
    t = t.reshape(-1) * max_period
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * freqs[None, :]
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)        # (B, dim)


def mlp(sizes, act=nn.SiLU, act_last=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or act_last:
            layers.append(act())
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# Vanilla CellFlow velocity field (MLP on the cell embedding)
# --------------------------------------------------------------------------- #
class CellFlowVF(nn.Module):
    def __init__(self, data_dim, cond_dim=256, time_dim=1024,
                 hidden=(1024, 1024, 1024), decoder=(1024, 1024, 1024)):
        super().__init__()
        self.time_dim = time_dim
        self.time_enc = mlp((time_dim,) + tuple(hidden), act_last=True)
        self.x_enc = mlp((data_dim,) + tuple(hidden), act_last=True)
        fused = hidden[-1] * 2 + cond_dim
        self.dec = mlp((fused,) + tuple(decoder), act_last=True)
        self.out = nn.Linear(decoder[-1], data_dim)

    def forward(self, t, x, cond_emb):
        te = self.time_enc(sinusoidal_time(t, self.time_dim))
        xe = self.x_enc(x)
        h = torch.cat([te, xe, cond_emb], dim=-1)
        return self.out(self.dec(h))


# --------------------------------------------------------------------------- #
# Gene-axis velocity field (for the +SPE arm): treats the 1000-gene vector as a
# sequence of gene tokens, so our Spectral Positional Encoding (WIRE) can attach.
# arm 'cellflow_gene' = gene-axis attention, NO SPE (isolates "gene axis" alone);
# arm 'cellflow_spe'  = + WIRE spectral rotary on q/k (SPEAR's actual mechanism).
# --------------------------------------------------------------------------- #
class GeneAttnBlock(nn.Module):
    def __init__(self, d_model, n_heads, use_spe, eigvec_dim, use_repo=False):
        super().__init__()
        self.n_heads, self.hd = n_heads, d_model // n_heads
        self.norm1, self.norm2 = RMSNorm(d_model), RMSNorm(d_model)
        self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model); self.o = nn.Linear(d_model, d_model)
        self.use_spe, self.use_repo = use_spe, use_repo
        if use_spe:
            self.wire = WireRotaryEncoding(eigvec_dim, self.hd, n_heads)
        if use_repo:                                     # SPEAR's Adaptive Repositioning
            self.repo = GeneREPO(d_model, n_heads)
            half = self.hd // 2
            self.register_buffer('freqs', 1.0 / (10000 ** (torch.arange(half).float() / half)))
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.SiLU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, h, coords):                       # h (B,G,d); coords (B,G,E) or None
        B, G, _ = h.shape
        x = self.norm1(h)
        q = self.q(x).view(B, G, self.n_heads, self.hd).transpose(1, 2)   # (B,H,G,hd)
        k = self.k(x).view(B, G, self.n_heads, self.hd).transpose(1, 2)
        v = self.v(x).view(B, G, self.n_heads, self.hd).transpose(1, 2)
        ang = None
        if self.use_spe and coords is not None:
            ang = torch.einsum('ble,hfe->bhlf', coords, self.wire.omega)  # (B,H,G,half) WIRE
        if self.use_repo:
            z = self.repo(x)                             # (B,G,H) perturbation-conditioned scalar
            rep = z.permute(0, 2, 1).unsqueeze(-1) * self.freqs           # (B,H,G,half) RoPE
            ang = rep if ang is None else ang + rep
        if ang is not None:
            q = apply_rotary(q, ang); k = apply_rotary(k, ang)
        out = F.scaled_dot_product_attention(q, k, v)                     # (B,H,G,hd)
        out = out.transpose(1, 2).reshape(B, G, -1)
        h = h + self.o(out)
        h = h + self.mlp(self.norm2(h))
        return h


class GeneAxisVF(nn.Module):
    def __init__(self, n_genes, d_model=128, n_layers=4, n_heads=8, cond_dim=256,
                 time_dim=1024, use_spe=False, eigvec_dim=32, spectral_coords=None,
                 use_repo=False):
        super().__init__()
        self.time_dim = time_dim
        self.gene_embed = nn.Embedding(n_genes, d_model)
        self.val_proj = nn.Linear(1, d_model)
        self.time_mlp = mlp((time_dim, d_model, d_model), act_last=True)
        self.cond_mlp = nn.Linear(cond_dim, d_model)
        self.blocks = nn.ModuleList([GeneAttnBlock(d_model, n_heads, use_spe, eigvec_dim, use_repo)
                                     for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.register_buffer('gene_ids', torch.arange(n_genes))
        self.use_spe = use_spe
        if use_spe:
            assert spectral_coords is not None
            self.register_buffer('coords', spectral_coords)              # (n_genes, E)

    def forward(self, t, x, cond_emb):                  # x (B, n_genes)
        B, G = x.shape
        h = self.gene_embed(self.gene_ids)[None].expand(B, -1, -1) + self.val_proj(x[..., None])
        h = h + self.time_mlp(sinusoidal_time(t, self.time_dim))[:, None, :]
        h = h + self.cond_mlp(cond_emb)[:, None, :]
        coords = self.coords[None].expand(B, -1, -1) if self.use_spe else None
        for blk in self.blocks:
            h = blk(h, coords)
        return self.head(self.norm(h)).squeeze(-1)      # (B, n_genes)


# --------------------------------------------------------------------------- #
# Learned encoder/decoder for the AE-latent arms: REPLACE CellFlow's fixed PCA
# encode / inverse-PCA decode with our SPE-aware gene-axis encoder + a learned
# decoder, keeping the efficient LOW-DIM latent flow (the reason vanilla CellFlow
# works). SPE attaches inside the encoder so it shapes the latent itself, instead
# of being bolted onto a hard 1000-dim gene-space flow.
# --------------------------------------------------------------------------- #
class GeneEncoder(nn.Module):
    """Gene-axis transformer over gene tokens -> mean-pool -> latent (B,G)->(B,L)."""
    def __init__(self, n_genes, latent_dim, d_model=128, n_layers=3, n_heads=8,
                 use_spe=False, eigvec_dim=32, spectral_coords=None):
        super().__init__()
        self.gene_embed = nn.Embedding(n_genes, d_model)
        self.val_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([GeneAttnBlock(d_model, n_heads, use_spe, eigvec_dim, False)
                                     for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.to_latent = nn.Linear(d_model, latent_dim)
        self.register_buffer('gene_ids', torch.arange(n_genes))
        self.use_spe = use_spe
        if use_spe:
            assert spectral_coords is not None
            self.register_buffer('coords', spectral_coords)              # (n_genes, E)

    def forward(self, x):                               # x (B, n_genes)
        B, G = x.shape
        h = self.gene_embed(self.gene_ids)[None].expand(B, -1, -1) + self.val_proj(x[..., None])
        coords = self.coords[None].expand(B, -1, -1) if self.use_spe else None
        for blk in self.blocks:
            h = blk(h, coords)
        return self.to_latent(self.norm(h).mean(dim=1))                  # (B, latent_dim)


class GeneDecoder(nn.Module):
    """Learned latent -> gene expression (replaces inverse-PCA); (B,L)->(B,G)."""
    def __init__(self, latent_dim, n_genes, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, n_genes))

    def forward(self, z):
        return self.net(z)


# --------------------------------------------------------------------------- #
# minibatch OT coupling (CellFlow's match_linear): sqeuclidean cost -> reorder tgt
# --------------------------------------------------------------------------- #
def ot_reorder(x0, x1, reg=0.0):
    """Return x1 reordered so rows are OT-matched to x0 (squared-euclidean cost)."""
    with torch.no_grad():
        a = x0.detach().cpu().numpy().astype(np.float64)
        b = x1.detach().cpu().numpy().astype(np.float64)
        M = pot.dist(a, b, metric='sqeuclidean')
        M = M / (M.mean() + 1e-8)                     # scale_cost='mean'
        n = a.shape[0]
        u = np.ones(n) / n
        if reg and reg > 0:
            P = pot.sinkhorn(u, u, M, reg)
        else:
            P = pot.emd(u, u, M)
        tgt_idx = P.argmax(axis=1)                    # sample_joint (argmax)
    return x1[torch.as_tensor(tgt_idx, device=x1.device)]


# --------------------------------------------------------------------------- #
class PCAProj:
    def __init__(self, X, k):
        self.pca = PCA(n_components=k, svd_solver='randomized', random_state=0).fit(X)
        self.mean = torch.tensor(self.pca.mean_, dtype=torch.float32, device=DEV)
        self.comp = torch.tensor(self.pca.components_, dtype=torch.float32, device=DEV)  # (k, G)
    def to_z(self, x):    return (x - self.mean) @ self.comp.T          # (B,G)->(B,k)
    def to_x(self, z):    return z @ self.comp + self.mean              # (B,k)->(B,G)


def build_model(arm, data_dim, num_pert, control_id, cfg, spectral_coords=None):
    cond = SetConditionEncoder(num_pert, emb_dim=cfg.cond_emb, out_dim=cfg.cond_dim,
                               control_id=control_id).to(DEV)
    if arm in ('cellflow', 'cellflow_ae', 'cellflow_ae_spe'):
        # PCA arm and AE arms both flow in a LOW-DIM latent -> MLP velocity field.
        vf = CellFlowVF(data_dim, cond_dim=cfg.cond_dim,
                        hidden=(cfg.hidden,) * 3, decoder=(cfg.hidden,) * 3).to(DEV)
    elif arm in ('cellflow_gene', 'cellflow_spe', 'cellflow_spear'):
        use_spe = arm in ('cellflow_spe', 'cellflow_spear')
        use_repo = arm == 'cellflow_spear'
        vf = GeneAxisVF(data_dim, d_model=cfg.d_model, n_layers=cfg.n_layers,
                        n_heads=cfg.n_heads, cond_dim=cfg.cond_dim, time_dim=1024,
                        use_spe=use_spe, eigvec_dim=cfg.eigvec_dim,
                        spectral_coords=spectral_coords, use_repo=use_repo).to(DEV)
    else:
        raise ValueError(arm)
    return cond, vf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='cellflow',
                    choices=['cellflow', 'cellflow_gene', 'cellflow_spe', 'cellflow_spear',
                             'cellflow_ae', 'cellflow_ae_spe'])
    ap.add_argument('--split_file', default='')
    ap.add_argument('--data_name', default='norman')
    ap.add_argument('--split_method', default='additive')
    ap.add_argument('--fold', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--pca_dim', type=int, default=128)
    ap.add_argument('--cond_emb', type=int, default=128)
    ap.add_argument('--cond_dim', type=int, default=256)
    ap.add_argument('--hidden', type=int, default=1024)      # MLP arm width
    ap.add_argument('--d_model', type=int, default=128)      # gene-axis arms
    ap.add_argument('--n_layers', type=int, default=4)
    ap.add_argument('--n_heads', type=int, default=8)
    ap.add_argument('--eigvec_dim', type=int, default=32)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--ot_reg', type=float, default=0.0)   # 0 -> exact EMD; >0 -> sinkhorn
    ap.add_argument('--use_ot', type=int, default=1)
    ap.add_argument('--eval_every', type=int, default=5000)
    ap.add_argument('--n_eval_ctrl', type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--eval_threads', type=int, default=32)
    ap.add_argument('--gene_norm', default='none', choices=['none', 'zscore'])  # standardize gene space
    ap.add_argument('--ae_latent', type=int, default=128)  # AE-arm latent dim (learned enc/dec)
    ap.add_argument('--ae_recon_w', type=float, default=1.0)
    ap.add_argument('--ae_steps', type=int, default=4000)  # staged AE reconstruction pretrain steps (warm start)
    ap.add_argument('--ae_lr', type=float, default=1e-3)
    ap.add_argument('--ae_joint', type=int, default=0)     # 1: don't freeze AE; co-train enc/dec with flow
                                                           #    (loss = flow + ae_recon_w*recon). 0: staged/frozen.
    ap.add_argument('--out', default='result_cf/dbg')
    cfg = ap.parse_args()

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    os.makedirs(cfg.out, exist_ok=True)
    print(f"[cfg] {vars(cfg)}", flush=True)
    t0 = time.time()

    # ---- data (identical to SPEAR) ----
    dm = Data('./data'); dm.load_data(cfg.data_name)
    dm.process_data(n_top_genes=5000, infer_top_gene=1000, split_method=cfg.split_method,
                    fold=cfg.fold, use_negative_edge=True, k=30,
                    split_file=cfg.split_file, train_frac=1.0)
    train_sampler, test_dl, _ = dm.load_flow_data(batch_size=cfg.batch_size)
    pert_dict = dm.perturbation_dict
    num_pert = max(pert_dict.values()) + 1
    control_id = pert_dict.get('control', None)

    # SPEAR trains on n_top_genes (~5029) but cell_eval scores the infer_top_gene (~1000)
    # subset. Operate CellFlow in that SAME 1000-gene eval space so predictions are directly
    # comparable on the exact genes SPEAR is scored on (the 1000 test genes are a subset of
    # the 5029 train genes). gene_idx maps test-gene order -> train-column index.
    test_genes = list(test_dl.adata.var_names)
    train_genes = list(train_sampler.adata.var_names)
    pos = {g: i for i, g in enumerate(train_genes)}
    missing = [g for g in test_genes if g not in pos]
    assert not missing, f"{len(missing)} eval genes not in train space, e.g. {missing[:5]}"
    gene_idx_np = np.array([pos[g] for g in test_genes], dtype=np.int64)
    gene_idx = torch.as_tensor(gene_idx_np, device=DEV)
    G = len(test_genes)
    print(f"[data] {time.time()-t0:.0f}s  train_genes={len(train_genes)} eval_genes={G} "
          f"num_pert={num_pert} control_id={control_id}", flush=True)

    # ---- training expression restricted to the 1000 eval genes ----
    Xtr = dm.adata_train.X
    Xtr = Xtr.toarray() if hasattr(Xtr, 'toarray') else np.asarray(Xtr)
    Xtr = Xtr[:, gene_idx_np].astype(np.float32)

    # Representation: PCA cell-embedding (vanilla CellFlow) vs raw gene space (gene-axis arms).
    spectral = None
    ae_encoder = ae_decoder = None       # set (kept trainable) only in the ae_joint path
    if cfg.arm == 'cellflow':
        pca = PCAProj(Xtr, cfg.pca_dim)
        data_dim = cfg.pca_dim
        enc, dec = pca.to_z, pca.to_x
        print(f"[rep] PCA k={cfg.pca_dim} explained_var={pca.pca.explained_variance_ratio_.sum():.3f}", flush=True)
    elif cfg.arm in ('cellflow_ae', 'cellflow_ae_spe'):
        # OUR learned encoder/decoder replacing PCA. Flow runs in this learned latent.
        data_dim = cfg.ae_latent
        use_spe = cfg.arm == 'cellflow_ae_spe'
        if use_spe:
            coords_np = compute_laplacian_eigvecs(Xtr, cfg.eigvec_dim, k=30)
            spectral = torch.tensor(np.asarray(coords_np), dtype=torch.float32, device=DEV)
        encoder = GeneEncoder(G, cfg.ae_latent, d_model=cfg.d_model, n_layers=cfg.n_layers,
                              n_heads=cfg.n_heads, use_spe=use_spe, eigvec_dim=cfg.eigvec_dim,
                              spectral_coords=spectral).to(DEV)
        decoder = GeneDecoder(cfg.ae_latent, G, hidden=cfg.hidden).to(DEV)
        # ---- STAGE 1: pretrain the AE with reconstruction, then FREEZE (staged training) ----
        ae_opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.ae_lr)
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
        encoder.train(); decoder.train()
        for astep in range(cfg.ae_steps):
            idx = torch.randint(0, Xtr_t.shape[0], (cfg.batch_size,), device=DEV)
            xb = Xtr_t[idx]
            xhat = decoder(encoder(xb))
            aloss = cfg.ae_recon_w * F.mse_loss(xhat, xb)
            ae_opt.zero_grad(); aloss.backward(); ae_opt.step()
            if (astep + 1) % 1000 == 0:
                print(f"[ae {astep+1}/{cfg.ae_steps}] recon_mse={aloss.item():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        if cfg.ae_joint:
            enc = lambda x: encoder(x)                  # LIVE: grad flows -> latent shaped by flow too
            dec = lambda z: decoder(z)
            ae_encoder, ae_decoder = encoder, decoder   # -> added to optimizer + recon-anchor loss
            print(f"[rep] LEARNED AE latent={cfg.ae_latent} spe={use_spe} JOINT "
                  f"(warm-start {cfg.ae_steps}, then co-train w/ flow; recon_w={cfg.ae_recon_w})", flush=True)
        else:
            encoder.eval(); decoder.eval()
            for p in encoder.parameters(): p.requires_grad_(False)
            for p in decoder.parameters(): p.requires_grad_(False)
            enc = lambda x: encoder(x).detach()         # frozen learned encoder (staged)
            dec = lambda z: decoder(z)                  # frozen learned decoder
            print(f"[rep] LEARNED AE latent={cfg.ae_latent} spe={use_spe} "
                  f"(staged pretrain {cfg.ae_steps} steps, then frozen)", flush=True)
        # ---- CEILING probe: decode(encode(REAL)) doubles-only DE-LFC. The frozen decoder
        # caps the whole arm -- flow can NEVER beat this. If < PCA's 0.718 the AE latent is
        # losing the DE genes (recon-MSE is housekeeping-dominated) and the approach can't win.
        try:
            def _rec(x):
                with torch.no_grad():
                    return dec(enc(torch.tensor(x, dtype=torch.float32, device=DEV))).cpu().numpy()
            ctrlc = test_dl.get_control_data()['src_cell_data'].numpy()
            allp, allr = [_rec(ctrlc)], [ctrlc]
            npn, nrn = ['control'] * ctrlc.shape[0], ['control'] * ctrlc.shape[0]
            for c in list(test_dl._perturbation_covariates):
                tg = test_dl.get_perturbation_data(c)['tgt_cell_data'].numpy()
                allp.append(_rec(tg)); allr.append(tg)
                npn += [c] * tg.shape[0]; nrn += [c] * tg.shape[0]
            P = np.nan_to_num(np.concatenate(allp, 0), nan=0., posinf=0., neginf=0.)
            R = np.concatenate(allr, 0)
            cdir = os.path.join(cfg.out, 'ae_ceiling'); os.makedirs(cdir, exist_ok=True)
            evc = MetricsEvaluator(adata_pred=ad.AnnData(X=P, obs=pd.DataFrame({'perturbation': npn})),
                                   adata_real=ad.AnnData(X=R, obs=pd.DataFrame({'perturbation': nrn})),
                                   control_pert='control', pert_col='perturbation',
                                   num_threads=cfg.eval_threads, outdir=cdir)
            resc, _ = evc.compute()
            rdf = resc.to_pandas() if hasattr(resc, 'to_pandas') else pd.DataFrame(resc)
            rdf.to_csv(os.path.join(cdir, 'results.csv'), index=False)
            pcol = 'perturbation' if 'perturbation' in rdf.columns else rdf.columns[0]
            db = rdf[rdf[pcol].apply(cls) == 'double']
            kk = ['de_spearman_lfc_sig', 'de_direction_match', 'discrimination_score_cosine', 'pearson_delta']
            print("[ae_ceiling] " + "  ".join(f"{k}={db[k].mean():.4f}" for k in kk if k in db.columns)
                  + f"   (recon of REAL, doubles n={len(db)}; flow can't beat this)", flush=True)
        except Exception as e:
            print(f"[ae_ceiling] FAILED: {e}", flush=True)
    else:
        data_dim = G
        if cfg.gene_norm == 'zscore':
            # Standardize gene space (per-gene mean/std) so flow-matching MSE weights all
            # genes equally (not dominated by high-expression housekeeping genes) -- mirrors
            # what PCA centering/whitening does, but keeps the gene axis so SPE can attach.
            gmean = torch.tensor(Xtr.mean(0), dtype=torch.float32, device=DEV)
            gstd = torch.tensor(Xtr.std(0) + 1e-6, dtype=torch.float32, device=DEV)
            enc = lambda x: (x - gmean) / gstd
            dec = lambda z: z * gstd + gmean
            print("[rep] gene-space Z-SCORED (per-gene mean/std)", flush=True)
        else:
            enc = (lambda x: x); dec = (lambda z: z)            # flow directly in raw gene space
        if cfg.arm in ('cellflow_spe', 'cellflow_spear'):
            coords_np = compute_laplacian_eigvecs(Xtr, cfg.eigvec_dim, k=30)   # (G, eigvec_dim)
            spectral = torch.tensor(np.asarray(coords_np), dtype=torch.float32, device=DEV)
            print(f"[rep] gene-space arm={cfg.arm} SPE coords {tuple(spectral.shape)} "
                  f"repo={cfg.arm=='cellflow_spear'}", flush=True)
        else:
            print(f"[rep] gene-space (no SPE)", flush=True)

    cond_enc, vf = build_model(cfg.arm, data_dim, num_pert, control_id, cfg, spectral)
    params = list(cond_enc.parameters()) + list(vf.parameters())
    if ae_encoder is not None:                          # ae_joint: co-train encoder+decoder
        params += list(ae_encoder.parameters()) + list(ae_decoder.parameters())
        ae_encoder.train(); ae_decoder.train()
    opt = torch.optim.Adam(params, lr=cfg.lr)
    nparam = sum(p.numel() for p in params)
    print(f"[model] arm={cfg.arm} params={nparam/1e6:.2f}M", flush=True)

    train_ds = PerturbationDataset(train_sampler, cfg.batch_size, cell_type_col='cell_type')
    dl = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers,
                    pin_memory=True, persistent_workers=(cfg.num_workers > 0))

    def run_eval(tag):
        cond_enc.eval(); vf.eval()
        ctrl = test_dl.get_control_data()['src_cell_data'].numpy()   # (n,G)
        conds = list(test_dl._perturbation_covariates)
        rng = np.random.default_rng(0)
        allp, allr = [ctrl], [ctrl]
        np_names, nr_names = ['control'] * ctrl.shape[0], ['control'] * ctrl.shape[0]
        for c in conds:
            pdata = test_dl.get_perturbation_data(c)
            target = pdata['tgt_cell_data'].numpy()
            cid = pdata['condition_id']                              # (n_tgt, S)
            cid0 = cid[0:1].to(DEV).long()                          # one condition's set
            idx = rng.permutation(ctrl.shape[0])[:cfg.n_eval_ctrl]
            src = torch.tensor(ctrl[idx], dtype=torch.float32, device=DEV)
            with torch.no_grad():
                z0 = enc(src)                                        # (n, data_dim)
                ce = cond_enc(cid0.expand(z0.shape[0], -1))          # (n,cond)
                def odef(t, z):
                    tt = t.expand(z.shape[0]) if t.ndim == 0 else t
                    return vf(tt, z, ce)
                zT = torchdiffeq.odeint(odef, z0, torch.tensor([0.0, 1.0], device=DEV),
                                        method='dopri5', rtol=1e-4, atol=1e-4)[-1]
                pred = dec(zT).cpu().numpy()
            allp.append(pred); np_names += [c] * pred.shape[0]
            allr.append(target); nr_names += [c] * target.shape[0]
        P = np.nan_to_num(np.concatenate(allp, 0), nan=0.0, posinf=0.0, neginf=0.0)
        R = np.concatenate(allr, 0)
        pred = ad.AnnData(X=P, obs=pd.DataFrame({'perturbation': np_names}))
        real = ad.AnnData(X=R, obs=pd.DataFrame({'perturbation': nr_names}))
        odir = os.path.join(cfg.out, f'eval_{tag}'); os.makedirs(odir, exist_ok=True)
        ev = MetricsEvaluator(adata_pred=pred, adata_real=real, control_pert='control',
                              pert_col='perturbation', num_threads=cfg.eval_threads, outdir=odir)
        results, agg = ev.compute()
        keys = ['de_spearman_lfc_sig', 'de_direction_match', 'de_spearman_sig',
                'pearson_delta', 'mse_delta', 'discrimination_score_cosine']
        adf = agg.to_pandas() if hasattr(agg, 'to_pandas') else pd.DataFrame(agg)
        m = adf[adf['statistic'] == 'mean'].iloc[0]
        row = {k: float(m[k]) for k in keys if k in m}
        print(f"[evalAll {tag}] " + "  ".join(f"{k}={v:.4f}" for k, v in row.items()), flush=True)
        adf.to_csv(os.path.join(odir, 'agg.csv'), index=False)
        # ---- SPEAR-LOCKED doubles-only metric: identical to eval_spear.py (per-condition
        # results.csv -> cls() -> average the A+B (both non-control) rows). This is THE
        # number directly comparable to SPEAR's 0.852; [evalAll] above is context only.
        drow = row
        rdf = results.to_pandas() if hasattr(results, 'to_pandas') else pd.DataFrame(results)
        rdf.to_csv(os.path.join(odir, 'results.csv'), index=False)
        pcol = 'perturbation' if 'perturbation' in rdf.columns else rdf.columns[0]
        rdf['_c'] = rdf[pcol].apply(cls)
        dbl = rdf[rdf['_c'] == 'double']
        if len(dbl):
            drow = {k: float(dbl[k].mean()) for k in keys if k in dbl.columns}
            print(f"[eval {tag}] " + "  ".join(f"{k}={v:.4f}" for k, v in drow.items())
                  + f"   (doubles-only n={len(dbl)})", flush=True)
        else:
            print(f"[eval {tag}] WARN no double conditions in results", flush=True)
        cond_enc.train(); vf.train()
        return drow

    # ---- training ----
    step = 0; loss_ema = None
    cond_enc.train(); vf.train()
    while step < cfg.steps:
        for batch in dl:
            src = batch['src_cell_data'].squeeze(0).to(DEV)      # (B, 5029)
            tgt = batch['tgt_cell_data'].squeeze(0).to(DEV)      # (B, 5029)
            cid = batch['condition_id'].squeeze(0).to(DEV).long()  # (B,S)
            if src.shape[0] < 4 or tgt.shape[0] < 4:
                continue
            src = src.index_select(1, gene_idx)                  # (B, 1000) eval-gene space
            tgt = tgt.index_select(1, gene_idx)
            z0 = enc(src); z1 = enc(tgt)
            n = min(z0.shape[0], z1.shape[0])
            z0, z1, cid = z0[:n], z1[:n], cid[:n]
            recon = None
            if ae_encoder is not None:                  # recon anchor (BEFORE OT reorders z1)
                recon = F.mse_loss(dec(z0), src[:n]) + F.mse_loss(dec(z1), tgt[:n])
            if cfg.use_ot:
                z1 = ot_reorder(z0, z1, reg=cfg.ot_reg)
            t = torch.rand(n, device=DEV)
            x_t = (1 - t)[:, None] * z0 + t[:, None] * z1
            u_t = z1 - z0
            ce = cond_enc(cid)
            v = vf(t, x_t, ce)
            loss = F.mse_loss(v, u_t)
            if recon is not None:
                loss = loss + cfg.ae_recon_w * recon
            opt.zero_grad(); loss.backward(); opt.step()
            loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
            step += 1
            if step % 500 == 0:
                print(f"[step {step}/{cfg.steps}] loss={loss.item():.4f} ema={loss_ema:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if step % cfg.eval_every == 0 or step == cfg.steps:
                run_eval(str(step))
            if step >= cfg.steps:
                break
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
