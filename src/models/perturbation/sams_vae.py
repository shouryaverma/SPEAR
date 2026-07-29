"""
Sparse Additive Mechanism Shift (SAMS-VAE style) head for SPEAR.

Motivation
----------
This is the *second* latent-causal route for SPEAR v2, running in parallel to IPLC
(`latent_scm.py`). Both target the same PSB "mechanism + interpretability" pitch,
but they draw identifiability from DIFFERENT sources so they can cross-validate:

  - IPLC   : latent factors + an explicit factor->factor causal DAG, made
             identifiable by *hard-anchoring* each factor to a gene (A == gene x
             gene regulatory matrix, directly comparable to TRRUST).
  - SAMS-VAE: no DAG. Every perturbation is a *sparse additive mechanism shift* in
             a latent space, and identifiability comes from the SPARSE-MECHANISM
             PRIOR + additive structure + perturbation diversity (Bereket &
             Karaletsos, NeurIPS 2023; Lachapelle et al.).

What it does (adapted to flow-matching, not a VAE)
--------------------------------------------------
We add a latent mechanism code as a residual on the mean-pooled perturbation embedding:

    z = sum_{p perturbed}  mask_p (.) shift_hat_p          # sparse additive shift
    code = to_dmodel(z)                                    # decode to the d_model residual
    perturbation_emb <- perturbation_emb + code

mask_p and shift_p are GLOBAL, per-perturbable-gene parameters (indexed by the gene's
vocab token id), NOT amortized from the token embedding -- so `mask_p` is a stable,
readable property of perturbation p ("which latent dims gene p engages"). control /
non-perturbable slots map to row -1 and contribute nothing, so z(A+B) = z(A) + z(B)
holds by construction (compositional). `to_dmodel` is zero-initialized => the whole
head is an EXACT no-op residual at init (fair baseline comparison).

IDENTIFIABILITY -- the scale gauge and how it is pinned (preflight-review fix)
-----------------------------------------------------------------------------
The data path only sees the PRODUCT mask (.) shift, which admits a scale gauge:
(mask/c, shift*c) leaves the mechanism (hence the fit) invariant. Left unpinned, an
L1 penalty on a *continuous* mask is minimized for free by shrinking every mask
toward 0 while shift (unregularized) inflates to compensate -- the exported mask then
reads numerically empty for GAUGE reasons, not biology, and z_norm/code_norm (both
gauge-invariant) cannot detect it. Two defenses, applied here:

  1. `shift` is L2-normalized per row (unit direction) inside forward. The magnitude
     can no longer migrate from mask into the raw shift parameter -> the per-element
     (mask/c, shift*c) gauge is killed for BOTH priors.
  2. DEFAULT prior is 'concrete' (hard-concrete L0, Louizos et al. 2018). Its penalty
     is expected-L0 = sum sigmoid(logit - const), a function of the LOGITS ONLY
     (structure: which dims are on), invariant to any magnitude rescaling of shift or
     to_dmodel. So it rewards only genuine STRUCTURAL sparsification and is immune to
     both the per-element gauge AND the residual (z/c, to_dmodel*c) decode gauge.
     The 'l1' prior is kept as an ablation: shift-normalization kills its per-element
     gauge, but it retains the decode-scale gauge, so read it via the gauge-INVARIANT
     `mech_l1` metric in stats() (not the raw mask), and prefer 'concrete' for claims.

We deliberately do NOT normalize z (BatchNorm-on-z), which would make the fit
invariant to mask scale and thereby *aid* the L1 collapse.

Collapse observability: stats() reports gauge-invariant `mech_l1` and (for concrete)
the hard active-dim count, so a genuine collapse to an empty mechanism is visible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hard-concrete (Louizos et al. 2018) stretch constants.
_HC_BETA = 2.0 / 3.0     # temperature of the binary-concrete relaxation
_HC_GAMMA = -0.1         # stretch lower bound (< 0 so gates can hit exact 0)
_HC_ZETA = 1.1           # stretch upper bound (> 1 so gates can hit exact 1)


class SparseAdditiveMechanismShift(nn.Module):
    def __init__(self, d_model: int, latent_dim: int = 64,
                 pert_vocab_ids=None, vocab_size: int = None,
                 control_id: int = None, mask_prior: str = 'concrete',
                 shift_init_std: float = 0.1, gen_seed: int = 20230608):
        super().__init__()
        assert mask_prior in ('l1', 'concrete')
        assert pert_vocab_ids is not None and len(pert_vocab_ids) > 0, \
            "SAMS-VAE needs the perturbable-gene vocab ids (from run.py _scm_pert_vocab_ids)."
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.mask_prior = mask_prior
        self.control_id = control_id
        self._gen_seed = gen_seed
        self._generators = {}          # lazy per-device torch.Generator (RNG-clean concrete)

        # Global per-perturbable-gene parameter tables. Row k == perturbation gene
        # pert_vocab_ids[k]. The vocab-token -> row map sends control / any token
        # not in the perturbable set to -1 (contributes nothing).
        n_pert = len(pert_vocab_ids)
        self.n_pert = n_pert
        vs = vocab_size or (int(max(list(pert_vocab_ids) + [control_id or 0])) + 1)
        p2r = torch.full((vs,), -1, dtype=torch.long)
        for _k, _vid in enumerate(pert_vocab_ids):
            p2r[int(_vid)] = _k
        self.register_buffer('pert_to_row', p2r)
        self.register_buffer('row_to_vocab', torch.tensor([int(v) for v in pert_vocab_ids],
                                                          dtype=torch.long))

        # shift_p (mechanism DIRECTION; L2-normalized per row in forward -> the scale
        # gauge cannot hide magnitude here) and mask logits (sparse gate) per gene.
        self.shift = nn.Parameter(torch.randn(n_pert, latent_dim) * shift_init_std)
        # logit 0 -> sigmoid 0.5 (masks start half-on); L1/L0 penalty prunes them.
        self.mask_logits = nn.Parameter(torch.zeros(n_pert, latent_dim))

        # Decode the latent mechanism code back to d_model. Zero-init => exact no-op
        # residual at init (baseline-preserving), same convention as IPLC.
        self.to_dmodel = nn.Linear(latent_dim, d_model)
        nn.init.zeros_(self.to_dmodel.weight)
        nn.init.zeros_(self.to_dmodel.bias)

    # -- RNG (dedicated generator so 'concrete' does NOT consume the global stream) --
    def _get_generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        gen = self._generators.get(key)
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(self._gen_seed)
            self._generators[key] = gen
        return gen

    def _shift_hat(self) -> torch.Tensor:
        """Per-row unit-normalized shift direction (kills the per-element scale gauge)."""
        return F.normalize(self.shift, dim=-1, eps=1e-8)

    # -- mask probabilities -------------------------------------------------
    def _mask_table(self) -> torch.Tensor:
        """Deterministic mask gates for the full (n_pert, L) table.
        'l1' -> sigmoid(logits); 'concrete' -> hard-concrete median (deterministic) gate."""
        if self.mask_prior == 'l1':
            return torch.sigmoid(self.mask_logits)
        s_bar = torch.sigmoid(self.mask_logits) * (_HC_ZETA - _HC_GAMMA) + _HC_GAMMA
        return s_bar.clamp(0.0, 1.0)

    def _mask_rows(self, rows_idx: torch.Tensor) -> torch.Tensor:
        """Per-token mask gates for gathered rows (B, n_comb) -> (B, n_comb, L).
        In 'concrete' training mode draws relaxed-Bernoulli gates from the dedicated
        generator (does not touch the global RNG stream a base/l1 run sees)."""
        logits = self.mask_logits[rows_idx]                      # (B, n_comb, L)
        if self.mask_prior == 'l1':
            return torch.sigmoid(logits)
        if self.training:
            gen = self._get_generator(logits.device)
            u = torch.rand(logits.shape, generator=gen, device=logits.device,
                           dtype=logits.dtype).clamp(1e-6, 1.0 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1.0 - u) + logits) / _HC_BETA)
        else:
            s = torch.sigmoid(logits)
        s_bar = s * (_HC_ZETA - _HC_GAMMA) + _HC_GAMMA
        return s_bar.clamp(0.0, 1.0)

    # -- forward ------------------------------------------------------------
    def forward(self, pert_ids: torch.Tensor, tok_emb: torch.Tensor = None) -> torch.Tensor:
        """
        pert_ids: (B, n_comb) vocab token ids. tok_emb is accepted for interface
        parity with IPLC but unused (the shift is a global gene property, by design).
        Returns the latent mechanism code as a d_model residual (B, d_model).
        """
        rows = self.pert_to_row[pert_ids]                        # (B, n_comb), -1 = control
        valid = (rows >= 0).to(self.shift.dtype).unsqueeze(-1)   # (B, n_comb, 1)
        idx = rows.clamp(min=0)                                  # safe gather index
        shift = self._shift_hat()[idx]                           # (B, n_comb, L) unit dirs
        mask = self._mask_rows(idx)                              # (B, n_comb, L)
        mech = mask * shift * valid                              # control slots zeroed
        z = mech.sum(dim=1)                                      # (B, L) additive shift
        code = self.to_dmodel(z)                                 # (B, d_model), no-op at init
        self._last_z = z
        self._last_mech = mech
        return code

    # -- regularization -----------------------------------------------------
    def sparsity_penalty(self, lambda_mask: float) -> torch.Tensor:
        """Sparse-mechanism-shift prior on the GLOBAL mask table (per-gene, batch
        independent for a stable objective). 'l1' -> L1 on mask probs; 'concrete' ->
        expected-L0 = sum sigmoid(logit - beta*log(-gamma/zeta)), a gauge-invariant
        count of structurally-active dims. Summed over dims, averaged over genes."""
        if lambda_mask <= 0:
            return self.shift.new_zeros(())
        if self.mask_prior == 'l1':
            pen = torch.sigmoid(self.mask_logits).sum(dim=-1).mean()
        else:
            shift_c = _HC_BETA * torch.log(torch.tensor(-_HC_GAMMA / _HC_ZETA,
                                                        device=self.mask_logits.device))
            p_active = torch.sigmoid(self.mask_logits - shift_c)
            pen = p_active.sum(dim=-1).mean()
        return lambda_mask * pen

    # -- diagnostics / interpretability readout -----------------------------
    @torch.no_grad()
    def stats(self, pert_ids: torch.Tensor) -> dict:
        """Activity probe. z_norm/code_norm are gauge-INVARIANT (cannot see a scale
        collapse); `mech_l1` and `active_dims` are gauge-invariant STRUCTURE metrics
        that DO reveal an empty-mechanism collapse. shift is unit-norm per row."""
        code = self.forward(pert_ids)
        mask_tbl = self._mask_table()                            # (n_pert, L) deterministic
        # hard structural gate (concrete): dim is "on" if its median gate exceeds 0.5.
        active = (mask_tbl > 0.5).float()
        return {
            "code_norm_mean": float(code.norm(dim=-1).mean()),
            "code_norm_max": float(code.norm(dim=-1).max()),
            "z_norm_mean": float(self._last_z.norm(dim=-1).mean()),
            # gauge-invariant collapse probes: mean L1 of the actual mechanism per pert,
            # and mean #structurally-active dims per pert. If these -> 0 the mask is empty.
            "mech_l1_mean": float(self._last_mech.abs().sum(dim=-1).sum(dim=-1).mean()),
            "active_dims_mean": float(active.sum(dim=-1).mean()),
            "mask_mean": float(mask_tbl.mean()),
            "shift_absmax": float(self.shift.abs().max()),
        }

    @torch.no_grad()
    def export_masks(self):
        """Interpretability export for GRN/DE alignment + seed-stability analysis:
        (mask_probs [n_pert, L], shift_hat [n_pert, L] unit dirs,
         mechanism = mask (.) shift_hat [n_pert, L], row_to_vocab [n_pert])."""
        mask = self._mask_table().detach()
        shat = self._shift_hat().detach()
        mech = (mask * shat)
        return mask.cpu(), shat.cpu(), mech.cpu(), self.row_to_vocab.detach().cpu()
