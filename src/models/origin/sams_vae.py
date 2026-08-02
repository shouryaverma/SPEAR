"""
Sparse Additive Mechanism Shift (SAMS-VAE style) head for SPEAR.
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

    @torch._dynamo.disable
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
        """Activity probe. Deterministic: reads the median gate table directly instead of
        calling forward(), so instrumentation never advances the concrete gate generator
        and the training trajectory is identical with or without stats printing.
        z_norm/code_norm are gauge-INVARIANT; mech_l1/active_dims are gauge-invariant
        STRUCTURE metrics that DO reveal an empty-mechanism collapse."""
        mask_tbl = self._mask_table()                            # (n_pert, L) deterministic
        shat = self._shift_hat()
        rows = self.pert_to_row[pert_ids]                        # (B, n_comb), -1 = control
        valid = (rows >= 0).to(shat.dtype).unsqueeze(-1)
        idx = rows.clamp(min=0)
        mech = mask_tbl[idx] * shat[idx] * valid                 # (B, n_comb, L)
        z = mech.sum(dim=1)
        code = self.to_dmodel(z)
        active = (mask_tbl > 0.5).float()
        return {
            "code_norm_mean": float(code.norm(dim=-1).mean()),
            "code_norm_max": float(code.norm(dim=-1).max()),
            "z_norm_mean": float(z.norm(dim=-1).mean()),
            "mech_l1_mean": float(mech.abs().sum(dim=-1).sum(dim=-1).mean()),
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
