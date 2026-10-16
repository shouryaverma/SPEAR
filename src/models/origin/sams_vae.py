"""Sparse Additive Mechanism Shift head (SAMS-VAE style) for the SPEAR origin model.

Every perturbable gene k owns a unit-norm latent direction shift_hat[k] and a sparse gate
mask[k]. The mechanism of a gene is mask[k] * shift_hat[k]. A condition is the SUM of the
mechanisms of its genes (additive in latent space); control slots contribute nothing. A
zero-initialised linear layer decodes the latent sum to d_model, so the head is an exact
no-op residual at initialisation.

Gauge notes:
  * shift is L2-normalised per row, so its scale is not free.
  * the 'concrete' prior penalises expected-L0 of the gates, which depends only on the
    logits (not on any scale). The 'l1' prior penalises sigmoid gates and is not fully
    scale-free because the decoder can absorb gate scale.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hard-concrete (Louizos et al. 2018) stretch constants.
HC_BETA = 2.0 / 3.0
HC_GAMMA = -0.1
HC_ZETA = 1.1
# P(gate != 0) = sigmoid(logit - HC_L0_OFFSET)
HC_L0_OFFSET = HC_BETA * math.log(-HC_GAMMA / HC_ZETA)


def mask_gate(mask_logits, prior):
    """Deterministic gate for a logit table. 'l1': sigmoid. 'concrete': hard-concrete median."""
    if prior == 'l1':
        return torch.sigmoid(mask_logits)
    gate = torch.sigmoid(mask_logits) * (HC_ZETA - HC_GAMMA) + HC_GAMMA
    return gate.clamp(0.0, 1.0)


def normalize_shift(shift):
    """Per-row unit-normalised shift direction."""
    return F.normalize(shift, dim=-1, eps=1e-8)


class SparseAdditiveMechanismShift(nn.Module):
    def __init__(self, d_model, vocab_size, pert_vocab_ids, latent_dim=64,
                 mask_prior='concrete', shift_init_std=0.1, generator_seed=20230608):
        super().__init__()
        assert mask_prior in ('l1', 'concrete'), mask_prior
        ids = [int(v) for v in pert_vocab_ids] if pert_vocab_ids is not None else []
        assert len(ids) > 0, "SAMS-VAE needs the perturbable-gene vocab ids"
        assert len(set(ids)) == len(ids), "duplicate perturbable vocab ids"
        assert min(ids) >= 0 and max(ids) < vocab_size, "perturbable vocab id outside vocab"

        self.d_model = d_model
        self.latent_dim = latent_dim
        self.mask_prior = mask_prior
        self.n_pert = len(ids)
        self.generator_seed = generator_seed
        self.generators = {}

        # Row k of every table is the perturbation gene with vocab id ids[k]. Tokens that
        # are not perturbable (control, any other gene) map to row -1 and contribute nothing.
        pert_to_row = torch.full((vocab_size,), -1, dtype=torch.long)
        pert_to_row[torch.tensor(ids, dtype=torch.long)] = torch.arange(len(ids), dtype=torch.long)
        self.register_buffer('pert_to_row', pert_to_row)
        self.register_buffer('row_to_vocab', torch.tensor(ids, dtype=torch.long))

        self.shift = nn.Parameter(torch.randn(self.n_pert, latent_dim) * shift_init_std)
        # logit 0 -> gate 0.5 for both priors; the sparsity penalty prunes from there.
        self.mask_logits = nn.Parameter(torch.zeros(self.n_pert, latent_dim))

        self.to_dmodel = nn.Linear(latent_dim, d_model)
        nn.init.zeros_(self.to_dmodel.weight)
        nn.init.zeros_(self.to_dmodel.bias)

    def get_generator(self, device):
        """Dedicated generator so concrete gate sampling never touches the global RNG stream."""
        key = str(device)
        generator = self.generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.generator_seed)
            self.generators[key] = generator
        return generator

    @torch._dynamo.disable
    def gate_rows(self, rows):
        """Gates for gathered rows (B, n_comb) -> (B, n_comb, L). Concrete + training mode
        samples relaxed-Bernoulli gates; every other case returns the deterministic gate."""
        logits = self.mask_logits[rows]
        if self.mask_prior == 'l1':
            return torch.sigmoid(logits)
        if not self.training:
            return mask_gate(logits, 'concrete')
        generator = self.get_generator(logits.device)
        u = torch.rand(logits.shape, generator=generator, device=logits.device,
                       dtype=logits.dtype).clamp(1e-6, 1.0 - 1e-6)
        s = torch.sigmoid((torch.log(u) - torch.log(1.0 - u) + logits) / HC_BETA)
        return (s * (HC_ZETA - HC_GAMMA) + HC_GAMMA).clamp(0.0, 1.0)

    def latent(self, pert_ids):
        """pert_ids: (B, n_comb) vocab token ids -> additive latent shift z: (B, L)."""
        rows = self.pert_to_row[pert_ids]
        valid = (rows >= 0).to(self.shift.dtype).unsqueeze(-1)
        idx = rows.clamp(min=0)
        mechanism = self.gate_rows(idx) * normalize_shift(self.shift[idx]) * valid
        return mechanism.sum(dim=1)

    def forward(self, pert_ids):
        """Latent mechanism code decoded to a d_model residual: (B, d_model)."""
        return self.to_dmodel(self.latent(pert_ids))

    def sparsity_penalty(self, lambda_mask, pert_ids):
        """Sparse-mechanism-shift prior evaluated on the rows used by the current batch.

        Only rows that receive a data gradient in this step receive a penalty gradient. A
        penalty on the whole table gives every untouched row a tiny constant gradient, and
        Adam rescales a constant gradient to a full-size step, which makes the effective
        sparsity pressure independent of lambda. 'l1': L1 of sigmoid gates. 'concrete':
        expected L0 = sum sigmoid(logit - HC_L0_OFFSET). Summed over dims, mean over rows."""
        if lambda_mask <= 0:
            return self.shift.new_zeros(())
        rows = self.pert_to_row[pert_ids].reshape(-1)
        rows = torch.unique(rows[rows >= 0])
        if rows.numel() == 0:
            return self.shift.new_zeros(())
        logits = self.mask_logits[rows]
        if self.mask_prior == 'l1':
            per_gene = torch.sigmoid(logits).sum(dim=-1)
        else:
            per_gene = torch.sigmoid(logits - HC_L0_OFFSET).sum(dim=-1)
        return lambda_mask * per_gene.mean()

    @torch.no_grad()
    def stats(self, pert_ids):
        """Activity probe on the deterministic gate table; never advances the gate generator.
        mech_l1_mean and active_dims_mean reveal an empty-mechanism collapse."""
        gate = mask_gate(self.mask_logits, self.mask_prior)
        direction = normalize_shift(self.shift)
        rows = self.pert_to_row[pert_ids]
        valid = (rows >= 0).to(direction.dtype).unsqueeze(-1)
        idx = rows.clamp(min=0)
        mechanism = gate[idx] * direction[idx] * valid
        z = mechanism.sum(dim=1)
        code = self.to_dmodel(z)
        return {
            "code_norm_mean": float(code.norm(dim=-1).mean()),
            "code_norm_max": float(code.norm(dim=-1).max()),
            "z_norm_mean": float(z.norm(dim=-1).mean()),
            "mech_l1_mean": float(mechanism.abs().sum(dim=-1).sum(dim=-1).mean()),
            "active_dims_mean": float((gate > 0.5).float().sum(dim=-1).mean()),
            "mask_mean": float(gate.mean()),
            "shift_norm_mean": float(self.shift.norm(dim=-1).mean()),
        }

    @torch.no_grad()
    def export_masks(self):
        """(gate [n_pert, L], shift_hat [n_pert, L], mechanism = gate * shift_hat, row_to_vocab)."""
        gate = mask_gate(self.mask_logits, self.mask_prior)
        direction = normalize_shift(self.shift)
        return (gate.cpu(), direction.cpu(), (gate * direction).cpu(),
                self.row_to_vocab.detach().cpu())