"""Standalone unit test for the SAMS-VAE head (no data / GPU needed)."""
import torch
from src.models.perturbation.sams_vae import SparseAdditiveMechanismShift

torch.manual_seed(0)
d_model, L = 128, 32
control_id = 0
pert_vocab_ids = [5, 6, 7, 8, 9]          # 5 perturbable genes, rows 0..4
vocab_size = 32

for prior in ('l1', 'concrete'):
    print(f"\n===== mask_prior={prior} =====")
    m = SparseAdditiveMechanismShift(
        d_model, latent_dim=L, pert_vocab_ids=pert_vocab_ids,
        vocab_size=vocab_size, control_id=control_id, mask_prior=prior)
    m.eval()  # deterministic

    # (1) no-op at init: to_dmodel zero-init => code == 0 for any input.
    pid = torch.tensor([[5, 6], [7, control_id], [control_id, control_id]])  # A+B, single, control-only
    code = m(pid)
    assert code.shape == (3, d_model)
    assert torch.allclose(code, torch.zeros_like(code)), "NOT a no-op at init!"
    print("  no-op at init: OK (code all zero)")

    # break the no-op so we can test the latent geometry z
    with torch.no_grad():
        m.to_dmodel.weight.normal_(0, 0.1)

    # (2) additivity in latent space: z(A+B) == z(A) + z(B); read m._last_z per call
    _ = m(torch.tensor([[5, control_id]])); zA = m._last_z.clone()
    _ = m(torch.tensor([[6, control_id]])); zB = m._last_z.clone()
    _ = m(torch.tensor([[5, 6]]));          zAB = m._last_z.clone()
    add_err = (zAB - (zA + zB)).abs().max().item()
    print(f"  additivity z(A+B) vs z(A)+z(B): max err = {add_err:.2e}")
    assert add_err < 1e-5, "additivity broken!"

    # (3) control-only slots contribute nothing
    _ = m(torch.tensor([[control_id, control_id]])); zc = m._last_z
    assert torch.allclose(zc, torch.zeros_like(zc)), "control leaked signal!"
    print("  control masking: OK")

    # (3b) shift is unit-normalized per row (scale gauge killed at the per-element level)
    shat = m._shift_hat()
    norms = shat.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "shift not unit-norm!"
    print(f"  shift unit-norm per row: OK (norms ~ {float(norms.mean()):.4f})")

    # (4) sparsity penalty + stats run and are finite; gauge-invariant probes present
    pen = m.sparsity_penalty(1e-3)
    st = m.stats(pid)
    assert torch.isfinite(pen)
    for k in ('mech_l1_mean', 'active_dims_mean', 'z_norm_mean'):
        assert k in st, f"missing stats key {k}"
    print(f"  sparsity_penalty={float(pen):.4f}  stats={ {k: round(v,3) for k,v in st.items()} }")

    # (4b) GAUGE ROBUSTNESS: for 'concrete', the penalty must depend ONLY on logits
    # (structure), not on shift magnitude. Blow up shift 100x -> penalty unchanged.
    if prior == 'concrete':
        p0 = float(m.sparsity_penalty(1.0))
        with torch.no_grad():
            m.shift.mul_(100.0)
        p1 = float(m.sparsity_penalty(1.0))
        assert abs(p0 - p1) < 1e-6, f"concrete L0 penalty NOT gauge-invariant ({p0} vs {p1})!"
        with torch.no_grad():
            m.shift.div_(100.0)
        print(f"  concrete L0 gauge-invariance: OK (penalty {p0:.4f} == {p1:.4f} under 100x shift)")

    # (5) gradient flows to shift, mask_logits, to_dmodel
    m.train()
    code = m(torch.tensor([[5, 6], [7, 8]]))
    loss = code.pow(2).sum() + m.sparsity_penalty(1e-2)
    loss.backward()
    for name in ('shift', 'mask_logits'):
        g = getattr(m, name).grad
        assert g is not None and torch.isfinite(g).all(), f"no grad on {name}"
    assert m.to_dmodel.weight.grad is not None
    print("  gradient flow: OK")

# (6) export_masks shape (mask, shift_hat, mechanism, row_to_vocab)
tbl, shat, mech, r2v = m.export_masks()
print(f"\nexport_masks: mask {tuple(tbl.shape)}, shift_hat {tuple(shat.shape)}, "
      f"mech {tuple(mech.shape)}, row_to_vocab {r2v.tolist()}")
assert tbl.shape == (len(pert_vocab_ids), L) and mech.shape == (len(pert_vocab_ids), L)
print("\nALL SAMS-VAE UNIT TESTS PASSED")
