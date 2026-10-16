import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import math
import random
import re

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchdiffeq
import tqdm
import tyro
from accelerate import Accelerator, DistributedDataParallelKwargs
from cell_eval import MetricsEvaluator
from torch.utils.data import DataLoader

from config.config_flow import FlowConfig as Config
from src.data_process.data import Data, PerturbationDataset
from src.flow_matching.path import AffineProbPath
from src.flow_matching.path.scheduler import CondOTScheduler
from src.models.instantiate_model import instantiate_model
from src.utils.utils import (save_checkpoint, load_checkpoint, make_lognorm_poisson_noise,
                             pick_eval_score, process_vocab, set_requires_grad_for_p_only)

path = AffineProbPath(scheduler=CondOTScheduler())

EVAL_SEED = 0          # evaluation draws noise and control cells from its own fixed seed
EVAL_CELLS = 128       # control cells sampled per evaluated condition


# ----------------------------------------------------------------------------------------
# Distribution distances
# ----------------------------------------------------------------------------------------
def pairwise_sq_dists(X, Y):
    return torch.cdist(X, Y, p=2) ** 2


@torch.no_grad()
def median_sigmas(X, scales=(0.5, 1.0, 2.0, 4.0)):
    D2 = pairwise_sq_dists(X, X)
    tri = D2[~torch.eye(D2.size(0), dtype=bool, device=D2.device)]
    m = torch.median(tri).clamp_min(1e-12)
    s2 = torch.tensor(scales, device=X.device) * m
    return [float(s.item()) for s in torch.sqrt(s2)]


def mmd2_unbiased_multi_sigma(X, Y, sigmas):
    X = X.float()
    Y = Y.float()
    m, n = X.size(0), Y.size(0)
    Dxx = pairwise_sq_dists(X, X)
    Dyy = pairwise_sq_dists(Y, Y)
    Dxy = pairwise_sq_dists(X, Y)
    vals = []
    for sigma in sigmas:
        beta = 1.0 / (2.0 * (sigma ** 2) + 1e-12)
        Kxx = torch.exp(-beta * Dxx)
        Kyy = torch.exp(-beta * Dyy)
        Kxy = torch.exp(-beta * Dxy)
        term_xx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1) + 1e-12)
        term_yy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1) + 1e-12)
        term_xy = Kxy.mean()
        vals.append(term_xx + term_yy - 2.0 * term_xy)
    return torch.stack(vals).mean()


def mmd_endpoint(X, Y, scales=(0.5, 1.0, 2.0, 4.0)):
    sigmas = median_sigmas(Y, scales=scales)
    return mmd2_unbiased_multi_sigma(X, Y, sigmas)


def weighted_sqdist(X, Y, w=None):
    if w is None:
        return torch.cdist(X, Y, p=2) ** 2
    s = w.clamp_min(0).sqrt()
    return torch.cdist(X * s, Y * s, p=2) ** 2


def sinkhorn_value(C, eps, n_iters):
    C = C.float()
    eps = float(eps)
    m, n = C.shape
    log_a = torch.full((m,), -math.log(m), device=C.device, dtype=C.dtype)
    log_b = torch.full((n,), -math.log(n), device=C.device, dtype=C.dtype)
    f = torch.zeros(m, device=C.device, dtype=C.dtype)
    g = torch.zeros(n, device=C.device, dtype=C.dtype)
    for _ in range(n_iters):
        f = eps * (log_a - torch.logsumexp((g[None, :] - C) / eps + log_b[None, :], dim=1))
        g = eps * (log_b - torch.logsumexp((f[:, None] - C) / eps + log_a[:, None], dim=0))
    return torch.dot(f, log_a.exp()) + torch.dot(g, log_b.exp())


def sinkhorn_divergence(X, Y, eps, n_iters, w=None):
    X = X.float()
    Y = Y.float()
    if w is not None:
        w = w.float()
    return (sinkhorn_value(weighted_sqdist(X, Y, w), eps, n_iters)
            - 0.5 * sinkhorn_value(weighted_sqdist(X, X, w), eps, n_iters)
            - 0.5 * sinkhorn_value(weighted_sqdist(Y, Y, w), eps, n_iters))


def sliced_w2(X, Y, w=None, n_projections=128, top_frac=0.1):
    X = X.float()
    Y = Y.float()
    if w is not None:
        s = w.float().clamp_min(0).sqrt()
        X = X * s
        Y = Y * s
    d = X.shape[1]
    theta = torch.randn(d, n_projections, device=X.device)
    theta = theta / theta.norm(dim=0, keepdim=True).clamp_min(1e-12)
    Xp = torch.sort(X @ theta, dim=0).values          # (n, L)
    Yp = torch.sort(Y @ theta, dim=0).values
    w2 = ((Xp - Yp) ** 2).mean(dim=0)                 # (L,)
    k = max(1, int(round(top_frac * n_projections)))
    return torch.topk(w2, k).values.mean()


# ----------------------------------------------------------------------------------------
# Endpoint-loss weights
# ----------------------------------------------------------------------------------------
def dense_expression(train_sampler):
    adata = train_sampler.adata
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return adata, np.asarray(X, dtype=np.float32)


def control_mask(cond):
    is_ctrl = cond == 'control+control'
    if is_ctrl.sum() == 0:
        is_ctrl = np.array(['control' in c.lower() for c in cond])
    return is_ctrl


def compute_deg_weights(train_sampler, epsilon=0.1):
    adata, X = dense_expression(train_sampler)
    cond = adata.obs['perturbation_covariates'].astype(str).values
    pair_ids = np.asarray(train_sampler.perturbation_covariates_id)        # (N, 2)

    is_ctrl = control_mask(cond)
    ctrl_mean = X[is_ctrl].mean(0) if is_ctrl.sum() > 0 else X.mean(0)

    string_to_pair = {}
    for i, name in enumerate(cond):
        if name not in string_to_pair:
            string_to_pair[name] = tuple(sorted(int(v) for v in pair_ids[i]))

    weights = {}
    for name in np.unique(cond):
        if name == 'control+control':
            continue
        mask = cond == name
        lfc = np.abs(X[mask].mean(0) - ctrl_mean)
        w_max = max(float(lfc.max()), 1e-12)
        w = epsilon + (1.0 - epsilon) * (lfc / w_max)
        weights[string_to_pair[name]] = torch.from_numpy(w.astype(np.float32))
    return weights


def compute_precision_weights(train_sampler, epsilon=0.1):
    adata, X = dense_expression(train_sampler)
    cond = adata.obs['perturbation_covariates'].astype(str).values
    is_ctrl = control_mask(cond)
    ctrl = X[is_ctrl] if is_ctrl.sum() > 0 else X
    prec = 1.0 / (ctrl.var(0) + 1e-8)
    p_max = max(float(prec.max()), 1e-12)
    w = epsilon + (1.0 - epsilon) * (prec / p_max)
    return torch.from_numpy(w.astype(np.float32))


# ----------------------------------------------------------------------------------------
# Helpers that read module-level state set in __main__: config, accelerator, device,
# vocab, gene_ids, inverse_dict
# ----------------------------------------------------------------------------------------
def unwrap(vf):
    """Underlying nn.Module with no accelerate / DDP / torch.compile wrapper."""
    model = accelerator.unwrap_model(vf)
    return getattr(model, '_orig_mod', model)

class WeightEMA:
    """Exponential moving average of the model weights. The averaged weights are swapped in
    for checkpointing and evaluation and swapped back before training continues."""
    def __init__(self, module, decay):
        self.decay = decay
        self.params = list(module.parameters())
        self.shadow = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def update(self, iteration):
        decay = min(self.decay, (1.0 + iteration) / (10.0 + iteration))
        torch._foreach_lerp_(self.shadow, [p.detach() for p in self.params], 1.0 - decay)

    @torch.no_grad()
    def swap(self):
        for shadow, param in zip(self.shadow, self.params):
            held = param.detach().clone()
            param.copy_(shadow)
            shadow.copy_(held)

def vocab_ids_for_condition(condition_row):
    """perturbation_dict ids of one condition (n_comb,) -> vocab token ids (n_comb,)."""
    names = [inverse_dict[int(i)] for i in condition_row.cpu().tolist()]
    missing = [n for n in names if n not in vocab.stoi]
    if missing:
        raise RuntimeError(
            f"condition {names} has genes missing from the vocab ({missing}); "
            f"perturbation_id would index the gene embedding table with unrelated integers")
    return torch.tensor(vocab.encode(names), dtype=torch.long, device=device)


# ----------------------------------------------------------------------------------------
# Training step
# ----------------------------------------------------------------------------------------
def train_step(source, target, perturbation_id, vf, deg_weights=None, condition_idx=None,
               precision_weights=None, samsvae_lambda=0.0):
    B = source.shape[0]

    input_gene_ids = torch.randperm(source.shape[-1], device=device)[:config.infer_top_gene]
    source = source[:, input_gene_ids]
    target = target[:, input_gene_ids]
    gene_input = gene_ids.repeat(B, 1)[:, input_gene_ids]

    if config.mode == "predict_y":
        t = torch.rand(B, device=device)
        if config.noise_type == "Gaussian":
            target_noise = torch.randn_like(source)
        elif config.noise_type == "Poisson":
            target_noise = make_lognorm_poisson_noise(
                target_log=source,
                alpha=getattr(config, "poisson_alpha", 0.8),
                per_cell_L=getattr(config, "poisson_target_sum", 1e4),
            )
        else:
            raise ValueError(f"Unknown noise_type: {config.noise_type}")
        path_x1 = path.sample(t=t, x_0=target_noise, x_1=target)

        predicted_velocity = vf(gene_input, path_x1.x_t, path_x1.t, source,
                                perturbation_id, gene_input, mode="predict_y")
        loss = ((predicted_velocity - path_x1.dx_t) ** 2).mean()
        x1_hat = path_x1.x_t + predicted_velocity * (1 - t).unsqueeze(-1)

        endpoint_type = getattr(config, 'endpoint_loss', 'mmd')
        if endpoint_type != 'none':
            w = None
            deg_types = ('deg_mse', 'deg_sinkhorn', 'deg_max_sliced',
                         'deg_sinkhorn_sparse', 'deg_mmd')
            if endpoint_type in deg_types:
                assert deg_weights is not None and condition_idx is not None, \
                    f"{endpoint_type} selected but DEG weights were never precomputed"
                ci = condition_idx
                if ci.dim() > 1:
                    assert bool((ci == ci[0]).all()), \
                        "batch mixes conditions but condition_idx[0] is used as the DEG key"
                    ci = ci[0]
                key = tuple(sorted(int(v) for v in ci.reshape(-1).tolist()))
                w_full = deg_weights.get(key, None)
                w = w_full[input_gene_ids.cpu()].to(device) if w_full is not None else None
            elif endpoint_type == 'precision_sinkhorn':
                assert precision_weights is not None, \
                    "precision_sinkhorn selected but precision weights were never precomputed"
                w = precision_weights[input_gene_ids.cpu()].to(device)

            eps_scale = getattr(config, 'sinkhorn_eps', 0.1)
            n_iters = getattr(config, 'sinkhorn_iters', 50)

            if endpoint_type == 'mmd' or (endpoint_type in ('deg_sinkhorn', 'deg_mse') and w is None):
                endpoint_loss = mmd_endpoint(x1_hat, target)
            elif endpoint_type == 'deg_mmd':
                if w is None:
                    endpoint_loss = mmd_endpoint(x1_hat, target)
                else:
                    s = w.clamp_min(0).sqrt()
                    endpoint_loss = mmd_endpoint(x1_hat * s, target * s)
            elif endpoint_type == 'deg_mse':
                weighted_mse = (w * (x1_hat - target) ** 2).mean()
                endpoint_loss = weighted_mse + 0.1 * mmd_endpoint(x1_hat, target, scales=(1.0, 2.0))
            elif endpoint_type in ('sinkhorn', 'deg_sinkhorn', 'precision_sinkhorn'):
                x1_hat_f, target_f = x1_hat.float(), target.float()
                with torch.no_grad():
                    med = torch.median(weighted_sqdist(x1_hat_f, target_f, w)).clamp_min(1e-6)
                    eps = (eps_scale * med).clamp_min(1e-3)
                endpoint_loss = sinkhorn_divergence(x1_hat_f, target_f, eps, n_iters, w=w)
            elif endpoint_type in ('max_sliced', 'deg_max_sliced'):
                endpoint_loss = sliced_w2(
                    x1_hat, target, w=w,
                    n_projections=getattr(config, 'n_slices', 128),
                    top_frac=getattr(config, 'sliced_top_frac', 0.1),
                )
            elif endpoint_type == 'deg_sinkhorn_sparse':
                x1_hat_f, target_f = x1_hat.float(), target.float()
                with torch.no_grad():
                    med = torch.median(weighted_sqdist(x1_hat_f, target_f, w)).clamp_min(1e-6)
                    eps = (eps_scale * med).clamp_min(1e-3)
                sink = sinkhorn_divergence(x1_hat_f, target_f, eps, n_iters, w=w)
                if not torch.isfinite(sink):
                    sink = torch.zeros((), device=device)
                mean_delta = x1_hat.mean(0) - source.mean(0)                   # (G,)
                gate_w = (1.0 - w) if w is not None else torch.ones_like(mean_delta)
                sparse = (gate_w * mean_delta.abs()).mean()
                endpoint_loss = sink + getattr(config, 'sparse_lambda', 0.1) * sparse
            else:
                raise ValueError(f"Unknown endpoint_loss: {endpoint_type}")

            if not torch.isfinite(endpoint_loss):
                endpoint_loss = torch.zeros((), device=device)

            loss = loss + endpoint_loss * config.gamma

        # Sparse-mechanism-shift prior on the rows of this batch. Computed on the unwrapped
        # model so the gradient reaches the head regardless of DDP / torch.compile wrappers.
        if samsvae_lambda > 0:
            loss = loss + unwrap(vf).samsvae_regularizer(perturbation_id, samsvae_lambda)

    elif config.mode == "predict_p":
        t_p = torch.ones(B, device=device)
        predicted_p_embed = vf(gene_input, target, t_p, source, perturbation_id, gene_input,
                               mode="predict_p")
        p_embed_gt = unwrap(vf).get_perturbation_emb(perturbation_id=perturbation_id, cell_1=source)
        pred = F.normalize(predicted_p_embed, dim=-1)
        tgt = F.normalize(p_embed_gt.detach(), dim=-1)
        loss = 1 - (pred * tgt).sum(dim=-1).mean()
    else:
        raise ValueError(f"Unknown mode: {config.mode}")

    return loss


# ----------------------------------------------------------------------------------------
# Sampling and evaluation
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def generate_sample(vf, source, perturbation_id, gene_ids_eval, steps=20, method="rk4"):
    if config.noise_type == "Gaussian":
        initial_noise = torch.randn(source.shape, device=source.device, dtype=source.dtype)
    elif config.noise_type == "Poisson":
        initial_noise = make_lognorm_poisson_noise(
            target_log=source,
            alpha=getattr(config, "poisson_alpha", 0.8),
            per_cell_L=getattr(config, "poisson_target_sum", 1e4),
        )
    else:
        raise ValueError(f"Unknown noise_type: {config.noise_type}")

    gene = gene_ids_eval.repeat(source.shape[0], 1)

    def ode_fn(t, x):
        return vf(gene, x, t, source, perturbation_id, gene).clone()

    traj = torchdiffeq.odeint(
        ode_fn, initial_noise, torch.linspace(0, 1, steps).to(source.device),
        atol=1e-4, rtol=1e-4, method=method,
    )
    return torch.clamp(traj[-1], min=0)


def predict_conditions(data_sampler, vf):
    """Generate predictions for every evaluated condition. Returns (pred, real) AnnData."""
    gene_ids_eval = torch.tensor(vocab.encode(list(data_sampler.adata.var_names)),
                                 dtype=torch.long, device=device)
    condition_names = data_sampler._perturbation_covariates
    control_cells = data_sampler.get_control_data()['src_cell_data']
    control_source = control_cells.to(device)

    pred_blocks, real_blocks = [control_cells], [control_cells]
    pred_labels = ['control'] * control_cells.shape[0]
    real_labels = ['control'] * control_cells.shape[0]

    print('perturbation_name_list:', len(condition_names))
    for name in condition_names:
        data = data_sampler.get_perturbation_data(name)
        target = data['tgt_cell_data']
        condition_row = data['condition_id'].to(device)[0]
        if config.perturbation_function == 'crisper':
            condition_row = vocab_ids_for_condition(condition_row)

        order = torch.randperm(control_source.shape[0])
        source = control_source[order][:EVAL_CELLS]

        chunks = []
        for i in range(0, source.shape[0], config.batch_size):
            batch_source = source[i:i + config.batch_size]
            batch_condition = condition_row.repeat(batch_source.shape[0], 1)
            chunks.append(generate_sample(vf, batch_source, batch_condition, gene_ids_eval))
        pred_expressions = torch.cat(chunks, dim=0).cpu().numpy()

        pred_blocks.append(pred_expressions)
        real_blocks.append(target)
        pred_labels.extend([name] * pred_expressions.shape[0])
        real_labels.extend([name] * target.shape[0])

    pred_matrix = np.concatenate(pred_blocks, axis=0)
    n_nan = np.isnan(pred_matrix).sum()
    n_inf = np.isinf(pred_matrix).sum()
    print(f"[eval] pred non-finite: nan={n_nan} inf={n_inf} "
          f"min={np.nanmin(pred_matrix):.4f} max={np.nanmax(pred_matrix):.4f}")
    pred_matrix = np.nan_to_num(pred_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    real_matrix = np.concatenate(real_blocks, axis=0)
    pred = ad.AnnData(X=pred_matrix, obs=pd.DataFrame({'perturbation': pred_labels}))
    real = ad.AnnData(X=real_matrix, obs=pd.DataFrame({'perturbation': real_labels}))
    return pred, real


@torch.inference_mode()
def evaluate(data_sampler, vf, out_dir, scheme='mse'):
    """Evaluation runs in eval mode (no dropout, deterministic SAMS-VAE gates) with its own
    fixed RNG stream, and restores both the training mode and the training RNG afterwards."""
    cuda_devices = []
    if device.type == 'cuda':
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    vf.eval()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(EVAL_SEED)
            pred, real = predict_conditions(data_sampler, vf)
    finally:
        vf.train()

    eval_score = None
    if accelerator.is_main_process:
        evaluator = MetricsEvaluator(adata_pred=pred, adata_real=real, control_pert="control",
                                     pert_col="perturbation",
                                     num_threads=min(32, os.cpu_count() or 1), outdir=out_dir)
        results, agg_results = evaluator.compute()
        results.write_csv(os.path.join(out_dir, 'results.csv'))
        agg_results.write_csv(os.path.join(out_dir, 'agg_results.csv'))
        pred.write_h5ad(os.path.join(out_dir, 'pred.h5ad'))
        real.write_h5ad(os.path.join(out_dir, 'real.h5ad'))
        eval_score = pick_eval_score(agg_results, scheme)
        print(f"Current evaluation score: {eval_score:.4f}")
    return eval_score


# ----------------------------------------------------------------------------------------
# Checkpoint discovery
# ----------------------------------------------------------------------------------------
def find_latest_checkpoint(save_path: str) -> str | None:
    if not os.path.isdir(save_path):
        return None
    candidates = []
    for name in os.listdir(save_path):
        match = re.match(r"iteration_(\d+)$", name)
        if match:
            ckpt_path = os.path.join(save_path, name, "checkpoint.pt")
            if os.path.isfile(ckpt_path):
                candidates.append((int(match.group(1)), ckpt_path))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


if __name__ == "__main__":
    config = tyro.cli(Config)
    torch.set_float32_matmul_precision("high")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='bf16', kwargs_handlers=[ddp_kwargs])

    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(config.seed)

    save_path = config.make_path()
    if accelerator.is_main_process:
        print(config)
        os.makedirs(save_path, exist_ok=True)
    device = accelerator.device

    if config.use_samsvae and config.perturbation_function != 'crisper':
        raise ValueError("use_samsvae requires perturbation_function='crisper'")

    data_manager = Data('./data')
    data_manager.load_data(config.data_name)
    data_manager.process_data(
        n_top_genes=config.n_top_genes, infer_top_gene=config.infer_top_gene,
        split_method=config.split_method, fold=config.fold,
        use_negative_edge=config.use_negative_edge, k=config.topk,
        split_file=getattr(config, 'split_file', ''),
        train_frac=getattr(config, 'train_frac', 1.0),
    )
    train_sampler, valid_sampler, test_dl = data_manager.load_flow_data(batch_size=config.batch_size)

    train_dataset = PerturbationDataset(
        train_sampler, config.batch_size,
        cell_type_col='cell_type',
        plate_col=config.plate_col if config.plate_col else None,
    )
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True)

    mask_name = ('mask_fold_' + str(config.fold) + 'topk_' + str(config.topk) + config.split_method
                 + ('_negative_edge' if config.use_negative_edge else '') + '.pt')
    mask_path = os.path.join(data_manager.data_path, data_manager.data_name, mask_name)

    vocab = process_vocab(data_manager, config)

    _adj_shape = torch.load(mask_path).shape[0]
    assert _adj_shape == len(vocab), \
        f"co-expression mask rows {_adj_shape} != vocab size {len(vocab)}; " \
        f"mask row order must equal vocab token-id order"

    if config.use_wire:
        wire_path = mask_path.replace('.pt', '_wire_eigvecs.pt')
        if not os.path.exists(wire_path):
            from src.data_process.precompute_wire import compute_laplacian_eigvecs
            eigvec_dim = getattr(config, 'wire_eigvec_dim', 32)
            hvg_gene_names = list(data_manager.adata.var_names)
            gene_token_ids = vocab.encode(hvg_gene_names)
            X = data_manager.adata_train.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
            X = X.astype(np.float32)
            coords_hvg = compute_laplacian_eigvecs(X, eigvec_dim, k=config.topk)
            spectral_coords = torch.zeros(len(vocab), eigvec_dim, dtype=torch.float32)
            for pos, token_id in enumerate(gene_token_ids):
                spectral_coords[token_id] = torch.from_numpy(coords_hvg[pos])
            torch.save(spectral_coords, wire_path)
        _wire_rows = torch.load(wire_path, weights_only=True).shape[0]
        assert _wire_rows == len(vocab), (
            f"wire eigvec rows {_wire_rows} != vocab {len(vocab)}; delete {wire_path} "
            f"and rebuild it")
        model_wire_path = wire_path
    else:
        model_wire_path = None

    gene_ids = torch.tensor(vocab.encode(list(data_manager.adata.var_names)),
                            dtype=torch.long, device=device)

    # Row k of the SAMS-VAE tables is the k-th perturbable gene in sorted order, so the
    # row <-> gene correspondence is identical across seeds and arms.
    samsvae_pert_vocab_ids = []
    if config.use_samsvae:
        pert_names = sorted(g for g in data_manager.perturbation_dict if g != 'control')
        samsvae_pert_vocab_ids = [vocab[g] for g in pert_names if g in vocab.stoi]

    vf = instantiate_model(
        config.model_type,
        ntoken=len(vocab),
        d_model=config.d_model,
        fusion_method=config.fusion_method,
        perturbation_function=config.perturbation_function,
        mask_path=mask_path,
        wire_path=model_wire_path,
        use_wire=config.use_wire,
        use_repo=config.use_repo,
        neighbor_cap=config.neighbor_cap,
        use_velocity_gate=config.use_velocity_gate,
        use_samsvae=config.use_samsvae,
        samsvae_pert_vocab_ids=samsvae_pert_vocab_ids,
        samsvae_latent_dim=config.samsvae_latent_dim,
        samsvae_mask_prior=config.samsvae_mask_prior,
    )

    if config.use_samsvae:
        head_params = list(vf.samsvae.parameters())
        head_ids = {id(p) for p in head_params}
        other_params = [p for p in vf.parameters() if id(p) not in head_ids]
        head_lr = config.samsvae_lr if config.samsvae_lr > 0 else config.lr
        optimizer = torch.optim.Adam([
            {'params': other_params, 'lr': config.lr},
            {'params': head_params, 'lr': head_lr},
        ])
        print(f"[SAMSVAE] dedicated param group: {len(head_params)} tensors @ lr={head_lr}")
    else:
        optimizer = torch.optim.Adam(vf.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.steps, eta_min=config.eta_min)

    if config.checkpoint_path == '':
        config.checkpoint_path = find_latest_checkpoint(save_path) or ''

    start_iteration = 0
    if config.checkpoint_path != '':
        if config.finetune:
            # Weights only; fresh optimizer / scheduler; training restarts at iteration 0.
            if accelerator.is_main_process:
                print(f"Finetuning from checkpoint (weights only): {config.checkpoint_path}")
            load_checkpoint(config.checkpoint_path, vf)
        else:
            if accelerator.is_main_process:
                print(f"Resuming from checkpoint: {config.checkpoint_path}")
            load_checkpoint(config.checkpoint_path, vf, optimizer, scheduler)
            match = re.search(r"iteration_(\d+)", config.checkpoint_path)
            if match:
                start_iteration = int(match.group(1)) + 1

    if start_iteration >= config.steps:
        if accelerator.is_main_process:
            print(f"Checkpoint already at iteration {start_iteration - 1}, "
                  f"target steps={config.steps} reached. Nothing to do.")
        start_iteration = config.steps

    deg_weights = None
    if getattr(config, 'endpoint_loss', 'mmd') in ('deg_mmd', 'deg_sinkhorn', 'deg_max_sliced',
                                                    'deg_sinkhorn_sparse', 'deg_mse'):
        if accelerator.is_main_process:
            print("Precomputing DEG weights...")
        deg_weights = compute_deg_weights(train_sampler, epsilon=getattr(config, 'deg_epsilon', 0.1))
        if accelerator.is_main_process:
            print(f"DEG weight conditions: {len(deg_weights)}; sample keys: {list(deg_weights)[:3]}")
            _w_len = next(iter(deg_weights.values())).shape[0]
            _n_genes = train_sampler.adata.shape[1]
            assert _w_len == _n_genes, f"DEG weight length {_w_len} != adata n_genes {_n_genes}"

    precision_weights = None
    if getattr(config, 'endpoint_loss', 'mmd') == 'precision_sinkhorn':
        if accelerator.is_main_process:
            print("Precomputing precision weights...")
        precision_weights = compute_precision_weights(
            train_sampler, epsilon=getattr(config, 'precision_epsilon', 0.1))

    vf = accelerator.prepare(vf)
    if os.environ.get("SPEAR_COMPILE", "0") == "1":
        torch._dynamo.config.cache_size_limit = 64
        vf = torch.compile(vf, mode="max-autotune")
    optimizer, scheduler, dataloader = accelerator.prepare(optimizer, scheduler, dataloader)
    inverse_dict = {v: str(k) for k, v in data_manager.perturbation_dict.items()}
    ema = WeightEMA(unwrap(vf), config.ema_decay)
    pbar = tqdm.tqdm(total=config.steps, initial=start_iteration)
    iteration = start_iteration

    while iteration < config.steps:
        for batch_data in dataloader:
            if iteration >= config.steps:
                break
            source = batch_data['src_cell_data'].squeeze(0)
            target = batch_data['tgt_cell_data'].squeeze(0)
            condition_idx_raw = batch_data['condition_id'].squeeze(0)        # before vocab conversion
            perturbation_id = condition_idx_raw.to(device)
            if config.perturbation_function == 'crisper':
                perturbation_id = vocab_ids_for_condition(perturbation_id[0]).repeat(source.shape[0], 1)

            set_requires_grad_for_p_only(vf, p_only=config.mode)

            samsvae_lambda = 0.0
            if config.use_samsvae:
                warmup = config.samsvae_warmup_steps
                scale = min(1.0, iteration / float(warmup)) if warmup > 0 else 1.0
                samsvae_lambda = config.samsvae_lambda_mask * scale

            loss = train_step(source, target, perturbation_id, vf,
                              deg_weights=deg_weights, condition_idx=condition_idx_raw,
                              precision_weights=precision_weights, samsvae_lambda=samsvae_lambda)

            optimizer.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(vf.parameters(), max_norm=1.0)
                optimizer.step()
                ema.update(iteration)
            elif accelerator.is_main_process:
                print(f"[warn] non-finite loss at iteration {iteration}, step skipped")
            scheduler.step()

            if config.use_samsvae and accelerator.is_main_process and iteration % 200 == 0:
                try:
                    st = unwrap(vf).samsvae_stats(perturbation_id)
                    # mech_l1 and active_dims trending to 0 means the sparse mask is empty.
                    print(f"[SAMSVAE] it={iteration} code_norm(mean/max)="
                          f"{st['code_norm_mean']:.4f}/{st['code_norm_max']:.4f} "
                          f"z_norm={st['z_norm_mean']:.4f} mech_l1={st['mech_l1_mean']:.4f} "
                          f"active_dims={st['active_dims_mean']:.4f} mask_mean={st['mask_mean']:.4f} "
                          f"shift_norm={st['shift_norm_mean']:.4f}")
                except Exception as e:
                    print(f"[SAMSVAE] stats error: {e}")

            is_final = iteration + 1 == config.steps
            if (iteration > 0 and iteration % config.print_every == 0) or is_final:
                # The final model is saved as iteration_<steps> so it is never skipped.
                # Checkpoints and evaluation use the EMA weights.
                tag = config.steps if is_final else iteration
                save_path_ = os.path.join(save_path, f'iteration_{tag}')
                os.makedirs(save_path_, exist_ok=True)
                ema.swap()
                try:
                    if accelerator.is_main_process:
                        print(f"saving {tag}'s checkpoint...")
                        save_checkpoint(
                            model=unwrap(vf), optimizer=optimizer, scheduler=scheduler,
                            iteration=iteration, eval_score=None, save_path=save_path_, is_best=False)
                    evaluate(valid_sampler, vf, save_path_)
                finally:
                    ema.swap()

            accelerator.wait_for_everyone()
            pbar.update(1)
            pbar.set_description(f'loss: {loss.item():.4f}, iteration: {iteration}')
            iteration += 1