from dataclasses import dataclass
import os
@dataclass
class FlowConfig:
    # Flow model type
    model_type: str = 'origin'
    # Flow Matching specific parameters
    batch_size: int = 32
    ntoken: int = 512
    d_model: int = 512
    lr: float = 1e-5
    steps: int = 5000 # iterations
    eta_min: float = 1e-7
    devices: str = "1"
    test_only: bool = False
    seed: int = 42
    # Perturbation related parameters
    data_name: str = "combosciplex"
    perturbation_function: str = 'crisper'
    noise_type: str = "Gaussian"
    poisson_alpha: float = 0.8
    poisson_target_sum: int = -1
    print_every: int = 5000
    mode: str = 'predict_y'  # predict_y, predict_p
    result_path: str = './result'
    perturbation_fusion_method: str = 'sum'  # mlp, sum
    fusion_method: str = 'cross'  # cross , concat, add
    infer_top_gene: int = 1000
    n_top_genes: int = 5000
    checkpoint_path: str = ''
    finetune: bool = False
    gamma: float = 0.0
    split_method: str = 'additive'
    split_file: str = ''
    train_frac: float = 1.0
    plate_col: str = ''  # Donor/plate column for batch-controlled pairing. Empty disables it.
    use_mmd_loss: bool = False
    fold: int = 0
    use_negative_edge: bool = False
    topk: int = 15
    neighbor_cap: int = 128
    use_wire: bool = False
    wire_eigvec_dim: int = 32
    use_repo: bool = False
    use_velocity_gate: bool = True
    # Endpoint distribution loss selector (replaces use_mmd_loss as the active gate).
    endpoint_loss: str = 'mmd'
    deg_epsilon: float = 0.1
    sinkhorn_eps: float = 0.1
    sinkhorn_iters: int = 50
    # Sliced Wassertein distance parameters
    n_slices: int = 128
    sliced_top_frac: float = 0.1        # lower -> more max-sliced (worst-case direction)
    precision_epsilon: float = 0.1
    sparse_lambda: float = 0.1
    # SAMS-VAE sparse additive mechanism shift head
    use_samsvae: bool = False
    samsvae_latent_dim: int = 64
    samsvae_mask_prior: str = 'concrete'   # 'concrete' (gauge-robust) | 'l1' (ablation)
    samsvae_lambda_mask: float = 1e-4
    samsvae_lr: float = 1e-3
    samsvae_warmup_steps: int = 500
    samsvae_freeze_base: bool = False

    def __post_init__(self):
        if self.data_name == 'norman_umi_go_filtered':
            self.n_top_genes = 5054
        if self.data_name == 'norman':
            self.n_top_genes = 5000
        assert self.samsvae_mask_prior in ('l1', 'concrete')
        path = self.make_path()

    def make_path(self):
        _LOSS = {
            'mmd': 'mmd', 'deg_mmd': 'dmmd', 'sinkhorn': 'sink', 'deg_sinkhorn': 'dsink',
            'max_sliced': 'msl', 'deg_max_sliced': 'dmsl',
            'precision_sinkhorn': 'psink', 'deg_sinkhorn_sparse': 'dsinksp',
            'none': 'none',
        }
        _PRIOR = {'concrete': 'cz', 'l1': 'l1'}
        flags = (f"{int(self.use_negative_edge)}{int(self.use_wire)}"
                 f"{int(self.use_repo)}{int(self.use_velocity_gate)}")
        parts = [
            self.data_name,
            self.split_method[:3],
            f'f{self.fold}',
            _LOSS.get(self.endpoint_loss, self.endpoint_loss),
            f'g{self.gamma:g}',
            f'nwrv{flags}',
            f'pe{self.precision_epsilon:g}',
            f'tf{self.sliced_top_frac:g}',
            f'ns{self.n_slices}',
            f'sl{self.sparse_lambda:g}',
        ]
        if self.use_samsvae:
            parts.append(f'{_PRIOR[self.samsvae_mask_prior]}L{self.samsvae_latent_dim}'
                         f'lm{self.samsvae_lambda_mask:g}lr{self.samsvae_lr:g}')
        else:
            parts.append('nocausal')
        parts.append(f's{self.seed}')
        name = '-'.join(parts)
        assert len(name) <= 200, f'result dir component too long ({len(name)}): {name}'
        return os.path.join(self.result_path, name)