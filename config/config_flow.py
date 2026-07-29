from dataclasses import dataclass
import os
@dataclass
class FlowConfig:
    # Flow model type
    model_type: str = 'origin'
    seed: int = 42
    # Flow Matching specific parameters
    batch_size: int = 32
    ntoken: int = 512
    d_model: int = 512
    lr: float = 1e-5
    steps: int = 5000 # iterations
    eta_min: float = 1e-7
    devices: str = "1"
    test_only: bool = False
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
    split_file: str = ''     # override the split pickle path (for custom unseen splits)
    train_frac: float = 1.0  # data-efficiency regime: fraction of train perturbations to keep
    finetune: bool = False   # load ONLY model weights from checkpoint_path (strip _orig_mod,
                             # keep new zero-init heads at init, fresh optimizer/scheduler,
                             # start_iteration=0). For finetuning a pretrained baseline + causal head.
    gamma: float = 0.0
    split_method: str = 'additive'
    plate_col: str = ''  # Donor/plate column for batch-controlled pairing. Empty disables it.
    use_mmd_loss: bool = False
    fold: int = 0
    use_negative_edge: bool = False
    topk: int = 15
    use_wire: bool = False
    use_repo: bool = False
    use_velocity_gate: bool = True
    # Endpoint distribution loss selector (replaces use_mmd_loss as the active gate).
    # 'mmd' | 'deg_mse' | 'sinkhorn' | 'deg_sinkhorn' | 'none'. Requires gamma > 0 to take effect.
    endpoint_loss: str = 'mmd'
    deg_epsilon: float = 0.1
    sinkhorn_eps: float = 0.1
    sinkhorn_iters: int = 50
    # Sliced Wassertein distance parameters
    # For 'deg_max_sliced' loss
    n_slices: int = 128
    sliced_top_frac: float = 0.1        # lower -> more max-sliced (worst-case direction)
    precision_epsilon: float = 0.1
    sparse_lambda: float = 0.1
    # SAMS-VAE head: per-perturbation sparse additive mechanism shift in a latent space
    # (Bereket & Karaletsos 2023). Identifiability from the sparse-mask prior (not
    # anchoring). Zero-init decode => exact no-op residual when off (byte-identical to the
    # baseline). samsvae_* = its structural params.
    use_samsvae: bool = False
    samsvae_lambda_mask: float = 1e-4  # sparse mechanism-shift prior weight (+ warmup ramp)
    samsvae_lr: float = 1e-3           # dedicated (higher) lr for the head; <=0 -> use global lr
    samsvae_warmup_steps: int = 500    # ramp the sparsity penalty from 0 over these steps
    samsvae_freeze_base: bool = False  # freeze pretrained baseline; train ONLY the causal head
    samsvae_latent_dim: int = 64      # latent mechanism dimension L
    samsvae_mask_prior: str = 'concrete'  # 'concrete' (hard-concrete L0, gauge-ROBUST default) |
                                          # 'l1' (sigmoid+L1 ablation; shift-normalized but retains a
                                          #  decode-scale gauge -> read via gauge-invariant mech_l1 metric)
    def __post_init__(self):
        if self.data_name == 'norman_umi_go_filtered':
            self.n_top_genes = 5054
        if self.data_name == 'norman':
            self.n_top_genes = 5000
        path = self.make_path()
    def make_path(self):
        exp_name = '-'.join([
                            # 'flow',
                            #  f'fusion_{self.fusion_method}',
                             f'{self.data_name}',
                             self.model_type,
                            #  self.mode,
                             f'gamma_{self.gamma}',
                             f'pertfunc_{self.perturbation_function}',
                            #  f'lr_{self.lr}',
                            #  f'dim_{self.d_model}',
                            #  f'infertop_{self.infer_top_gene}',
                             f'split_{self.split_method}',
                            #  f'mmd_{self.use_mmd_loss}',
                             f'fold_{self.fold}',
                             f'negedge_{self.use_negative_edge}',
                            #  f'topk_{self.topk}',
                             f'wire_{self.use_wire}',
                             f'repo_{self.use_repo}',
                             f'endpoint_{self.endpoint_loss}',
                             f'precision_epsilon_{self.precision_epsilon}',
                             f'sliced_top_frac_{self.sliced_top_frac}',
                             f'n_slices_{self.n_slices}',
                             f'sparse_lambda_{self.sparse_lambda}',
                             f'velocity_gate_{self.use_velocity_gate}',
                             f'samsvae_{self.use_samsvae}',
                             f'seed_{self.seed}',
                             ])
        return os.path.join(self.result_path, exp_name)