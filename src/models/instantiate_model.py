from src.models.origin.model import model as OriginModel

LAYERS_BY_FUSION = {'differential_transformer': 8, 'differential_perceiver': 4}
DEFAULT_LAYERS = 8


def instantiate_model(model_type: str, **kwargs):
    if model_type != 'origin':
        raise ValueError(f"Invalid model type: {model_type}")

    use_samsvae = kwargs.get('use_samsvae', False)
    pert_vocab_ids = kwargs.get('samsvae_pert_vocab_ids') or []
    if use_samsvae and not pert_vocab_ids:
        raise ValueError(
            "use_samsvae=True but samsvae_pert_vocab_ids is empty; check that "
            "perturbation_dict keys are single gene symbols present in the vocab.")

    return OriginModel(
        ntoken=kwargs['ntoken'],
        d_model=kwargs['d_model'],
        fusion_method=kwargs['fusion_method'],
        nlayers=LAYERS_BY_FUSION.get(kwargs['fusion_method'], DEFAULT_LAYERS),
        perturbation_function=kwargs['perturbation_function'],
        mask_path=kwargs['mask_path'],
        wire_path=kwargs.get('wire_path', None),
        use_wire=kwargs.get('use_wire', False),
        use_repo=kwargs.get('use_repo', False),
        neighbor_cap=kwargs.get('neighbor_cap', 128),
        use_velocity_gate=kwargs.get('use_velocity_gate', True),
        use_samsvae=use_samsvae,
        samsvae_pert_vocab_ids=pert_vocab_ids,
        samsvae_latent_dim=kwargs.get('samsvae_latent_dim', 64),
        samsvae_mask_prior=kwargs.get('samsvae_mask_prior', 'concrete'),
    )