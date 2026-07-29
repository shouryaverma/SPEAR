from src.models.scGPT.model import TransformerModel
from src.models.perturbation.model import Model as FlowModel
from src.models.perturbation.model import TimedTransformer
from src.models.origin.model import model as OriginModel
import torch
def instantiate_model(model_type: str, **kwargs):
    if model_type == 'origin':
        if kwargs['fusion_method'] == 'differential_transformer':
            layers = 8
        elif kwargs['fusion_method'] == 'differential_perceiver':
            layers = 4
        else:
            layers = 8
        m = OriginModel(
            ntoken=kwargs['ntoken'],
            d_model=kwargs['d_model'],
            fusion_method=kwargs['fusion_method'],
            nlayers=layers,
            perturbation_function=kwargs['perturbation_function'],
            mask_path=kwargs['mask_path'],
            wire_path=kwargs.get('wire_path', None),
            use_wire=kwargs.get('use_wire', False),
            use_repo=kwargs.get('use_repo', False),
            use_velocity_gate=kwargs.get('use_velocity_gate', True),
            use_samsvae=kwargs.get('use_samsvae', False),
            samsvae_control_id=kwargs.get('samsvae_control_id', None),
            samsvae_pert_vocab_ids=kwargs.get('samsvae_pert_vocab_ids', None),
            samsvae_latent_dim=kwargs.get('samsvae_latent_dim', 64),
            samsvae_mask_prior=kwargs.get('samsvae_mask_prior', 'l1'),
        )
        return m
    else:
        raise ValueError(f"Invalid model type: {model_type}")
if __name__ == "__main__":
    model = instantiate_model("punet128")
    x = torch.randn(32, 128, 128)
    t = torch.randn(32)
    out = model(x, t)
    print(out.shape)