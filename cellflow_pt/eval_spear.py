"""
Evaluate trained SPEAR (e7, 295k) using the REPO'S OWN eval (run.test / _test_impl,
N=128, generate_sample) -> results.csv, then score with the repo's LOCKED metric:
doubles-only (A+B, both non-control) pearson_delta, plus doubles-only DE metrics.
No invented eval loop.

  cd /home/wangmengbo/SPEAR/SPEAR-main
  PYTHONPATH=. HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=0 \
    /home/wangmengbo/SPEAR/venv/bin/python cellflow_pt/eval_spear.py
"""
import os, time, argparse
import numpy as np, pandas as pd, torch

from config.config_flow import FlowConfig
from src.data_process.data import Data
from src.models.instantiate_model import instantiate_model
from src.utils.utils import load_checkpoint, process_vocab
import src.script.run as R

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DE = ['de_spearman_lfc_sig', 'de_direction_match', 'de_spearman_sig', 'pearson_delta',
      'mse_delta', 'discrimination_score_cosine']


def cls(p):                       # EXACT copy of night_scan.py / final_report.py
    p = str(p)
    if "+" not in p:
        return "ctrl"
    a, b = p.split("+", 1)
    return "ctrl" if (a == "control" and b == "control") else \
           ("single" if (a == "control" or b == "control") else "double")


def report(results_csv, tag):
    d = pd.read_csv(results_csv)
    d["_c"] = d["perturbation"].apply(cls)
    for subset in ("double", "single"):
        s = d[d["_c"] == subset]
        if not len(s):
            continue
        vals = {m: float(s[m].mean()) for m in DE if m in s.columns}
        print(f"[{tag} {subset} n={len(s)}] " + "  ".join(f"{k}={v:.4f}" for k, v in vals.items()), flush=True)


class StubAcc:
    device = DEV
    is_main_process = True
    def unwrap_model(self, m):
        return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='result/checkpoint.pt')
    ap.add_argument('--out', default='result_cf/spear_native')
    ap.add_argument('--eval_seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    cfg = FlowConfig(
        data_name='norman', split_method='additive', fold=1, model_type='origin',
        d_model=128, fusion_method='differential_perceiver', perturbation_function='crisper',
        noise_type='Gaussian', mode='predict_y', infer_top_gene=1000, n_top_genes=5000,
        topk=30, use_negative_edge=True, use_wire=True, use_repo=True, use_velocity_gate=True,
        endpoint_loss='deg_sinkhorn_sparse', gamma=0.1, sparse_lambda=0.1,
        deg_epsilon=0.1, sinkhorn_eps=0.1, sinkhorn_iters=50, eta_min=1e-6, lr=5e-5,
    )
    R.device = DEV
    R.config = cfg

    dm = Data('./data'); dm.load_data('norman')
    dm.process_data(n_top_genes=5000, infer_top_gene=1000, split_method='additive', fold=1,
                    use_negative_edge=True, k=30, split_file='', train_frac=1.0)
    train_sampler, test_dl, _ = dm.load_flow_data(batch_size=48)
    vocab = process_vocab(dm, cfg)
    R.inverse_dict = {v: str(k) for k, v in dm.perturbation_dict.items()}
    print(f"[data] {time.time()-t0:.0f}s vocab={len(vocab)} eval_genes={test_dl.adata.shape[1]}", flush=True)

    mask_path = 'data/norman/mask_fold_1topk_30additive_negative_edge.pt'
    wire_path = mask_path.replace('.pt', '_wire_eigvecs.pt')
    model = instantiate_model(
        'origin', ntoken=len(vocab), d_model=128, d_perturbation=128,
        fusion_method='differential_perceiver', perturbation_function='crisper',
        mask_path=mask_path, wire_path=wire_path,
        use_wire=True, use_repo=True, use_velocity_gate=True, use_scm=False,
        scm_control_id=(vocab['control'] if 'control' in vocab.stoi else None),
    ).to(DEV)
    load_checkpoint(a.ckpt, model)
    model.eval()
    print(f"[model] SPEAR loaded (params={sum(p.numel() for p in model.parameters())/1e6:.2f}M)", flush=True)

    # Repeat the repo's native eval over several draws (SPEAR generates from random noise
    # -> pearson_delta is high-variance); report mean +/- std of doubles-only metrics.
    dbl = {m: [] for m in DE}
    for es in a.eval_seeds:
        torch.manual_seed(es)        # control subsample + generation noise draw
        odir = os.path.join(a.out, f'eseed{es}'); os.makedirs(odir, exist_ok=True)
        R.test(test_dl, model, StubAcc(), batch_size=128, path=odir, vocab=vocab, scheme='mse')
        report(os.path.join(odir, 'results.csv'), f'SPEAR(eseed={es})')
        d = pd.read_csv(os.path.join(odir, 'results.csv'))
        d = d[d['perturbation'].apply(cls) == 'double']
        for m in DE:
            if m in d.columns:
                dbl[m].append(float(d[m].mean()))
    print(f"\n[SPEAR doubles-only, {len(a.eval_seeds)} eval draws — mean +/- std]", flush=True)
    for m in DE:
        if dbl[m]:
            print(f"  {m:<28} {np.mean(dbl[m]):.4f} ± {np.std(dbl[m]):.4f}", flush=True)
    print(f"[all done] {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
