"""
Precompute Laplacian eigenvectors for WIRE gene position encoding.
Run from project root with PYTHONPATH=./ same as run.py.

Usage:
    python precompute_wire.py \
        --data_path_root ./data \
        --data_name combosciplex \
        --out_path data/combosciplex/mask_fold_1topk_30additive_negative_edge.pt \
        --n_top_genes 2000 \
        --d_model 128 \
        --nhead 8 \
        --ntoken 6000 \
        --split_method additive \
        --fold 1 \
        --topk 30 \
        --use_negative_edge
"""

import argparse
import numpy as np
import torch

from src.data_process.data import Data
from src.utils.utils import process_vocab


def compute_laplacian_eigvecs(X: np.ndarray, eigvec_dim: int, k: int = 30) -> np.ndarray:
    n_genes = X.shape[1]

    X = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    X = X / norms
    corr = (X.T @ X) / X.shape[0]
    np.fill_diagonal(corr, 1.0)

    A = np.abs(corr).astype(np.float64)
    np.fill_diagonal(A, 0.0)

    topk_indices = np.argpartition(A, -k, axis=1)[:, -k:]
    A_sparse = np.zeros_like(A)
    rows = np.arange(n_genes)[:, None]
    A_sparse[rows, topk_indices] = A[rows, topk_indices]
    A = np.maximum(A_sparse, A_sparse.T)

    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 1e-12, 1.0 / np.sqrt(d), 0.0)
    L = np.eye(n_genes) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
    L = (L + L.T) / 2.0

    _, eigenvectors = np.linalg.eigh(L)
    coords = eigenvectors[:, 1:eigvec_dim + 1].astype(np.float32)

    if coords.shape[1] < eigvec_dim:
        pad = np.zeros((n_genes, eigvec_dim - coords.shape[1]), dtype=np.float32)
        coords = np.concatenate([coords, pad], axis=1)

    return coords


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path_root", type=str, default="./data")
    parser.add_argument("--data_name",      type=str, required=True)
    parser.add_argument("--out_path",       type=str, required=True)
    parser.add_argument("--n_top_genes",    type=int, default=2000)
    parser.add_argument("--d_model",        type=int, default=128)
    parser.add_argument("--nhead",          type=int, default=8)
    parser.add_argument("--split_method",   type=str, default="additive")
    parser.add_argument("--fold",           type=int, default=0)
    parser.add_argument("--topk",           type=int, default=30)
    parser.add_argument("--infer_top_gene", type=int, default=256)
    parser.add_argument("--use_negative_edge", action="store_true")
    parser.add_argument("--eigvec_dim", type=int, default=None,
                        help="Number of Laplacian eigenvectors. "
                            "Defaults to (d_model // nhead) // 2 if not set. "
                            "Override to 32 for large HVG sets regardless of head_dim.")
    args = parser.parse_args()

    eigvec_dim = args.eigvec_dim if args.eigvec_dim is not None else (args.d_model // args.nhead) // 2
    print(f"eigvec_dim = {eigvec_dim}  (d_model={args.d_model} // nhead={args.nhead} // 2)")

    # Replicate exactly what run.py does: same Data pipeline, same vocab
    data_manager = Data(args.data_path_root)
    data_manager.load_data(args.data_name)
    data_manager.process_data(
        n_top_genes=args.n_top_genes,
        infer_top_gene=args.infer_top_gene,
        split_method=args.split_method,
        fold=args.fold,
        use_negative_edge=args.use_negative_edge,
        k=args.topk,
    )

    # process_vocab builds vocab from adata.var_names after HVG filtering —
    # same call as run.py, guarantees identical gene → token_id mapping
    class _FakeConfig:
        data_name = args.data_name
        n_top_genes = args.n_top_genes

    vocab = process_vocab(data_manager, _FakeConfig())

    hvg_gene_names = list(data_manager.adata.var_names)
    gene_token_ids = vocab.encode(hvg_gene_names)   # list[int], same order as var_names
    print(f"HVG genes: {len(hvg_gene_names)}, "
          f"token ID range: {min(gene_token_ids)} .. {max(gene_token_ids)}")

    ntoken = len(vocab)
    print(f"ntoken from vocab: {ntoken}")

    X = data_manager.adata_train.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype(np.float32)
    print(f"Computing correlation matrix from training cells: {X.shape}")

    coords_hvg = compute_laplacian_eigvecs(X, eigvec_dim, k=args.topk)   # (n_hvg, eigvec_dim)
    print(f"Eigvec tensor: {coords_hvg.shape}")

    spectral_coords = torch.zeros(ntoken, eigvec_dim, dtype=torch.float32)
    for pos, token_id in enumerate(gene_token_ids):
        spectral_coords[token_id] = torch.from_numpy(coords_hvg[pos])

    print(f"Full lookup tensor: {spectral_coords.shape}")
    torch.save(spectral_coords, args.out_path)
    print(f"Saved to {args.out_path}")