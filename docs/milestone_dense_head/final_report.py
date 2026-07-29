#!/usr/bin/env python
"""最终报告: latent-causal (linear additive) vs SPEAR baseline 的决定性对比。
主判据 = doubles-only (未见 A+B) pearson_delta, 确定性 eval, 同 seed 同 regime。
用法: ~/spear_venv/bin/python final_report.py
"""
import glob, pandas as pd, numpy as np

NIGHT = "/home/wangmengbo/p1/SPEAR/SPEAR-main/result/night"

def cls(p):
    p = str(p)
    if "+" not in p: return "ctrl"
    a, b = p.split("+", 1)
    return "ctrl" if (a == "control" and b == "control") else ("single" if (a == "control" or b == "control") else "double")

def metric(name, ck, subset="double", col="pearson_delta"):
    fs = glob.glob(f"{NIGHT}/{name}/**/iteration_{ck}/results.csv", recursive=True)
    if not fs: return None
    d = pd.read_csv(fs[0]); d = d[d["perturbation"].apply(cls) == subset]
    return float(d[col].mean()) if len(d) else None

def seed_table(prefix_base, prefix_var, label_var, ck):
    print(f"\n@{ck} 步  doubles-only pearson_delta (↑越高越好):")
    print(f"  {'seed':>4}  {'baseline':>10}  {label_var:>12}  {'Δ':>9}  {'相对':>7}")
    bs, vs = [], []
    for s in [42, 43, 44]:
        b = metric(f"{prefix_base}_s{s}", ck); v = metric(f"{prefix_var}_s{s}", ck)
        if b is not None and v is not None:
            bs.append(b); vs.append(v)
            print(f"  {s:>4}  {b:>10.4f}  {v:>12.4f}  {v-b:>+9.4f}  {(v-b)/b*100:>+6.1f}%")
    if bs:
        db, dv = np.mean(bs), np.mean(vs)
        print(f"  {'均值':>4}  {db:>10.4f}  {dv:>12.4f}  {dv-db:>+9.4f}  {(dv-db)/db*100:>+6.1f}%")
        print(f"        (std base={np.std(bs):.4f}, {label_var}={np.std(vs):.4f}; n={len(bs)} seeds)")
    return bs, vs

print("=" * 68)
print("  SPEAR latent-causal 结果报告 — norman/additive/fold1")
print("  主判据: doubles-only (未见 A+B 组合) pearson_delta")
print("=" * 68)

print("\n【决定性对比 1】 linear-K256 (additive 潜层组合) vs baseline")
seed_table("base5k", "lin256_5k", "linear-K256", 2500)
seed_table("base5k", "lin256_5k", "linear-K256", 5000)

print("\n【对照 A】 gelu-K256 (非加法, 同容量) vs baseline  —— decode 线性性是否关键")
seed_table("base5k", "gelu256_5k", "gelu-K256", 5000)

print("\n【对照 B】 dense-head (per-gene MLP, 无稀疏因子, 参数量更多) vs baseline  —— 结构 vs 纯参数量")
seed_table("base5k", "dense_5k", "dense-head", 5000)

print("\n【K 容量确认】 linear-K512 vs baseline")
seed_table("base5k", "lin512_5k", "linear-K512", 5000)

print("\n【K 扫描 @3000, seed42】")
for k, n in [(0, "baseline"), (128, "linear"), (256, "linear_K256"), (384, "lin_K384"), (512, "lin_K512")]:
    v = metric(n, 3000)
    print(f"  K={k if k else '(baseline)':>10}: {v:.4f}" if v is not None else f"  K={k}: -")
print("\n" + "=" * 68)
