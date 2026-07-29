#!/usr/bin/env python
"""非加性分层分析: 检验 IPLC 的交互建模是否在真正非加性(上位)的 doubles 上胜过纯加性 dense。
用真值 Δ(A+B) vs Δ(A)+Δ(B) 的偏差度量非加性, 分层比较各模型 per-condition pearson_delta。
用法: cd SPEAR-main && ~/spear_venv/bin/python ../analyze_nonadditive.py"""
import scanpy as sc, numpy as np, pandas as pd, glob
a=sc.read_h5ad("data/norman.h5ad"); cond=a.obs['condition'].astype(str).values; X=a.X
def mo(c):
    i=np.where(cond==c)[0]; return None if len(i)==0 else np.asarray(X[i].mean(0)).ravel()
ctrl=mo('ctrl')
def dl(c):
    m=mo(c); return None if m is None else m-ctrl
def load(n):
    f=glob.glob(f"result/night/{n}/**/iteration_5000/results.csv",recursive=True)
    return dict(zip(pd.read_csv(f[0])['perturbation'],pd.read_csv(f[0])['pearson_delta'])) if f else {}
M={n:load(n) for n in ["base5k_s42","dense_5k_s42","gelu256_5k_s42","lin512_5k_s42"]}
doubles=[p for p in M["dense_5k_s42"] if '+' in p and 'control' not in p]
rows=[]
for p in doubles:
    A,B=p.split('+'); dAB=dl(f"{A}+{B}") or dl(f"{B}+{A}"); dA=dl(f"{A}+ctrl"); dB=dl(f"{B}+ctrl")
    if dAB is None or dA is None or dB is None: continue
    na=np.linalg.norm(dAB-(dA+dB))/(np.linalg.norm(dAB)+1e-8)
    rows.append(dict(pert=p,nonadd=na,**{k.split('_')[0]:M[k].get(p,np.nan) for k in M}))
df=pd.DataFrame(rows).dropna(); med=df['nonadd'].median()
print(f"doubles={len(df)}, 非加性中位数={med:.3f}")
for g,nm in [(df[df.nonadd<=med],"低非加性"),(df[df.nonadd>med],"高非加性/上位")]:
    print(f"\n{nm} (n={len(g)}): baseline={g['base5k'].mean():.4f} dense={g['dense5k'].mean():.4f} "
          f"gelu256={g['gelu2565k'].mean():.4f} lin512={g['lin5125k'].mean():.4f} | gelu256−dense={g['gelu2565k'].mean()-g['dense5k'].mean():+.4f}")
