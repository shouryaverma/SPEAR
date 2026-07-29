"""Non-inferiority performance table: SAMS-VAE vs from-scratch base vs IPLC.

Main judge (locked): doubles-only (unseen A+B) pearson_delta, deterministic eval, matched
step count. SAMS-VAE is pure-additive, so on the additive split the expected result is
NON-INFERIORITY (samsvae ~= base ~= iplc), not a win -- performance is a footnote to the
interpretability headline (see samsvae_interpret.py). Reuses the existing byte-identical-code
comparators in ../SPEAR-main (no retraining).

Usage: python src/script/samsvae_perf_table.py [--ck 50000]
"""
import glob, argparse, numpy as np, pandas as pd
from scipy import stats

SAMS_DIR = "result/scratch2"                                   # this codebase (samsvae runs)
COMP_DIR = "../SPEAR-main/result/scratch2"                      # reused base/iplc comparators
METRICS = ['pearson_delta', 'mse_delta', 'de_direction_match']  # doubles-only


def cls(p):
    p = str(p)
    if "+" not in p:
        return "ctrl"
    a, b = p.split("+", 1)
    return "ctrl" if (a == "control" and b == "control") else \
        ("single" if (a == "control" or b == "control") else "double")


def run_metrics(run_dir, ck):
    fs = glob.glob(f"{run_dir}/**/iteration_{ck}/results.csv", recursive=True)
    if not fs:
        return None
    d = pd.read_csv(fs[0])
    d = d[d["perturbation"].apply(cls) == "double"]
    if not len(d):
        return None
    return {m: float(d[m].mean()) for m in METRICS if m in d.columns}


def collect(dir_, patterns, ck):
    """patterns: list of glob stems (each a run family across seeds). Returns per-metric arrays."""
    out = {m: [] for m in METRICS}
    used = []
    for stem in patterns:
        for run in sorted(glob.glob(f"{dir_}/{stem}")):
            r = run_metrics(run, ck)
            if r:
                used.append(run.split('/')[-1])
                for m in METRICS:
                    if m in r:
                        out[m].append(r[m])
    return out, used


def fmt(arr):
    a = np.array(arr, dtype=float)
    return f"{a.mean():.4f} ± {a.std():.4f} (n={len(a)})" if len(a) else "  -  "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ck', default='50000')
    args = ap.parse_args()
    ck = args.ck

    groups = {
        "SAMS-VAE (concrete,L64)": (SAMS_DIR, ["samsvae_concrete_L64_lm1e-4_s*"]),
        "base (from-scratch)":     (COMP_DIR, ["base_s*"]),
        "IPLC":                    (COMP_DIR, ["iplc_s*"]),
    }
    data = {}
    print(f"### Non-inferiority performance @ iteration {ck} — norman/additive/fold1 ###")
    print("Main judge: doubles-only pearson_delta (higher=better)\n")
    print(f"{'config':26s}  {'doubles pearson_delta':24s}  {'mse_delta(lower better)':24s}  {'de_direction':18s}")
    for name, (dir_, pats) in groups.items():
        m, used = collect(dir_, pats, ck)
        data[name] = m
        print(f"{name:26s}  {fmt(m['pearson_delta']):24s}  {fmt(m['mse_delta']):24s}  {fmt(m['de_direction_match']):18s}")
        print(f"{'':26s}  seeds: {used}")

    # non-inferiority read on the main judge
    sv = np.array(data["SAMS-VAE (concrete,L64)"]['pearson_delta'], dtype=float)
    print("\n--- Main-judge (doubles pearson_delta) comparison ---")
    for ref in ["base (from-scratch)", "IPLC"]:
        rv = np.array(data[ref]['pearson_delta'], dtype=float)
        if len(sv) and len(rv):
            delta = sv.mean() - rv.mean()
            # small-n: Mann-Whitney U (rank) as a robustness indicator; report delta primarily
            try:
                p = stats.mannwhitneyu(sv, rv, alternative='two-sided').pvalue
            except Exception:
                p = float('nan')
            tag = ("~= (non-inferior)" if abs(delta) < 0.01 else
                   ("SAMS-VAE higher" if delta > 0 else "SAMS-VAE lower"))
            print(f"  SAMS-VAE vs {ref:22s}: Δ={delta:+.4f}  (MWU p={p:.3f}, small-n) -> {tag}")
    print("\nNote: pure-additive head on an additive split -> non-inferiority is the expected "
          "and honest result; the contribution is interpretability (see samsvae_interpret.py).")


if __name__ == '__main__':
    main()
