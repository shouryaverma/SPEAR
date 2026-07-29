"""
Collect the CellFlow-PT / SPEAR comparison from eval logs -> final table:
per-arm mean +/- std across seeds (best ckpt per seed by stable de_direction_match),
plus paired tests (spe vs gene, spear vs gene, spear vs spe) isolating SPE / SPE+AR.

  # additive (defaults): reads ms_cellflow*_s{42,43,44}.log + spear_eval.log
  python cellflow_pt/collect_results.py
  # unseen split:
  python cellflow_pt/collect_results.py --tag ms_unseen --seeds 42 43 44 45 --no_spear
"""
import re, os, argparse
import numpy as np
from scipy import stats

LOGDIR = '/home/wangmengbo/SPEAR'
METRICS = ['de_spearman_lfc_sig', 'de_direction_match', 'de_spearman_sig',
           'pearson_delta', 'mse_delta', 'discrimination_score_cosine']
SELECT_BY = 'de_direction_match'
EVAL_RE = re.compile(r'\[(?:eval \d+|SPEAR eval)\]\s+(.*)')
KV_RE = re.compile(r'(\w+)=(-?\d+\.?\d*)')


def parse_log(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path):
        m = EVAL_RE.search(line)
        if m:
            d = {k: float(v) for k, v in KV_RE.findall(m.group(1))}
            if d:
                rows.append(d)
    return rows


def best_row(rows, by=SELECT_BY):
    rows = [r for r in rows if by in r]
    return max(rows, key=lambda r: r[by]) if rows else None


SPEAR_SUMMARY_RE = re.compile(r'^\s+(\w+)\s+(-?\d+\.?\d*)\s*(?:±|\+/-)')


def parse_spear(path):
    """Parse eval_spear.py's final 'doubles-only, mean +/- std' summary block."""
    if not os.path.exists(path):
        return None
    out = {}
    for line in open(path):
        m = SPEAR_SUMMARY_RE.match(line)
        if m and m.group(1) in METRICS:
            out[m.group(1)] = float(m.group(2))
    return out or None


def collect_arm(tag, arm_sfx, seeds):
    out = {}
    for s in seeds:
        name = f"{tag}_cellflow{arm_sfx}_s{s}.log"
        b = best_row(parse_log(os.path.join(LOGDIR, name)))
        if b:
            out[s] = b
    return out


def agg(d, m):
    vals = [r[m] for r in d.values() if m in r]
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (None, None, 0)


def paired(a, b, label):
    shared = sorted(set(a) & set(b))
    print(f"\n{'-'*104}\nPAIRED {label}  (shared seeds={shared})\n{'-'*104}")
    if len(shared) < 2:
        print(f"  need >=2 shared seeds (have {len(shared)})"); return
    for m in METRICS:
        av = np.array([a[s][m] for s in shared]); bv = np.array([b[s][m] for s in shared])
        diff = av - bv
        try:
            p = stats.ttest_rel(av, bv).pvalue
        except Exception:
            p = float('nan')
        star = ' *' if (p == p and p < 0.05) else ''
        print(f"  {m:<28} {label.split(' vs ')[0]}-{label.split(' vs ')[1]} = "
              f"{diff.mean():+.4f} (per-seed {np.round(diff,3)})  p={p:.3f}{star}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='ms')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--no_spear', action='store_true', help='skip the trained-SPEAR row')
    a = ap.parse_args()

    arms = {
        'CellFlow (PCA, vanilla)': collect_arm(a.tag, '', a.seeds),
        'CellFlow_gene (no SPE)':  collect_arm(a.tag, '_gene', a.seeds),
        'CellFlow_spe (+SPE)':     collect_arm(a.tag, '_spe', a.seeds),
        'CellFlow_spear (+SPE+AR)': collect_arm(a.tag, '_spear', a.seeds),
        'CellFlow_ae (learn enc/dec)': collect_arm(a.tag, '_ae', a.seeds),
        'CellFlow_ae_spe (+SPE enc)':  collect_arm(a.tag, '_ae_spe', a.seeds),
    }
    spear = None
    if not a.no_spear:
        spear = parse_spear(os.path.join(LOGDIR, 'spear_eval.log'))

    print(f"\n{'='*104}\nCOMPARISON tag={a.tag} seeds={a.seeds}  "
          f"(best ckpt per seed by {SELECT_BY}; mean +/- std)\n{'='*104}")
    print(f"{'arm':<26}" + "".join(f"{m[:14]:>15}" for m in METRICS) + f"{'seeds':>7}")
    for name, d in arms.items():
        if not d:
            print(f"{name:<26}  (no runs yet)"); continue
        cells = "".join(f"{(f'{agg(d,m)[0]:.3f}±{agg(d,m)[1]:.3f}' if agg(d,m)[0] is not None else '-'):>15}"
                        for m in METRICS)
        print(f"{name:<26}{cells}{len(d):>7}")
    if spear:
        print(f"{'SPEAR (full, 295k)':<26}" +
              "".join(f"{spear.get(m, float('nan')):>15.3f}" for m in METRICS) + f"{'1':>7}")

    g, s, sp = arms['CellFlow_gene (no SPE)'], arms['CellFlow_spe (+SPE)'], arms['CellFlow_spear (+SPE+AR)']
    if g and s:
        paired(s, g, 'spe vs gene')
    if g and sp:
        paired(sp, g, 'spear vs gene')
    print()


if __name__ == '__main__':
    main()
