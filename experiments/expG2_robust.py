"""Experiment G2: robustness for the production-function comparison.

Two threats to G:
  (1) labs whose ||A_i|| is indistinguishable from zero have no measurable scaling
      response, so cosines involving them are undefined in practice and drag the
      average toward zero;
  (2) within a lab, bigger models tend to be NEWER, so a slope on log(params) can be
      picking up generational progress rather than scale. Add a release-date control.
"""
import sys, numpy as np, pandas as pd
from itertools import combinations

DIMS = ['ifeval', 'bbh', 'math', 'gpqa', 'musr', 'mmlu_pro']
LABS = ['qwen', 'google', 'meta-llama', 'mistralai', 'deepseek-ai', 'microsoft',
        '01-ai', 'ibm-granite', 'allenai', 'huggingfacetb', 'tiiuae', 'eleutherai',
        'nvidia', 'cohereforai', 'nousresearch', 'bigscience', 'arcee-ai']

b = pd.read_parquet('data/benchmarks.parquet')
b['org'] = b.hf_model_id.str.split('/').str[0].str.lower()
b = b[b.params_b.notna() & (b.params_b > 0) & ~b.flagged & ~b.merged].dropna(subset=DIMS)
b['lp'] = np.log(b.params_b)
b['t'] = (pd.to_datetime(b.upload_date) - pd.Timestamp('2023-01-01')).dt.days / 365.25
for k in DIMS:
    b[k] = (b[k] - b[k].mean()) / b[k].std()
b = b[b.org.isin(LABS)].copy()
b['is_instruct'] = (b.model_type == 'instruct').astype(float)
b = b.dropna(subset=['t'])

keep = [o for o, g in b.groupby('org')
        if len(g) >= 8 and np.log10(g.params_b.max() / g.params_b.min()) >= 0.6
        and g.params_b.nunique() >= 4]
b = b[b.org.isin(keep)]


def fit_A(g, with_time):
    cols = [np.ones(len(g)), g.lp.values, g.is_instruct.values]
    if with_time:
        cols.append(g.t.values)
    X = np.column_stack(cols)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    return np.array([np.linalg.lstsq(X, g[k].values, rcond=None)[0][1] for k in DIMS])


def cos(u, v):
    n = np.linalg.norm(u) * np.linalg.norm(v)
    return u @ v / n if n > 0 else np.nan


B, rng = 2000, np.random.default_rng(0)
for with_time in (False, True):
    tag = 'WITH release-date control' if with_time else 'no time control (as in G)'
    print(f'\n{"="*68}\n=== {tag} ===')
    A, boots = {}, {}
    for org, g in b.groupby('org'):
        a = fit_A(g, with_time)
        if a is None:
            continue
        A[org] = a
        bs = []
        for _ in range(B):
            gg = g.iloc[rng.integers(0, len(g), len(g))]
            r = fit_A(gg, with_time)
            bs.append(r if r is not None else a)
        boots[org] = np.array(bs)

    # which labs have a measurable scaling response at all?
    print(f'{"lab":16s} {"||A||":>7s} {"95% CI":>20s}   measurable?')
    strong = []
    for o in sorted(A, key=lambda o: -np.linalg.norm(A[o])):
        nb = np.linalg.norm(boots[o], axis=1)
        lo, hi = np.percentile(nb, [2.5, 97.5])
        # "measurable" = bootstrap lower bound clears a small-effect threshold
        ok = lo > 0.5
        if ok:
            strong.append(o)
        print(f'{o:16s} {np.linalg.norm(A[o]):7.3f}  [{lo:6.3f}, {hi:6.3f}]   {"yes" if ok else "NO"}')

    print(f'\nlabs with a measurable scaling response: {len(strong)}/{len(A)}')
    pr = [(o1, o2, cos(A[o1], A[o2])) for o1, o2 in combinations(strong, 2)]
    cs = np.array([p[2] for p in pr])
    print(f'  mean pairwise cosine among these = {cs.mean():+.3f}'
          f'  median {np.median(cs):+.3f}  min {cs.min():+.3f}')
    allp = np.array([cos(A[o1], A[o2]) for o1, o2 in combinations(A, 2)])
    print(f'  (all labs, for contrast)         = {allp.mean():+.3f}')

    # bootstrap CI on the MEAN cosine among strong labs
    mb = []
    for t in range(B):
        v = [cos(boots[o1][t], boots[o2][t]) for o1, o2 in combinations(strong, 2)]
        mb.append(np.nanmean(v))
    print(f'  95% CI on mean cosine = [{np.percentile(mb,2.5):+.3f}, '
          f'{np.percentile(mb,97.5):+.3f}]')

    M = np.array([A[o] for o in strong])
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    s = np.linalg.svd(Mn, compute_uv=False)
    ev = s**2 / (s**2).sum()
    print(f'  leading common direction explains {ev[0]:.1%} of variation in A'
          f'   (eff. dims {(s**2).sum()**2/((s**2)**2).sum():.2f})')
    print(f'  mean direction loadings: ' +
          ', '.join(f'{k}={v:+.2f}' for k, v in zip(DIMS, Mn.mean(0)/np.linalg.norm(Mn.mean(0)))))
