"""How large is the mechanical regression-to-the-mean in the gap-chasing
regression, measured rather than assumed?

beta is estimated from  dtheta = theta_new - theta_old  on  gap = frontier - theta_old.
Both contain theta_old, so noise in theta_old induces a positive slope even when
no firm responds to anything.  The clean fix is to estimate theta_old twice on
disjoint halves of the battles.  We cannot redo the event regression that way --
the local battle dump covers only 2025-04-17 to 2025-07-24 and tags only math,
instruction-following and creative writing -- but the same split measures the
bias itself.

Split the battles 50/50, fit Bradley-Terry on each half, and regress

    y = theta_B - theta_A        (pure noise: its expectation is zero)
    x = frontier_A - theta_A     (the gap, built from the SAME half as enters y)

The slope of that regression is exactly the spurious component of beta.  We
compare it with what the errors-in-variables correction in the draft predicts,
which is var(noise in theta_old) / var(gap).
"""
import glob
import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

FILES = sorted(glob.glob('data/140k/data/*.parquet'))
TAGS = {'math': ('math_v0.1', 'math'),
        'instruction_following': ('if_v0.1', 'if'),
        'creative_writing': ('creative_writing_v0.1', 'creative_writing')}
MIN_BATTLES = 60
SCALE = 400 / np.log(10)


def load():
    cols = ['model_a', 'model_b', 'winner', 'category_tag']
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in FILES],
                  ignore_index=True)
    d = d[d.winner.isin(['model_a', 'model_b', 'tie', 'both_bad'])].copy()
    d['s'] = np.where(d.winner == 'model_a', 1.0,
                      np.where(d.winner == 'model_b', 0.0, 0.5))
    for cat, (block, key) in TAGS.items():
        d[cat] = d.category_tag.map(
            lambda t: bool(t.get(block, {}).get(key)) if isinstance(t, dict) else False)
    return d


def fit_bt(df, models, ridge=1e-3, iters=400, tol=1e-9):
    """Bradley-Terry by Newton steps on the logistic likelihood, ties = 0.5."""
    idx = {m: i for i, m in enumerate(models)}
    a = df.model_a.map(idx).values
    b = df.model_b.map(idx).values
    s = df.s.values
    n = len(models)
    th = np.zeros(n)
    for _ in range(iters):
        z = th[a] - th[b]
        p = 1.0 / (1.0 + np.exp(-z))
        r = s - p
        g = np.bincount(a, r, n) - np.bincount(b, r, n) - ridge * th
        w = p * (1 - p)
        h = np.bincount(a, w, n) + np.bincount(b, w, n) + ridge
        step = g / np.maximum(h, 1e-9)
        th += step
        th -= th.mean()
        if np.abs(step).max() < tol:
            break
    # inverse-Hessian diagonal as the sampling variance, on the Arena scale
    z = th[a] - th[b]
    p = 1.0 / (1.0 + np.exp(-z))
    w = p * (1 - p)
    h = np.bincount(a, w, n) + np.bincount(b, w, n) + ridge
    return th * SCALE, (1.0 / np.maximum(h, 1e-9)) * SCALE ** 2


def one_split(d, rng):
    half = rng.random(len(d)) < 0.5
    rows = []
    for cat in ['overall'] + list(TAGS):
        sub = d if cat == 'overall' else d[d[cat]]
        cnt = pd.concat([sub.model_a, sub.model_b]).value_counts()
        models = sorted(cnt[cnt >= MIN_BATTLES].index)
        if len(models) < 8:
            continue
        sA = sub[half[sub.index]]
        sB = sub[~half[sub.index]]
        sA = sA[sA.model_a.isin(models) & sA.model_b.isin(models)]
        sB = sB[sB.model_a.isin(models) & sB.model_b.isin(models)]
        if len(sA) < 500 or len(sB) < 500:
            continue
        thA, vA = fit_bt(sA, models)
        thB, vB = fit_bt(sB, models)
        thB = thB - thB.mean() + thA.mean()
        frontA = thA.max()
        for i, m in enumerate(models):
            rows.append(dict(cat=cat, model=m,
                             gap_A=frontA - thA[i],
                             y=thB[i] - thA[i],
                             vA=vA[i], vB=vB[i]))
    return pd.DataFrame(rows)


def slope(x, y):
    b1, b0 = np.polyfit(x, y, 1)
    e = y - (b0 + b1 * x)
    se = np.sqrt((e ** 2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
    return b1, se


d = load()
print(f'battles={len(d)}  models={pd.concat([d.model_a, d.model_b]).nunique()}')
rng = np.random.default_rng(0)
allres, bias, pred = [], [], []
N_SPLIT = 20
for s in range(N_SPLIT):
    t = one_split(d, rng)
    if len(t) == 0:
        continue
    t['split'] = s
    allres.append(t)
    # pooled over categories, category means removed
    x = t.gap_A - t.groupby('cat').gap_A.transform('mean')
    y = t.y - t.groupby('cat').y.transform('mean')
    b1, _ = slope(x.values, y.values)
    bias.append(b1)
    # what the EIV correction predicts: var(noise in theta_old)/var(gap)
    pred.append(t.vA.mean() / x.var(ddof=1))

R = pd.concat(allres, ignore_index=True)
bias = np.array(bias); pred = np.array(pred)
print(f'\nmodels per category (half-sample fits, >= {MIN_BATTLES} battles):')
print(R[R.split == 0].groupby('cat').size().to_string())

print(f'\nMEASURED spurious slope over {len(bias)} random 50/50 splits:')
print(f'  mean {bias.mean():+.4f}   sd {bias.std():.4f}   '
      f'range [{bias.min():+.4f}, {bias.max():+.4f}]')
print(f'EIV-predicted spurious slope  var(noise)/var(gap):')
print(f'  mean {pred.mean():+.4f}   sd {pred.std():.4f}')

print('\nby category (pooled over splits):')
rows = []
for cat, g in R.groupby('cat'):
    b1, se = slope(g.gap_A.values, g.y.values)
    rows.append((cat, b1, se, g.vA.mean() / g.gap_A.var(ddof=1),
                 np.sqrt(g.vA.mean()), g.gap_A.std(), g.model.nunique()))
print(pd.DataFrame(rows, columns=['category', 'measured_bias', 'se',
                                  'EIV_predicted', 'noise_sd', 'gap_sd', 'models'])
      .to_string(index=False, float_format=lambda x: f'{x:8.3f}'))

# is the half-sample noise consistent with the reported Arena variances?
print('\nhalf-sample noise check: sd(theta_B - theta_A) vs sqrt(vA+vB)')
for cat, g in R.groupby('cat'):
    print(f'  {cat:<22} empirical {g.y.std():6.2f}   '
          f'from Hessian {np.sqrt(g.vA.mean()+g.vB.mean()):6.2f}')
