"""RQ3, continued. The same recovery under other production functions.

theta_i,k = a_i,k phi(x_i,k), budget sum_k x_i,k = c_i, firm maximises
sum_k w_k theta_i,k.  The first-order condition gives

    psi(x_k) = phi'(x_k)/phi(x_k) = lambda / (w_k theta_k)

so x_k = psi^{-1}(lambda / (w_k theta_k)) and lambda is set by the budget.  Any
increasing concave phi with invertible psi works.  Two families:

  phi = log(1+x)   psi = 1/((1+x)log(1+x))   -> Lambert W, no closed form
  phi = x^alpha    psi = alpha/x             -> x_k = c w_k theta_k / sum_j w_j theta_j
                                                a_k = theta_k^(1-alpha) (S/(c w_k))^alpha

The power case is worth stating: the ALLOCATION does not depend on alpha at all,
and a_k scales as c^(-alpha) uniformly across k, so the shape of A_i is free of
the budget -- unlike the log case, where the shape drifts with c.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
from scipy.special import lambertw
from scipy.optimize import brentq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import load_panel, flagships, CORE, FRONTIER

DO = {'openai': 'OpenAI', 'anthropic': 'Anthropic', 'google': 'Google',
      'alibaba': 'Alibaba', 'meta': 'Meta', 'mistral': 'Mistral', 'xai': 'xAI',
      'deepseek': 'DeepSeek'}
SNAP = pd.Timestamp('2026-03-19')
C_GRID = np.logspace(-1, 3, 41)


def a_log(theta, w, c):
    def spend(ll):
        u = np.real(lambertw(w * theta / np.exp(ll)))
        return np.sum(np.exp(u) - 1) - c
    if spend(-40) < 0 or spend(40) > 0:
        return None
    ll = brentq(spend, -40, 40, xtol=1e-12, rtol=1e-14)
    u = np.real(lambertw(w * theta / np.exp(ll)))
    return theta / u, np.exp(u) - 1


def a_pow(theta, w, c, alpha):
    S = np.sum(w * theta)
    x = c * w * theta / S
    return theta / x ** alpha, x


d = load_panel(cats=CORE, firms_only=False, since=None)
d = d[d.dt <= SNAP]
flag = flagships(d)
rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')
t = d.dt.max()
field = d[(d.dt == t) & (d.category.isin(CORE))].dropna(subset=['rating'])
floor = field.groupby('category').rating.min()[CORE].values
vc = field.groupby('category').vote_count.sum().reindex(CORE)
W = (vc / vc.sum()).values

firms, TH = [], []
for f in FRONTIER:
    m = flag.get((t, f))
    if m is None:
        continue
    try:
        r = rat.loc[(t, m)][CORE]
    except KeyError:
        continue
    if r.isna().any():
        continue
    firms.append(f)
    TH.append(r.values.astype(float) - floor)
TH = np.array(TH)
print(f'snapshot {pd.Timestamp(t).date()}, {len(firms)} firms, 8 categories')

FAMS = [('log(1+x)', None), ('sqrt(x)', 0.5), ('x^0.25', 0.25), ('x^0.75', 0.75)]


def decomp(N):
    tot = ((N - N.mean()) ** 2).sum()
    cat = len(N) * ((N.mean(0) - N.mean()) ** 2).sum()
    frm = N.shape[1] * ((N.mean(1) - N.mean()) ** 2).sum()
    return 100 * cat / tot, 100 * frm / tot, 100 * (tot - cat - frm) / tot


print('\n(1) IDENTIFIED SET over c in [0.1, 1000]: max/min of a_i,k')
rows = []
for lab, alpha in FAMS:
    ratios, shapes = [], []
    for i, f in enumerate(firms):
        vals = []
        for c in C_GRID:
            out = a_log(TH[i], W, c) if alpha is None else a_pow(TH[i], W, c, alpha)
            if out is None:
                continue
            vals.append(out[0])
        V = np.array(vals)
        ratios.extend((V.max(0) / V.min(0)).tolist())
        shapes.append(V / V.mean(1, keepdims=True))
    ratios = np.array(ratios)
    # does the shape move with the budget?
    drift = np.max([np.abs(s[0] - s[-1]).max() for s in shapes])
    rows.append((lab, np.median(ratios), ratios.min(), ratios.max(), drift))
print(pd.DataFrame(rows, columns=['family', 'median max/min', 'min', 'max',
                                  'max shape drift over c'])
      .to_string(index=False, float_format=lambda x: f'{x:12.3f}'))

print('\n(2) SHAPE OF A AT c = 10, and where its variance comes from')
for lab, alpha in FAMS:
    M = []
    for i, f in enumerate(firms):
        out = a_log(TH[i], W, 10.0) if alpha is None else a_pow(TH[i], W, 10.0, alpha)
        M.append(out[0])
    M = np.array(M)
    N = M / M.mean(1, keepdims=True)
    cat, frm, inter = decomp(N)
    print(f'  {lab:<10} category {cat:5.1f}%   firm {frm:5.1f}%   '
          f'firm x category {inter:5.1f}%   '
          f'max across-firm spread within a category {np.max(N.max(0)-N.min(0)):.4f}')
    print(f'             mean profile: ' +
          ', '.join(f'{CORE[j][:6]}={N.mean(0)[j]:.2f}' for j in range(8)))

print('\n(3) POWER FAMILY: the allocation does not depend on alpha')
i = firms.index('openai')
for alpha in (0.25, 0.5, 0.75):
    _, x = a_pow(TH[i], W, 10.0, alpha)
    print(f'  alpha={alpha}: x = ' + ', '.join(f'{v:.3f}' for v in x))

print('\n(4) FIRM SCALE vs FIRM SHAPE, sqrt(x), c = 10')
M = np.array([a_pow(TH[i], W, 10.0, 0.5)[0] for i in range(len(firms))])
N = M / M.mean(1, keepdims=True)
sc = pd.Series(M.mean(1), index=[DO.get(f, f) for f in firms]).sort_values(ascending=False)
print('  scale (mean over k): ' + ', '.join(f'{k}={v:.1f}' for k, v in sc.items()))
print(f'  scale spread max/min = {sc.max()/sc.min():.3f}')
C = np.corrcoef(N)
iu = np.triu_indices(len(firms), 1)
print(f'  pairwise correlation of the normalised shapes: '
      f'min {C[iu].min():.4f}, median {np.median(C[iu]):.4f}')
print('\n  normalised shape per firm (sqrt):')
print(pd.DataFrame(N, index=[DO.get(f, f) for f in firms],
                   columns=[c[:8] for c in CORE]).round(3).to_string())
