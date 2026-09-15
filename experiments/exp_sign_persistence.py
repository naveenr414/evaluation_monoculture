"""How many firm-category comparative advantages hold the SAME SIGN every
quarter, and is that more than chance?

The naive null -- each quarter's sign is an independent coin flip, so
P(all same sign) = 2^{1-T} -- is far too strict here, because the quarterly
two-way residual is autocorrelated (rho_1 ~ 0.7).  An autocorrelated zero-mean
series holds one sign for long stretches on its own.  We therefore calibrate
the null by simulation: zero-mean AR(1) series with the observed rho and
dispersion, pushed through the same two-way demeaning (which imposes the row/
column adding-up constraints that make signs dependent across pairs).

Per-pair inference uses a Newey-West (HAC) t-test on the mean residual, with
Benjamini-Hochberg FDR across all pairs.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'experiments')
from common import load_panel, two_way_resid, flagships, CORE

DO = {'openai': 'OpenAI', 'anthropic': 'Anthropic', 'google': 'Google',
      'alibaba': 'Alibaba', 'meta': 'Meta', 'mistral': 'Mistral', 'xai': 'xAI',
      'deepseek': 'DeepSeek', 'nvidia': 'NVIDIA', 'amazon': 'Amazon',
      'stepfun': 'StepFun', 'cohere': 'Cohere', 'microsoft': 'Microsoft',
      'ibm': 'IBM', 'minimax': 'MiniMax', 'moonshot': 'Moonshot', 'zai': 'Z.ai',
      'meituan': 'Meituan', 'tencent': 'Tencent', 'ant-group': 'Ant Group',
      'baidu': 'Baidu', 'xiaomi': 'Xiaomi', 'bytedance': 'ByteDance',
      'allenai': 'AI2'}
DC = {'coding': 'Coding', 'math': 'Math', 'hard_prompts': 'HardPrompts',
      'instruction_following': 'InstrFollow', 'creative_writing': 'CreativeWr',
      'longer_query': 'LongerQuery', 'multi_turn': 'MultiTurn',
      'english': 'English'}


def build(since):
    d = load_panel(cats=CORE, firms_only=False, since=since)
    flag = flagships(d)
    rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')
    dates = sorted(d.dt.unique())
    last = d.dt.max()
    snap = d[(d.dt == last) & (d.category == 'overall')].dropna(subset=['rating'])
    ranked = (snap.sort_values('rating', ascending=False).groupby('org').head(1)
                  .sort_values('rating', ascending=False))
    orgs = [o for o in ranked.org if o and o not in ('inception-ai',)][:23]
    rows = []
    for t in dates:
        prof = {}
        for o in orgs:
            m = flag.get((t, o))
            if m is None:
                continue
            try:
                r = rat.loc[(t, m)][CORE]
            except KeyError:
                continue
            if r.isna().any():
                continue
            prof[o] = r.values.astype(float)
        if len(prof) < 6:
            continue
        nm = sorted(prof)
        R = two_way_resid(np.array([prof[n] for n in nm]))
        for i, o in enumerate(nm):
            for j, c in enumerate(CORE):
                rows.append((t, o, c, R[i, j]))
    p = pd.DataFrame(rows, columns=['dt', 'org', 'cat', 'resid'])
    p['q'] = pd.PeriodIndex(p.dt, freq='Q')
    return p.groupby(['q', 'org', 'cat']).resid.mean().reset_index()


def nw_tstat(x, lags=1):
    """Newey-West t-stat for the mean of a short autocorrelated series."""
    x = np.asarray(x, float)
    T = len(x)
    mu = x.mean()
    e = x - mu
    g0 = (e @ e) / T
    var = g0
    for L in range(1, lags + 1):
        if L >= T:
            break
        gl = (e[L:] @ e[:-L]) / T
        var += 2 * (1 - L / (lags + 1)) * gl
    var = max(var, 1e-12)
    return mu, mu / np.sqrt(var / T)


def bh(pvals, alpha=0.05):
    p = np.asarray(pvals)
    n = len(p)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, n + 1)) / n
    passed = p[order] <= thresh
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    rej = np.zeros(n, bool)
    rej[order[:k]] = True
    q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return rej, np.clip(q, 0, 1)


def analyse(qp, label, n_sim=20000, seed=0):
    print('\n' + '=' * 78)
    print(label)
    print('=' * 78)
    quarters = sorted(qp.q.unique())
    W = qp.pivot_table(index=['org', 'cat'], columns='q', values='resid')
    bal = W.dropna()
    T = len(quarters)
    print(f'{T} quarters {quarters[0]}..{quarters[-1]};  '
          f'{len(bal)} firm-category pairs observed in ALL quarters '
          f'({bal.index.get_level_values(0).nunique()} firms x {len(CORE)} cats)')

    A = bal.values
    signs = np.sign(A)
    allpos = (signs > 0).all(1)
    allneg = (signs < 0).all(1)
    obs = int(allpos.sum() + allneg.sum())
    print(f'\nOBSERVED all-same-sign pairs: {obs}/{len(bal)} = '
          f'{100*obs/len(bal):.1f}%   ({allpos.sum()} all-positive, '
          f'{allneg.sum()} all-negative)')

    # ------------------------------------------------ nulls -----------------
    naive_p = 2.0 ** (1 - T)
    print(f'\nNAIVE null (independent coin flips): P(all same sign) = '
          f'{naive_p:.4f}  ->  expected {naive_p*len(bal):.1f} pairs')

    # observed pooled lag-1 autocorrelation of the quarterly residual
    num = den = 0.0
    for r in A:
        e = r - r.mean()
        num += (e[1:] * e[:-1]).sum()
        den += (e ** 2).sum()
    rho = num / den
    sd = A.std()
    print(f'pooled within-pair lag-1 autocorrelation rho = {rho:+.3f}, '
          f'residual sd = {sd:.2f}')

    rng = np.random.default_rng(seed)
    n_firm = bal.index.get_level_values(0).nunique()
    K = len(CORE)
    sims = np.empty(n_sim, int)
    for s in range(n_sim):
        # zero-mean AR(1) panel, then two-way demeaned each quarter so the
        # adding-up constraints match the real residual
        X = np.empty((T, n_firm, K))
        prev = rng.normal(0, sd, (n_firm, K))
        for t in range(T):
            prev = rho * prev + rng.normal(0, sd * np.sqrt(1 - rho ** 2), (n_firm, K))
            X[t] = two_way_resid(prev)
        sg = np.sign(X)
        same = ((sg > 0).all(0) | (sg < 0).all(0)).sum()
        sims[s] = same
    pval = (sims >= obs).mean()
    print(f'\nCALIBRATED null (zero-mean AR(1), rho={rho:.2f}, two-way demeaned):')
    print(f'  expected all-same-sign pairs = {sims.mean():.1f} '
          f'(sd {sims.std():.1f}, 95th pct {np.percentile(sims,95):.0f})')
    print(f'  observed {obs}  ->  p = {pval:.4f}   '
          f'z = {(obs - sims.mean())/sims.std():+.2f}')
    print(f'  [naive null would have said expected {naive_p*len(bal):.1f}, '
          f'i.e. {sims.mean()/max(naive_p*len(bal),1e-9):.1f}x too few]')

    # ------------------------------- per-pair HAC t-tests + BH-FDR ----------
    from scipy import stats
    recs = []
    for (org, cat), r in zip(bal.index, A):
        mu, t = nw_tstat(r, lags=1)
        p_ = 2 * stats.t.sf(abs(t), df=len(r) - 1)
        recs.append((DO.get(org, org), DC[cat], mu, t, p_,
                     int((np.sign(r) > 0).all() or (np.sign(r) < 0).all())))
    tab = pd.DataFrame(recs, columns=['firm', 'category', 'mean', 't_HAC', 'p', 'same_sign'])
    rej, q = bh(tab.p.values, 0.05)
    tab['q_BH'] = q
    tab['sig'] = rej
    print(f'\nPER-PAIR HAC t-test on the mean residual, BH-FDR at 5%:')
    print(f'  {tab.sig.sum()}/{len(tab)} pairs significant '
          f'({100*tab.sig.mean():.1f}%)')
    print(f'  of the {obs} all-same-sign pairs, {int(tab[tab.same_sign==1].sig.sum())} '
          f'are also FDR-significant')
    print('\n  strongest persistent specializations (|t| HAC, FDR-significant):')
    top = tab[tab.sig].reindex(tab[tab.sig].t_HAC.abs().sort_values(ascending=False).index)
    print(top.head(15).to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    return tab


qp25 = build('2025-01-01')
analyse(qp25, 'A.  2025Q1-2026Q2  (the window used in the draft figures)')

qp24 = build('2024-01-01')
analyse(qp24, 'B.  2024Q1-2026Q2  (longer window, more quarters, fewer firms balanced)')


# ---------------------------------------------------------------------------
# Sensitivity: the null's rho is estimated AFTER removing each pair's own mean,
# which at T=6 is downward-biased (Nickell bias ~ -(1+rho)/(T-1) ~ -0.24).  The
# null gets easier to beat as rho falls, so re-run it across a range of rho up
# to the raw AR(1) slope (0.71) measured on the undemeaned series.
# ---------------------------------------------------------------------------
def rho_sensitivity(qp, label, obs=None, n_sim=20000, seed=1):
    print('\n' + '=' * 78)
    print('RHO SENSITIVITY OF THE NULL -- ' + label)
    print('=' * 78)
    quarters = sorted(qp.q.unique())
    bal = qp.pivot_table(index=['org', 'cat'], columns='q', values='resid').dropna()
    A = bal.values
    T = len(quarters)
    sg = np.sign(A)
    obs = int(((sg > 0).all(1) | (sg < 0).all(1)).sum())
    n_firm = bal.index.get_level_values(0).nunique()
    K = len(CORE)
    sd = A.std()
    rng = np.random.default_rng(seed)
    print(f'  observed = {obs} of {len(bal)} pairs, T={T}')
    print(f'  {"rho":>6} {"E[null]":>9} {"sd":>6} {"95th":>6} {"p":>8} {"z":>7}')
    for rho in (0.22, 0.35, 0.46, 0.60, 0.71, 0.85):
        sims = np.empty(n_sim, int)
        for s in range(n_sim):
            X = np.empty((T, n_firm, K))
            prev = rng.normal(0, sd, (n_firm, K))
            for t in range(T):
                prev = rho * prev + rng.normal(0, sd * np.sqrt(1 - rho ** 2),
                                               (n_firm, K))
                X[t] = two_way_resid(prev)
            s2 = np.sign(X)
            sims[s] = ((s2 > 0).all(0) | (s2 < 0).all(0)).sum()
        p = (sims >= obs).mean()
        print(f'  {rho:>6.2f} {sims.mean():>9.1f} {sims.std():>6.1f} '
              f'{np.percentile(sims,95):>6.0f} {p:>8.4f} '
              f'{(obs-sims.mean())/sims.std():>+7.2f}')


rho_sensitivity(qp25, 'A. 2025Q1-2026Q2')
rho_sensitivity(qp24, 'B. 2024Q4-2026Q2')
