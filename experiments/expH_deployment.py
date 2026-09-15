"""Experiment H: deployment monoculture and the evaluation/usage divergence.

The paper measures homogeneity in capability profiles. But the welfare-relevant
monoculture is whether everyone USES the same model. OpenRouter gives that directly.

H1  concentration of deployment over time (HHI, effective number of models/firms)
H2  KL(v || w) at model granularity, where
        v = share of tokens actually served      (OpenRouter)
        w = share of evaluation attention        (Arena vote counts)
    This is the paper's welfare term, computable today without the category-level
    OpenRouter pull -- at model rather than category granularity.
H3  is the frontier that people USE more homogeneous than the frontier at large?
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'experiments')
from common import load_panel, two_way_resid, firm_profiles, CORE, FRONTIER
import or_arena_match

o = pd.read_parquet('data/openrouter_daily.parquet')
o['date'] = pd.to_datetime(o.date)
o['org'] = o.model_permaslug.str.split('/').str[0].str.lower()

# ---------------- H1: deployment concentration ----------------
print('=== H1  concentration of deployment ===')
rows = []
for per, g in o.groupby(o.date.dt.to_period('M')):
    tm = g.groupby('model_permaslug').total_tokens.sum()
    tf = g.groupby('org').total_tokens.sum()
    sm, sf = tm / tm.sum(), tf / tf.sum()
    rows.append(dict(month=per.to_timestamp(), n_models=len(sm), n_orgs=len(sf),
                     hhi_model=(sm ** 2).sum(), hhi_firm=(sf ** 2).sum(),
                     eff_models=1 / (sm ** 2).sum(), eff_firms=1 / (sf ** 2).sum(),
                     top1=sm.max(), top3=sm.nlargest(3).sum(),
                     top1_firm=sf.max(), top3_firm=sf.nlargest(3).sum()))
h = pd.DataFrame(rows).set_index('month')
print(h[['n_models', 'eff_models', 'eff_firms', 'top1_firm', 'top3_firm']].round(3).to_string())

t = (h.index - h.index[0]).days.values / 365.25
for c in ['eff_models', 'eff_firms', 'top3_firm']:
    b = np.polyfit(t, h[c].values, 1)[0]
    r = np.corrcoef(t, h[c].values)[0, 1]
    print(f'  {c:12s} slope/yr={b:+7.3f}  r={r:+.3f}')
print('\n  effective number of firms is the deployment analogue of monoculture:')
print(f'  {h.eff_firms.iloc[0]:.2f} firms at the start -> {h.eff_firms.iloc[-1]:.2f} at the end')

# ---------------- match OpenRouter slugs to Arena model names ----------------
print('\n=== matching OpenRouter models to the Arena panel ===')
d = load_panel(cats=CORE, firms_only=False, since='2025-01-01')
arena_models = sorted(d.model_name.unique())

or_models = o.groupby('model_permaslug').total_tokens.sum().sort_values(ascending=False)
match_result = or_arena_match.match(or_models.index, arena_models)
matched = {slug: name for slug, (name, method) in match_result.items()}

cov = or_models[or_models.index.isin(matched)].sum() / or_models.sum()
print(f'matched {len(matched)}/{len(or_models)} OpenRouter models')
print(f'   covering {cov:.1%} of all tokens served')
from collections import Counter
print('   by method:', Counter(m for _, m in match_result.values()))
print('   top unmatched by volume:')
un = or_models[~or_models.index.isin(matched)].head(8)
for s, v in un.items():
    print(f'     {s:45s} {v/or_models.sum():6.2%}')

# ---------------- H2: KL(v || w) ----------------
print('\n=== H2  KL(usage || evaluation attention), model granularity ===')
res = []
for per, g in o.groupby(o.date.dt.to_period('M')):
    tm = g.groupby('model_permaslug').total_tokens.sum()
    tm = tm[tm.index.isin(matched)]
    if len(tm) < 15:
        continue
    tm.index = [matched[s] for s in tm.index]
    tm = tm.groupby(level=0).sum()

    snapd = d[d.dt <= per.to_timestamp('M')]
    if snapd.empty:
        continue
    last = snapd.dt.max()
    votes = (d[(d.dt == last) & (d.category == 'overall')]
             .set_index('model_name').vote_count.dropna())
    common = tm.index.intersection(votes.index)
    if len(common) < 15:
        continue
    v = tm[common] / tm[common].sum()
    w = votes[common] / votes[common].sum()
    kl_vw = float((v * np.log(v / w)).sum())
    kl_wv = float((w * np.log(w / v)).sum())
    res.append(dict(month=per.to_timestamp(), n=len(common), kl_vw=kl_vw, kl_wv=kl_wv,
                    tv=0.5 * float((v - w).abs().sum())))
k = pd.DataFrame(res).set_index('month')
print(k.round(3).to_string())
if len(k) > 3:
    tt = (k.index - k.index[0]).days.values / 365.25
    for c in ['kl_vw', 'tv']:
        b = np.polyfit(tt, k[c].values, 1)[0]
        print(f'  {c:6s} mean={k[c].mean():.3f}  slope/yr={b:+.3f}'
              f'  r={np.corrcoef(tt,k[c].values)[0,1]:+.3f}')
    print('\n  KL(v||w) is the paper\'s welfare loss term. Values well above 0 mean')
    print('  evaluation attention is allocated very differently from actual usage.')

# ---------------- H3: usage-weighted homogeneity ----------------
print('\n=== H3  is the frontier people USE more homogeneous than the frontier at large? ===')
dd = load_panel(cats=CORE, since='2025-01-01')
rows = []
for per, g in o.groupby(o.date.dt.to_period('M')):
    tf = g.groupby('org').total_tokens.sum()
    sf = tf / tf.sum()
    snaps = sorted(dd.dt[dd.dt <= per.to_timestamp('M')].unique())
    if not snaps:
        continue
    P, V, names = firm_profiles(dd, CORE, snaps[-1])
    if len(P) < 5:
        continue
    R = two_way_resid(P)
    wts = np.array([sf.get(n if n != 'xai' else 'x-ai',
                           sf.get('meta-llama', 0) if n == 'meta' else 0) for n in names])
    if wts.sum() <= 0:
        continue
    wts = wts / wts.sum()
    unw = R.var(axis=0, ddof=1).mean()
    mu = (R * wts[:, None]).sum(0)
    wtd = (wts[:, None] * (R - mu) ** 2).sum(0).mean() * len(P) / (len(P) - 1)
    rows.append(dict(month=per.to_timestamp(), unweighted=unw, usage_weighted=wtd,
                     ratio=wtd / unw))
u = pd.DataFrame(rows).set_index('month')
if len(u):
    print(u.round(2).to_string())
    print(f'\n  mean ratio (usage-weighted / unweighted) = {u.ratio.mean():.2f}')
    print('  ratio < 1 => the models people actually use are MORE similar than the')
    print('  frontier at large, i.e. effective monoculture exceeds measured monoculture.')
