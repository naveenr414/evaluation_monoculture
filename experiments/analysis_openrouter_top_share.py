"""What fraction of OpenRouter usage goes to models that are NOT the (Arena-ELO)
top model of the moment?

Rather than solving the full slug->Arena-name matching problem (expH's blocker:
only 29.7% of tokens match), we only need to match the *current top-K* Arena
model(s) each month into the OpenRouter slug list -- a much smaller problem.
Every token not attributed to one of those slugs is, by construction, usage of
a non-top model (whether or not it separately matches).

Output: experiments/plot_openrouter_top_share.png
"""
import sys, re, difflib
sys.path.insert(0, 'experiments')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

o = pd.read_parquet('data/openrouter_daily.parquet')
o['date'] = pd.to_datetime(o.date)

d = pd.read_parquet('data/leaderboard/text/full-00000-of-00001.parquet')
d['dt'] = pd.to_datetime(d.leaderboard_publish_date)
ov = d[d.category == 'overall'].dropna(subset=['rating'])


STRIP_TOKENS = {'thinking', 'high', 'low', 'medium', 'preview', 'latest',
                'instruct', 'it', 'hf', 'exp', 'experimental'}


def canon(s):
    """Order-independent, suffix-tolerant key: family/version tokens only.

    Arena names carry reasoning-effort/thinking suffixes and a different
    family-version word order than OpenRouter's date-suffixed permaslugs
    (e.g. arena 'claude-opus-4-6-thinking' vs OpenRouter
    'anthropic/claude-4.6-opus-20260217'); this reduces both to the same
    sorted token set {'4','6','claude','opus'} so they match.
    """
    s = s.lower().split('/')[-1]
    s = s.split(':')[0]
    toks = re.split(r'[^a-z0-9]+', s)
    toks = [t for t in toks if t and t not in STRIP_TOKENS and not re.fullmatch(r'\d{8}', t)]
    return '-'.join(sorted(toks))


all_slugs = o.model_permaslug.unique()
slug_canon = {s: canon(s) for s in all_slugs}
canon_to_slugs = {}
for s, c in slug_canon.items():
    canon_to_slugs.setdefault(c, []).append(s)


def slugs_for(model_name, cutoff=0.90):
    """All OpenRouter permaslugs (any provider suffix, any :variant) that refer
    to this Arena model name -- exact canonical match first, fuzzy fallback."""
    target = canon(model_name)
    if target in canon_to_slugs:
        return canon_to_slugs[target]
    close = difflib.get_close_matches(target, list(canon_to_slugs), n=1, cutoff=cutoff)
    return canon_to_slugs[close[0]] if close else []


K_LIST = [1, 3, 5]
rows = []
for per, g in o.groupby(o.date.dt.to_period('M')):
    total = g.total_tokens.sum()
    tm = g.groupby('model_permaslug').total_tokens.sum()

    snaps = ov.dt[ov.dt <= per.to_timestamp('M')]
    if snaps.empty:
        continue
    last = snaps.max()
    top_ranked = (ov[ov.dt == last].sort_values('rating', ascending=False)
                  .model_name.tolist())

    row = dict(month=per.to_timestamp(), n_tokens=total, arena_snapshot=last,
               top1_name=top_ranked[0], top1_matched=bool(slugs_for(top_ranked[0])))
    for K in K_LIST:
        top_slugs = set()
        for name in top_ranked[:K]:
            top_slugs.update(slugs_for(name))
        top_tokens = tm[tm.index.isin(top_slugs)].sum()
        row[f'share_top{K}'] = top_tokens / total
        row[f'frac_not_top{K}'] = 1 - top_tokens / total
    rows.append(row)

res = pd.DataFrame(rows).set_index('month')
print(res[['top1_name', 'top1_matched'] + [f'frac_not_top{K}' for K in K_LIST]].round(3).to_string())
print(f"\ntop-1 model found in OpenRouter's slug list in {res.top1_matched.mean():.0%} of months")

t = (res.index - res.index[0]).days.values / 365.25
for K in K_LIST:
    c = f'frac_not_top{K}'
    b = np.polyfit(t, res[c].values, 1)[0]
    r = np.corrcoef(t, res[c].values)[0, 1]
    print(f'frac_not_top{K}: mean={res[c].mean():.3f}  slope/yr={b:+.3f}  r={r:+.3f}')

# ---- plot ----
COLOR = {1: '#0072B2', 3: '#009E73', 5: '#E69F00'}
fig, ax = plt.subplots(figsize=(9, 5.2))
for K in K_LIST:
    ax.plot(res.index, res[f'frac_not_top{K}'], color=COLOR[K], lw=2.2,
             marker='o', ms=3.5, label=f'not in top {K} by Arena rating')
ax.set_ylim(0, 1)
ax.set_ylabel('Share of OpenRouter tokens NOT served by\nthe current top-K Arena-rated model(s)')
ax.set_title('Most LLM usage does not go to the top-rated model')
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, loc='lower left', fontsize=9)
ax.axhline(0.5, color='#888888', lw=0.8, ls='--', zorder=0)
fig.tight_layout()
fig.savefig('experiments/plot_openrouter_top_share.png', dpi=150, bbox_inches='tight')
print('\nwrote experiments/plot_openrouter_top_share.png')
