"""Does the gap still predict improvement once the model's LEVEL is controlled?

A weak outgoing model has large gaps in every category and also more room to
improve, so beta could be picking up level rather than the category-specific
gap.  We add the outgoing model's overall rating at t_old and control for it
three ways: linearly, as decile x category fixed effects, and by estimating
beta separately inside each level quartile.

Specification is otherwise the draft's: firm x category and category x quarter
fixed effects, standard errors clustered on the release event.
Writes beta_by_level.pdf.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import load_panel, build_events, fe_regress, CORE

INK, INK_2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BASELINE, SURFACE, BLUE = '#c3c2b7', '#ffffff', '#2a78d6'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 11, 'text.color': INK,
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK, 'axes.labelsize': 11.5,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'axes.linewidth': 0.7, 'pdf.fonttype': 42,
})

d = load_panel(cats=CORE)
ev = build_events(d, cats=CORE)
ev['q'] = pd.PeriodIndex(ev.switch, freq='Q').astype(str)
ev['ik'] = ev.firm + '|' + ev.cat
ev['kt'] = ev.cat + '|' + ev.q

# level of the outgoing model: its 'overall' rating at the pre-release snapshot
ov = (d[d.category == 'overall'].dropna(subset=['rating'])
        .set_index(['dt', 'model_name']).rating)
ev['old_model'] = ev.event.str.split(':').str[1].str.split('->').str[0]
ev['t_old'] = ev.groupby('event').switch.transform('min')
snaps = np.sort(d.dt.unique())


def level_of(row):
    prior = [s for s in snaps if s < row.switch]
    for t in reversed(prior):
        try:
            return ov.loc[(t, row.old_model)]
        except KeyError:
            continue
    return np.nan


lvl = ev.drop_duplicates('event').apply(level_of, axis=1)
ev = ev.merge(pd.DataFrame({'event': ev.drop_duplicates('event').event.values,
                            'level': lvl.values}), on='event', how='left')
ev = ev.dropna(subset=['level'])
print(f'events={ev.event.nunique()}  obs={len(ev)}  '
      f'level range {ev.level.min():.0f}-{ev.level.max():.0f}')

# level bins defined on EVENTS, not observations, so every category of an event
# falls in the same bin
evl = ev.drop_duplicates('event')[['event', 'level']]
for nb, name in ((4, 'q4'), (10, 'd10')):
    edges = np.quantile(evl.level, np.linspace(0, 1, nb + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    evl[name] = np.digitize(evl.level, edges[1:-1])
ev = ev.merge(evl[['event', 'q4', 'd10']], on='event', how='left')
ev['dk'] = ev.d10.astype(str) + '|' + ev.cat


def fit(df, extra=(), groups=('ik', 'kt')):
    cols = ['gap'] + list(extra)
    y, X = df.dtheta.values, df[cols].values.astype(float)
    gs = [df[g].values for g in groups]
    n = len(df)
    dfac = (n - (sum(df[g].nunique() for g in groups) - len(groups) + 1)) / n
    nc = np.zeros(len(cols)); nc[0] = df.v_old.mean() * dfac
    nv = np.zeros((len(cols), len(cols)))
    nv[0, 0] = (df.v_old.mean() + df.v_front.mean()) * dfac
    beta, se, bc, resid, _, _ = fe_regress(y, X, gs, noise_cov=nc, noise_var=nv,
                                        cluster=df.event.values)
    return dict(beta=beta[0], se=se[0], t=beta[0] / se[0], corr=bc[0],
                n=n, events=df.event.nunique())


print('\nCONTROLLING FOR LEVEL')
rows = [dict(spec='(A) baseline, firm x cat + cat x quarter FE', **fit(ev)),
        dict(spec='(B) + level, linear', **fit(ev, extra=['level'])),
        dict(spec='(C) level decile x category FE',
             **fit(ev, groups=('ik', 'kt', 'dk')))]
print(pd.DataFrame(rows).to_string(index=False,
                                   float_format=lambda x: f'{x:7.3f}'))

print('\nBETA WITHIN EACH LEVEL QUARTILE (1 = weakest outgoing model)')
qrows = []
for b in sorted(ev.q4.unique()):
    sub = ev[ev.q4 == b]
    r = fit(sub)
    qrows.append(dict(quartile=b + 1,
                      level_lo=sub.level.min(), level_hi=sub.level.max(),
                      mean_gap=sub.gap.mean(), mean_dtheta=sub.dtheta.mean(), **r))
qt = pd.DataFrame(qrows)
print(qt.to_string(index=False, float_format=lambda x: f'{x:7.3f}'))

fig, ax = plt.subplots(figsize=(5.2, 3.3))
x = qt.quartile.values
ax.axhline(0, color=BASELINE, lw=0.8)
b0 = rows[0]['beta']
ax.axhline(b0, color=MUTED, lw=1.0, ls=(0, (3, 2)))
ax.text(4.35, b0, 'pooled', fontsize=9, color=MUTED, va='center')
ax.errorbar(x, qt.beta, yerr=1.96 * qt.se, fmt='o', color=BLUE, ms=5,
            lw=1.3, capsize=3)
ax.set_xticks(x)
ax.set_xticklabels([f'Q{i}\n{lo:.0f}-{hi:.0f}'
                    for i, lo, hi in zip(x, qt.level_lo, qt.level_hi)])
ax.set_xlabel('quartile of outgoing model overall rating')
ax.set_ylabel(r'$\hat{\beta}$ (gap)')
ax.set_xlim(0.5, 4.8)
ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
for side, spine in ax.spines.items():
    spine.set_visible(side in ('left', 'bottom'))
ax.tick_params(length=3, width=0.6)
fig.tight_layout(pad=0.6)
fig.savefig('beta_by_level.pdf')
print('\nwrote beta_by_level.pdf')
