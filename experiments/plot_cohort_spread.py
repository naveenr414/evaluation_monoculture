"""Fix the set of competing firms and watch the spread among their flagships.

The falling leader-minus-rank-k spread could be composition: new firms and new
models keep arriving.  Here the cohort is frozen.  A firm is in the cohort if it
has a rated flagship in every month of the window, and at each snapshot it is
represented by its own current flagship, so firms keep improving but no firm
enters or leaves.

Spread measures across the cohort's flagships, per snapshot per category:
  sd, leader minus the median, leader minus the worst.

The all-firms series is drawn behind for contrast.
Writes cohort_spread.pdf.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, 'experiments')
from common import load_panel, flagships, CORE

INK, INK_2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BASELINE, SURFACE = '#c3c2b7', '#ffffff'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'

DC = {'coding': 'Coding', 'math': 'Math', 'hard_prompts': 'Hard Prompts',
      'instruction_following': 'Instruction Following',
      'creative_writing': 'Creative Writing', 'longer_query': 'Longer Query',
      'multi_turn': 'Multi-Turn', 'english': 'English'}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 10.5, 'text.color': INK,
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK, 'axes.labelsize': 10.5,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'axes.linewidth': 0.7, 'pdf.fonttype': 42,
})

START = '2024-06-01'
# the `organization` field goes blank for 50-61 tail models from 2026-04-20
# onward (56 distinct orgs -> 25, with no change in the model count), so every
# org-keyed series is truncated before that break
END = '2026-03-19'
d = load_panel(cats=CORE, firms_only=False, since=None)
d = d[(d.dt >= pd.Timestamp(START)) & (d.dt <= pd.Timestamp(END))]
flag = flagships(d)
rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')
dates = sorted(d.dt.unique())
months = sorted(set(pd.PeriodIndex(dates, freq='M')))

# cohort: firms with a flagship in every month of the window
have = {}
for t in dates:
    m = pd.Timestamp(t).to_period('M')
    for org in flag.xs(t, level='dt').index if t in flag.index.get_level_values('dt') else []:
        have.setdefault(org, set()).add(m)
PRESENT = sorted([o for o, ms in have.items() if len(ms) == len(months) and o])
# presence is not activity: dormant leaderboard entries (lmsys, openassistant,
# stability ai, ...) keep a rating forever without ever shipping, and fall
# behind mechanically.  Require at least MIN_RELEASES distinct flagships.
MIN_RELEASES = 3
nflag = {o: flag.xs(o, level='org').nunique()
         for o in PRESENT if o in flag.index.get_level_values('org')}
COHORT = sorted([o for o, k in nflag.items() if k >= MIN_RELEASES])
DORMANT = sorted([o for o, k in nflag.items() if k < MIN_RELEASES])
print(f'window {pd.Timestamp(dates[0]).date()} .. {pd.Timestamp(dates[-1]).date()}, '
      f'{len(months)} months')
print(f'present every month: {len(PRESENT)}')
print(f'cohort ({len(COHORT)} active, >= {MIN_RELEASES} distinct flagships): {COHORT}')
print(f'dropped as dormant ({len(DORMANT)}): {DORMANT}')


def spreads(t, orgs):
    out = {}
    for cat in CORE:
        vals = []
        for o in orgs:
            m = flag.get((t, o))
            if m is None:
                continue
            try:
                v = rat.loc[(t, m)][cat]
            except KeyError:
                continue
            if pd.notna(v):
                vals.append(float(v))
        if len(vals) < 5:
            continue
        v = np.array(vals)
        out[cat] = dict(sd=v.std(ddof=1), lead_med=v.max() - np.median(v),
                        lead_min=v.max() - v.min(), n=len(v))
    return out


rows = []
allorgs = sorted(set(flag.index.get_level_values('org')) - {''})
for t in dates:
    for lab, orgs in (('cohort', COHORT), ('all', allorgs)):
        for cat, s in spreads(t, orgs).items():
            rows.append(dict(dt=t, set=lab, cat=cat, **s))
S = pd.DataFrame(rows)
S['q'] = pd.PeriodIndex(S.dt, freq='Q')

print('\nquarterly means, pooled over the 8 categories')
piv = S.groupby(['set', 'q'])[['sd', 'lead_med', 'lead_min', 'n']].mean().round(1)
print(piv.unstack(0).to_string())

print('\ntrend per year')
for lab in ('cohort', 'all'):
    sub = S[S.set == lab]
    x = (sub.dt - sub.dt.min()).dt.days.values / 365.25
    line = []
    for c in ('sd', 'lead_med', 'lead_min'):
        b1 = np.polyfit(x, sub[c].values, 1)[0]
        r = np.corrcoef(x, sub[c].values)[0, 1]
        line.append(f'{c} {b1:+6.1f} (r {r:+.2f})')
    print(f'  {lab:<7} ' + '   '.join(line))

print('\ncohort only, trend in sd by category')
for cat in CORE:
    sub = S[(S.set == 'cohort') & (S.cat == cat)]
    x = (sub.dt - sub.dt.min()).dt.days.values / 365.25
    b1 = np.polyfit(x, sub.sd.values, 1)[0]
    r = np.corrcoef(x, sub.sd.values)[0, 1]
    print(f'  {DC[cat]:<22} {sub.sd.iloc[:20].mean():6.1f} -> '
          f'{sub.sd.iloc[-20:].mean():6.1f}   {b1:+6.1f}/yr  r {r:+.2f}')

fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3), sharex=True)
for ax, (col, lab) in zip(axes, (('sd', 'standard deviation'),
                                 ('lead_med', 'leader $-$ median'),
                                 ('lead_min', 'leader $-$ worst'))):
    for setlab, color, lw in (('all', MUTED, 1.2), ('cohort', BLUE, 1.8)):
        g = S[S.set == setlab].groupby('q')[col].mean()
        t = g.index.to_timestamp() + pd.Timedelta(days=45)
        ax.plot(t, g.values, color=color, lw=lw, marker='o', ms=3.4,
                label=f'{setlab} firms')
    ax.set_title(lab, fontsize=11, color=INK_2, pad=6)
    ax.set_ylabel('rating points')
    ax.set_ylim(0, None)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ('left', 'bottom'))
    ax.tick_params(length=3, width=0.6)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
axes[0].legend(frameon=False, fontsize=9, loc='upper right')
fig.tight_layout(pad=0.7, w_pad=1.8)
fig.savefig('cohort_spread.pdf')
plt.close(fig)
print('\nwrote cohort_spread.pdf')
