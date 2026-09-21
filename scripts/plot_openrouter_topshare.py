"""What fraction of REAL OpenRouter usage actually goes to near-top models,
as a function of how strict "near-top" is?

  1. openrouter_share_by_threshold.pdf   x = Arena threshold (within x rating
                                         points of the top),
                                          y = avg. share of OpenRouter token
                                          volume served by models within x% of
                                          that month's top 'overall' rating
  2. near_top_single.pdf                 the single "avg # models within x% of
                                          top" series (from plot_rank_dynamics.py),
                                          at the x picked as the elbow/inflection
                                          of plot 1 -- full-size, not small multiples

Usage is matched at the MODEL level (OpenRouter has no per-category breakdown),
so thresholds here are against the 'overall' Arena rating, using the same
or_arena_match mapping built for Experiment H. Unmatched OpenRouter volume
(~16% of tokens; stealth codenames, embeddings, etc. -- see or_arena_match.py)
is counted in the denominator but never in the numerator, i.e. treated as NOT
in the top-x% -- a conservative floor on the true share.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import load_panel, CORE, OPENROUTER
import or_arena_match

INK, INK_2, MUTED, GRID, BASELINE, SURFACE = \
    '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ffffff'
BLUE, ORANGE = '#2a78d6', '#eb6834'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 15, 'text.color': INK,
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK, 'axes.labelsize': 17,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelsize': 17, 'ytick.labelsize': 17,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.linewidth': 1.0, 'pdf.fonttype': 42,
})


def strip_axes(ax, keep=('left', 'bottom')):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
    ax.tick_params(length=4, width=0.7)


# ---------------------------------------------------------------- data -----
d = load_panel(cats=CORE, firms_only=False, since='2025-01-01')
o = pd.read_parquet(OPENROUTER)
o['date'] = pd.to_datetime(o.date)

arena_models = sorted(d.model_name.unique())
or_models_all = o.model_permaslug.unique()
match_result = or_arena_match.match(or_models_all, arena_models)
matched = {slug: name for slug, (name, method) in match_result.items()}
print(f'matched {len(matched)}/{len(or_models_all)} OpenRouter models to Arena names')

# Rating points below the top, not percentages: a Bradley-Terry rating has an
# arbitrary origin, so only differences on it mean anything. See crowding.py.
X_VALS = [10, 25, 50, 100, 150, 200, 300]

rows = []
for per, g in o.groupby(o.date.dt.to_period('M')):
    tm = g.groupby('model_permaslug').total_tokens.sum()
    total = tm.sum()
    if total <= 0:
        continue
    snapd = d[d.dt <= per.to_timestamp('M')]
    if snapd.empty:
        continue
    last = snapd.dt.max()
    ov = d[(d.dt == last) & (d.category == 'overall')].dropna(subset=['rating'])
    if len(ov) < 5:
        continue
    top = ov.rating.max()
    rating_of = ov.set_index('model_name').rating

    tm_matched = tm[[s in matched for s in tm.index]]
    arena_names = tm_matched.index.map(lambda s: matched[s])
    ratings = arena_names.map(lambda n: rating_of.get(n, np.nan))

    row = {'month': per.to_timestamp()}
    for x in X_VALS:
        thresh = top - x
        row[x] = tm_matched[ratings >= thresh].sum() / total
    rows.append(row)

share = pd.DataFrame(rows).set_index('month').sort_index()
avg_share = share.mean()
print('\navg. OpenRouter token share served by models within x% of top:')
print(avg_share.round(3).to_string())

# the curve can never exceed the avg. matched-token coverage: unmatched slugs
# (stealth codenames, embeddings, brand-new releases not yet in or_arena_match)
# have no Arena rating at all, so they can never satisfy the threshold no
# matter how permissive x is -- this is a coverage ceiling, not a claim that
# ~20% of usage genuinely stays outside the top half of the leaderboard.
match_cov = []
for per, g in o.groupby(o.date.dt.to_period('M')):
    tm = g.groupby('model_permaslug').total_tokens.sum()
    match_cov.append(tm[[s in matched for s in tm.index]].sum() / tm.sum())
avg_match_cov = np.mean(match_cov)
print(f'\navg. matched-token coverage (the ceiling this curve saturates at): '
      f'{avg_match_cov:.3f}')

# ---- pick the elbow/inflection of the threshold curve (max distance from the
# chord connecting the first and last points -- the standard elbow heuristic) ----
xs = np.array(X_VALS, dtype=float)
ys = avg_share.values
p0, p1 = np.array([xs[0], ys[0]]), np.array([xs[-1], ys[-1]])
line_vec = p1 - p0
line_vec_norm = line_vec / np.linalg.norm(line_vec)
dists = []
for xi, yi in zip(xs, ys):
    pt = np.array([xi, yi]) - p0
    proj = (pt @ line_vec_norm) * line_vec_norm
    dists.append(np.linalg.norm(pt - proj))
elbow_x = int(xs[np.argmax(dists)])
print(f'\nelbow/inflection threshold: x={elbow_x} points')

# ============================================================= FIGURE 1 ====
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.plot(xs, ys * 100, color=BLUE, lw=2.6, marker='o', markersize=8,
        markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
ax.fill_between(xs, ys * 100, color=BLUE, alpha=0.08, zorder=1)
ax.axvline(elbow_x, color=ORANGE, lw=1.6, ls=(0, (4, 2)), zorder=2)
ax.axhline(avg_match_cov * 100, color=MUTED, lw=1.3, ls=(0, (2, 2)), zorder=2)
ax.set_xlabel('Arena threshold $x$ (within $x$ points of top)')
ax.set_ylabel('avg. OpenRouter share (%)')
ax.grid(color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
strip_axes(ax)
fig.tight_layout(pad=0.8)
fig.savefig('openrouter_share_by_threshold.pdf')
plt.close(fig)

# ============================================================= FIGURE 2 ====
dates = sorted(d.dt.unique())
rows2 = []
for t_ in dates:
    snap = d[d.dt == t_]
    counts = []
    for cat in CORE:
        sub = snap[(snap.category == cat)].dropna(subset=['rating'])
        if len(sub) < 5:
            continue
        top = sub.rating.max()
        thresh = top - elbow_x
        counts.append(int((sub.rating >= thresh).sum()))
    if not counts:
        continue
    rows2.append(dict(dt=t_, n=np.mean(counts)))
nt = pd.DataFrame(rows2).set_index('dt').sort_index()
nt_monthly = nt.n.resample('MS').mean().dropna()

tt = (nt_monthly.index - nt_monthly.index[0]).days.values / 365.25
b, r = np.polyfit(tt, nt_monthly.values, 1)[0], np.corrcoef(tt, nt_monthly.values)[0, 1]

fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.plot(nt_monthly.index, nt_monthly.values, color=ORANGE, lw=2.6, marker='o',
        markersize=7, markerfacecolor=ORANGE, markeredgecolor=SURFACE,
        markeredgewidth=1.0, zorder=3)
ax.fill_between(nt_monthly.index, nt_monthly.values, color=ORANGE, alpha=0.08, zorder=1)
ax.set_ylim(0, None)
ax.set_ylabel(f'avg. # of models\nwithin {elbow_x} points of top')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
strip_axes(ax)
fig.tight_layout(pad=0.8)
fig.savefig('near_top_single.pdf', bbox_inches='tight')
plt.close(fig)
print(f'\nwrote openrouter_share_by_threshold.pdf, near_top_single.pdf '
      f'(elbow x={elbow_x} points)')
