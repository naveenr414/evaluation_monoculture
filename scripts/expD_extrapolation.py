"""Fit simple trend models to the headline series and extrapolate them forward.

Six series, each monthly (or quarterly where the underlying event count is thin),
pooled over the 8 core categories where relevant:

  A  frontier rating          mean over categories of the top rating
  B  models within 10% of top mean over categories
  C  leader minus 10th best   mean over categories
  D  effective firms          OpenRouter monthly token shares
  E  months behind frontier   new frontier-firm releases, quarterly
  F  entry gap in points      new frontier-firm releases, quarterly

Three candidate forms are fitted to each: linear in time, linear in log time, and
a saturating exponential a - b exp(-c t).  The form is chosen by BACKTEST: fit on
everything up to the last BACKTEST_MONTHS and score on the tail.  The winner is
then refit on the full series and extrapolated, with a band of +/-1.96 times the
residual standard deviation.

Extrapolation is a curve-fitting exercise, not a forecast with a model of firm
behaviour behind it.  The backtest column is the only evidence that any of these
forms carries out of sample at all.

Writes extrapolation_curves.pdf.
"""
import sys
import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import load_panel, CORE, FRONTIER, OPENROUTER

warnings.filterwarnings('ignore')

INK, INK_2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BASELINE, SURFACE = '#c3c2b7', '#ffffff'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 10.5, 'text.color': INK,
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK, 'axes.labelsize': 10.5,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'axes.linewidth': 0.7, 'pdf.fonttype': 42,
})

BACKTEST = 5          # last k points held out to choose the functional form
HORIZON_YEARS = 1.5


def f_lin(t, a, b):
    return a + b * t


def f_log(t, a, b):
    return a + b * np.log1p(np.maximum(t, 0))


def f_sat(t, a, b, c):
    return a - b * np.exp(-np.clip(c, 1e-4, 5) * t)


FORMS = [('linear', f_lin, [0.0, 0.0]),
         ('log-time', f_log, [0.0, 0.0]),
         ('saturating', f_sat, None)]


def fit_one(t, y, form, p0):
    if p0 is None:
        p0 = [y[-1] + (y[-1] - y[0]), max(abs(y[-1] - y[0]), 1e-3), 0.5]
    return curve_fit(form, t, y, p0=p0, maxfev=40000)[0]


def choose(t, y):
    """Pick the form by backtest, then refit on everything."""
    n = len(t)
    k = min(BACKTEST, max(2, n // 4))
    out = []
    for name, form, p0 in FORMS:
        try:
            p = fit_one(t[:-k], y[:-k], form, p0)
            rmse = float(np.sqrt(np.mean((y[-k:] - form(t[-k:], *p)) ** 2)))
        except Exception:
            rmse = np.inf
        out.append((name, form, p0, rmse))
    out.sort(key=lambda r: r[3])
    name, form, p0, rmse = out[0]
    p = fit_one(t, y, form, p0)
    resid = y - form(t, *p)
    return dict(name=name, form=form, params=p, backtest_rmse=rmse,
                sd=float(resid.std(ddof=1)),
                all_rmse={r[0]: r[3] for r in out})


# ------------------------------------------------------------- series ------
d = load_panel(cats=CORE, firms_only=False, since=None)
d = d[d.category.isin(CORE)].dropna(subset=['rating'])
d = d[d.dt >= pd.Timestamp('2024-01-01')]

rows = []
for (t_, cat), g in d.groupby(['dt', 'category']):
    v = np.sort(g.rating.values)[::-1]
    if len(v) < 20:
        continue
    rows.append(dict(dt=t_, cat=cat, top=v[0],
                     near10=(v >= v[0] * 0.90).sum(),
                     sp10=v[0] - v[9]))
P = pd.DataFrame(rows)
P['m'] = P.dt.dt.to_period('M').dt.to_timestamp()
M = P.groupby('m')[['top', 'near10', 'sp10']].mean()

o = pd.read_parquet(OPENROUTER)
o['date'] = pd.to_datetime(o.date)
o['org'] = o.model_permaslug.str.split('/').str[0].str.lower()
effs = []
for per, g in o.groupby(o.date.dt.to_period('M')):
    s = g.groupby('org').total_tokens.sum()
    s = s / s.sum()
    effs.append((per.to_timestamp(), 1 / (s ** 2).sum()))
EFF = pd.Series(dict(effs)).sort_index()

# entrant series, quarterly
rows = []
for cat, g in d.groupby('category'):
    piv = g.pivot_table(index='dt', columns='model_name', values='rating').sort_index()
    front = piv.max(axis=1).cummax()
    ft, fv = front.index.values, front.values
    seen = piv.notna().cumsum() > 0
    fst = piv.notna().idxmax().where(piv.notna().any())
    orgs = g.groupby('model_name').org.first()
    for m_ in piv.columns:
        t0 = fst[m_]
        if pd.isna(t0) or orgs.get(m_) not in FRONTIER:
            continue
        row = piv.loc[t0]
        inc = row[[c for c in piv.columns
                   if c != m_ and seen.loc[t0, c] and pd.notna(row[c])]]
        if len(inc) < 20 or pd.isna(row[m_]):
            continue
        upto = np.searchsorted(ft, np.datetime64(t0), side='right')
        j = np.searchsorted(fv[:upto], row[m_], side='right') - 1
        if j < 0:
            continue
        rows.append(dict(t0=t0, months=(pd.Timestamp(t0) - pd.Timestamp(ft[j])).days / 30.44,
                         gap=inc.max() - row[m_]))
EN = pd.DataFrame(rows)
EN['q'] = pd.PeriodIndex(EN.t0, freq='Q')
ENq = EN.groupby('q')[['months', 'gap']].mean()
ENq.index = ENq.index.to_timestamp() + pd.Timedelta(days=45)

SERIES = [
    ('Frontier rating', M.top, 'rating points'),
    ('Models within 10% of top', M.near10, 'models'),
    ('Leader $-$ 10th best', M.sp10, 'rating points'),
    ('Effective firms (OpenRouter)', EFF, 'firms'),
    ('Months behind frontier', ENq.months, 'months'),
    ('Entry gap', ENq.gap, 'rating points'),
]

fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.2))
print(f'{"series":<30} {"form":<11} {"backtest RMSE":>14} {"resid sd":>9} '
      f'{"now":>8} {"+18mo":>9}')
for ax, (lab, s, ylab) in zip(axes.ravel(), SERIES):
    s = s.dropna()
    t0 = s.index[0]
    t = np.array([(x - t0).days for x in s.index], float) / 365.25
    y = s.values.astype(float)
    fit = choose(t, y)
    tf = np.linspace(0, t[-1] + HORIZON_YEARS, 200)
    yf = fit['form'](tf, *fit['params'])
    dates = [t0 + pd.Timedelta(days=365.25 * x) for x in tf]
    now = fit['form'](t[-1], *fit['params'])
    fut = fit['form'](t[-1] + HORIZON_YEARS, *fit['params'])
    print(f'{lab:<30} {fit["name"]:<11} {fit["backtest_rmse"]:>14.3f} '
          f'{fit["sd"]:>9.3f} {now:>8.1f} {fut:>9.1f}')
    ax.fill_between(dates, yf - 1.96 * fit['sd'], yf + 1.96 * fit['sd'],
                    color=BLUE, alpha=0.12, lw=0)
    ax.plot(dates, yf, color=BLUE, lw=1.6)
    ax.plot(s.index, y, color=INK, lw=0, marker='o', ms=3.2, alpha=0.7)
    ax.axvline(s.index[-1], color=BASELINE, lw=0.9, ls=(0, (3, 2)))
    ax.set_title(f'{lab}  ({fit["name"]})', fontsize=10, color=INK_2, pad=5)
    ax.set_ylabel(ylab)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ('left', 'bottom'))
    ax.tick_params(length=3, width=0.6)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%y'))

print('\nbacktest RMSE by form (lower is better; the chosen form is starred)')
for lab, s, _ in SERIES:
    s = s.dropna()
    t0 = s.index[0]
    t = np.array([(x - t0).days for x in s.index], float) / 365.25
    fit = choose(t, s.values.astype(float))
    parts = [f'{k}{"*" if k == fit["name"] else ""}={v:.3f}'
             for k, v in fit['all_rmse'].items()]
    print(f'  {lab:<30} ' + '  '.join(parts))

fig.tight_layout(pad=0.7, w_pad=1.6, h_pad=1.4)
fig.savefig('extrapolation_curves.pdf')
plt.close(fig)
print('\nwrote extrapolation_curves.pdf')
