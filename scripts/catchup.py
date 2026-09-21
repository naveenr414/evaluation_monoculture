"""Catch-up dynamics: does a firm improve most where it trails rivals?

Every flagship changeover is an event. For each event and category we measure
how much the new model gained, dtheta_{i,k}, and how far the outgoing model
trailed the best rival in that category, g_{i,k}. Regressing one on the other,

    dtheta_{i,k} = beta * g_{i,k} + (firm x category) + (category x quarter) + e

gives beta, the share of its per-category deficit a firm closes per release.
beta > 0 is eval-chasing: firms spend effort where they are behind, not where
they hold an advantage.

Two things the estimate has to survive. Both dtheta and g contain the outgoing
model's rating with opposite signs, so measurement error alone forces beta up;
`fit_beta` subtracts that spurious covariance using the leaderboard's own
reported variances, and reports the corrected value alongside the naive one.
And a weak model has large gaps everywhere AND more room to improve, so beta
could be level rather than gap -- `beta_by_level` controls for that.
"""
import numpy as np
import pandas as pd

from common import (BATTLES, CORE, FRONTIER, MATURE_DAYS, MIN_EVENTS,
                    fe_regress, flagships, load_panel, within)

BENCHMARKS = {
    'gap_best': ('v_best', 'best rival'),
    'gap_2nd': ('v_2nd', 'second-best rival'),
    'gap_top3': ('v_top3', 'top-3 average'),
}

# How each regressor loads on the outgoing model's rating, whose sampling error
# is shared with dtheta. A gap is (rival - theta_old), so it loads -1; the level
# theta_old itself loads +1; anything else here is treated as measured without
# error, which is what `fit_beta`'s `extra` argument used to assume throughout.
THETA_LOADING = {'theta': 1.0, **{g: -1.0 for g in BENCHMARKS}}


def build_event_panel(d, cats=CORE, mature_days=MATURE_DAYS, min_rivals=4):
    """One row per (flagship changeover, category)

    `gap_best` reproduces `common.build_events`' `gap`; the other benchmarks
    are added here because whether beta depends on which rival defines the
    frontier is itself a robustness check.

    Arguments:
        d: Panel DataFrame from common.load_panel
        cats: Categories defining the profile
        mature_days: Days to wait before reading the new model's rating
        min_rivals: Minimum rival firms required at the pre-release snapshot

    Returns: DataFrame with dtheta, level, gap_* and v_* per benchmark, plus
        the event / firm / cat keys and the ik, kt fixed-effect labels"""
    rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')
    var = d.pivot_table(index=['dt', 'model_name'], columns='category', values='variance')
    overall = d[d.category == 'overall'].dropna(subset=['rating'])
    overall = overall.set_index(['dt', 'model_name']).rating
    flag = flagships(d)
    snaps = np.sort(d.dt.unique())

    changeovers = []
    for firm in FRONTIER:
        if firm not in flag.index.get_level_values('org'):
            continue
        series = flag.xs(firm, level='org').sort_index()
        prev_model, prev_t = None, None
        for t, model in series.items():
            if prev_model is not None and model != prev_model:
                changeovers.append((firm, prev_model, model, t, prev_t))
            prev_model, prev_t = model, t

    rows = []
    for firm, old, new, switch, t_old in changeovers:
        later = [s for s in snaps if s >= switch + np.timedelta64(mature_days, 'D')]
        if not later:
            continue
        t_new = later[0]
        try:
            th_old, th_new = rat.loc[(t_old, old)][cats], rat.loc[(t_new, new)][cats]
            v_old = var.loc[(t_old, old)][cats]
        except KeyError:
            continue
        if th_old.isna().any() or th_new.isna().any():
            continue

        snap = rat.xs(t_old, level='dt')
        rivals = [flag.get((t_old, f)) for f in FRONTIER if f != firm]
        rivals = [m for m in rivals if m is not None and m in snap.index]
        riv = snap.loc[rivals][cats].dropna()
        if len(riv) < min_rivals:
            continue

        event = f'{firm}:{old}->{new}@{pd.Timestamp(switch).strftime("%Y%m%d")}'
        level = overall.get((t_old, old), np.nan)
        for k in cats:
            ranked = riv[k].sort_values(ascending=False)
            if len(ranked) < 3:
                continue
            top3 = ranked.index[:3]
            rows.append({
                'event': event, 'firm': firm, 'cat': k, 'switch': switch,
                'dtheta': th_new[k] - th_old[k], 'level': level,
                'theta': th_old[k],
                'gap_best': ranked.iloc[0] - th_old[k],
                'gap_2nd': ranked.iloc[1] - th_old[k],
                'gap_top3': ranked.iloc[:3].mean() - th_old[k],
                'v_old': v_old[k],
                'v_best': var.loc[(t_old, ranked.index[0])][k],
                'v_2nd': var.loc[(t_old, ranked.index[1])][k],
                'v_top3': np.mean([var.loc[(t_old, m)][k] for m in top3]) / 3.0,
            })

    ev = pd.DataFrame(rows)
    ev['year'] = pd.to_datetime(ev.switch).dt.year
    ev['q'] = pd.PeriodIndex(ev.switch, freq='Q').astype(str)
    ev['ik'] = ev.firm + '|' + ev.cat
    ev['kt'] = ev.cat + '|' + ev.q
    return ev


def fit_beta(ev, gap='gap_best', extra=(), fe=('ik', 'kt'), min_events=6,
             min_resid_df=10):
    """Regress dtheta on the gap with fixed effects and a noise correction

    The correction subtracts the covariance that measurement error alone
    induces: theta_old enters dtheta negatively and the gap positively, so its
    sampling variance inflates beta mechanically. Any regressor in
    `THETA_LOADING` -- including the level theta_old, which carries the same
    error with the opposite sign -- is corrected jointly with the gap; every
    other `extra` is taken as measured without error.

    `fe` matters for subsamples. The default two-way spec absorbs so many
    degrees of freedom that a single firm's 7 events leave none, and the fit
    comes back with se = 0 and a meaningless beta; callers splitting the panel
    should pass a lighter spec, and the `min_resid_df` guard rejects the fit
    rather than returning a degenerate one.

    Arguments:
        ev: Event panel from `build_event_panel`
        gap: Which gap column to use as the regressor
        extra: Additional regressor columns, uncorrected
        fe: Column names to absorb as fixed effects
        min_events: Skip the fit below this many distinct events
        min_resid_df: Skip the fit below this many residual degrees of freedom

    Returns: Dict with beta, se, t, corrected, n, events, firms, and the same
        four fields prefixed `d_` for the second regressor when `extra` names
        one -- or None if the subsample is too small, too saturated, or
        singular"""
    if ev.event.nunique() < min_events:
        return None

    cols = [gap, *extra]
    y = ev.dtheta.to_numpy(float)
    X = ev[cols].to_numpy(float)
    groups = [ev[g].to_numpy() for g in fe]

    n = len(ev)
    absorbed = sum(ev[g].nunique() for g in fe) - (len(fe) - 1)
    if n - absorbed - len(cols) < min_resid_df:
        return None
    v_old = ev.v_old.mean()
    load = np.array([THETA_LOADING.get(c, 0.0) for c in cols])
    indep = np.array([ev[BENCHMARKS[c][0]].mean() if c in BENCHMARKS else 0.0
                      for c in cols])
    # dtheta loads -1 on theta_old, so cov(err_x, err_y) = -load * v_old.
    # These are the raw per-observation noise moments; `fe_regress` applies the
    # degrees-of-freedom factor that the within transform costs them. Scaling
    # by it here as well applied it twice and left the correction at ~80%
    # strength.
    noise_cov = -load * v_old
    noise_var = v_old * np.outer(load, load) + np.diag(indep)

    try:
        beta, se, corrected, _, _, se_c = fe_regress(
            y, X, groups, noise_cov=noise_cov, noise_var=noise_var,
            cluster=ev.event.to_numpy())
    except np.linalg.LinAlgError:
        return None

    out = {'beta': beta[0], 'se': se[0], 't': beta[0] / se[0],
           'corrected': corrected[0], 'se_corrected': se_c[0],
           't_corrected': corrected[0] / se_c[0], 'n': n,
           'events': ev.event.nunique(), 'firms': ev.firm.nunique()}
    if len(cols) > 1:
        out.update({'delta': beta[1], 'se_delta': se[1],
                    't_delta': beta[1] / se[1], 'delta_corrected': corrected[1],
                    'se_delta_corrected': se_c[1],
                    't_delta_corrected': corrected[1] / se_c[1]})
    return out


def beta_per_firm(ev, gap='gap_best', min_events=MIN_EVENTS):
    """Beta for each firm separately, with event and category fixed effects

    Within one firm the pooled spec is saturated, so this absorbs event and
    category instead: the comparison that identifies beta is "which category
    lagged more within this release", which those two effects isolate.

    Arguments:
        ev: Event panel from `build_event_panel`
        gap: Which gap column to use
        min_events: Below this the event-clustered SE is degenerate;
            defaults to MIN_EVENTS, the same cut that builds the roster

    Returns: Dict keyed by firm, each with beta, se, corrected, and the
        residualized gap and dtheta the scatter plots"""
    out = {}
    for firm in FRONTIER:
        sub = ev[ev.firm == firm]
        if sub.event.nunique() < min_events:
            continue
        fit = fit_beta(sub, gap=gap, fe=('event', 'cat'), min_events=min_events)
        if fit is None:
            continue
        out[firm] = {**fit,
                     'gap': within(sub[gap].to_numpy(float), sub.event, sub.cat),
                     'dtheta': within(sub.dtheta.to_numpy(float), sub.event, sub.cat)}
    return out


def beta_by(ev, column, gap='gap_best', **kwargs):
    """Estimate beta separately within each value of a column

    Splitting by firm makes `ik` collinear with the category and leaves the
    category x quarter effects saturated, so those splits want fe=('ik',).

    Arguments:
        ev: Event panel from `build_event_panel`
        column: Column to split on, e.g. 'firm' or 'year'
        gap: Which gap column to use
        kwargs: Passed to `fit_beta`

    Returns: DataFrame indexed by the column's values, one row per subsample
        that was estimable, with a `dropped` attribute listing the rest"""
    rows, dropped = {}, []
    for value, group in ev.groupby(column):
        fit = fit_beta(group, gap=gap, **kwargs)
        if fit is None:
            dropped.append((value, group.event.nunique()))
        else:
            rows[value] = fit
    out = pd.DataFrame(rows).T
    out.attrs['dropped'] = dropped
    return out


def beta_by_level(ev, n_bins=4):
    """Does the gap still predict improvement once the model's level is fixed?

    Four controls, weakest to strongest: none, the outgoing model's overall
    rating as a linear covariate, its per-category rating theta_{ikt} as a
    linear covariate, and beta estimated freely inside each quantile bin of the
    overall rating.

    The theta_{ikt} row is the demanding one, because the gap is built from
    theta_{ikt} by subtraction: within a (category, quarter) cell the two are
    separated only by how far the rival frontier moves across events, so read
    it together with `level_collinearity`.

    Arguments:
        ev: Event panel from `build_event_panel`
        n_bins: Number of level bins to split into

    Returns: (overall, by_bin) -- a DataFrame of the pooled specifications and
        a DataFrame of one row per level bin"""
    ev = ev.dropna(subset=['level']).copy()

    pooled = pd.DataFrame({
        'no control': fit_beta(ev),
        'linear in level': fit_beta(ev, extra=['level']),
        'linear in theta_ikt': fit_beta(ev, extra=['theta']),
    }).T

    ev['bin'] = pd.qcut(ev.level, n_bins, labels=False, duplicates='drop')
    by_bin = beta_by(ev, 'bin', fe=('ik',))
    by_bin['level_mid'] = ev.groupby('bin')['level'].median()
    return pooled, by_bin


def level_collinearity(ev, gap='gap_best', fe=('ik', 'kt')):
    """How much independent variation the gap and the level still have

    The gap is the rival frontier minus theta_{ikt}, so after the fixed effects
    the two regressors are close to a sign flip of each other and beta and
    delta are identified only by movement in the frontier. This reports what is
    left: their correlation inside the fixed effects, the variance inflation
    factor they impose on each other, and the share of the gap's within
    variance that the frontier rather than theta_{ikt} supplies.

    Arguments:
        ev: Event panel from `build_event_panel`
        gap: Which gap column to use
        fe: Fixed effects to absorb before measuring the correlation

    Returns: Dict with corr, vif, frontier_share and the two within variances"""
    groups = [ev[g].to_numpy() for g in fe]
    g = within(ev[gap].to_numpy(float), *groups)
    th = within(ev.theta.to_numpy(float), *groups)
    front = within((ev[gap] + ev.theta).to_numpy(float), *groups)
    r = float(np.corrcoef(g, th)[0, 1])
    return {'corr': r, 'vif': 1.0 / (1.0 - r ** 2),
            'frontier_share': float(front.var() / g.var()),
            'var_gap': float(g.var()), 'var_theta': float(th.var())}


def robustness_table(ev):
    """Beta under every robustness cut, as one table

    Arguments:
        ev: Event panel from `build_event_panel`

    Returns: DataFrame with a `split` label and a `group` naming which
        robustness exercise each row belongs to"""
    blocks = []

    def add(group, label, fit):
        if fit is not None:
            blocks.append({'group': group, 'split': label, **fit})

    add('baseline', 'all events', fit_beta(ev))

    for gap, (_, label) in BENCHMARKS.items():
        add('benchmark', label, fit_beta(ev, gap=gap))

    for year, group in ev.groupby('year'):
        add('year', str(year), fit_beta(group, fe=('ik',)))

    cut = ev.drop_duplicates('event').switch.median()
    add('half', f'early (<= {pd.Timestamp(cut).date()})',
        fit_beta(ev[ev.switch <= cut], fe=('ik',)))
    add('half', 'late', fit_beta(ev[ev.switch > cut], fe=('ik',)))

    odd = sorted(ev.event.unique())[::2]
    add('sample split', 'events A', fit_beta(ev[ev.event.isin(odd)], fe=('ik',)))
    add('sample split', 'events B', fit_beta(ev[~ev.event.isin(odd)], fe=('ik',)))

    # Strategic candidacy: a release effect absorbs anything that made the whole
    # release good or bad at once -- a firm shipping only when its model is
    # strong across the board, or timing a release to a good quarter. Beta is
    # then identified purely within a release, across categories. This is the
    # last rung of SPECS, repeated here so the confounds table carries it.
    add('release', 'event fixed effect', fit_beta(ev, fe=('event', 'ik', 'kt')))

    return pd.DataFrame(blocks)


SPECS = [
    ('pooled', ()),
    ('additive', ('firm', 'cat', 'q')),
    ('firm x cat', ('ik', 'q')),
    ('firm x cat + cat x time', ('ik', 'kt')),
    ('+ release', ('event', 'ik', 'kt')),
]


def fe_ladder(ev, gap='gap_best'):
    """Beta under progressively more demanding fixed effects

    The ladder answers a different question from `robustness_table`: not
    whether beta survives cutting the sample, but whether it survives taking
    away the comparisons it is allowed to use. The fourth rung is the spec
    every other routine here uses.

    The last rung adds a release effect, delta_e, which is the answer to
    strategic candidacy: a firm that ships only when its model is good all
    round, or that times a release to a quarter when it happens to have gained
    everywhere, moves dtheta for every category at once, and delta_e absorbs
    exactly that. What is left is the comparison across categories inside one
    release -- did this model gain more in the categories this firm was behind
    in -- which is the same comparison `beta_per_firm` already runs within a
    firm.

    Arguments:
        ev: Event panel from `build_event_panel`
        gap: Which gap column to use as the regressor

    Returns: DataFrame with one row per specification, carrying beta, se,
        corrected, se_corrected and the fixed effects that were absorbed"""
    ev = ev.copy()
    ev['const'] = 0  # a single group demeans the column, i.e. an intercept

    rows = []
    for label, fe in SPECS:
        fit = fit_beta(ev, gap=gap, fe=fe or ('const',))
        if fit is None:
            continue
        rows.append({'spec': label, 'fe': fe, **fit})
    return pd.DataFrame(rows)


def split_half_bias(n_splits=20, min_battles=60, seed=0):
    """Measure the regression-to-the-mean in beta instead of assuming it

    dtheta and the gap share theta_old, so noise in theta_old alone induces a
    positive slope. Splitting the battles in two and fitting Bradley-Terry on
    each half isolates that slope: regress (theta_B - theta_A), which is pure
    noise, on the gap built from half A. Whatever comes back is the spurious
    component, and it is compared with what the errors-in-variables correction
    predicts, var(noise) / var(gap).

    The battle dump covers 2025-04-17 to 2025-07-24 and tags only three
    categories, so this sizes the bias rather than re-running the event panel.

    Arguments:
        n_splits: Random 50/50 splits to average over
        min_battles: Minimum battles for a model to enter a category fit
        seed: Seed for the split

    Returns: Dict with the measured and predicted slope, or None when the
        battle dump is not present"""
    files = sorted(BATTLES.glob('*.parquet'))
    if not files:
        return None

    tags = {'math': ('math_v0.1', 'math'),
            'instruction_following': ('if_v0.1', 'if'),
            'creative_writing': ('creative_writing_v0.1', 'creative_writing')}
    cols = ['model_a', 'model_b', 'winner', 'category_tag']
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in files],
                  ignore_index=True)
    d = d[d.winner.isin(['model_a', 'model_b', 'tie', 'both_bad'])].copy()
    d['s'] = np.where(d.winner == 'model_a', 1.0,
                      np.where(d.winner == 'model_b', 0.0, 0.5))
    for cat, (block, key) in tags.items():
        d[cat] = d.category_tag.map(
            lambda t: bool(t.get(block, {}).get(key)) if isinstance(t, dict) else False)

    rng = np.random.default_rng(seed)
    measured, predicted = [], []
    for _ in range(n_splits):
        half = rng.random(len(d)) < 0.5
        rows = []
        for cat in ['overall', *tags]:
            sub = d if cat == 'overall' else d[d[cat]]
            cnt = pd.concat([sub.model_a, sub.model_b]).value_counts()
            models = sorted(cnt[cnt >= min_battles].index)
            if len(models) < 8:
                continue
            a, b = sub[half[sub.index]], sub[~half[sub.index]]
            a = a[a.model_a.isin(models) & a.model_b.isin(models)]
            b = b[b.model_a.isin(models) & b.model_b.isin(models)]
            if len(a) < 500 or len(b) < 500:
                continue
            th_a, v_a = _fit_bt(a, models)
            th_b, _ = _fit_bt(b, models)
            th_b = th_b - th_b.mean() + th_a.mean()
            for i in range(len(models)):
                rows.append({'gap': th_a.max() - th_a[i],
                             'y': th_b[i] - th_a[i], 'v': v_a[i]})
        t = pd.DataFrame(rows)
        if len(t) < 10:
            continue
        measured.append(np.polyfit(t.gap, t.y, 1)[0])
        predicted.append(t.v.mean() / t.gap.var())

    return {'measured': float(np.mean(measured)), 'measured_sd': float(np.std(measured)),
            'predicted': float(np.mean(predicted)), 'splits': len(measured),
            'battles': len(d)}


def _fit_bt(df, models, ridge=1e-3, iters=400, tol=1e-9):
    """Bradley-Terry by Newton steps on the logistic likelihood, ties = 0.5

    Arguments:
        df: Battles with model_a, model_b and the score column s
        models: Ordered model names defining the parameter vector
        ridge: Ridge penalty keeping the fit identified
        iters: Maximum Newton steps
        tol: Stop once the largest step falls below this

    Returns: (theta, variance) on the Arena scale"""
    scale = 400 / np.log(10)
    idx = {m: i for i, m in enumerate(models)}
    a, b = df.model_a.map(idx).to_numpy(), df.model_b.map(idx).to_numpy()
    s, n = df.s.to_numpy(), len(models)

    th = np.zeros(n)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(th[a] - th[b])))
        r = s - p
        g = np.bincount(a, r, n) - np.bincount(b, r, n) - ridge * th
        w = p * (1 - p)
        h = np.bincount(a, w, n) + np.bincount(b, w, n) + ridge
        step = g / np.maximum(h, 1e-9)
        th += step
        th -= th.mean()
        if np.abs(step).max() < tol:
            break

    p = 1.0 / (1.0 + np.exp(-(th[a] - th[b])))
    w = p * (1 - p)
    h = np.bincount(a, w, n) + np.bincount(b, w, n) + ridge
    return th * scale, (1.0 / np.maximum(h, 1e-9)) * scale ** 2


def load_events(since='2024-01-01'):
    """Build the event panel from the raw leaderboard in one call

    Arguments:
        since: Left edge of the panel

    Returns: Event panel DataFrame"""
    return build_event_panel(load_panel(cats=CORE, since=since))
