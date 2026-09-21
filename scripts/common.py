"""Shared machinery for the Evaluation & Monoculture empirical experiments.

Maps the paper's objects onto the LMArena per-category leaderboard panel:
    theta_{i,k}  = BT rating of firm i's flagship model in category k
    K            = CORE categories
    N            = FRONTIER firms (derived; see `frontier_firms`)
Every routine here noise-corrects using the `variance` column (the BT estimator's
sampling variance), which is 100% populated from 2024 onward.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# Repository layout, resolved from this file so scripts run from any directory.
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'raw_data'
PROCESSED = ROOT / 'processed_data'
PLOTS = ROOT / 'plots'

PANEL = RAW / 'leaderboard.parquet'
OPENROUTER = RAW / 'openrouter_daily.parquet'
# Raw battles, needed only by the split-half check in `catchup`. They live in
# the archive rather than raw_data/ because nothing else reads them and they
# are 1.3 GB; the check reports itself as unavailable when they are missing.
BATTLES = ROOT / '_archive' / 'data' / '140k' / 'data'

CORE = ['coding', 'math', 'hard_prompts', 'instruction_following',
        'creative_writing', 'longer_query', 'multi_turn', 'english']
INDUSTRY = ['industry_business_and_management_and_financial_operations',
            'industry_entertainment_and_sports_and_media',
            'industry_legal_and_government',
            'industry_life_and_physical_and_social_science',
            'industry_mathematical',
            'industry_medicine_and_healthcare',
            'industry_software_and_it_services',
            'industry_writing_and_literature_and_language']
MATURE_DAYS = 56

# A firm enters the sample once it has enough flagship changeovers to fit a
# coefficient of its own. This is the same threshold `catchup.beta_per_firm`
# needs, so the roster and the estimator turn on one number rather than two.
MIN_EVENTS = 4


def load_panel(path=PANEL, cats=None, firms_only=True, since='2024-01-01'):
    d = pd.read_parquet(path)
    d['dt'] = pd.to_datetime(d.leaderboard_publish_date)
    d['org'] = d.organization.str.lower()
    d = d[d.org.notna() & (d.org.str.strip() != '')]
    if cats is not None:
        d = d[d.category.isin(list(cats) + ['overall'])]
    if firms_only:
        d = d[d.org.isin(frontier_firms())]
    if since:
        d = d[d.dt >= since]
    return d


def changeover_counts(cats=CORE, mature_days=MATURE_DAYS, since='2024-01-01'):
    """Usable flagship changeovers per organization, over every organization

    A changeover counts only if it would survive into the event panel: a
    snapshot exists at least `mature_days` after the switch, and both the
    outgoing and incoming models are rated in every category of `cats`.

    Arguments:
        cats: Categories the two models must both be rated in
        mature_days: Days to wait before reading the incoming model
        since: Left edge of the panel

    Returns: Series of counts indexed by organization, descending"""
    d = load_panel(cats=cats, firms_only=False, since=since)
    rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')
    flag = flagships(d)
    snaps = np.sort(d.dt.unique())
    orgs = flag.index.get_level_values('org').unique()

    counts = {}
    for org in orgs:
        series = flag.xs(org, level='org').sort_index()
        prev_model, prev_t, n = None, None, 0
        for t, model in series.items():
            if prev_model is not None and model != prev_model:
                later = [s for s in snaps if s >= t + np.timedelta64(mature_days, 'D')]
                if later:
                    try:
                        old = rat.loc[(prev_t, prev_model)][cats]
                        new = rat.loc[(later[0], model)][cats]
                        if not (old.isna().any() or new.isna().any()):
                            n += 1
                    except KeyError:
                        pass
            prev_model, prev_t = model, t
        counts[org] = n
    return pd.Series(counts).sort_values(ascending=False)


@lru_cache(maxsize=None)
def frontier_firms(min_events=MIN_EVENTS):
    """Organizations with at least `min_events` usable flagship changeovers

    Replaces the hand-picked roster this module used to carry. The list is
    derived from the panel, so a data refresh moves it rather than silently
    leaving it stale.

    Arguments:
        min_events: Minimum usable changeovers to enter the sample

    Returns: Sorted list of organization slugs"""
    counts = changeover_counts()
    return sorted(counts[counts >= min_events].index)


def __getattr__(name):
    """Compute FRONTIER on first use, so importing this module stays cheap."""
    if name == 'FRONTIER':
        return frontier_firms()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def flagships(d):
    """Highest 'overall'-rated model per (snapshot, firm)."""
    ov = d[d.category == 'overall'].dropna(subset=['rating'])
    return (ov.sort_values('rating').groupby(['dt', 'org']).tail(1)
              .set_index(['dt', 'org']).model_name)


def build_events(d, cats=CORE, mature_days=MATURE_DAYS):
    """One row per (flagship changeover event, category).

    dtheta = new flagship rating - outgoing flagship rating
    gap    = rivals' per-category frontier - outgoing flagship rating
    gap_teacher = best-overall rival's rating - outgoing flagship rating
    """
    rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')
    var = d.pivot_table(index=['dt', 'model_name'], columns='category', values='variance')
    flag = flagships(d)
    snaps = np.sort(d.dt.unique())

    events = []
    for firm in FRONTIER:
        if firm not in flag.index.get_level_values('org'):
            continue
        ser = flag.xs(firm, level='org').sort_index()
        prev_m, prev_t = None, None
        for t, m in ser.items():
            if prev_m is not None and m != prev_m:
                events.append((firm, prev_m, m, t, prev_t))
            prev_m, prev_t = m, t

    ov_rat = d[d.category == 'overall'].pivot_table(
        index='dt', columns='model_name', values='rating')

    rows = []
    for firm, old, new, switch, t_old in events:
        cand = [s for s in snaps if s >= switch + np.timedelta64(mature_days, 'D')]
        if not cand:
            continue
        t_new = cand[0]
        try:
            th_old, th_new = rat.loc[(t_old, old)][cats], rat.loc[(t_new, new)][cats]
            v_old, v_new = var.loc[(t_old, old)][cats], var.loc[(t_new, new)][cats]
        except KeyError:
            continue
        if th_old.isna().any() or th_new.isna().any():
            continue

        snap = rat.xs(t_old, level='dt')
        rivals = [flag.get((t_old, f)) for f in FRONTIER if f != firm]
        rivals = [m for m in rivals if m is not None and m in snap.index]
        riv = snap.loc[rivals][cats].dropna()
        if len(riv) < 3:
            continue
        frontier = riv.max(axis=0)
        v_front = np.array([var.loc[(t_old, riv[k].idxmax())][k] for k in cats])

        # distillation candidate: single best-overall rival at t_old
        ov_t = ov_rat.loc[t_old, [m for m in riv.index if m in ov_rat.columns]].dropna()
        teacher = ov_t.idxmax() if len(ov_t) else None
        th_teach = riv.loc[teacher] if teacher in riv.index else pd.Series(np.nan, index=cats)
        # per-category frontier excluding the teacher
        riv_x = riv.drop(index=[teacher]) if teacher in riv.index else riv
        front_x = riv_x.max(axis=0) if len(riv_x) else pd.Series(np.nan, index=cats)

        for j, k in enumerate(cats):
            rows.append(dict(
                # event label disambiguated by switch date: a firm can toggle between the
                # same two model names at different dates (e.g. a thinking-mode variant
                # launched, reverted, relaunched), which otherwise collides two independent
                # release episodes under one label and gets silently pooled by groupby('event')
                event=f'{firm}:{old}->{new}@{pd.Timestamp(switch).strftime("%Y%m%d")}',
                firm=firm, cat=k, switch=switch,
                dtheta=th_new[k] - th_old[k],
                gap=frontier[k] - th_old[k],
                gap_teacher=th_teach[k] - th_old[k],
                gap_excl_teacher=front_x[k] - th_old[k],
                teacher=teacher,
                v_old=v_old[k], v_new=v_new[k], v_front=v_front[j]))
    return pd.DataFrame(rows)


def within(series, *group_arrays, iters=60):
    """Iterative demeaning = multi-way fixed effects within transform."""
    s = pd.Series(np.asarray(series, dtype=float)).reset_index(drop=True)
    gs = [pd.Series(np.asarray(g)).reset_index(drop=True) for g in group_arrays]
    for _ in range(iters):
        for g in gs:
            s = s - s.groupby(g).transform('mean')
    return s.values


def fe_regress(y, X, groups, noise_cov=None, noise_var=None, cluster=None):
    """Multi-way FE regression with optional errors-in-variables correction.

    noise_cov : vector, per-regressor spurious covariance with y (subtracted)
    noise_var : matrix, spurious variance/covariance among regressors (subtracted)

    Returns (beta, se, beta_c, resid, dfac, se_c). The corrected estimate solves
    a system with a shrunken denominator, so it carries more sampling error than
    the raw one; `se_c` propagates the same Var(X'y) through that system, which
    for one regressor is just se scaled by Sxx / (Sxx - noise). It treats the
    reported variances as known, so it is a lower bound on the true uncertainty.
    """
    n = len(y)
    yw = within(y, *groups)
    Xw = np.column_stack([within(X[:, j], *groups) for j in range(X.shape[1])])
    dfac = (n - sum(len(pd.unique(g)) for g in groups) + len(groups) - 1) / n

    XtX = Xw.T @ Xw / n
    Xty = Xw.T @ yw / n
    beta = np.linalg.solve(XtX, Xty)

    beta_c, A = None, None
    if noise_cov is not None:
        A = XtX - (np.asarray(noise_var) * dfac if noise_var is not None else 0)
        b = Xty - np.asarray(noise_cov) * dfac
        beta_c = np.linalg.solve(A, b)

    resid = yw - Xw @ beta
    if cluster is not None:
        meat = np.zeros((Xw.shape[1], Xw.shape[1]))
        for _, idx in pd.Series(range(n)).groupby(np.asarray(cluster)):
            xi, ri = Xw[idx.values], resid[idx.values]
            s = xi.T @ ri
            meat += np.outer(s, s)
        bread = np.linalg.inv(XtX * n)
        V = bread @ meat @ bread
    else:
        V = np.linalg.inv(XtX * n) * (resid @ resid) / (n - Xw.shape[1])

    se_c = None
    if A is not None:
        # beta_c = A^-1 (X'y - const), so Var(beta_c) = A^-1 XtX V XtX A^-1.
        M = np.linalg.solve(A, XtX)
        se_c = np.sqrt(np.diag(M @ V @ M.T))
    return beta, np.sqrt(np.diag(V)), beta_c, resid, dfac, se_c


def two_way_resid(P):
    R = P - P.mean(1, keepdims=True)
    return R - R.mean(0, keepdims=True)


def firm_profiles(d, cats, date, value='rating'):
    """One flagship per firm at `date`; returns (profiles, variances, firm names)."""
    rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values=value)
    var = d.pivot_table(index=['dt', 'model_name'], columns='category', values='variance')
    flag = flagships(d)
    names, P, V = [], [], []
    for f in FRONTIER:
        m = flag.get((date, f))
        if m is None:
            continue
        try:
            r, v = rat.loc[(date, m)][cats], var.loc[(date, m)][cats]
        except KeyError:
            continue
        if r.isna().any():
            continue
        names.append(f); P.append(r.values); V.append(v.values)
    return np.array(P), np.array(V), names
