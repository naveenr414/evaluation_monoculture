"""Shared machinery for the Evaluation & Monoculture empirical experiments.

Maps the paper's objects onto the LMArena per-category leaderboard panel:
    theta_{i,k}  = BT rating of firm i's flagship model in category k
    K            = CORE categories
    N            = the firms `frontier_firms` derives from the panel
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
MATURE_DAYS = 56

# A firm enters the sample once it has enough flagship changeovers to fit a
# coefficient of its own. This is the same threshold `catchup.beta_per_firm`
# needs, so the roster and the estimator turn on one number rather than two.
MIN_EVENTS = 4


@lru_cache(maxsize=None)
def _read_panel(path):
    """The raw leaderboard with its derived columns, read once per process

    Every routine here starts from the same file, and a full run reads it half
    a dozen times, so the parse is cached. Callers only ever see the result of
    a boolean mask over it, which is a fresh frame, so the cached copy cannot
    be mutated from underneath them.

    Arguments:
        path: Path to the leaderboard parquet

    Returns: DataFrame with `dt` (snapshot date) and `org` (lowercase
        organization) added, and rows with a blank organization dropped"""
    d = pd.read_parquet(path)
    d['dt'] = pd.to_datetime(d.leaderboard_publish_date)
    d['org'] = d.organization.str.lower()
    # The organization field goes blank from 2026-04-02 onward, ~5% of rows.
    # They are unusable for anything firm-keyed, so they never enter the panel.
    return d[d.org.notna() & (d.org.str.strip() != '')].reset_index(drop=True)


def load_panel(path=PANEL, cats=None, firms_only=True, since='2024-01-01'):
    """The leaderboard panel, filtered to the rows an experiment wants

    Arguments:
        path: Path to the leaderboard parquet
        cats: Categories to keep, `overall` always included, or None for all
        firms_only: Keep only the organizations `frontier_firms` derives
        since: Left edge of the panel, or None for no cut

    Returns: DataFrame, one row per (snapshot, model, category)"""
    d = _read_panel(path)
    keep = pd.Series(True, index=d.index)
    if cats is not None:
        keep &= d.category.isin(list(cats) + ['overall'])
    if firms_only:
        keep &= d.org.isin(frontier_firms())
    if since:
        keep &= d.dt >= pd.Timestamp(since)
    return d[keep]


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

    counts = {org: 0 for org in flag.index.get_level_values('org').unique()}
    for org, old, new, switch, t_old in changeovers(flag):
        t_new = first_mature_snapshot(snaps, switch, mature_days)
        if t_new is None:
            continue
        try:
            th_old = rat.loc[(t_old, old)][cats]
            th_new = rat.loc[(t_new, new)][cats]
        except KeyError:
            continue
        if not (th_old.isna().any() or th_new.isna().any()):
            counts[org] += 1
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


def flagships(d):
    """The model standing in for each firm at each snapshot

    A firm ships several models at once, so one has to stand for the firm.
    That is its best `overall` model: the one the evaluator itself ranks
    highest, rather than whichever model happens to lead a given category.

    Arguments:
        d: Panel DataFrame from `load_panel`

    Returns: Series of model names indexed by (dt, org)"""
    ov = d[d.category == 'overall'].dropna(subset=['rating'])
    return (ov.sort_values('rating').groupby(['dt', 'org']).tail(1)
              .set_index(['dt', 'org']).model_name)


def changeovers(flag, orgs=None):
    """Every point at which a firm's flagship model changes

    One release episode per tuple. The switch date is carried because a firm
    can toggle between the same two model names at different dates -- a
    thinking-mode variant launched, reverted, then relaunched -- and those are
    independent episodes that a name-only label would pool into one.

    Arguments:
        flag: Series from `flagships`, indexed by (dt, org)
        orgs: Organizations to walk, or None for every one in `flag`

    Returns: List of (org, outgoing model, incoming model, switch date,
        last snapshot the outgoing model was the flagship at)"""
    present = flag.index.get_level_values('org').unique()
    out = []
    for org in (present if orgs is None else [o for o in orgs if o in present]):
        series = flag.xs(org, level='org').sort_index()
        prev_model, prev_t = None, None
        for t, model in series.items():
            if prev_model is not None and model != prev_model:
                out.append((org, prev_model, model, t, prev_t))
            prev_model, prev_t = model, t
    return out


def first_mature_snapshot(snaps, switch, mature_days=MATURE_DAYS):
    """The first snapshot far enough after a switch to read the new model at

    A model's rating churns for weeks after release while votes accumulate, so
    the incoming model is read once it has had `mature_days` to settle.

    Arguments:
        snaps: Sorted array of snapshot dates
        switch: Date the flagship changed
        mature_days: Days to wait before reading the incoming model

    Returns: Timestamp, or None when the panel ends before the model matures"""
    ready_at = pd.Timestamp(switch).to_datetime64() + np.timedelta64(mature_days, 'D')
    i = int(np.searchsorted(snaps, ready_at))
    return snaps[i] if i < len(snaps) else None


def event_label(firm, old, new, switch):
    """The key that identifies one release episode

    Arguments:
        firm: Organization slug
        old: Outgoing flagship model name
        new: Incoming flagship model name
        switch: Date the flagship changed

    Returns: String, unique per episode (see `changeovers` on why the date is
        part of it)"""
    return f'{firm}:{old}->{new}@{pd.Timestamp(switch).strftime("%Y%m%d")}'


def within(series, *group_arrays, iters=60):
    """Residual of a column after absorbing several sets of fixed effects

    Alternating projections: demeaning by one grouping undoes part of the
    previous one, so the passes are repeated until the column stops moving.
    For one or two groupings this converges long before the default iteration
    count, which is why there is no tolerance to tune.

    Arguments:
        series: Values to transform
        group_arrays: One array of group labels per set of fixed effects
        iters: Demeaning passes over the full set of groupings

    Returns: Array of within-transformed values, in the input order"""
    s = pd.Series(np.asarray(series, dtype=float)).reset_index(drop=True)
    gs = [pd.Series(np.asarray(g)).reset_index(drop=True) for g in group_arrays]
    for _ in range(iters):
        for g in gs:
            s = s - s.groupby(g).transform('mean')
    return s.values


def fe_regress(y, X, groups, noise_cov=None, noise_var=None, cluster=None):
    """Multi-way FE regression, optionally corrected for measurement error

    The correction subtracts the moments that noise alone contributes, so it
    shrinks the denominator of the estimate: the corrected coefficient carries
    more sampling error than the raw one, and `se_c` propagates the same
    Var(X'y) through the shrunken system rather than reusing the raw standard
    error. For a single regressor that works out to `se` scaled by
    Sxx / (Sxx - noise). It treats the leaderboard's reported variances as
    known, so the corrected interval is a lower bound on the true uncertainty.

    The within transform costs degrees of freedom, and the noise moments are
    per-observation, so both are scaled by `dfac` here -- callers pass the raw
    moments and must not scale them again.

    Arguments:
        y: Outcome vector
        X: Regressor matrix, one column per regressor
        groups: One array of group labels per set of fixed effects
        noise_cov: Per-regressor spurious covariance with y, or None to skip
            the correction entirely
        noise_var: Spurious variance/covariance matrix among the regressors,
            or None to treat the regressors as measured without error
        cluster: Cluster labels for the standard errors, or None for
            homoskedastic ones

    Returns: (beta, se, beta_c, resid, dfac, se_c) -- the raw coefficients and
        their standard errors, the corrected coefficients, the within-residual,
        the degrees-of-freedom factor, and the corrected standard errors.
        beta_c and se_c are None when `noise_cov` is not given"""
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
