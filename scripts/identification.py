"""What the data can and cannot pin down about the production function F_i.

Two questions, one file.

`identified_set` asks what a single snapshot leaves free. With
theta_{i,k} = a_{i,k} log(1 + x_{i,k}), an unobserved budget sum_k x_{i,k} = c_i
and a firm maximising sum_k w_k theta_{i,k}, the first-order condition is
w_k a_{i,k} = lambda exp(u_k) with u_k = log(1 + x_{i,k}). Substituting
a_{i,k} = theta_{i,k} / u_k gives

    u_k exp(u_k) = w_k theta_{i,k} / lambda    ->    u_k = W(w_k theta_{i,k} / lambda)

with W the Lambert function and lambda fixed by the budget. So for each (c_i, w)
the matrix is point identified, and the identified set is swept out by varying
them. Sweeping the budget is Theorem 9 made arithmetic: the LEVEL of A_i moves
freely, and the question worth asking is whether the SHAPE across categories
survives.

`family_cv` asks which restrictions on F_i actually earn their parameters. The
theory table distinguishes four families by whether they admit firm-by-category
heterogeneity and cross-category spillovers. Each maps to a restriction on the
local response matrix B_i in dtheta = B_i gap, which is the one part of F_i the
release panel observes directly:

    isotropic       B_i = a_i beta_0 I          a_i log(1+x) 1, one scalar per firm
    shared          B   = B_pool (K x K)        A log(1+x), spillovers, no firm terms
    diagonal        B_i = diag(b_i1..b_iK)      A_i diagonal, no spillovers
    general         B_i = full K x K per firm   A_i general

Scored by leave-one-event-out prediction of the held-out release's full
K-vector. The ridge for each family is the best one on that same loop, which
flatters the flexible families; they lose anyway, which is the point.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import lambertw

from common import CORE, FRONTIER, flagships, load_panel

# The organization field goes blank after this, so firm-keyed snapshots stop.
ORG_BREAK = pd.Timestamp('2026-03-19')
BUDGETS = np.logspace(-1, 3, 41)
RIDGES = (0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)
FAMILIES = ('isotropic', 'shared', 'diagonal', 'general')


def _solve_u(theta, w, budget):
    """Interior allocation u_k = W(w_k theta_k / lambda) at a given budget

    Arguments:
        theta: Category ratings above the floor, one per category
        w: Evaluation weights, one per category
        budget: Total spend sum_k x_k

    Returns: Array of u_k, or None when no interior solution exists"""
    def spend(loglam):
        u = np.real(lambertw(w * theta / np.exp(loglam)))
        return np.sum(np.exp(u) - 1) - budget

    if spend(-40.0) < 0 or spend(40.0) > 0:
        return None
    loglam = brentq(spend, -40.0, 40.0, xtol=1e-12, rtol=1e-14)
    return np.real(lambertw(w * theta / np.exp(loglam)))


def identified_set(cats=CORE, budgets=BUDGETS, snapshot=ORG_BREAK, weights=None):
    """Range of a_{i,k} consistent with one snapshot, over a budget sweep

    Theta is measured above a floor, the weakest rated model in that category,
    so that theta > 0 and the logarithm is defined. The floor is itself a free
    normalisation.

    Arguments:
        cats: Categories spanning the production matrix
        budgets: Budgets to sweep; each one point identifies A_i
        snapshot: Read firms' flagships at the last snapshot on or before this
        weights: Evaluation weights over cats; defaults to Arena vote shares

    Returns: DataFrame with one row per (firm, category, budget) carrying the
        implied a, the allocation u and the spend x"""
    d = load_panel(cats=cats, firms_only=False, since=None)
    d = d[d.dt <= snapshot]
    rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')
    flag = flagships(d)
    t = d.dt.max()

    field = d[(d.dt == t) & (d.category.isin(cats))].dropna(subset=['rating'])
    floor = field.groupby('category').rating.min()[cats].to_numpy()
    if weights is None:
        votes = field.groupby('category').vote_count.sum().reindex(cats)
        weights = (votes / votes.sum()).to_numpy()

    rows = []
    for firm in FRONTIER:
        model = flag.get((t, firm))
        if model is None:
            continue
        try:
            theta = rat.loc[(t, model)][cats]
        except KeyError:
            continue
        if theta.isna().any():
            continue
        theta = theta.to_numpy(float) - floor

        for budget in budgets:
            u = _solve_u(theta, weights, budget)
            if u is None or np.any(u <= 0):
                continue
            a = theta / u
            for j, cat in enumerate(cats):
                rows.append({'firm': firm, 'cat': cat, 'budget': budget,
                             'a': a[j], 'u': u[j], 'x': np.exp(u[j]) - 1})
    return pd.DataFrame(rows)


def identified_ranges(setdf):
    """Level and shape of A_i across the budget sweep, per firm

    The level is free by Theorem 9, so the number that matters is whether the
    SHAPE -- a_{i,k} divided by the firm's own mean -- holds still while the
    level moves.

    Arguments:
        setdf: Output of `identified_set`

    Returns: (levels, shapes) -- per-firm ranges of a, and the spread of the
        normalized shape across budgets"""
    setdf = setdf.copy()
    setdf['shape'] = setdf.a / setdf.groupby(['firm', 'budget']).a.transform('mean')

    levels = setdf.groupby('firm').a.agg(['min', 'max', 'median'])
    levels['fold'] = levels['max'] / levels['min']

    spread = setdf.groupby(['firm', 'cat']).shape.agg(['min', 'max'])
    spread['fold'] = spread['max'] / spread['min']
    shapes = spread.groupby('firm').fold.agg(['min', 'max', 'median'])
    return levels, shapes


# ------------------------------------------------------------- families ----
def _design(events, cats=CORE):
    """Gap and dtheta matrices, one row per release event

    Arguments:
        events: Event panel carrying gap and dtheta per (event, category)
        cats: Column order for both matrices

    Returns: (G, D, firms) -- n x K gaps, n x K improvements, firm per row"""
    gap_col = 'gap_best' if 'gap_best' in events.columns else 'gap'
    g = events.pivot_table(index='event', columns='cat', values=gap_col)[cats]
    d = events.pivot_table(index='event', columns='cat', values='dtheta')[cats]
    firms = events.drop_duplicates('event').set_index('event').firm.reindex(g.index)
    return g.to_numpy(float), d.to_numpy(float), firms.to_numpy()


def _fit_predict(family, G, D, firms, train, test, ridge):
    """Fit one family on `train` rows and predict the `test` rows

    Every family carries a free per-category intercept, fitted as the training
    mean of dtheta. Without it a family is charged for the average size of a
    release as well as for its shape, and the mean baseline wins by default.

    Arguments:
        family: One of FAMILIES
        G: n x K gap matrix
        D: n x K improvement matrix
        firms: Firm label per row
        train: Boolean mask of training rows
        test: Boolean mask of rows to predict
        ridge: Ridge strength, shrinking each firm toward the pooled fit

    Returns: Array of predictions for the test rows, K columns wide"""
    Gt, Dt, Ft = G[train], D[train], firms[train]
    K = G.shape[1]
    eye = np.eye(K)

    gbar, dbar = Gt.mean(0), Dt.mean(0)
    Gc, Dc = Gt - gbar, Dt - dbar
    Gq = G[test] - gbar
    pooled = np.linalg.solve(Gc.T @ Gc + ridge * eye, Gc.T @ Dc)

    if family == 'shared':
        return dbar + Gq @ pooled

    if family == 'isotropic':
        beta0 = (Gc * Dc).sum() / (Gc * Gc).sum()
        scale = {}
        for f in np.unique(Ft):
            m = Ft == f
            scale[f] = (((beta0 * Gc[m] * Dc[m]).sum() + ridge)
                        / (((beta0 * Gc[m]) ** 2).sum() + ridge))
        return np.array([dbar + scale.get(f, 1.0) * beta0 * g
                         for g, f in zip(Gq, firms[test])])

    if family == 'diagonal':
        diag_pool = (Gc * Dc).sum(0) / ((Gc * Gc).sum(0) + ridge)
        diag = {}
        for f in np.unique(Ft):
            m = Ft == f
            diag[f] = (((Gc[m] * Dc[m]).sum(0) + ridge * diag_pool)
                       / ((Gc[m] * Gc[m]).sum(0) + ridge))
        return np.array([dbar + diag.get(f, diag_pool) * g
                         for g, f in zip(Gq, firms[test])])

    mats = {}
    for f in np.unique(Ft):
        m = Ft == f
        mats[f] = np.linalg.solve(Gc[m].T @ Gc[m] + ridge * eye,
                                  Gc[m].T @ Dc[m] + ridge * pooled)
    return np.array([dbar + g @ mats.get(f, pooled) for g, f in zip(Gq, firms[test])])


def family_cv(events, families=FAMILIES, ridges=RIDGES, cats=CORE):
    """Leave-one-event-out prediction accuracy for each F_i family

    Every held-out release is predicted as a full K-vector, and the baseline is
    the training mean, so R^2 <= 0 means a family does no better than "every
    release improves by the average amount".

    Arguments:
        events: Event panel from catchup.build_event_panel
        families: Which restrictions to score
        ridges: Ridge grid; the best value per family is reported
        cats: Column order for the design matrices

    Returns: DataFrame with one row per family carrying the best ridge, the
        out-of-sample R^2, and the free parameters per firm"""
    G, D, firms = _design(events, cats)
    n, K = G.shape
    n_firms = len(np.unique(firms))
    per_firm = {'isotropic': 1, 'shared': 0, 'diagonal': K, 'general': K * K}

    rows = []
    for family in families:
        best = None
        for ridge in ridges:
            sse = base = 0.0
            for i in range(n):
                train = np.ones(n, bool)
                train[i] = False
                test = ~train
                pred = _fit_predict(family, G, D, firms, train, test, ridge)
                sse += ((D[test] - pred) ** 2).sum()
                base += ((D[test] - D[train].mean(0)) ** 2).sum()
            r2 = 1 - sse / base
            if best is None or r2 > best[1]:
                best = (ridge, r2)
        rows.append({'family': family, 'ridge': best[0], 'oos_r2': best[1],
                     'params_per_firm': per_firm[family],
                     'total_params': per_firm[family] * n_firms
                                     + (K * K if family == 'shared' else 0),
                     'events': n, 'firms': n_firms})
    return pd.DataFrame(rows)
