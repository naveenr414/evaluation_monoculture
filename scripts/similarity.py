"""Firm similarity: how alike are firms' capability profiles, and is what's
left after the obvious structure any more alike than chance?

A snapshot gives a matrix P of shape (firms x categories), one Bradley-Terry
rating per cell.  Raw profiles are nearly identical -- every firm is better at
the easy categories and worse at the hard ones -- so the object of interest is
the two-way residual, P with its firm and category means removed:

    P = grand + firm_effect + category_effect + residual

The catch is that the residual has zero row and column sums by construction, so
its rows are forced to be slightly anti-correlated: the mean pairwise cosine of
N such rows sits near -1/(N-1) no matter what the data says.  Any claim that
firms are alike, or converging, has to be read against that null rather than
against zero, which is what `permutation_null` is for.

The pieces here are deliberately separate so they compose:

    decompose          the additive two-way fit, all four terms
    variance_shares    how much variance each term absorbs
    pairwise_cosines   the observed similarities
    permutation_null   the same statistic under a null that keeps the structure

`panel_cosines` runs the last two over every snapshot, which is the form the
figures want.
"""
from collections import namedtuple

import numpy as np
import pandas as pd

from common import CORE, flagships, load_panel

TwoWay = namedtuple('TwoWay', 'grand firm category residual')
MeanCosine = namedtuple('MeanCosine', 'names matrix counts null')


def profiles_at(rat, flag, date, orgs, cats=CORE):
    """Each firm's flagship profile at one snapshot

    Firms whose flagship is missing, or is missing any category, are dropped
    rather than imputed -- a partial row would change that firm's norm and so
    its cosine with everyone else.

    Arguments:
        rat: DataFrame indexed by (date, model_name), columns are categories
        flag: Series mapping (date, org) to that firm's flagship model name
        date: Timestamp, the snapshot to read
        orgs: List of organization names to attempt
        cats: List of categories, in the column order to return

    Returns: (names, P) where names is the list of firms actually present and
        P is their (firms x categories) rating matrix"""
    names, rows = [], []
    for org in orgs:
        model = flag.get((date, org))
        if model is None:
            continue
        try:
            row = rat.loc[(date, model)][cats]
        except KeyError:
            continue
        if row.isna().any():
            continue
        names.append(org)
        rows.append(row.values.astype(float))
    return names, np.array(rows)


def decompose(P):
    """Additive two-way fit of a firm x category matrix

    This is the least-squares fit of P on firm and category indicators, which
    for a complete matrix is just iterated demeaning done in closed form.  The
    residual is what every similarity measure here is computed on.

    Arguments:
        P: Array of shape (firms, categories)

    Returns: TwoWay(grand, firm, category, residual), where grand is a scalar,
        firm has shape (firms, 1), category has shape (1, categories), and the
        four terms sum back to P"""
    grand = P.mean()
    firm = P.mean(axis=1, keepdims=True) - grand
    category = P.mean(axis=0, keepdims=True) - grand
    residual = P - grand - firm - category
    return TwoWay(grand, firm, category, residual)


def two_way_residual(P):
    """The two-way residual on its own

    Arguments:
        P: Array of shape (firms, categories)

    Returns: Array of shape (firms, categories), rows and columns summing to 0"""
    return decompose(P).residual


def variance_shares(P):
    """How much of the raw variance the firm and category means absorb

    Sums of squares about the grand mean.  The three terms are orthogonal for
    a complete matrix, so they partition the total exactly.  The headline
    number is `explained`: firm and category means together account for ~96%
    of the spread in raw ratings, which is why the residual is the only place
    a firm-specific profile could live.

    Arguments:
        P: Array of shape (firms, categories)

    Returns: Dict with keys firm, category, residual, explained -- each a
        fraction of the total sum of squares -- plus residual_sd"""
    fit = decompose(P)
    n_cats, n_firms = P.shape[1], P.shape[0]

    ss_firm = float((fit.firm ** 2).sum() * n_cats)
    ss_category = float((fit.category ** 2).sum() * n_firms)
    ss_residual = float((fit.residual ** 2).sum())
    total = ss_firm + ss_category + ss_residual

    return {
        'firm': ss_firm / total,
        'category': ss_category / total,
        'residual': ss_residual / total,
        'explained': (ss_firm + ss_category) / total,
        'residual_sd': float(fit.residual.std(ddof=0)),
    }


def variance_shares_by_period(resid_panel, freq='QE'):
    """Variance shares within each period, pooling that period's snapshots

    The single-snapshot version answers "at this date, how much of the spread
    is firm and category means".  Pooling a quarter instead is the more stable
    estimand, and the one the writeups quote, because a single snapshot's
    residual sd is noisy in the firms that happen to have shipped that week.

    The residuals are the ones `residual_panel` already fitted SNAPSHOT BY
    SNAPSHOT, and this only aggregates them.  Re-fitting firm and category
    means across a whole quarter instead would charge the residual for every
    flagship release inside that quarter -- which is real movement in theta,
    not a firm-specific profile -- and roughly triples the residual share.

    Arguments:
        resid_panel: Long DataFrame from `residual_panel`
        freq: String, pandas period alias for the grouping

    Returns: DataFrame indexed by period with columns explained, residual,
        residual_sd, n_obs"""
    rows = []
    for period, g in resid_panel.groupby(pd.Grouper(key='dt', freq=freq)):
        if g.empty:
            continue
        y = g.rating.to_numpy(float)
        resid = g.residual.to_numpy(float)
        ss_total = float(((y - y.mean()) ** 2).sum())
        ss_resid = float((resid ** 2).sum())
        rows.append({'period': period, 'explained': 1 - ss_resid / ss_total,
                     'residual': ss_resid / ss_total,
                     'residual_sd': float(resid.std(ddof=0)), 'n_obs': len(g)})
    return pd.DataFrame(rows).set_index('period')


def pairwise_cosines(R):
    """Cosine similarity of every distinct pair of rows

    Arguments:
        R: Array of shape (firms, categories), normally a two-way residual

    Returns: Array of length n(n-1)/2, upper triangle in row-major order.
        Rows of zero norm are dropped rather than returning NaN"""
    norms = np.linalg.norm(R, axis=1)
    keep = norms > 0
    Rn = R[keep] / norms[keep, None]
    C = Rn @ Rn.T
    iu = np.triu_indices(len(Rn), k=1)
    return C[iu]


def cosine_matrix(R):
    """Full pairwise cosine matrix, for the heatmap

    Arguments:
        R: Array of shape (firms, categories)

    Returns: Array of shape (firms, firms), diagonal set to exactly 1"""
    norms = np.linalg.norm(R, axis=1, keepdims=True)
    Rn = R / np.where(norms == 0, 1, norms)
    C = Rn @ Rn.T
    np.fill_diagonal(C, 1.0)
    return C


def permutation_null(R, draws=20, rng=None):
    """Cosines under a null that shuffles firms within each category

    Permuting independently within each column destroys any firm-specific
    profile while preserving each category's marginal distribution -- and,
    crucially, preserving the zero row and column sums that pin the mean
    cosine near -1/(N-1).  Comparing against this rather than against zero is
    the whole point: an observed mean of -0.05 is not evidence of firms being
    dissimilar, it is what the arithmetic forces.

    Arguments:
        R: Array of shape (firms, categories)
        draws: Int, permutations per call
        rng: numpy Generator, or None for a fresh default

    Returns: Array of pooled cosines, length draws * n(n-1)/2"""
    rng = rng or np.random.default_rng()
    out = []
    for _ in range(draws):
        Rp = np.column_stack([rng.permutation(R[:, k]) for k in range(R.shape[1])])
        out.append(pairwise_cosines(Rp))
    return np.concatenate(out) if out else np.array([])


def rank_organizations(d, exclude=()):
    """Every organization rated at the panel's last snapshot, best first

    No cutoff: `profiles_at` already drops any firm without a complete profile
    in every category, and that requirement is the one that should decide who
    appears. An arbitrary top-n on top of it silently discards firms that
    would have qualified.

    Arguments:
        d: Panel DataFrame from common.load_panel
        exclude: Iterable of organization names to drop

    Returns: List of organization names, ordered by best `overall` rating"""
    last = d.dt.max()
    snap = d[(d.dt == last) & (d.category == 'overall')].dropna(subset=['rating'])
    ranked = (snap.sort_values('rating', ascending=False)
                  .groupby('org').head(1)
                  .sort_values('rating', ascending=False))
    return [o for o in ranked.org if o and o not in set(exclude)]


def mean_cosine_matrix(d, orgs, cats=CORE, min_snapshots=10):
    """Pairwise cosine averaged over every snapshot in the panel

    Each snapshot's matrix is computed on that snapshot's own residual, and
    the average is taken per PAIR over the snapshots where both firms have a
    complete profile. Firms enter and leave the panel at different dates, so
    averaging whole matrices would compare pairs measured over different
    windows; this way each cell is an average over exactly the dates its two
    firms share, with `counts` reporting how many that was.

    Arguments:
        d: Panel DataFrame from common.load_panel
        orgs: List of organizations to include
        cats: List of categories defining the profile
        min_snapshots: Int, drop firms appearing in fewer snapshots than this

    Returns: MeanCosine(names, matrix, counts, null) -- the surviving firms,
        their mean cosine matrix with diagonal 1, the per-pair snapshot counts,
        and the mechanical null averaged the same way. `null` is the mean over
        snapshots of -1/(N_t - 1), NOT -1/(N-1) on the final firm count: N_t
        varies a lot across the panel, and using the final count understates
        how negative the null is on the early, smaller snapshots"""
    rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')
    flag = flagships(d)

    index = {org: i for i, org in enumerate(orgs)}
    n = len(orgs)
    total = np.zeros((n, n))
    counts = np.zeros((n, n), dtype=int)
    appearances = np.zeros(n, dtype=int)
    nulls = []

    for date in sorted(d.dt.unique()):
        names, P = profiles_at(rat, flag, date, orgs, cats)
        if len(names) < 2:
            continue
        idx = [index[name] for name in names]
        C = cosine_matrix(two_way_residual(P))
        total[np.ix_(idx, idx)] += C
        counts[np.ix_(idx, idx)] += 1
        appearances[idx] += 1
        nulls.append(-1 / (len(names) - 1))

    keep = [i for i in range(n) if appearances[i] >= min_snapshots]
    names = [orgs[i] for i in keep]
    total, counts = total[np.ix_(keep, keep)], counts[np.ix_(keep, keep)]

    with np.errstate(invalid='ignore', divide='ignore'):
        C = np.where(counts > 0, total / counts, 0.0)
    np.fill_diagonal(C, 1.0)
    return MeanCosine(names, C, counts, float(np.mean(nulls)))


def panel_cosines(d, orgs, cats=CORE, draws=20, seed=0, min_firms=4):
    """Observed and null cosines pooled over every snapshot in the panel

    Arguments:
        d: Panel DataFrame from common.load_panel
        orgs: List of organizations to include
        cats: List of categories defining the profile
        draws: Int, null permutations per snapshot
        seed: Int, seed for the permutation draws
        min_firms: Int, skip snapshots with fewer firms than this

    Returns: (observed, null, per_date) -- two pooled arrays, and a DataFrame
        with one row per snapshot giving that snapshot's mean observed cosine,
        mean null cosine, and firm count"""
    rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')
    flag = flagships(d)
    rng = np.random.default_rng(seed)

    observed, null, rows = [], [], []
    for date in sorted(d.dt.unique()):
        names, P = profiles_at(rat, flag, date, orgs, cats)
        if len(names) < min_firms:
            continue
        R = two_way_residual(P)
        obs_t = pairwise_cosines(R)
        null_t = permutation_null(R, draws=draws, rng=rng)
        observed.append(obs_t)
        null.append(null_t)
        rows.append({'dt': date, 'n_firms': len(names),
                     'observed': obs_t.mean(), 'null': null_t.mean()})

    return (np.concatenate(observed), np.concatenate(null),
            pd.DataFrame(rows))


def residual_panel(d, orgs, cats=CORE):
    """Every (snapshot, firm, category) residual as a long table

    The form the time-series figure and the persistence work want, as opposed
    to the per-snapshot matrices the cosines are computed on.

    Arguments:
        d: Panel DataFrame from common.load_panel
        orgs: List of organizations to include
        cats: List of categories defining the profile

    Returns: DataFrame with columns dt, org, category, residual, rating"""
    rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')
    flag = flagships(d)

    rows = []
    for date in sorted(d.dt.unique()):
        names, P = profiles_at(rat, flag, date, orgs, cats)
        if len(names) < 2:
            continue
        R = two_way_residual(P)
        for i, org in enumerate(names):
            for k, cat in enumerate(cats):
                rows.append({'dt': date, 'org': org, 'category': cat,
                             'residual': R[i, k], 'rating': P[i, k]})
    return pd.DataFrame(rows)


def load_default_panel(since='2025-01-01'):
    """The panel every similarity figure starts from

    Arguments:
        since: String date, the panel's left edge

    Returns: (d, orgs) -- the panel DataFrame and the ranked organization list"""
    d = load_panel(cats=CORE, firms_only=False, since=since)
    return d, rank_organizations(d)
