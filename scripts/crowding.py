"""How crowded the top of each category is, and how that has changed.

Two views of the same question, both per snapshot and averaged over the CORE
categories.

    near_top_counts  how many models sit within x points of the category leader
    top_rank_gaps    how far the r-th best model trails the leader

The first fixes a rating band and counts models; the second fixes a rank and
measures a distance. They move in opposite directions when the pool grows, so
neither is worth reading alone: a band that swallows more models could mean the
top has tightened or simply that more models exist, and the rank-r gap is
immune to the pool size but says nothing about how many models are close.

A note on the band. The band is additive, rating >= top rating - x, because a
Bradley-Terry rating has an arbitrary origin: only differences carry meaning,
so a percentage of a rating is not a scale-free quantity. The draft's earlier
"within x%" convention also let the band drift with the scale -- its 1% band
was 12.6 points wide at the start of the panel and 15.2 at the end, so part of
the rise in the counts was the band widening underneath them. The point
thresholds are set near the widths the old bands averaged, and `band_pct`
reports what fraction of the top rating each one currently is.

Nothing here is noise-corrected. A count has no obvious correction, and the
leader's rating is a maximum over a noisy sample, so both series are biased
toward looking more crowded (counts) and more spread out (gaps) than the truth.
The bias is a level, not a trend, so the movement over time is still readable.
"""
import numpy as np
import pandas as pd

import or_arena_match
from common import CORE, OPENROUTER, load_panel

# Rating points below the leader. Chosen at the mean widths the old 1%, 5%
# and 10% bands covered, so the counts stay comparable with the draft.
THRESHOLDS = (15, 75, 150)
RANKS = (10, 20, 50)


def near_top_counts(d, thresholds=THRESHOLDS, cats=CORE, by_cat=False):
    """Models rated within x points of the category leader, per snapshot

    Arguments:
        d: Panel DataFrame from common.load_panel
        thresholds: Rating points below the leader defining each band
        cats: Categories to count within
        by_cat: Return one row per (snapshot, category) instead of the
            average over categories

    Returns: DataFrame indexed by snapshot with one column per threshold, or
        the per-category long form when by_cat is set"""
    rows = []
    for (dt, cat), g in d[d.category.isin(cats)].groupby(['dt', 'category']):
        r = g.rating.dropna()
        if len(r) < 2:
            continue
        top = r.max()
        row = {'dt': dt, 'cat': cat, 'n_rated': len(r)}
        for x in thresholds:
            row[x] = int((r >= top - x).sum())
            row[f'pct_{x}'] = 100.0 * x / top
        rows.append(row)

    out = pd.DataFrame(rows)
    if by_cat:
        return out
    return out.groupby('dt')[list(thresholds)].mean()


def top_rank_gaps(d, ranks=RANKS, cats=CORE, by_cat=False):
    """Rating distance from the category leader to the r-th best model

    A category contributes to rank r only when it has at least r rated models,
    so the early snapshots carry fewer categories at r = 50 than at r = 10.

    Arguments:
        d: Panel DataFrame from common.load_panel
        ranks: Which order statistics to measure against the leader
        cats: Categories to measure within
        by_cat: Return one row per (snapshot, category) instead of the
            average over categories

    Returns: DataFrame indexed by snapshot with one column per rank, or the
        per-category long form when by_cat is set"""
    rows = []
    for (dt, cat), g in d[d.category.isin(cats)].groupby(['dt', 'category']):
        r = np.sort(g.rating.dropna().to_numpy())[::-1]
        if len(r) < 2:
            continue
        row = {'dt': dt, 'cat': cat, 'n_rated': len(r)}
        for k in ranks:
            row[k] = r[0] - r[k - 1] if len(r) >= k else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    if by_cat:
        return out
    return out.groupby('dt')[list(ranks)].mean()


def band_pct(d, thresholds=THRESHOLDS, cats=CORE):
    """Percentage of the leader's rating each point band covers

    This is the mirror of what the percentage convention had to report. There
    the band was fixed in percent and its width in points was carried
    alongside; here the band is fixed in points and this says what share of the
    scale it amounts to. That share drifts, which is why the bands are no
    longer set in percent.

    Arguments:
        d: Panel DataFrame from common.load_panel
        thresholds: Rating points below the leader defining each band
        cats: Categories to measure within

    Returns: DataFrame indexed by snapshot with one column per threshold"""
    wide = near_top_counts(d, thresholds=thresholds, cats=cats, by_cat=True)
    cols = [f'pct_{x}' for x in thresholds]
    out = wide.groupby('dt')[cols].mean()
    return out.rename(columns={f'pct_{x}': x for x in thresholds})


def load_series(since='2024-01-01', cats=CORE):
    """Both crowding series from the raw leaderboard in one call

    Arguments:
        since: Left edge of the panel
        cats: Categories to measure within

    Returns: (counts, gaps, pcts) -- near-top counts, top-rank gaps and each
        point band as a percentage of the leader's rating"""
    d = load_panel(cats=cats, firms_only=False, since=since)
    return (near_top_counts(d, cats=cats),
            top_rank_gaps(d, cats=cats),
            band_pct(d, cats=cats))


def quantile_profile(d, years=(2025, 2026), cats=CORE, n_grid=101, relative=True):
    """Performance by quantile of the field, normalized against the leader

    Reading the count series alone cannot separate a tightening top from a
    growing pool, because both push the counts up. This does separate them: a
    quantile is a position in the field regardless of how big the field is, so
    if the extra models are simply more of the same, the curves for two years
    lie on top of each other. A curve that rises toward the leader means the
    field genuinely closed in.

    Arguments:
        d: Panel DataFrame from common.load_panel
        years: Calendar years to profile, one curve each
        cats: Categories to profile within
        n_grid: Points on the quantile grid, 0 = leader, 1 = worst rated
        relative: Normalize as a ratio to the top rating; when False, report
            the shortfall from the top in rating points

    Returns: DataFrame indexed by quantile with one column per year"""
    grid = np.linspace(0, 1, n_grid)
    d = d[d.category.isin(cats)]

    out = {}
    for year in years:
        curves = []
        sub = d[d.dt.dt.year == year]
        for _, g in sub.groupby(['dt', 'category']):
            r = np.sort(g.rating.dropna().to_numpy())[::-1]
            if len(r) < 10:
                continue
            q = np.arange(len(r)) / (len(r) - 1)
            y = r / r[0] if relative else r[0] - r
            curves.append(np.interp(grid, q, y))
        if curves:
            out[year] = np.mean(curves, axis=0)
    return pd.DataFrame(out, index=pd.Index(grid, name='quantile'))


def load_profiles(since='2024-01-01', cats=CORE, years=(2025, 2026)):
    """Both quantile profiles from the raw leaderboard in one call

    Arguments:
        since: Left edge of the panel
        cats: Categories to profile within
        years: Calendar years to profile, one curve each

    Returns: (ratio, points) -- the profile as a fraction of the leader's
        rating and as a shortfall from it in rating points"""
    d = load_panel(cats=cats, firms_only=False, since=since)
    return (quantile_profile(d, years=years, cats=cats),
            quantile_profile(d, years=years, cats=cats, relative=False))


# --------------------------------------------------------------- deployment --
# Also rating points; the grid spans the field, whose bottom sits about 675
# points below the top.
USAGE_THRESHOLDS = (10, 25, 50, 100, 150, 200, 300)


def _openrouter(since='2025-01-01'):
    """OpenRouter daily token volume with a firm column

    Arguments:
        since: Left edge of the usage panel

    Returns: DataFrame with date, model_permaslug, org and total_tokens"""
    o = pd.read_parquet(OPENROUTER)
    o['date'] = pd.to_datetime(o.date)
    o['org'] = o.model_permaslug.str.split('/').str[0].str.lower()
    return o[o.date >= since] if since else o


def usage_share_by_threshold(thresholds=USAGE_THRESHOLDS, since='2025-01-01'):
    """Share of OpenRouter tokens served by models within x points of the top

    Usage has no category breakdown, so the threshold is against the `overall`
    rating at the last Arena snapshot on or before each month.

    Unmatched OpenRouter volume sits in the denominator and never in the
    numerator: a slug with no Arena name cannot clear any threshold. The curve
    is therefore a floor on the true share, and it cannot rise above the
    matched share of tokens, which is reported alongside.

    Arguments:
        thresholds: Rating points below the top rating defining each band
        since: Left edge of both panels

    Returns: (share, coverage) -- a DataFrame indexed by month with one column
        per threshold, and the monthly matched share of tokens"""
    d = load_panel(cats=CORE, firms_only=False, since=since)
    o = _openrouter(since)
    matched = {slug: name for slug, (name, _) in
               or_arena_match.match(o.model_permaslug.unique(),
                                    sorted(d.model_name.unique())).items()}

    rows = []
    for per, g in o.groupby(o.date.dt.to_period('M')):
        tokens = g.groupby('model_permaslug').total_tokens.sum()
        total = tokens.sum()
        seen = d[d.dt <= per.to_timestamp('M')]
        if total <= 0 or seen.empty:
            continue
        ov = d[(d.dt == seen.dt.max()) & (d.category == 'overall')].dropna(subset=['rating'])
        if len(ov) < 5:
            continue

        rating_of = ov.set_index('model_name').rating
        hit = tokens[[s in matched for s in tokens.index]]
        ratings = hit.index.map(lambda s: rating_of.get(matched[s], np.nan))
        top = ov.rating.max()

        row = {'month': per.to_timestamp(), 'coverage': hit.sum() / total}
        for x in thresholds:
            row[x] = hit[ratings >= top - x].sum() / total
        rows.append(row)

    out = pd.DataFrame(rows).set_index('month').sort_index()
    return out[list(thresholds)], out.coverage


def effective_counts(since='2025-01-01'):
    """Effective number of deployed models and firms, month by month

    The inverse Herfindahl of token shares: if three firms split the market
    evenly the effective number is three, and one dominant firm pulls it toward
    one however many firms exist. Firms are the OpenRouter provider prefix, so
    this is unaffected by the blank `organization` field on the Arena side.

    Arguments:
        since: Left edge of the usage panel

    Returns: DataFrame indexed by month with effective and raw counts of models
        and firms"""
    def inverse_hhi(v):
        s = np.asarray(v, float)
        s = s[s > 0]
        return 1.0 / ((s / s.sum()) ** 2).sum()

    o = _openrouter(since)
    rows = []
    for per, g in o.groupby(o.date.dt.to_period('M')):
        by_model = g.groupby('model_permaslug').total_tokens.sum()
        by_firm = g.groupby('org').total_tokens.sum()
        rows.append({'month': per.to_timestamp(),
                     'eff_models': inverse_hhi(by_model),
                     'eff_firms': inverse_hhi(by_firm),
                     'n_models': len(by_model), 'n_firms': len(by_firm)})
    return pd.DataFrame(rows).set_index('month').sort_index()
