"""Every figure and table in the paper, drawn from `similarity`, `catchup`
and `crowding`.

Each figure is a small function that pulls what it needs from those three
modules, draws it, and saves a PDF into `plots/`. Nothing here computes a
statistic: if a number appears in a figure it came from one of them, so the
figures and the text cannot drift apart.

Experiment A, specialization:
    cosine_kde           observed pairwise cosines against the permutation null
    cosine_heatmap       the mean pairwise cosine matrix, hierarchically ordered
    residual_timeseries  (org, category) residual trajectories over the panel

Experiment A, catch-up:
    gap_vs_dtheta        improvement against the gap, one line per firm
    beta_forest          the same betas with their intervals
    beta_by_level        beta once the outgoing model's rating is controlled for
    confounds_table      beta under every robustness cut (LaTeX to stdout)
    spec_ladder_table    beta under each fixed-effect spec (LaTeX to stdout)

Experiment B, the shrinking top:
    top_rank_gaps        leader-to-rank-r distance, per snapshot
    top_distribution     shortfall from the leader by quantile of the field
    usage_within_points  token share served from within x points of the top
    effective_number     effective number of deployed models and firms

The snapshot panel is loaded once and shared, and the event panel is built once
on the first figure that needs it.

Usage:
    python3 scripts/make_figures.py --figure all
    python3 scripts/make_figures.py --figure cosine_kde
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import gaussian_kde

import catchup
import crowding
import similarity as S
from common import CORE, PLOTS

INK, MUTED, GRID, BASELINE, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7', '#ffffff'
BLUE, ORANGE = '#2a78d6', '#eb6834'
N_CLUSTERS = 4

# Okabe-Ito (colour-vision-safe); a firm keeps its colour across figures.
FIRM_COLOR = {
    'openai': '#E69F00', 'anthropic': '#CC79A7', 'google': '#0072B2',
    'meta': '#009E73', 'mistral': '#56B4E9', 'xai': '#D55E00',
    'deepseek': '#B8A800', 'alibaba': '#000000',
}

# Firms outside that list are drawn in a neutral slate: with a derived roster
# there are too many for a distinct colour each, and the named firms are the
# ones the text refers to.
OTHER_FIRM = '#9aa3ad'

# gap_vs_dtheta.pdf and beta_forest.pdf are placed side by side in the text, so
# they are drawn at one size: same width, same height, same font scale. Text
# then comes out the same size in both without either being rescaled in LaTeX.
PAIR_WIDTH = 5.2
PAIR_SIZE = (3.8, 2.7)
PAIR_FONT = 4

# top_distribution.pdf, usage_within_points.pdf and effective_number.pdf are
# laid out as a row of three. They are drawn one-third of the width a
# three-panel figure gets here, so including each at 0.33\textwidth renders
# their text at the same size as a three-panel strip. The font offset keeps
# labels legible once the panel is this narrow.
TRIO_SIZE = (PAIR_WIDTH * 1.35 / 3, 1.75)
TRIO_FONT = -2

# residual_cosine_kde.pdf and residual_cosine_heatmap.pdf are the other pair.
# Both are set short; type sizes are left alone, so shortening them raises the
# text size relative to the panel rather than lowering it.
#
# `save` crops with bbox_inches='tight', so what a figure renders at is its
# cropped aspect times the width LaTeX gives it, not its figsize. The KDE
# crops to 3.87 x 1.54in, i.e. 0.397, and the heatmap to 3.19 x 2.42in, i.e.
# 0.759. The heatmap's matrix is square and its aspect is therefore fixed: it
# can only be made shorter by being made narrower.
#
# Side by side across the full text width, equal heights need widths in the
# ratio a_heat : a_kde, which is 0.66 : 0.34, and both then render 0.26 of the
# text width tall. That is the floor for this pair -- at full width the square
# matrix sets it -- and the text lays them out at those two numbers.
KDE_W, KDE_H = 4.0, 1.15
HEAT_H = 2.66
HEAT_FONT = 16

# Fixed y limits for beta_forest.pdf, so a wide interval cannot flatten the rest.
YLIM = (-2.0, 2.0)

DISPLAY_ORG = {
    'openai': 'OpenAI', 'anthropic': 'Anthropic', 'google': 'Google',
    'alibaba': 'Alibaba', 'meta': 'Meta', 'mistral': 'Mistral', 'xai': 'xAI',
    'deepseek': 'DeepSeek', 'nvidia': 'NVIDIA', 'amazon': 'Amazon',
    'stepfun': 'StepFun', 'cohere': 'Cohere', 'microsoft': 'Microsoft',
    'ibm': 'IBM', 'minimax': 'MiniMax', 'moonshot': 'Moonshot', 'zai': 'Z.ai',
    'meituan': 'Meituan', 'tencent': 'Tencent', 'ant-group': 'Ant Group',
    'baidu': 'Baidu', 'xiaomi': 'Xiaomi', 'bytedance': 'ByteDance',
    'allenai': 'AI2', 'thinky': 'Thinky', 'upstage': 'Upstage',
    'reka ai': 'Reka AI',
}
DISPLAY_CAT = {
    'coding': 'Coding', 'math': 'Math', 'hard_prompts': 'Hard Prompts',
    'instruction_following': 'Instruction Following',
    'creative_writing': 'Creative Writing', 'longer_query': 'Longer Query',
    'multi_turn': 'Multi-Turn', 'english': 'English',
}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 11, 'text.color': INK,
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK, 'axes.labelsize': 12,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'axes.linewidth': 0.7, 'pdf.fonttype': 42,
})


def firm_color(org):
    """Colour for an organization, neutral if it has no assigned one

    Arguments:
        org: String, the lowercase organization key

    Returns: String hex colour"""
    return FIRM_COLOR.get(org, OTHER_FIRM)


def label_org(org):
    """Display name for an organization

    Arguments:
        org: String, the lowercase organization key

    Returns: String, title-cased fallback if the name is not in DISPLAY_ORG"""
    return DISPLAY_ORG.get(org, org.replace('-', ' ').title())


def strip_axes(ax, keep=('left', 'bottom')):
    """Hide the spines a plot does not need

    Arguments:
        ax: Axes to modify
        keep: Tuple of spine names to leave visible

    Returns: None; modifies ax in place"""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
    ax.tick_params(length=3, width=0.6)


def bigger_text(delta=PAIR_FONT):
    """rcParams with every text size raised by the same number of points

    Arguments:
        delta: Number of points to add to each size

    Returns: Dict suitable for plt.rc_context"""
    keys = ('font.size', 'axes.labelsize', 'axes.titlesize',
            'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize')
    base = plt.rcParams['font.size']
    scalings = matplotlib.font_manager.font_scalings
    sizes = {}
    for k in keys:
        v = plt.rcParams[k]
        sizes[k] = (base * scalings[v] if isinstance(v, str) else v) + delta
    return sizes


def save(fig, name):
    """Write a figure into plots/ and close it

    Arguments:
        fig: Figure to save
        name: String filename, including the .pdf suffix

    Returns: None; prints the path written"""
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / name
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path.relative_to(PLOTS.parent)}')


def per_firm_fits(events):
    """`catchup.beta_per_firm`, computed once per event panel

    gap_vs_dtheta and beta_forest draw the same estimates two ways, and the
    fits take about a second, so the second figure reuses the first's.

    Arguments:
        events: Event panel from catchup.build_event_panel

    Returns: Dict keyed by firm, as `catchup.beta_per_firm` returns it"""
    if 'per_firm_fits' not in events.attrs:
        events.attrs['per_firm_fits'] = catchup.beta_per_firm(events)
    return events.attrs['per_firm_fits']


def make_cosine_kde(panel):
    """Distribution of pairwise residual cosines against the permutation null

    The two curves sitting on top of each other is the result: firms' residual
    profiles are no more alike than a within-category shuffle makes them. The
    dotted lines mark the two means, which differ by less than 0.005.

    Arguments:
        panel: (d, orgs) from similarity.load_default_panel

    Returns: None; writes residual_cosine_kde.pdf"""
    d, orgs = panel
    observed, null, _ = S.panel_cosines(d, orgs)
    print(f'  observed mean {observed.mean():+.4f}  null mean {null.mean():+.4f}'
          f'  (n={len(observed):,})')

    fig, ax = plt.subplots(figsize=(KDE_W, KDE_H))
    xs = np.linspace(-1, 1, 500)
    for values, color, style, label in [
            (null, MUTED, (0, (3, 1.5)), 'permutation null'),
            (observed, BLUE, '-', 'observed')]:
        density = gaussian_kde(values, bw_method=0.15)(xs)
        ax.fill_between(xs, density, color=color, alpha=0.17, lw=0)
        ax.plot(xs, density, color=color, lw=1.5, ls=style, label=label)
        ax.axvline(values.mean(), color=color, lw=0.9, ls=':')

    ax.set_xlim(-1, 1)
    # Headroom for the legend. Both densities are broad, so there is no empty
    # corner to drop it into; the alternative, anchoring it above the frame,
    # added 0.4in to the cropped height, which at this panel's height is a
    # quarter of the figure. Stretching the y axis is the cheaper of the two.
    peak = max(gaussian_kde(v, bw_method=0.15)(xs).max() for v in (null, observed))
    ax.set_ylim(0, peak * 1.42)
    ax.set_xlabel(r'$\cos(R_i, R_{i\prime})$')
    ax.set_ylabel('density')
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    strip_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc='upper left', ncol=2,
              handlelength=1.6, borderaxespad=0.3, columnspacing=1.2)
    save(fig, 'residual_cosine_kde.pdf')


def make_cosine_heatmap(panel):
    """Pairwise cosine averaged over the panel, hierarchically ordered

    Averaged rather than taken at one snapshot: a single date is one draw, and
    a pair's cosine moves a lot between snapshots. Each cell averages only the
    dates both firms were present for, so the counts are printed alongside.

    Ordering by the clustering is what makes the blocks visible; in firm order
    the same matrix looks like noise. Read it against the null, not against 0:
    the off-diagonal mean is pinned near -1/(N_t-1) at each snapshot, averaged
    the same way the matrix is.

    Arguments:
        panel: (d, orgs) from similarity.load_default_panel

    Returns: None; writes residual_cosine_heatmap.pdf"""
    d, orgs = panel
    names, C, counts, null = S.mean_cosine_matrix(d, orgs)
    off = ~np.eye(len(names), dtype=bool)
    print(f'  {len(names)} firms averaged over {d.dt.nunique()} snapshots; '
          f'per-pair count median {int(np.median(counts[off]))} '
          f'(min {counts[off].min()}, max {counts[off].max()})')
    print(f'  mean off-diagonal {C[off].mean():+.4f}, '
          f'mechanical null {null:+.4f}')

    dist = np.clip(1 - C, 0, 2)
    np.fill_diagonal(dist, 0)
    order = dendrogram(linkage(squareform(dist, checks=False), method='average'),
                       no_plot=True)['leaves']
    Co = C[np.ix_(order, order)]

    # The figure carries no firm labels, so membership is reported here: this
    # print is the source for the appendix table.
    groups = fcluster(linkage(squareform(dist, checks=False), method='average'),
                      N_CLUSTERS, criterion='maxclust')
    resid = S.residual_panel(d, names)
    profile = resid.groupby(['org', 'category']).residual.mean().unstack()[CORE]
    for g in range(1, N_CLUSTERS + 1):
        members = [names[i] for i in range(len(names)) if groups[i] == g]
        block = C[np.ix_([names.index(m) for m in members],
                         [names.index(m) for m in members])]
        within_cos = block[~np.eye(len(members), dtype=bool)]
        mean_profile = profile.loc[members].mean()
        strong = mean_profile.sort_values(ascending=False)
        print(f'  cluster {g} (n={len(members)}, within-cos '
              f'{within_cos.mean():+.2f}): {", ".join(label_org(m) for m in members)}')
        print(f'      strongest {DISPLAY_CAT[strong.index[0]]} {strong.iloc[0]:+.0f}, '
              f'{DISPLAY_CAT[strong.index[1]]} {strong.iloc[1]:+.0f} | '
              f'weakest {DISPLAY_CAT[strong.index[-1]]} {strong.iloc[-1]:+.0f}, '
              f'{DISPLAY_CAT[strong.index[-2]]} {strong.iloc[-2]:+.0f}')

    # No firm labels: at the width this runs beside the KDE they render at ~5pt
    # and are unreadable anyway, and dropping them lets the matrix itself fill
    # the panel. Cluster membership is reported in the appendix instead.
    fig, ax = plt.subplots(figsize=(4.9, HEAT_H))
    mesh = ax.pcolormesh(Co, cmap='RdBu_r', vmin=-1, vmax=1, edgecolors=SURFACE,
                         linewidth=0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    # The colorbar carries the only text in this figure, and the panel is
    # rescaled harder than the KDE beside it -- a third of the text width
    # against two thirds -- so it is set larger here to come out the same size
    # on the page. At the widths below it renders at about 9.5pt against the
    # KDE's 9.8pt. Widening the colorbar also lowers the figure's aspect
    # slightly, which is why HEAT_FONT appears in the width split.
    cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r'$\cos(R_i, R_{i\prime})$', fontsize=HEAT_FONT)
    cb.ax.tick_params(labelsize=HEAT_FONT - 2)
    cb.outline.set_visible(False)
    save(fig, 'residual_cosine_heatmap.pdf')


def trailing_run(series):
    """How long the residual has held its current sign, in snapshots

    Arguments:
        series: Residual series for one (org, category), NaNs already dropped

    Returns: Int, length of the run of one sign ending at the last value"""
    same_sign = np.sign(series) == np.sign(series.iloc[-1])
    return int(same_sign.iloc[::-1].cumprod().sum())


def make_residual_timeseries(panel, n_highlight=2, smooth=5):
    """Every (org, category) residual over the panel, two series highlighted

    Highlighted are the pairs with the longest run of one sign, which is the
    closest thing in the data to a firm holding a specialization. Everything
    else is drawn faintly to show how little structure the rest carries.

    Arguments:
        panel: (d, orgs) from similarity.load_default_panel
        n_highlight: Int, how many series to pick out
        smooth: Int, rolling window in snapshots

    Returns: None; writes residual_timeseries.pdf"""
    d, orgs = panel
    rp = S.residual_panel(d, orgs)
    wide = rp.pivot_table(index='dt', columns=['org', 'category'], values='residual')
    wide = wide.rolling(smooth, min_periods=2).mean()

    streaks = {col: trailing_run(wide[col].dropna()) for col in wide.columns
               if wide[col].notna().sum() > smooth}
    top = sorted(streaks, key=streaks.get, reverse=True)[:n_highlight]
    print('  highlighted:', ', '.join(f'{label_org(o)}/{DISPLAY_CAT.get(c, c)}'
                                      for o, c in top))

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for col in wide.columns:
        if col not in top:
            ax.plot(wide.index, wide[col], color=BASELINE, lw=0.5, alpha=0.5)
    for col, color in zip(top, (BLUE, ORANGE)):
        ax.plot(wide.index, wide[col], color=color, lw=1.8,
                label=f'{label_org(col[0])} / {DISPLAY_CAT.get(col[1], col[1])}')

    ax.axhline(0, color=MUTED, lw=0.8, ls=':')
    ax.set_ylabel('two-way residual (rating points)')
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    strip_axes(ax)
    ax.legend(frameon=False, fontsize=10, loc='upper center', ncol=2,
              bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout(pad=0.6)
    save(fig, 'residual_timeseries.pdf')


def make_gap_vs_dtheta(events):
    """Improvement against the gap, one noise-corrected line per firm

    Both axes are residualized on event and category, so the comparison is
    "within this release, which category lagged more" rather than how big the
    release was overall. Each firm's line is its own beta through the origin.
    The roster is large enough that labelling every firm is unreadable, so all
    firm lines are drawn as one band and only the pooled fit is called out.
    Singling out individual firms here invited reading a single line as a
    result; the per-firm estimates and their intervals are in beta_forest.pdf.

    Arguments:
        events: Event panel from catchup.load_events

    Returns: None; writes gap_vs_dtheta.pdf"""
    fits = per_firm_fits(events)
    order = sorted(fits, key=lambda f: -fits[f]['corrected'])
    for f in order:
        r = fits[f]
        print(f'  {f:10s} corrected {r["corrected"]:+.2f} (se {r["se_corrected"]:.2f})  '
              f'raw {r["beta"]:+.2f}  events {r["events"]}')

    pooled = catchup.fit_beta(events)
    gaps = np.concatenate([fits[f]['gap'] for f in order])
    dthetas = np.concatenate([fits[f]['dtheta'] for f in order])
    # cap at the 97th percentile: a few single-event outliers otherwise stretch
    # the lines into a fan that is mostly extrapolation
    xmax, ymax = np.percentile(np.abs(gaps), 97), np.percentile(np.abs(dthetas), 97)

    with plt.rc_context(bigger_text()):
        fig, ax = plt.subplots(figsize=PAIR_SIZE)
        grid = np.linspace(-xmax, xmax, 50)

        ax.scatter(gaps, dthetas, s=7, color=OTHER_FIRM, alpha=0.30, lw=0, zorder=2)
        for firm in order:
            ax.plot(grid, fits[firm]['corrected'] * grid, color=OTHER_FIRM, lw=1.0,
                    alpha=0.85, zorder=3)
        ax.plot(grid, pooled['corrected'] * grid, color=INK, lw=2.4, zorder=5,
                label=rf'pooled $\hat\beta$={pooled["corrected"]:+.2f}')

        ax.axhline(0, color=BASELINE, lw=0.8, zorder=1)
        ax.axvline(0, color=BASELINE, lw=0.8, zorder=1)
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-ymax, ymax)
        ax.set_xlabel(r'$g_{i,k}$')
        ax.set_ylabel(r'$\Delta\theta_{i,k}$')
        ax.grid(color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        strip_axes(ax)
        # The lines fan out through the origin, so every corner of the axes is
        # occupied; the legend goes above the frame instead of over the data.
        ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0),
                  ncol=1, handlelength=1.1, columnspacing=1.2,
                  handletextpad=0.5, borderaxespad=0.1)
        fig.tight_layout(pad=0.4)
        save(fig, 'gap_vs_dtheta.pdf')


def make_beta_forest(events):
    """Noise-corrected beta per firm with 95% intervals, coloured to match

    Firms run along the x axis and beta up the y axis, so the panel is wide
    rather than tall and sits beside gap_vs_dtheta.pdf at the same size. The y
    axis is fixed to YLIM; intervals running past it are drawn with an arrow.

    Arguments:
        events: Event panel from catchup.load_events

    Returns: None; writes beta_forest.pdf"""
    fits = per_firm_fits(events)
    order = sorted(fits, key=lambda f: -fits[f]['corrected'])

    with plt.rc_context(bigger_text()):
        # gap_vs_dtheta hangs its legend outside the frame, which tight bounds
        # then widen; the nudge lands both PDFs on the same final width so the
        # pair renders at one scale when each is set to half the text width.
        fig, ax = plt.subplots(figsize=(PAIR_SIZE[0] + 0.19, PAIR_SIZE[1]))
        for i, firm in enumerate(order):
            r, color = fits[firm], firm_color(firm)
            lo = r['corrected'] - 1.96 * r['se_corrected']
            hi = r['corrected'] + 1.96 * r['se_corrected']
            # One or two firms have intervals several times the width of the
            # rest, and letting them set the limits squashes every other firm
            # into the middle of the panel. The axis is fixed instead and the
            # long intervals are drawn to the edge with an arrow, which says
            # "continues" without costing the other ten firms their resolution.
            ax.plot([i, i], [max(lo, YLIM[0]), min(hi, YLIM[1])], color=color,
                    lw=3.0, zorder=2, solid_capstyle='round')
            for bound, end in ((YLIM[0], lo), (YLIM[1], hi)):
                if (end - bound) * np.sign(bound) > 0:
                    ax.plot([i], [bound], color=color, ms=6, zorder=3,
                            marker='v' if bound < 0 else '^')
            # A point estimate off the axis is printed rather than drawn: a dot
            # pinned to the frame reads as an estimate sitting at the limit.
            if YLIM[0] <= r['corrected'] <= YLIM[1]:
                ax.scatter([i], [r['corrected']], color=color, s=45, zorder=3,
                           edgecolor=SURFACE, lw=1.0)
            else:
                top = r['corrected'] > YLIM[1]
                ax.annotate(f'{r["corrected"]:+.1f}', xy=(i, YLIM[1 if top else 0]),
                            xytext=(0, -11 if top else 11), textcoords='offset points',
                            ha='center', va='center', fontsize=8, color=color)

        ax.axhline(0, color=BASELINE, lw=0.9, zorder=1)
        ax.axhline(1, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.set_xticks(range(len(order)))
        # Twelve firm names will not fit horizontally at this width, and at 45
        # degrees the long ones still overlap, so they are set vertically.
        ax.set_xticklabels([label_org(f) for f in order], rotation=90,
                           ha='center', va='top')
        ax.set_xlim(-0.7, len(order) - 0.3)
        ax.set_ylim(*YLIM)
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_ylabel(r'$\hat\beta$ (95% CI)')
        ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        strip_axes(ax, keep=('left',))
        ax.tick_params(axis='x', length=0)
        fig.tight_layout(pad=0.4)
        save(fig, 'beta_forest.pdf')


def make_beta_by_level(events):
    """Beta once the outgoing model's overall rating is controlled for

    A weak model trails everywhere and has more room to improve, so beta could
    be level rather than gap. Left: the pooled fit with and without the
    outgoing flagship's overall rating theta_i as a linear covariate. Right:
    beta fitted freely inside each quartile of that rating, which lets the
    level effect take any shape rather than assuming it is linear.

    The control is the OUTGOING model's rating. The incoming model's rating is
    downstream of the outcome -- dtheta is measured against it -- so
    conditioning on it is a bad control and inflates beta.

    Arguments:
        events: Event panel from catchup.load_events

    Returns: None; writes beta_by_level.pdf"""
    pooled, by_bin = catchup.beta_by_level(events)
    print(pooled[['corrected', 'se_corrected', 'beta', 'se', 'n']].round(3).to_string())
    # theta_{ikt} is reported by `beta_by_level` but kept out of the figure: the
    # gap is built from theta_{ikt} by subtraction, so that row answers a
    # different question from the other two and reads as a contradiction beside
    # them. `level_collinearity` is where it belongs.
    pooled = pooled.drop(index='linear in theta_ikt', errors='ignore')
    print(by_bin[['corrected', 'se_corrected', 'level_mid', 'events', 'n']]
          .round(3).to_string())

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 1.4), width_ratios=[1, 1.1])

    ypos = np.arange(len(pooled))
    ax.hlines(ypos, pooled.corrected - 1.96 * pooled.se_corrected,
              pooled.corrected + 1.96 * pooled.se_corrected, color=INK, lw=1.4)
    ax.plot(pooled.corrected, ypos, 'o', ms=6, color=BLUE, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.8, ls=':')
    ax.set_yticks(ypos)
    ax.set_yticklabels(pooled.index, fontsize=10)
    ax.set_ylim(-0.6, len(pooled) - 0.4)
    ax.set_xlabel(r'$\hat\beta$')
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    strip_axes(ax)

    xs = by_bin.level_mid.to_numpy(float)
    ax2.errorbar(xs, by_bin.corrected, yerr=1.96 * by_bin.se_corrected, fmt='o',
                 ms=5, color=ORANGE, ecolor=INK, elinewidth=1.2, capsize=3)
    pooled_beta = pooled.loc['no control', 'corrected']
    ax2.axhline(pooled_beta, color=BLUE, lw=1.2)
    # The gap between the first two bins is the one stretch of the panel with
    # no marker and no interval in it, so the label goes there, sitting just
    # above the line rather than on top of it.
    ax2.annotate('pooled', xy=(xs[:2].mean(), pooled_beta), xytext=(0, 4),
                 textcoords='offset points', ha='center', va='bottom',
                 fontsize=9.5, color=BLUE)
    ax2.axhline(0, color=MUTED, lw=0.8, ls=':')
    ax2.set_xlabel(r'outgoing model rating ($\theta_i$, bin median)')
    ax2.set_ylabel(r'$\hat\beta$')
    ax2.grid(color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    strip_axes(ax2)

    fig.tight_layout(pad=0.6)
    save(fig, 'beta_by_level.pdf')


def _stars(t):
    """Significance markers for a t statistic

    Arguments:
        t: Float t statistic

    Returns: String of LaTeX stars, empty when |t| < 1.96"""
    a = abs(t)
    return '$^{***}$' if a >= 2.58 else '$^{**}$' if a >= 1.96 else ''


def make_confounds_table(events):
    """Beta under each robustness cut, as a LaTeX table

    Covers the three confounds the text raises: which rival defines the gap,
    when the release happened, and the mechanical regression to the mean that
    the shared baseline rating induces. The last is measured on the raw battle
    dump rather than assumed, and reported in its own panel because it is a
    bias in beta, not another estimate of beta.

    Arguments:
        events: Event panel from catchup.load_events

    Returns: None; prints LaTeX to stdout"""
    table = catchup.robustness_table(events)
    panels = [('baseline', 'Baseline'), ('benchmark', 'Comparator firm'),
              ('year', 'Release year'), ('sample split', 'Split-sample'),
              ('release', 'Release effect')]

    lines = [r'\begin{tabular}{llcrr}', r'\toprule',
             r' & & $\hat\beta$ (s.e.) & Events & $N$ \\', r'\midrule']
    for i, (group, heading) in enumerate(panels):
        rows = table[table.group == group]
        if rows.empty:
            continue
        if i:
            lines.append(r'\addlinespace')
        for j, (_, r) in enumerate(rows.iterrows()):
            head = rf'\emph{{{heading}}}' if j == 0 else ''
            lines.append(
                f'{head} & {r.split} & '
                f'{r.corrected:+.3f}{_stars(r.t_corrected)} ({r.se_corrected:.3f}) & '
                f'{int(r.events)} & {int(r.n)} \\\\')

    bias = catchup.split_half_bias()
    if bias is not None:
        lines += [r'\midrule',
                  f'\\emph{{Regression to the mean}} & measured spurious slope & '
                  f'{bias["measured"]:+.3f} ({bias["measured_sd"]:.3f}) & '
                  f'{bias["splits"]} splits & {bias["battles"]:,} \\\\',
                  f' & predicted by the correction & {bias["predicted"]:+.3f} & '
                  f'-- & -- \\\\']
    lines += [r'\bottomrule', r'\end{tabular}']
    print('\n'.join(lines))
    if bias is not None:
        print(f'\n  [split-half: measured {bias["measured"]:+.3f} '
              f'(sd {bias["measured_sd"]:.3f}), EIV-predicted {bias["predicted"]:+.3f}, '
              f'{bias["splits"]} splits over {bias["battles"]:,} battles]')


def make_spec_ladder_table(events):
    """Beta under progressively more demanding fixed effects, as LaTeX

    Arguments:
        events: Event panel from catchup.load_events

    Returns: None; prints LaTeX to stdout"""
    ladder = catchup.fe_ladder(events)
    labels = {'firm': 'Firm FE', 'cat': 'Category FE', 'q': 'Time (quarter) FE',
              'ik': r'Firm$\times$Category FE', 'kt': r'Category$\times$Time FE',
              'event': r'Release FE ($\delta_e$)'}
    heads = ['Pooled', '(A) Additive FE', r'(B) Firm $\times$ Cat.',
             r'(C) (B) + Cat. $\times$ Time', r'(D) (C) + Release']

    def row(name, cells):
        return f'{name} & ' + ' & '.join(cells) + r' \\'

    lines = [r'\begin{tabular}{l' + 'c' * len(ladder) + '}', r'\toprule',
             row('', heads[:len(ladder)]), r'\midrule',
             row(r'$\hat\beta$ (Gap)',
                 [f'{r.corrected:+.3f}{_stars(r.t_corrected)}'
                  for _, r in ladder.iterrows()]),
             row('', [f'({r.se_corrected:.3f})' for _, r in ladder.iterrows()]),
             r'\addlinespace',
             row(r'$\hat\beta$ uncorrected',
                 [f'{r.beta:+.3f}' for _, r in ladder.iterrows()]),
             row('', [f'({r.se:.3f})' for _, r in ladder.iterrows()]),
             r'\midrule']
    for key, label in labels.items():
        lines.append(row(label, [r'\checkmark' if key in r.fe else '--'
                                 for _, r in ladder.iterrows()]))
    lines += [r'\midrule',
              row('$N$', [str(int(r.n)) for _, r in ladder.iterrows()]),
              row('Events', [str(int(r.events)) for _, r in ladder.iterrows()]),
              r'\bottomrule', r'\end{tabular}']
    print('\n'.join(lines))


def _crowding_panels(series, keys, titles, ylabel, color, name, note=None):
    """Three time-series panels sharing an x axis, one per threshold or rank

    Arguments:
        series: DataFrame indexed by snapshot, one column per key
        keys: Column keys to draw, left to right
        titles: Panel titles, same order as keys
        ylabel: Label for the leftmost y axis
        color: Line colour
        name: Output filename
        note: Optional line printed under the axes

    Returns: None; writes `name` into plots/"""
    fig, axes = plt.subplots(1, len(keys), figsize=(PAIR_WIDTH * 1.35, 1.25),
                             sharex=True)
    x = series.index
    for ax, key, title in zip(axes, keys, titles):
        y = series[key]
        ax.plot(x, y, color=color, lw=1.6, zorder=3)
        ax.fill_between(x, 0, y, color=color, alpha=0.10, zorder=2)
        ax.set_title(title, fontsize=11, pad=4)
        ax.set_ylim(0, float(np.nanmax(y)) * 1.12)
        ax.grid(color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        strip_axes(ax)
        # Half-year ticks: a panel this narrow fits about five dates.
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        ax.tick_params(axis='x', labelsize=9)
    axes[0].set_ylabel(ylabel)
    if note:
        fig.text(0.5, -0.06, note, ha='center', fontsize=9, color=MUTED)
    fig.tight_layout(pad=0.6)
    save(fig, name)


def make_top_rank_gaps():
    """Rating distance from the category leader to the r-th best model

    The mirror image of the count figure, and immune to the growth of the pool
    that inflates it: a fixed rank, a measured distance.

    Returns: None; writes top_rank_gaps.pdf. Reads its own series from the raw
        leaderboard rather than the shared snapshot panel, because the crowding
        measures count every rated model and not only the flagships"""
    _, gaps, _ = crowding.load_series()
    print('  gaps: ' + ', '.join(
        f'r={r} {gaps[r].iloc[0]:.0f} -> {gaps[r].iloc[-1]:.0f}'
        for r in crowding.RANKS))

    _crowding_panels(
        gaps, list(crowding.RANKS),
        [rf'$\theta_{{(1),k}} - \theta_{{({r}),k}}$' for r in crowding.RANKS],
        'rating points', ORANGE, 'top_rank_gaps.pdf')


def trio_axes():
    """A single panel sized to sit one-third of the way across the text

    Returns: (fig, ax) at TRIO_SIZE, gridded and stripped like the rest"""
    fig, ax = plt.subplots(figsize=TRIO_SIZE)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    strip_axes(ax)
    return fig, ax


def make_top_distribution():
    """Shortfall from the category leader by quantile of the field

    Separates a tightening top from a growing pool. A quantile is a position in
    the field whatever its size, so if the extra models were simply more of the
    same the two curves would coincide.

    The shortfall is in rating points. The ratio to the leader's rating, which
    this figure used to carry beside it, is still printed: it is the same curve
    divided by an arbitrary origin, so it says nothing the points version does
    not and it made the figure twice as wide.

    Returns: None; writes top_distribution.pdf"""
    ratio, points = crowding.load_profiles()
    years = list(ratio.columns)

    for q in (0.05, 0.10, 0.25, 0.50):
        print(f'  q={q:.2f}  ratio ' + ' '.join(f'{y}:{ratio[y].loc[q]:.4f}' for y in years)
              + '  |  points ' + ' '.join(f'{y}:{points[y].loc[q]:.1f}' for y in years))

    with plt.rc_context(bigger_text(TRIO_FONT)):
        fig, ax = trio_axes()
        colors = {years[0]: MUTED, years[-1]: BLUE}
        for y in years:
            ax.plot(points.index, points[y], color=colors[y], lw=1.8,
                    label=str(y), zorder=3)
        ax.set_xlabel('quantile of the field')
        ax.set_ylabel(r'$\theta_{(1),k} - \theta$ (points)')
        ax.set_xlim(0, 1)
        ax.legend(frameon=False, loc='upper left')
        fig.tight_layout(pad=0.4)
        save(fig, 'top_distribution.pdf')


def make_usage_within_points():
    """Share of OpenRouter tokens served from within x points of the Arena top

    Drawn at the trio width. The matched-token share is marked because the
    curve cannot rise above it: a slug with no Arena name clears no
    threshold.

    Returns: None; writes usage_within_points.pdf"""
    share, coverage = crowding.usage_share_by_threshold()
    avg = share.mean()
    print('  token share within x points of top: ' +
          ', '.join(f'{x}pt {avg[x]:.3f}' for x in share.columns))
    print(f'  matched coverage {coverage.mean():.3f}')

    with plt.rc_context(bigger_text(TRIO_FONT)):
        fig, ax = trio_axes()
        xs = list(share.columns)
        ax.plot(xs, avg.values, color=BLUE, lw=1.8, marker='o', ms=3.5, zorder=3)
        ax.axhline(coverage.mean(), color=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=2)
        # Labelled at the left end, where the curve is still near the floor;
        # at the right end it collides with both the curve and the top spine.
        ax.annotate('matched tokens', xy=(xs[0], coverage.mean()), xytext=(2, 4),
                    textcoords='offset points', ha='left', fontsize=7.5, color=MUTED)
        ax.set_xlabel('within $x$ points of the top')
        ax.set_ylabel('share of tokens')
        ax.set_ylim(0, 0.85)
        fig.tight_layout(pad=0.4)
        save(fig, 'usage_within_points.pdf')


def make_effective_number():
    """Effective number of deployed models and firms, month by month

    Drawn at the trio width. The last month is partial and is dropped, since
    a half month of tokens concentrates on whatever shipped first.

    Returns: None; writes effective_number.pdf"""
    eff = crowding.effective_counts()
    eff = eff[eff.index < eff.index.max()]
    first, last = eff.iloc[0], eff.iloc[-1]
    print(f'  effective models {first.eff_models:.1f} -> {last.eff_models:.1f}, '
          f'firms {first.eff_firms:.1f} -> {last.eff_firms:.1f} '
          f'({eff.index[0]:%b %Y} to {eff.index[-1]:%b %Y})')

    with plt.rc_context(bigger_text(TRIO_FONT)):
        fig, ax = trio_axes()
        ax.plot(eff.index, eff.eff_models, color=BLUE, lw=1.8, label='models', zorder=3)
        ax.plot(eff.index, eff.eff_firms, color=ORANGE, lw=1.8, label='firms', zorder=3)
        ax.set_ylabel('effective number')
        ax.set_ylim(0, float(eff.eff_models.max()) * 1.35)
        ax.set_xlim(eff.index[0], eff.index[-1])
        ax.legend(frameon=False, loc='upper left', ncol=2, handlelength=1.2,
                  columnspacing=1.0, borderaxespad=0.2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        fig.tight_layout(pad=0.4)
        save(fig, 'effective_number.pdf')


# What each figure reads: the snapshot panel, the event panel, or nothing --
# the crowding figures load their own series. main() builds each source at most
# once, and only when a figure in the run asks for it.
FIGURES = {
    'cosine_kde': (make_cosine_kde, 'panel'),
    'cosine_heatmap': (make_cosine_heatmap, 'panel'),
    'residual_timeseries': (make_residual_timeseries, 'panel'),
    'gap_vs_dtheta': (make_gap_vs_dtheta, 'events'),
    'beta_forest': (make_beta_forest, 'events'),
    'beta_by_level': (make_beta_by_level, 'events'),
    'confounds_table': (make_confounds_table, 'events'),
    'spec_ladder_table': (make_spec_ladder_table, 'events'),
    'top_rank_gaps': (make_top_rank_gaps, None),
    'top_distribution': (make_top_distribution, None),
    'usage_within_points': (make_usage_within_points, None),
    'effective_number': (make_effective_number, None),
}


def load_source(name, since, built):
    """One of the two panels the figures share, built on first use

    Arguments:
        name: Either 'panel' (the snapshot panel) or 'events'
        since: Left edge of the snapshot panel
        built: Dict the panels are cached in for the rest of the run

    Returns: (d, orgs) for 'panel', or the event panel DataFrame for 'events'"""
    if name not in built:
        if name == 'panel':
            d, orgs = S.load_default_panel(since=since)
            print(f'  panel: {len(orgs)} organizations, '
                  f'{d.dt.nunique()} snapshots since {since}')
            built[name] = (d, orgs)
        else:
            events = catchup.load_events()
            print(f'  event panel: {events.event.nunique()} events, '
                  f'{len(events)} obs, {events.firm.nunique()} firms')
            built[name] = events
    return built[name]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--figure', default='all', choices=list(FIGURES) + ['all'])
    parser.add_argument('--since', default='2025-01-01',
                        help='left edge of the panel')
    args = parser.parse_args()

    targets = FIGURES if args.figure == 'all' else {args.figure: FIGURES[args.figure]}
    built = {}
    for name, (draw, source) in targets.items():
        print(f'{name}:')
        if source is None:
            draw()
        else:
            draw(load_source(source, args.since, built))


if __name__ == '__main__':
    main()
