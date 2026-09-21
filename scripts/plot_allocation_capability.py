"""Three firms' budget allocations, and the capabilities those allocations buy.

Two schematic figures meant to be read in order. `allocation_shares.pdf` gives
each firm one unit of budget split across categories, so every bar is the same
length and only the split differs; `capability_profiles.pdf` is what comes out
after the production function, one row per firm.

The second figure is computed from the first here rather than drawn by hand, so
a firm that invests in a category cannot fail to be good at it.

Usage:
    python3 scripts/plot_allocation_capability.py
"""
import numpy as np

from make_figures import GRID, INK, MUTED, SURFACE, plt, save, strip_axes

CATEGORIES = ['Math', 'Code', 'Reasoning', 'English']
FIRMS = ['Firm A', 'Firm B', 'Firm C']

# Rows sum to 1. A and B chase the same split; C spends elsewhere.
SHARES = np.array([
    [0.45, 0.20, 0.20, 0.15],
    [0.42, 0.22, 0.21, 0.15],
    [0.22, 0.13, 0.35, 0.30],
])

# Diminishing returns: doubling the budget on a category buys less than twice
# the capability, which is what keeps any firm from covering everything.
ALPHA = 0.5

# Both figures run small in the paper, so everything is set well above the
# shared rcParams size; at this width the labels are the only text there is.
FONT = 17

# Okabe-Ito again, one colour per category rather than per firm.
CATEGORY_COLOR = ['#0072B2', '#D55E00', '#E69F00', '#009E73']


def capabilities(shares, alpha=ALPHA):
    """Capability in each category from the budget spent on it

    Arguments:
        shares: Array of shape (firms, categories), rows summing to 1
        alpha: Float exponent below 1, the strength of diminishing returns

    Returns: Array of shape (firms, categories)"""
    return shares ** alpha


def draw_shares(shares):
    """Budget splits as stacked unit bars, one per firm

    Arguments:
        shares: Array of shape (firms, categories), rows summing to 1

    Returns: None; writes allocation_shares.pdf"""
    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    left = np.zeros(len(FIRMS))
    ys = np.arange(len(FIRMS))
    for k, (name, color) in enumerate(zip(CATEGORIES, CATEGORY_COLOR)):
        ax.barh(ys, shares[:, k], left=left, color=color, height=0.68,
                label=name, zorder=2)
        left += shares[:, k]

    ax.set_yticks(ys, FIRMS, fontsize=FONT)
    ax.set_ylim(len(FIRMS) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.tick_params(axis='x', labelsize=FONT)
    ax.set_xlabel('share of budget', fontsize=FONT)
    strip_axes(ax, keep=('bottom',))
    ax.tick_params(axis='y', length=0)
    ax.legend(frameon=False, fontsize=FONT, ncol=len(CATEGORIES), loc='lower center',
              bbox_to_anchor=(0.5, 1.0), handlelength=1.1, columnspacing=1.6)
    save(fig, 'allocation_shares.pdf')


def draw_capabilities(C):
    """Capability profiles as a firm-by-category heatmap

    Sequential rather than diverging: these are levels, and there is no
    meaningful zero in the middle to diverge from.

    Arguments:
        C: Array of shape (firms, categories)

    Returns: None; writes capability_profiles.pdf"""
    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    mesh = ax.pcolormesh(C, cmap='Blues', vmin=0, vmax=C.max(),
                         edgecolors=SURFACE, linewidth=1.2)
    ax.set_xticks(np.arange(len(CATEGORIES)) + 0.5, CATEGORIES, fontsize=FONT)
    ax.set_yticks(np.arange(len(FIRMS)) + 0.5, FIRMS, fontsize=FONT)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.025)
    cb.set_label('capability', fontsize=FONT)
    cb.ax.tick_params(labelsize=FONT - 3)
    cb.outline.set_visible(False)
    save(fig, 'capability_profiles.pdf')


def main():
    assert np.allclose(SHARES.sum(axis=1), 1), SHARES.sum(axis=1)
    C = capabilities(SHARES)
    for name, row, cap in zip(FIRMS, SHARES, C):
        print(f'{name}: ' + ', '.join(f'{c} {s:.2f}->{v:.2f}'
                                      for c, s, v in zip(CATEGORIES, row, cap)))
    draw_shares(SHARES)
    draw_capabilities(C)


if __name__ == '__main__':
    main()
