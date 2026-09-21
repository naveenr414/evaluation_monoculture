"""Capability by category for three firms, under monoculture and specialization.

The same picture twice, differing only in the numbers: bars are grouped by
category and coloured by firm, so monoculture is three equal bars in every
group and specialization is one firm owning each group. Both are schematic --
the levels are chosen, not estimated -- and both are drawn on the same y-axis
so the height of a bar means the same thing in either panel.

Colour keys the firm here, unlike allocation_shares.pdf where it keys the
category; these two figures are not meant to be read side by side.

Usage:
    python3 scripts/plot_profile_bars.py
"""
import numpy as np

from make_figures import MUTED, plt, save, strip_axes

CATEGORIES = ['Math', 'Code', 'Reasoning', '...']
FIRMS = ['Firm A', 'Firm B', 'Firm C']
FIRM_COLORS = ['#0072B2', '#E69F00', '#009E73']
FONT = 15

# Rows are firms, columns categories. Under monoculture the three firms sit on
# top of each other and only the category level moves; under specialization
# each category has an owner and the other two are well behind.
MONOCULTURE = np.array([
    [0.62, 0.68, 0.62, 0.40],
    [0.65, 0.63, 0.57, 0.38],
    [0.58, 0.60, 0.58, 0.41],
])
SPECIALIZED = np.array([
    [0.85, 0.14, 0.22, 0.30],
    [0.33, 0.80, 0.28, 0.16],
    [0.38, 0.40, 0.82, 0.24],
])


def draw(values, name):
    """One grouped bar chart, categories on x and firms within each group

    Arguments:
        values: Array of shape (firms, categories)
        name: String filename, including the .pdf suffix

    Returns: None; writes into plots/"""
    xs = np.arange(len(CATEGORIES))
    width = 0.26

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    for i, (firm, color) in enumerate(zip(FIRMS, FIRM_COLORS)):
        ax.bar(xs + (i - 1) * width, values[i], width=width * 0.92, color=color,
               label=firm, zorder=2)

    ax.set_xticks(xs, CATEGORIES, fontsize=FONT)
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.55, len(CATEGORIES) - 0.45)
    ax.set_ylabel('capability', fontsize=FONT)
    strip_axes(ax, keep=('left', 'bottom'))
    ax.tick_params(axis='x', length=4, color=MUTED)
    ax.legend(frameon=False, fontsize=FONT - 2, ncol=3, loc='lower center',
              bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=1.2)
    save(fig, name)


def main():
    for label, values, name in [
            ('monoculture', MONOCULTURE, 'profiles_monoculture.pdf'),
            ('specialized', SPECIALIZED, 'profiles_specialized.pdf')]:
        spread = np.abs(values - values.mean(axis=0)).mean()
        print(f'{label}: mean within-category spread across firms {spread:.2f}')
        draw(values, name)


if __name__ == '__main__':
    main()
