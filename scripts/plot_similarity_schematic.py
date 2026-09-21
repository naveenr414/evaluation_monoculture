"""Two illustrative cosine heatmaps: a monoculture and a specialized market.

Neither is data. They are the two pictures `residual_cosine_heatmap.pdf` could
have shown, drawn at the same size, palette and scale so the real one can be
held against them: if every firm chases the same profile the matrix is solid
red, and if every firm has its own the off-diagonal is pale. The observed
matrix looks like the second, which is the point of putting them side by side.

Profiles here are firm capability profiles, not the two-way residuals of
`similarity.py` -- double-demeaning pins the mean cosine near -1/(N-1) and so
rules out the monoculture panel by construction.

Usage:
    python3 scripts/plot_similarity_schematic.py
"""
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

import similarity as S
from make_figures import SURFACE, plt, save

N_FIRMS, N_CATEGORIES = 25, 8
SEED = 11


def monoculture_profiles(rng):
    """Profiles that all point along one shared direction

    Every firm is the same profile up to a scale and a little noise, so the
    cosines sit near 1 regardless of how good the firm is: the level differs,
    the shape does not.

    Arguments:
        rng: numpy Generator

    Returns: Array of shape (N_FIRMS, N_CATEGORIES)"""
    shared = rng.normal(size=N_CATEGORIES)
    scale = rng.uniform(0.6, 1.4, size=(N_FIRMS, 1))
    return scale * shared + 0.28 * rng.normal(size=(N_FIRMS, N_CATEGORIES))


def specialized_profiles(rng):
    """Profiles drawn independently, one per firm

    Nothing is shared, so the cosines scatter around 0 with spread ~1/sqrt(K)
    -- the pale mottled matrix, which is also what the observed one looks like.

    Arguments:
        rng: numpy Generator

    Returns: Array of shape (N_FIRMS, N_CATEGORIES)"""
    return rng.normal(size=(N_FIRMS, N_CATEGORIES))


def draw(C, name):
    """Draw one cosine matrix in the style of the observed heatmap

    Ordered by the same average-linkage clustering, so any block structure
    shows up here as it would there.

    Arguments:
        C: Array of shape (firms, firms) of cosines
        name: String filename, including the .pdf suffix

    Returns: None; writes into plots/"""
    off = ~np.eye(len(C), dtype=bool)
    print(f'  off-diagonal mean {C[off].mean():+.3f}, sd {C[off].std():.3f}, '
          f'range [{C[off].min():+.2f}, {C[off].max():+.2f}]')

    dist = np.clip(1 - C, 0, 2)
    np.fill_diagonal(dist, 0)
    order = dendrogram(linkage(squareform(dist, checks=False), method='average'),
                       no_plot=True)['leaves']

    fig, ax = plt.subplots(figsize=(4.9, 3.05))
    mesh = ax.pcolormesh(C[np.ix_(order, order)], cmap='RdBu_r', vmin=-1, vmax=1,
                         edgecolors=SURFACE, linewidth=0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r'$\cos(R_i, R_{i\prime})$', fontsize=10)
    cb.outline.set_visible(False)
    save(fig, name)


def main():
    rng = np.random.default_rng(SEED)
    for label, profiles, name in [
            ('monoculture', monoculture_profiles, 'schematic_monoculture.pdf'),
            ('specialized', specialized_profiles, 'schematic_specialized.pdf')]:
        print(f'{label}:')
        draw(S.cosine_matrix(profiles(rng)), name)


if __name__ == '__main__':
    main()
