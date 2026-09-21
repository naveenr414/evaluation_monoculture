# Evaluation and Monoculture in LLMs

N firms invest across K categories, a central evaluator ranks them with category
weights w, and the question is what that ranking does to the diversity of what
gets built. This repository holds the empirical side: two experiments on the
LMArena per-category leaderboard and OpenRouter usage data.

## Layout

    scripts/          analysis and plotting code for experiments A and B
    raw_data/         data as downloaded, never edited
    processed_data/   derived panels (currently empty; see note below)
    plots/            figures, one per result
    _archive/         everything from the earlier passes, kept temporarily
    expA_writeup.md   Experiment A results
    expB_writeup.md   Experiment B results

## The experiments

**A. Catch-up and specialization.** Do firms close the gap to the category
leader, and does any firm-specific profile survive? Residual similarity is
measured against a permutation null, because two-way FE residuals have zero row
and column sums and so sit mechanically near -1/(N-1).

**B. The shrinking top.** Where usage sits relative to the evaluated frontier,
using OpenRouter token volume matched to Arena names, and how crowded the top of
each category has become. `crowding.py` measures that two ways: the number of
models within x% of the category leader, and the rating distance from the leader
to the r-th best model. The first grows with the pool of rated models and the
second does not, so they are reported together.

## Running

Scripts resolve paths from their own location, so they run from anywhere, and
every figure lands in `plots/`:

    python3 scripts/make_figures.py --figure all

`scripts/make_figures.py --figure <name>|all` runs one figure or all of them;
`confounds_table` and `spec_ladder_table` print LaTeX to stdout rather than
writing a PDF. Every number in a figure or table comes from `similarity.py`,
`catchup.py` or `crowding.py`, so figures and text cannot drift apart.

The split-half check inside `confounds_table` needs the raw battle dump at
`_archive/data/140k/data/`, which is not on GitHub. Without it that panel is
skipped and the rest of the table still builds.

`scripts/common.py` holds the shared layer: `load_panel`, the flagship and
changeover helpers (`flagships`, `changeovers`, `first_mature_snapshot`,
`event_label`), the `within` transform and `fe_regress`, and the CORE category
list. The firm roster is not hand-picked: `frontier_firms()` derives it from
the panel as the organizations with at least `MIN_EVENTS` (4) flagship
changeovers that survive into the event panel, which is the same minimum the
per-firm fits need. It is computed on first use and cached, and
`changeover_counts()` shows the counts behind it. The leaderboard parquet is
parsed once per process and every panel is a filtered view of that one read.
Every routine noise-corrects using the BT estimator's `variance` column.

## Data

`scripts/fetch_data.py` rebuilds `raw_data/` from scratch. It fetches whatever
is missing and leaves existing files alone, so it is safe to re-run:

    python3 scripts/fetch_data.py            # fill in what is missing
    python3 scripts/fetch_data.py --force    # re-fetch everything

The leaderboard half is public. The OpenRouter half needs a key:

    export OPENROUTER_API_KEY=...

Both sources move, so `--force` will change any number computed from them. The
script prints the panel dimensions at the end for exactly that reason. The
writeups in this repo were produced on the panel as of 2026-05-27; `raw_data/`
now runs to 2026-09-13, so their numbers will not reproduce exactly.

`raw_data/leaderboard.parquet` is the workhorse: 1.14M rows, per-category
ratings for 422 models from 62 organizations across 28 categories, 244 snapshots
from 2023-05 to 2026-09, with `variance` and `vote_count`. The `variance` column
is fully populated.

`raw_data/openrouter_daily.parquet` is model x day token volume with no category
breakdown: 31,000 rows, 408 models from 46 organizations over 620 days from
2025-01 to 2026-09, the top 50 models each day. Matching its slugs to Arena
names is handled by `scripts/or_arena_match.py`.

Two cautions. The `organization` field goes blank from 2026-04-02 onward --
about 1,300 rows per snapshot, 5.2% of the panel -- and distinct organizations
per snapshot fall from 56 to 28. Firm-keyed series therefore still stop at
2026-03-19 even though the panel now runs to September. And
`arena_leaderboard.parquet` (in `_archive/`) is overall-only, so it cannot be
used for profile work.

`processed_data/` is empty because no current script caches an intermediate;
every experiment recomputes from the raw panel.

## Archive

`_archive/` holds the earlier exploratory passes, the superseded drafts, and the
~2 GB of datasets those passes pulled. Nothing there is referenced by the
experiments above. The code is on GitHub; the data in `_archive/data/` is not,
so move it rather than delete it.
