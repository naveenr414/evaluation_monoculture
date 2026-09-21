"""Rebuild raw_data/ from scratch, and report what was fetched.

Two sources, both treated as read-only once written:

  raw_data/leaderboard.parquet
      Every historical LMArena leaderboard snapshot, one row per
      (snapshot, model, category), from the `lmarena-ai/leaderboard-dataset`
      Hugging Face dataset, `text` config, `full` split.  Public, no token.

  raw_data/openrouter_daily.parquet
      Daily token volume per model from OpenRouter's rankings dataset, one row
      per (date, model_permaslug).  Needs OPENROUTER_API_KEY in the environment.

Run with no arguments to fetch whatever is missing, or --force to re-fetch:

    python3 scripts/fetch_data.py
    python3 scripts/fetch_data.py --force

Both sources move.  The leaderboard grows a snapshot every few days and
OpenRouter grows a day at a time, so a re-fetch will shift any number computed
from them.  The summary printed at the end reports the panel dimensions for
exactly that reason -- it is the set of numbers the data section quotes.
"""
import os
import shutil
import sys
import time
from datetime import date

import pandas as pd
import requests
from huggingface_hub import hf_hub_download

from common import RAW

FORCE = '--force' in sys.argv

ARENA_REPO = 'lmarena-ai/leaderboard-dataset'
ARENA_FILE = 'text/full-00000-of-00001.parquet'

RANKINGS_URL = 'https://openrouter.ai/api/v1/datasets/rankings-daily'
OPENROUTER_START = date(2025, 1, 1)
REQUEST_PAUSE = 2.5           # 30 requests/minute is the documented ceiling


def fetch_arena():
    """Download the Arena leaderboard panel to a flat file in raw_data/

    The repo nests its configs, so we let hf_hub_download put the file in its
    own cache and copy it out under the name we actually want.

    Returns: Path to raw_data/leaderboard.parquet"""
    dest = RAW / 'leaderboard.parquet'
    if dest.exists() and not FORCE:
        print(f'  have {dest.name}')
        return dest

    print(f'  downloading {ARENA_FILE} ...')
    cached = hf_hub_download(repo_id=ARENA_REPO, repo_type='dataset',
                            filename=ARENA_FILE)
    shutil.copyfile(cached, dest)
    print(f'    -> {dest.name}')
    return dest


def month_starts(start, end):
    """First day of each month spanned by [start, end], plus start itself

    Arguments:
        start: date, first day to cover
        end: date, last day to cover

    Returns: DatetimeIndex of window start dates, ascending"""
    return pd.date_range(start, end, freq='MS').union(
        pd.DatetimeIndex([pd.Timestamp(start)])).sort_values()


def fetch_openrouter_window(session, key, start, end):
    """Fetch one window of daily token totals from the rankings dataset

    Arguments:
        session: requests.Session, reused across windows
        key: String, OpenRouter API key
        start: date, first day of the window, inclusive
        end: date, last day of the window, inclusive

    Returns: List of dicts, each with date, model_permaslug, total_tokens"""
    r = session.get(
        RANKINGS_URL,
        headers={'Authorization': f'Bearer {key}'},
        params={'start_date': str(start), 'end_date': str(end), 'period': 'day'},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()['data']


def fetch_openrouter():
    """Walk the OpenRouter history a month at a time and write one parquet

    Monthly windows keep each response small and the request count low enough
    that the rate limit never binds -- about 21 requests for the full history.
    The windows are month-aligned and do not overlap, but we deduplicate before
    writing anyway: the API is inclusive at both ends, so any future change to
    the windowing would silently double-count a boundary day.  The file this
    replaces had 700 such rows across 7 days.

    The per-day `other` row aggregates everything outside the top 50.  We drop
    it, so a daily total here is top-50 volume, not all OpenRouter volume, and
    anything dividing by it is a share of the top 50.

    Returns: Path to raw_data/openrouter_daily.parquet, or None if skipped"""
    dest = RAW / 'openrouter_daily.parquet'
    if dest.exists() and not FORCE:
        print(f'  have {dest.name}')
        return dest

    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        print('  SKIPPED: set OPENROUTER_API_KEY to fetch OpenRouter volume')
        return None

    today = date.today()
    starts = month_starts(OPENROUTER_START, today)

    session = requests.Session()
    rows = []
    for i, window_start in enumerate(starts):
        window_end = min((window_start + pd.offsets.MonthEnd(0)).date(), today)
        got = fetch_openrouter_window(session, key, window_start.date(), window_end)
        rows.extend(got)
        print(f'  {window_start:%Y-%m}  {len(got):>5} rows')
        if i < len(starts) - 1:
            time.sleep(REQUEST_PAUSE)

    d = pd.DataFrame(rows)
    d['date'] = pd.to_datetime(d.date)
    d['total_tokens'] = d.total_tokens.astype(float)
    d = d[d.model_permaslug != 'other']
    d = (d.drop_duplicates(subset=['date', 'model_permaslug'])
           .sort_values(['date', 'total_tokens'], ascending=[True, False])
           .reset_index(drop=True))

    d.to_parquet(dest, index=False)
    print(f'    -> {dest.name}')
    return dest


def describe_leaderboard(path):
    """Print the panel dimensions the data section quotes

    Arguments:
        path: Path to raw_data/leaderboard.parquet

    Returns: None; prints to stdout"""
    d = pd.read_parquet(path)
    dt = pd.to_datetime(d.leaderboard_publish_date)
    core = d[d.category != 'overall']

    print(f'  snapshots      {dt.nunique()}')
    print(f'  date range     {dt.min():%B %Y} through {dt.max():%B %Y}')
    print(f'  models         {d.model_name.nunique()}')
    print(f'  organizations  {d.organization.nunique()}')
    print(f'  categories     {core.category.nunique()}  '
          f'({d.category.nunique()} including "overall")')
    print(f'  rows           {len(d):,}')
    print(f'  variance       {d.variance.notna().mean():.1%} populated '
          f'({d[dt.dt.year >= 2024].variance.notna().mean():.1%} from 2024)')


def describe_openrouter(path):
    """Print the usage-panel dimensions the data section quotes

    Organizations are the slug prefix, so `anthropic/claude-3.5-sonnet` counts
    as anthropic.

    Arguments:
        path: Path to raw_data/openrouter_daily.parquet

    Returns: None; prints to stdout"""
    d = pd.read_parquet(path)
    orgs = d.model_permaslug.str.split('/').str[0]
    per_day = d.groupby('date').size()

    print(f'  days           {d.date.nunique()}')
    print(f'  date range     {d.date.min():%B %Y} through {d.date.max():%B %Y}')
    print(f'  models         {d.model_permaslug.nunique()}')
    print(f'  organizations  {orgs.nunique()}')
    print(f'  rows           {len(d):,}')
    print(f'  models per day median {per_day.median():.0f} '
          f'(min {per_day.min()}, max {per_day.max()})')


RAW.mkdir(parents=True, exist_ok=True)

print('Arena leaderboard:')
arena_path = fetch_arena()

print('\nOpenRouter daily volume:')
openrouter_path = fetch_openrouter()

print('\n' + '=' * 62)
print('ARENA PANEL')
print('=' * 62)
describe_leaderboard(arena_path)

if openrouter_path is not None:
    print('\n' + '=' * 62)
    print('OPENROUTER PANEL')
    print('=' * 62)
    describe_openrouter(openrouter_path)

print('\nraw_data/ is ready.')
