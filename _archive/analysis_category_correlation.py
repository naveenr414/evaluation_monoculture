"""
Category-level error correlation in Arena battle data.

Analogous to the BBH subtask analysis (Finding 7b) but using Arena battles
instead of benchmark accuracy. For each model, compute win-rate across prompt
categories; pairwise Spearman rho of these vectors measures whether models
have convergent strengths/weaknesses across categories.

Three analyses:
A. 2025 battle data — per-model win-rate vectors across 5 Arena categories
   (math, coding, creative writing, instruction following, general)
   → cross-family pairwise rho, category-level discrimination

B. 2023 battle data — keyword-classified categories (math/coding/general)
   → same metric, compare rho to 2025 cross-family finding

C. Category pair correlation — which pairs of categories are least discriminating
   (i.e., models that win in X always win in Y)?
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import glob, ast, re

BATTLE_MIN = 50   # min battles per model per category to include that cell
N_BOOT = 5000

# ── helpers ───────────────────────────────────────────────────────────────────

def spearman_ci(x, y, n_boot=N_BOOT, seed=42):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, len(x)
    rho, pval = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if len(set(x[idx])) < 2 or len(set(y[idx])) < 2:
            continue
        r, _ = stats.spearmanr(x[idx], y[idx])
        boot.append(r)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return rho, lo, hi, len(x)

ORG_MAP = {
    'claude': 'Anthropic', 'chatgpt': 'OpenAI', 'gpt': 'OpenAI',
    'o1': 'OpenAI', 'o3': 'OpenAI', 'o4': 'OpenAI',
    'gemini': 'Google', 'gemma': 'Google',
    'llama': 'Meta',
    'grok': 'xAI',
    'qwen': 'Alibaba', 'qwq': 'Alibaba',
    'deepseek': 'DeepSeek',
    'mistral': 'Mistral', 'magistral': 'Mistral',
    'command': 'Cohere',
    'amazon': 'Amazon',
    'kimi': 'Moonshot',
    'minimax': 'MiniMax',
    'hunyuan': 'Tencent',
}

def get_org(model_name):
    m = model_name.lower()
    for prefix, org in ORG_MAP.items():
        if m.startswith(prefix):
            return org
    return 'Other'

def compute_winrate_matrix(battles_df, categories, min_battles=BATTLE_MIN):
    """
    For each model, compute win rate per category.
    Returns model x category DataFrame (NaN where battle count < min_battles).
    """
    records = {}
    for cat in categories:
        sub = battles_df[battles_df['category'] == cat].copy()
        # Win = model_a wins or model_b wins (exclude ties for win rate)
        # code win as: model_a gets 1, tie gets 0.5, model_b gets 0
        sub['wa'] = sub['winner'].map({'model_a': 1.0, 'tie': 0.5, 'model_b': 0.0})
        sub['wb'] = 1.0 - sub['wa']
        # Per-model records
        a_records = sub[['model_a', 'wa']].rename(columns={'model_a': 'model', 'wa': 'w'})
        b_records = sub[['model_b', 'wb']].rename(columns={'model_b': 'model', 'wb': 'w'})
        combined = pd.concat([a_records, b_records])
        agg = combined.groupby('model')['w'].agg(['mean', 'count'])
        agg.columns = [f'{cat}_wr', f'{cat}_n']
        records[cat] = agg
    result = pd.concat([records[c][f'{c}_wr'] for c in categories], axis=1)
    result.columns = categories
    # Mask cells with too few battles
    for cat in categories:
        n_col = records[cat][f'{cat}_n'].reindex(result.index)
        result.loc[n_col < min_battles, cat] = np.nan
    return result

def pairwise_rho_analysis(wr_matrix, label=''):
    """
    Compute pairwise Spearman rho of win-rate vectors across all model pairs.
    Returns summary stats split by cross-org vs. same-org.
    """
    models = wr_matrix.dropna(thresh=3).index.tolist()  # need >=3 non-NaN categories
    print(f"\n{label}")
    print(f"  Models with >=3 valid categories: {len(models)}")

    orgs = {m: get_org(m) for m in models}

    cross_rhos, same_rhos, all_rhos = [], [], []
    cross_pairs, same_pairs = [], []

    for ma, mb in combinations(models, 2):
        va = wr_matrix.loc[ma]
        vb = wr_matrix.loc[mb]
        mask = ~(va.isna() | vb.isna())
        if mask.sum() < 3:
            continue
        va, vb = va[mask].values, vb[mask].values
        if len(set(va)) < 2 or len(set(vb)) < 2:
            continue
        rho, _ = stats.spearmanr(va, vb)
        all_rhos.append(rho)
        if orgs[ma] != orgs[mb]:
            cross_rhos.append(rho)
            cross_pairs.append((ma, mb, rho))
        else:
            same_rhos.append(rho)
            same_pairs.append((ma, mb, rho))

    def summarize(rhos, tag):
        if not rhos:
            print(f"  {tag}: no pairs")
            return
        arr = np.array(rhos)
        # bootstrap CI on mean
        rng = np.random.default_rng(42)
        boot_means = [rng.choice(arr, len(arr)).mean() for _ in range(N_BOOT)]
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        print(f"  {tag}: mean ρ = {arr.mean():.3f} [{lo:.3f}, {hi:.3f}]  "
              f"median = {np.median(arr):.3f}  n_pairs = {len(arr)}")

    summarize(all_rhos, "All pairs")
    summarize(cross_rhos, "Cross-org pairs")
    summarize(same_rhos, "Same-org pairs")

    # Top divergent pairs (lowest rho) — most discriminating
    if cross_pairs:
        cross_pairs.sort(key=lambda x: x[2])
        print(f"\n  Most category-discriminating cross-org pairs (lowest ρ):")
        for ma, mb, rho in cross_pairs[:8]:
            print(f"    {ma} vs {mb}: ρ = {rho:.3f}")
        print(f"\n  Most convergent cross-org pairs (highest ρ):")
        for ma, mb, rho in reversed(cross_pairs[-8:]):
            print(f"    {ma} vs {mb}: ρ = {rho:.3f}")

    return all_rhos, cross_rhos, same_rhos


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS A: 2025 battle data with native category tags
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("A. 2025 BATTLE DATA — native Arena category tags")
print("=" * 65)

files_25 = sorted(glob.glob('data/140k/data/*.parquet'))
df25 = pd.concat([pd.read_parquet(f) for f in files_25], ignore_index=True)

def parse_tag(t):
    if isinstance(t, dict): return t
    try: return ast.literal_eval(str(t))
    except: return {}

tags = df25['category_tag'].apply(parse_tag)
df25['_math'] = tags.apply(lambda x: x.get('math_v0.1', {}).get('math', False))
df25['_creative'] = tags.apply(lambda x: x.get('creative_writing_v0.1', {}).get('creative_writing', False))
df25['_if'] = tags.apply(lambda x: x.get('if_v0.1', {}).get('if', False))

def categorize_25(row):
    if row['_math']: return 'math'
    if row['_creative']: return 'creative_writing'
    if row['is_code']: return 'coding'
    if row['_if']: return 'instruction_following'
    return 'general'

df25['category'] = df25.apply(categorize_25, axis=1)

CATS_25 = ['math', 'coding', 'creative_writing', 'instruction_following', 'general']
print(f"\nCategory distribution (2025, n={len(df25):,}):")
print(df25['category'].value_counts().to_string())

wr25 = compute_winrate_matrix(df25, CATS_25)
print(f"\nModels in matrix: {len(wr25)}")
print(f"Models with all 5 categories valid (>={BATTLE_MIN} battles each): "
      f"{wr25.dropna().shape[0]}")

print("\nWin-rate matrix (models with all categories valid):")
full = wr25.dropna().copy()
full['org'] = full.index.map(get_org)
print(full.round(3).to_string())

all25, cross25, same25 = pairwise_rho_analysis(wr25, "2025 pairwise Spearman rho")

# ── C. Category pair correlation ──────────────────────────────────────────────
print("\n" + "=" * 65)
print("C. CATEGORY PAIR CORRELATION — which categories discriminate?")
print("=" * 65)

full_wr = wr25.dropna().copy()
print(f"\nSpearman ρ between category win-rate vectors (n={len(full_wr)} models):")
cat_corr = []
for ca, cb in combinations(CATS_25, 2):
    rho, lo, hi, n = spearman_ci(full_wr[ca], full_wr[cb])
    cat_corr.append((ca, cb, rho, lo, hi))
    print(f"  {ca} vs {cb}: ρ = {rho:.3f} [{lo:.3f}, {hi:.3f}]")

print("\nPearson r between category win-rate vectors:")
for ca, cb in combinations(CATS_25, 2):
    r, pval = stats.pearsonr(full_wr[ca], full_wr[cb])
    print(f"  {ca} vs {cb}: r = {r:.3f}, p = {pval:.4f}")

print("\nCategory win-rate standard deviations (higher = more discriminating):")
for cat in CATS_25:
    std = full_wr[cat].std()
    rng_val = full_wr[cat].max() - full_wr[cat].min()
    print(f"  {cat}: std = {std:.4f}, range = {rng_val:.4f}")

# ── Org-level aggregation ─────────────────────────────────────────────────────
print("\nOrg-level mean win rates per category (n>=50 battles filter):")
full_wr['org'] = full_wr.index.map(get_org)
org_means = full_wr.groupby('org')[CATS_25].mean()
print(org_means.round(3).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS B: 2023 battle data — keyword category classification
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("B. 2023 BATTLE DATA — keyword-classified categories")
print("=" * 65)

df23 = pd.read_parquet('data/data/train-00000-of-00001-cced8514c7ed782a.parquet')
print(f"2023 battles: {len(df23):,}, models: {df23['model_a'].nunique() + df23['model_b'].nunique()} (a+b unique)")

def extract_user_msg(conv):
    """Get first user message from conversation list."""
    if isinstance(conv, list):
        for turn in conv:
            if isinstance(turn, dict) and turn.get('role') == 'user':
                return str(turn.get('content', ''))
    return str(conv)

df23['prompt'] = df23['conversation_a'].apply(extract_user_msg)

# Keyword-based classification
MATH_RE = re.compile(
    r'\b(solve|calculate|compute|equation|integral|derivative|proof|theorem|'
    r'matrix|eigenvalue|probability|statistic|algebra|calculus|geometry|'
    r'sum|product|factorial|modulo|prime|fibonacci)\b|'
    r'[\d]+\s*[\+\-\*\/\^]\s*[\d]+', re.I
)
CODE_RE = re.compile(
    r'```|'
    r'\b(python|javascript|java|c\+\+|c#|golang|rust|sql|html|css|'
    r'function|def |class |import |return |algorithm|implement|code|program|'
    r'debug|error|exception|api|database|query|loop|array|list)\b', re.I
)
CREATIVE_RE = re.compile(
    r'\b(write a (story|poem|essay|letter|script|narrative|song|dialogue)|'
    r'creative writing|fiction|novel|character|plot|haiku|sonnet|'
    r'once upon a time|tell me a story|compose a)\b', re.I
)

def categorize_23(prompt):
    if CREATIVE_RE.search(prompt): return 'creative_writing'
    if MATH_RE.search(prompt): return 'math'
    if CODE_RE.search(prompt): return 'coding'
    return 'general'

df23['category'] = df23['prompt'].apply(categorize_23)
print("\nKeyword-classified category distribution (2023):")
print(df23['category'].value_counts().to_string())

CATS_23 = ['math', 'coding', 'creative_writing', 'general']
wr23 = compute_winrate_matrix(df23, CATS_23, min_battles=20)
print(f"\nModels in 2023 matrix (min 20 battles/cat): {wr23.dropna().shape[0]}")

all23, cross23, same23 = pairwise_rho_analysis(wr23, "2023 pairwise Spearman rho")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY COMPARISON: category rho vs. BBH subtask rho
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SUMMARY: Category win-rate rho vs. BBH subtask-level rho")
print("=" * 65)

def mean_ci(rhos, label):
    if not rhos:
        return
    arr = np.array(rhos)
    rng = np.random.default_rng(42)
    boots = [rng.choice(arr, len(arr)).mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  {label}: mean ρ = {arr.mean():.3f} [{lo:.3f}, {hi:.3f}]  (n_pairs={len(arr)})")

print()
print("Arena battle category win-rate vectors (this analysis):")
mean_ci(cross23, "2023 cross-org (4 keyword cats)")
mean_ci(cross25, "2025 cross-org (5 Arena cats)")

print()
print("BBH subtask-level accuracy vectors (Finding 7b, from analysis_subtask_correlation.py):")
print("  2023 cross-family: mean ρ = 0.782 [from paper]")
print("  2024 cross-family: mean ρ = 0.848 [from paper]")

print()
print("Interpretation note:")
print("  Category-level ρ uses 5 coarse categories vs. 24 BBH subtasks —")
print("  expect higher rho here from dimension reduction alone.")
print("  Key question: are categories less correlated than overall trend?")
print("  (Low category rho would mean Arena categories still discriminate)")
