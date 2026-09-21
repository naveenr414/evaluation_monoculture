"""
Test error correlation at the subtask level using BBH's 27 subtasks.
This separates saturation (all models score similarly) from correlation
(models fail on the same subtasks).

For each model we get a 27-dim vector of subtask accuracies.
We compute pairwise Spearman correlation of these vectors across models.
High correlation = models fail on the same subtasks (genuine error correlation).
This is independent of whether scores are near ceiling or floor.
"""

import pandas as pd
import numpy as np
import requests
import os
import json
from scipy import stats
from scipy.spatial.distance import cosine
import time

TOKEN = os.environ.get('HF_TOKEN', '')
headers = {'Authorization': f'Bearer {TOKEN}'} if TOKEN else {}

BASE_URL = 'https://huggingface.co/datasets/open-llm-leaderboard/results/resolve/main'

BBH_SUBTASKS = [
    'boolean_expressions', 'causal_judgement', 'date_understanding',
    'disambiguation_qa', 'formal_fallacies', 'geometric_shapes',
    'hyperbaton', 'logical_deduction_five_objects',
    'logical_deduction_seven_objects', 'logical_deduction_three_objects',
    'movie_recommendation', 'multistep_arithmetic_two',
    'navigate', 'object_counting', 'penguins_in_a_table',
    'reasoning_about_colored_objects', 'ruin_names',
    'salient_translation_error_detection', 'snarks',
    'sports_understanding', 'temporal_sequences',
    'tracking_shuffled_objects_five_objects',
    'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_three_objects',
    'web_of_lies', 'word_sorting'
]

def fetch_bbh_subtasks(hf_model_id):
    """Try to fetch BBH subtask scores for a model from the results dataset."""
    org, model = hf_model_id.split('/', 1) if '/' in hf_model_id else ('', hf_model_id)
    # Try to find result files via API
    api_url = f'https://huggingface.co/api/datasets/open-llm-leaderboard/results/parquet'
    # Instead, list files in folder
    list_url = f'https://huggingface.co/api/datasets/open-llm-leaderboard/results?full=true'

    # Direct attempt: guess the filename structure
    folder_url = f'https://huggingface.co/api/datasets/open-llm-leaderboard/results/tree/main/{org}/{model}'
    r = requests.get(folder_url, headers=headers, timeout=10)
    if r.status_code != 200:
        return None
    files = r.json()
    if not files:
        return None
    # Get the most recent result file
    result_files = [f['path'] for f in files if f.get('path','').endswith('.json')]
    if not result_files:
        return None
    result_files.sort(reverse=True)  # most recent first

    # Fetch the result file
    file_url = f'{BASE_URL}/{result_files[0]}'
    r2 = requests.get(file_url, headers=headers, timeout=15)
    if r2.status_code != 200:
        return None
    data = r2.json()
    results = data.get('results', {})

    # Extract BBH subtask scores
    subtask_scores = {}
    for subtask in BBH_SUBTASKS:
        key = f'leaderboard_bbh_{subtask}'
        if key in results:
            score = results[key].get('acc_norm,none', results[key].get('acc,none', None))
            if score is not None:
                subtask_scores[subtask] = score

    return subtask_scores if len(subtask_scores) >= 20 else None

# Load our benchmark data to pick representative models
bench = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/benchmarks.parquet')
bench = bench.dropna(subset=['ifeval','bbh','math','gpqa','musr','mmlu_pro','upload_date'])
bench = bench[~bench['flagged'] & ~bench['merged']]
bench['upload_year'] = bench['upload_date'].dt.year

# Pick official providers only for clean comparison
official_orgs = ['meta-llama','google','microsoft','mistralai','Qwen','deepseek-ai',
                 '01-ai','cohere','allenai','nvidia','stabilityai','tiiuae']
bench['org'] = bench['hf_model_id'].apply(lambda x: x.split('/')[0] if '/' in str(x) else '')
is_official = bench['org'].isin(official_orgs)

# Select target models: top official models from 2023 and 2024
targets_2023 = bench[(bench['upload_year']==2023) & is_official].nlargest(20,'avg_score')
targets_2024 = bench[(bench['upload_year']==2024) & is_official].nlargest(20,'avg_score')

# Also add some well-known models by name that might be in the dataset
extra = bench[bench['hf_model_id'].str.contains('Qwen2.5-72B|Llama-3.3|Llama-3.1-70B|Mistral-Large', na=False)].nlargest(10,'avg_score')

targets = pd.concat([targets_2023, targets_2024, extra]).drop_duplicates(subset='hf_model_id')
print(f'Attempting to fetch BBH subtask data for {len(targets)} models...\n')

subtask_data = {}
failed = []
for _, row in targets.iterrows():
    mid = row['hf_model_id']
    print(f'  Fetching {mid}...', end=' ')
    try:
        scores = fetch_bbh_subtasks(mid)
        if scores:
            subtask_data[mid] = {
                'year': row['upload_year'],
                'avg_score': row['avg_score'],
                'bbh_agg': row['bbh'],
                'subtasks': scores
            }
            print(f'OK ({len(scores)} subtasks)')
        else:
            failed.append(mid)
            print('NOT FOUND')
    except Exception as e:
        failed.append(mid)
        print(f'ERROR: {e}')
    time.sleep(0.3)

print(f'\nSuccessfully fetched: {len(subtask_data)} models')
print(f'Failed: {len(failed)}')
if failed:
    print('Failed models:', failed[:5])

if len(subtask_data) < 4:
    print('\nInsufficient data for correlation analysis.')
else:
    # Build subtask matrix
    models = list(subtask_data.keys())
    common_subtasks = set(BBH_SUBTASKS)
    for m in models:
        common_subtasks &= set(subtask_data[m]['subtasks'].keys())
    common_subtasks = sorted(common_subtasks)
    print(f'\nCommon subtasks: {len(common_subtasks)}')

    scores_matrix = pd.DataFrame(
        {m: [subtask_data[m]['subtasks'][s] for s in common_subtasks] for m in models},
        index=common_subtasks
    ).T  # rows=models, cols=subtasks
    scores_matrix['year'] = [subtask_data[m]['year'] for m in models]
    scores_matrix['bbh_agg'] = [subtask_data[m]['bbh_agg'] for m in models]

    print('\nBBH subtask score summary by year:')
    for yr in [2023, 2024]:
        sub = scores_matrix[scores_matrix['year']==yr][common_subtasks]
        if len(sub) < 2: continue
        stds = sub.std()
        print(f'  {yr}: mean subtask std={stds.mean():.4f}  '
              f'min={stds.min():.4f} ({stds.idxmin()})  '
              f'max={stds.max():.4f} ({stds.idxmax()})')

    # Key test: pairwise Spearman correlation of 27-dim subtask vectors
    # High correlation = models fail on same subtasks (genuine error correlation)
    # Independent of overall level (saturation doesn't affect this)
    print('\n=== Pairwise Spearman correlation of BBH subtask profiles ===')
    print('(This is independent of saturation — measures shared failure structure)')

    def mean_pairwise_spearman_matrix(df, cols):
        vals = df[cols].values.astype(float)
        n = len(vals)
        rhos = []
        for i in range(n):
            for j in range(i+1, n):
                if np.std(vals[i])>0 and np.std(vals[j])>0:
                    r, _ = stats.spearmanr(vals[i], vals[j])
                    rhos.append(r)
        return np.mean(rhos), np.std(rhos), len(rhos)

    for yr in [2023, 2024, None]:
        if yr is not None:
            sub = scores_matrix[scores_matrix['year']==yr]
            label = str(yr)
        else:
            sub = scores_matrix
            label = 'All'
        if len(sub) < 3: continue
        mean_r, std_r, n_pairs = mean_pairwise_spearman_matrix(sub, common_subtasks)
        print(f'  {label} (n={len(sub)} models, {n_pairs} pairs): '
              f'mean ρ={mean_r:.3f}  std={std_r:.3f}')

    # Which subtasks discriminate most vs. least across models?
    print(f'\nMost discriminating subtasks (highest std across all models):')
    all_scores = scores_matrix[common_subtasks]
    stds = all_scores.std().sort_values(ascending=False)
    for task, s in stds.head(5).items():
        means_by_yr = {yr: scores_matrix[scores_matrix['year']==yr][task].mean()
                       for yr in [2023,2024]}
        print(f'  {task}: std={s:.3f}  '
              f'2023 mean={means_by_yr.get(2023,float("nan")):.2f}  '
              f'2024 mean={means_by_yr.get(2024,float("nan")):.2f}')

    print(f'\nLeast discriminating subtasks (lowest std = most saturated):')
    for task, s in stds.tail(5).items():
        means_by_yr = {yr: scores_matrix[scores_matrix['year']==yr][task].mean()
                       for yr in [2023,2024]}
        print(f'  {task}: std={s:.3f}  '
              f'2023 mean={means_by_yr.get(2023,float("nan")):.2f}  '
              f'2024 mean={means_by_yr.get(2024,float("nan")):.2f}')

    # Save for inspection
    scores_matrix.to_csv('/Users/naveenr/Downloads/cyclic_optimization/data/bbh_subtask_scores.csv')
    print('\nSaved subtask matrix to data/bbh_subtask_scores.csv')
