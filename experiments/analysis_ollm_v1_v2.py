"""Does relative standing survive a real, sharp benchmark/weight swap?

HuggingFace's Open LLM Leaderboard discarded its entire v1 battery (ARC,
HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K) for a harder, unsaturated v2
battery (MMLU-Pro, GPQA, MuSR, BBH, MATH, IFEval) in June 2024. For static
open-weight checkpoints this is as clean a w-shock as exists in the wild:
theta = F_i(x_i) is literally frozen (same weights), so any change in a
model's relative standing between v1 and v2 is 100% attributable to the
benchmark/category swap, not new investment. Sample built by
pull_ollm_v1_sample.py (136 official lab releases matched to both batteries).

Output: experiments/plot_ollm_v1_v2.png
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_parquet('data/ollm_v1_v2_matched.parquet')
df['rank_v1'] = df.v1_avg.rank(ascending=False)
df['rank_v2'] = df.v2_avg.rank(ascending=False)
df['rank_shift'] = df.rank_v1 - df.rank_v2  # >0 = ranks BETTER (moves up) under v2

r = np.corrcoef(df.v1_avg, df.v2_avg)[0, 1]
rho, p = spearmanr(df.v1_avg, df.v2_avg)
print(f'n={len(df)}  pearson r={r:.3f}  spearman rho={rho:.3f} (p={p:.1e})')

org_shift = (df.groupby('org')['rank_shift'].agg(['mean', 'count'])
             .query('count >= 3').sort_values('mean'))
print(org_shift.round(1).to_string())

ORG_COLOR = {
    'google': '#0072B2', 'deepseek-ai': '#E69F00', 'allenai': '#009E73',
    'Qwen': '#CC79A7', 'meta-llama': '#56B4E9', 'mistralai': '#D55E00',
    'microsoft': '#F0E442', 'NousResearch': '#999999', 'teknium': '#666666',
    'HuggingFaceH4': '#333333',
}
DEFAULT = '#CCCCCC'

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={'width_ratios': [1.1, 1]})

ax = axes[0]
for org, g in df.groupby('org'):
    c = ORG_COLOR.get(org, DEFAULT)
    ax.scatter(g.v1_avg, g.v2_avg, color=c, s=40 if org in ORG_COLOR else 22,
               alpha=1.0 if org in ORG_COLOR else 0.5, zorder=3 if org in ORG_COLOR else 2,
               label=org if org in ORG_COLOR else None)
b, a = np.polyfit(df.v1_avg, df.v2_avg, 1)
xs = np.linspace(df.v1_avg.min(), df.v1_avg.max(), 50)
ax.plot(xs, a + b * xs, color='#888888', lw=1.2, ls='--', zorder=1)
ax.set_xlabel('v1 battery score (ARC/HellaSwag/MMLU/TruthfulQA/Winogrande/GSM8K)')
ax.set_ylabel('v2 battery score (MMLU-Pro/GPQA/MuSR/BBH/MATH/IFEval)')
ax.set_title(f'Same frozen checkpoints, two batteries\n(Spearman ρ={rho:.2f}, n={len(df)})')
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc='upper left')

ax = axes[1]
colors = [ORG_COLOR.get(o, DEFAULT) for o in org_shift.index]
ax.barh(org_shift.index, org_shift['mean'], color=colors)
ax.axvline(0, color='#888888', lw=0.8)
ax.set_xlabel('mean rank shift, v1 -> v2  (+ = ranks better under the harder battery)')
ax.set_title('Which labs gain/lose rank\nunder the w-shock')
ax.spines[['top', 'right']].set_visible(False)
for i, (org, row) in enumerate(org_shift.iterrows()):
    ax.text(row['mean'] + (1 if row['mean'] >= 0 else -1), i, f"n={int(row['count'])}",
            va='center', ha='left' if row['mean'] >= 0 else 'right', fontsize=7, color='#666666')

fig.suptitle('A real w-shock (Open LLM Leaderboard v1 -> v2, June 2024) on FIXED checkpoints', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('experiments/plot_ollm_v1_v2.png', dpi=150, bbox_inches='tight')
print('\nwrote experiments/plot_ollm_v1_v2.png')
