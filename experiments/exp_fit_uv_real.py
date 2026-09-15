"""Fit the SAME structure as the toy calibration -- theta = firm_effect_i + cat_effect_k
+ u_i*v_k -- to the REAL Arena panel. This is exactly log(A_i*A_k*D_ik) in log-space,
since Arena BT ratings are already a log-odds scale. Uses the same 8 firms / 8
categories as exp1_evalchasing.py so it lines up with the synthetic plot.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CORE = ['coding','math','hard_prompts','instruction_following',
        'creative_writing','longer_query','multi_turn','english']
FIRMS = ['openai','anthropic','google','meta','mistral','xai','deepseek','alibaba']

d = pd.read_parquet('data/leaderboard/text/full-00000-of-00001.parquet')
d['org'] = d.organization.str.lower()
d = d[d.org.isin(FIRMS) & d.category.isin(CORE)]
d['dt'] = pd.to_datetime(d.leaderboard_publish_date)

# recent window: last 4 available snapshot dates, frontier (max rating) model per org x cat
recent_dates = np.sort(d.dt.unique())[-4:]
d = d[d.dt.isin(recent_dates)]
frontier = d.groupby(['org', 'category'])['rating'].max().reset_index()
M_df = frontier.pivot(index='org', columns='category', values='rating').reindex(index=FIRMS, columns=CORE)
print('coverage (non-null cells):')
print(M_df.notna().sum(axis=1).to_string())
M_df = M_df.dropna(how='any')  # need full rows for a clean SVD
print(f'\nusing {len(M_df)} firms with full category coverage: {list(M_df.index)}')

M = M_df.values
grand = M.mean()
firm_effect = M.mean(axis=1) - grand
cat_effect = M.mean(axis=0) - grand
resid = M - grand - firm_effect[:, None] - cat_effect[None, :]

U, S, Vt = np.linalg.svd(resid)
u1_raw, s1, v1_raw = U[:, 0], S[0], Vt[0, :]
u1 = u1_raw * np.sqrt(s1)
v1 = v1_raw * np.sqrt(s1)
pr = (S.sum() ** 2) / (S ** 2).sum() if len(M_df) > 1 else np.nan
share = s1 ** 2 / np.sum(S ** 2)

print(f'\ngrand mean rating: {grand:.1f}')
print(f'top singular value s1={s1:.2f}, share of residual variance={share:.1%}')
print(f'participation ratio of residual (effective rank): {pr:.2f} of {min(M.shape)} possible')

firms_kept = list(M_df.index)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
ax.bar(firms_kept, u1, color='#4C72B0')
ax.axhline(0, color='gray', lw=0.8)
ax.set_title(f'u_i  (empirical firm loading, top SVD factor,\n{share:.0%} of residual variance)')

ax = axes[0, 1]
ax.bar(CORE, v1, color='#DD8452')
ax.axhline(0, color='gray', lw=0.8)
ax.set_title('v_k  (empirical category loading, top SVD factor)')

ax = axes[1, 0]
ax.bar(firms_kept, firm_effect, color='#55A868')
ax.axhline(0, color='gray', lw=0.8)
ax.set_title('firm_effect_i  (= log A_i, additive)')

ax = axes[1, 1]
ax.bar(CORE, cat_effect, color='#C44E52')
ax.axhline(0, color='gray', lw=0.8)
ax.set_title('cat_effect_k  (= log A_k, additive)')

for ax in axes.flat:
    ax.tick_params(axis='x', rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha('right')

fig.suptitle(f'REAL Arena data, frontier model per firm x category, last 4 snapshots\n'
             f'theta = grand + firm_effect_i + cat_effect_k + u_i*v_k  '
             f'(effective rank of residual: {pr:.2f} / {min(M.shape)})', fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.91])
plt.savefig('experiments/plot_uv_AiAk_REAL.png', dpi=140, bbox_inches='tight')
print('\nsaved experiments/plot_uv_AiAk_REAL.png')

print('\nresidual matrix (firms x categories), interaction not explained by firm/cat effects:')
print(pd.DataFrame(np.round(resid, 1), index=firms_kept, columns=CORE).to_string())
