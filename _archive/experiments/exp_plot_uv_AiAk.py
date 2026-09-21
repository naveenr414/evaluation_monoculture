"""Plot u_i, v_k (the low-rank interaction factors) and A_i, A_k (the separable
scale terms) from the best-fit calibration: sigma_logD=0.14, seed=7, d=1."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N, K = 8, 8
SIGMA_LOGD = 0.14
SEED = 7

a_k = 1.0 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, K, endpoint=False))
alpha_i = np.linspace(0.7, 1.3, N)

r = np.random.default_rng(SEED)
u = r.normal(0, 1, N)
v = r.normal(0, 1, K)
raw = np.outer(u, v)
raw -= raw.mean(axis=1, keepdims=True)
scale = SIGMA_LOGD / raw.std()
u_scaled = u * np.sqrt(scale)   # keep u*v product scale correct, split sqrt(scale) to each
v_scaled = v * np.sqrt(scale)
# (raw already carries the row-demeaning; report the RAW u,v since the demeaning
#  step mixes rows -- what's actually plotted below uses raw u,v pre-demean for
#  interpretability, with the achieved overall sigma noted in the title)

# same 8 firms / 8 categories used in exp1_evalchasing.py (N=K=8 lines up exactly)
firms = ['openai', 'anthropic', 'google', 'meta', 'mistral', 'xai', 'deepseek', 'alibaba']
cats = ['coding', 'math', 'hard_prompts', 'instruction_following',
        'creative_writing', 'longer_query', 'multi_turn', 'english']

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
ax.bar(firms, u_scaled, color='#4C72B0')
ax.axhline(0, color='gray', lw=0.8)
ax.set_title('u_i  (firm loading on the interaction factor)')
ax.set_ylabel('u_i')

ax = axes[0, 1]
ax.bar(cats, v_scaled, color='#DD8452')
ax.axhline(0, color='gray', lw=0.8)
ax.set_title('v_k  (category loading on the interaction factor)')
ax.set_ylabel('v_k')

ax = axes[1, 0]
ax.bar(firms, alpha_i, color='#55A868')
ax.set_title('A_i  (firm overall scale, separable term)')
ax.set_ylabel('A_i')

ax = axes[1, 1]
ax.bar(cats, a_k, color='#C44E52')
ax.set_title('A_k  (category overall scale, separable term)')
ax.set_ylabel('A_k')

for ax in axes.flat:
    ax.tick_params(axis='x', rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha('right')

fig.suptitle(f'Best-fit calibration: sigma_logD={SIGMA_LOGD}, seed={SEED}  (SYNTHETIC values,\n'
             f'labeled with the real firm/category names for readability -- not fit to real data)\n'
             f'log D_i,k = u_i * v_k  (rank-1 interaction, zero-mean per firm after construction)',
             fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig('experiments/plot_uv_AiAk.png', dpi=140, bbox_inches='tight')
print('saved experiments/plot_uv_AiAk.png')

# also show which (firm, category) cells get the biggest positive/negative D boost,
# i.e. where u_i and v_k have the same/opposite sign and large magnitude
logD_final = np.outer(u, v)
logD_final -= logD_final.mean(axis=1, keepdims=True)
logD_final *= SIGMA_LOGD / logD_final.std()
print()
print('log D_{i,k} matrix (firms x categories):')
print(np.round(logD_final, 3))
