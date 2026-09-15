"""
Simulate k-way rankings from Arena pairwise battle data.

Steps:
1. Parse category tags -> assign each battle to a category (math / creative / code / general)
2. Fit per-category Bradley-Terry scores via MLE
3. Simulate k-way softmax tournament using those BT scores
4. Compare rankings under k=2,4,8,N across categories
"""

import json, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import softmax as sp_softmax

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_parquet("data/140k/data/")

# Parse category tags
def parse_cat(tag):
    if isinstance(tag, str):
        tag = json.loads(tag)
    if tag.get("math_v0.1", {}).get("math", False):
        return "math"
    if tag.get("creative_writing_v0.1", {}).get("creative_writing", False):
        return "creative"
    if tag.get("if_v0.1", {}).get("if", False):
        return "instruct"
    return "general"

df["category"] = df["category_tag"].apply(parse_cat)

# Keep only decisive battles
df = df[df["winner"].isin(["model_a", "model_b"])].copy()
df["winner_model"] = df.apply(lambda r: r["model_a"] if r["winner"]=="model_a" else r["model_b"], axis=1)
df["loser_model"]  = df.apply(lambda r: r["model_b"] if r["winner"]=="model_a" else r["model_a"], axis=1)

print(f"Battles (decisive): {len(df)}")
print(df["category"].value_counts())

# ---------------------------------------------------------------------------
# Fit Bradley-Terry per category via MLE
# ---------------------------------------------------------------------------

CATS = ["math", "creative", "instruct", "general"]

# Use top models with enough battles
min_battles = 30
model_counts = pd.concat([df["model_a"], df["model_b"]]).value_counts()
models = list(model_counts[model_counts >= min_battles].index)
print(f"\nModels with >={min_battles} battles: {len(models)}")

def fit_bt(battles_df, models):
    """Fit Bradley-Terry log-strengths via MLE. Returns dict model->score."""
    model_idx = {m: i for i, m in enumerate(models)}
    N = len(models)

    rows = battles_df[battles_df["winner_model"].isin(models) &
                      battles_df["loser_model"].isin(models)]

    def neg_log_lik(s):
        ll = 0.0
        for _, r in rows.iterrows():
            wi, li = model_idx[r["winner_model"]], model_idx[r["loser_model"]]
            # P(i beats j) = exp(s_i) / (exp(s_i) + exp(s_j))
            ll += s[wi] - np.logaddexp(s[wi], s[li])
        return -ll

    s0 = np.zeros(N)
    res = minimize(neg_log_lik, s0, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-10})
    s = res.x - res.x.mean()  # center
    return {m: s[i] for i, m in enumerate(models)}

bt_scores = {}
for cat in CATS:
    sub = df[df["category"] == cat]
    print(f"\nFitting BT for {cat} ({len(sub)} battles)...")
    bt_scores[cat] = fit_bt(sub, models)

# Build score matrix: models x categories
score_mat = pd.DataFrame(bt_scores, index=models)  # shape: (N_models, 4)
score_mat = score_mat.fillna(0.0)

print("\nTop 10 models by mean BT score:")
score_mat["mean"] = score_mat.mean(axis=1)
top = score_mat.sort_values("mean", ascending=False).head(10)
print(top[CATS + ["mean"]].round(3).to_string())

# ---------------------------------------------------------------------------
# Simulate k-way tournament rankings
# ---------------------------------------------------------------------------

def kway_ranking(score_vec, k, tau=1.0, n_trials=50000, rng=None):
    """
    Monte Carlo k-way ranking.
    Each trial: sample k models, award softmax(scores/tau) win probability.
    Returns expected win share per model.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(score_vec)
    scores = np.array(score_vec)
    win_counts = np.zeros(N)
    for _ in range(n_trials):
        idx = rng.choice(N, size=k, replace=False)
        sm = sp_softmax(scores[idx] / tau)
        win_counts[idx] += sm
    return win_counts / win_counts.sum()

# Use mean BT score across categories for overall ranking
# Also do per-category
K_VALS = [2, 4, 8, len(models)]
TAU = 1.0
rng = np.random.default_rng(42)

print(f"\nSimulating k-way rankings (tau={TAU})...")
kway_ranks = {}
for k in K_VALS:
    label = "N" if k == len(models) else str(k)
    scores = score_mat["mean"].values
    win_share = kway_ranking(scores, k, tau=TAU, rng=rng)
    kway_ranks[k] = win_share
    rank_order = np.argsort(-win_share)
    print(f"\n  k={label} top-10:")
    for rank, idx in enumerate(rank_order[:10]):
        print(f"    {rank+1:2d}. {models[idx]:<45s}  share={win_share[idx]:.4f}  BT={scores[idx]:.3f}")

# ---------------------------------------------------------------------------
# Rank correlation across k
# ---------------------------------------------------------------------------

print("\nSpearman rank correlations between k-way rankings:")
from scipy.stats import spearmanr
k_list = K_VALS
for k1, k2 in itertools.combinations(k_list, 2):
    rho, p = spearmanr(kway_ranks[k1], kway_ranks[k2])
    l1 = "N" if k1==len(models) else str(k1)
    l2 = "N" if k2==len(models) else str(k2)
    print(f"  k={l1} vs k={l2}: rho={rho:.4f}  p={p:.2e}")

# ---------------------------------------------------------------------------
# Per-category k-way ranking (math vs creative)
# ---------------------------------------------------------------------------

print("\nPer-category k=2 vs k=8 rank shift (math):")
for cat in ["math", "creative"]:
    s_vec = score_mat[cat].values
    rk2 = kway_ranking(s_vec, 2,  tau=TAU, rng=rng)
    rk8 = kway_ranking(s_vec, 8,  tau=TAU, rng=rng)
    rank2 = pd.Series(np.argsort(-rk2), name="rank_k2")
    rank8 = pd.Series(np.argsort(-rk8), name="rank_k8")
    # rank change for top 15
    top15 = np.argsort(-rk2)[:15]
    print(f"\n  {cat}:")
    for idx in top15:
        r2 = int(np.where(np.argsort(-rk2)==idx)[0])
        r8 = int(np.where(np.argsort(-rk8)==idx)[0])
        delta = r2 - r8  # positive = moved up with k=8
        arrow = f"▲{delta}" if delta > 0 else (f"▼{abs(delta)}" if delta < 0 else "—")
        print(f"    {models[idx]:<45s}  k2_rank={r2+1:2d}  k8_rank={r8+1:2d}  {arrow}")

# ---------------------------------------------------------------------------
# Plot: rank scatter k=2 vs k=8, colored by BT score
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, cat in zip(axes, ["math", "creative"]):
    s_vec = score_mat[cat].values
    rk2 = kway_ranking(s_vec, 2, tau=TAU, rng=rng)
    rk8 = kway_ranking(s_vec, 8, tau=TAU, rng=rng)
    ranks_k2 = np.argsort(-rk2) + 1   # 1-indexed
    ranks_k8 = np.argsort(-rk8) + 1
    # invert: rank[i] = position of model i
    rank2_by_model = np.empty(len(models), dtype=int)
    rank8_by_model = np.empty(len(models), dtype=int)
    for pos, idx in enumerate(np.argsort(-rk2)): rank2_by_model[idx] = pos+1
    for pos, idx in enumerate(np.argsort(-rk8)): rank8_by_model[idx] = pos+1

    colors = s_vec - s_vec.min()
    colors = colors / colors.max()
    sc = ax.scatter(rank2_by_model, rank8_by_model,
                    c=colors, cmap="plasma", s=60, alpha=0.8, zorder=3)
    lim = len(models) + 1
    ax.plot([1, lim], [1, lim], "k--", lw=1, alpha=0.4, label="no change")
    # label top models
    top_idx = np.argsort(-rk2)[:12]
    for idx in top_idx:
        name = models[idx].split("-")[0] + "…" if len(models[idx])>15 else models[idx]
        ax.annotate(name, (rank2_by_model[idx], rank8_by_model[idx]),
                    fontsize=6, xytext=(4, 2), textcoords="offset points", alpha=0.8)
    ax.set_xlabel("Rank under k=2 (pairwise)", fontsize=10)
    ax.set_ylabel("Rank under k=8", fontsize=10)
    ax.set_title(f"{cat.capitalize()}  (tau={TAU})\nPoints above diagonal = penalised by k=8",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label="BT score (normalized)")

plt.suptitle("Rank shift: k=2 (pairwise) → k=8 (group match)\nArena battles, Bradley-Terry per category",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("plot_arena_kway.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("\nSaved: plot_arena_kway.pdf")
