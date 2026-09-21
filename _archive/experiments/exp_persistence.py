import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_parquet("data/leaderboard/text/full-00000-of-00001.parquet")

# drop non-firm buckets and the "overall"/meta categories
df = df[df["organization"].str.strip() != ""]
df = df[df["category"] != "overall"]
df = df[~df["category"].isin(["exclude_ties", "hard_prompts", "hard_prompts_english", "longer_query"])]

df["date"] = pd.to_datetime(df["leaderboard_publish_date"])
df["quarter"] = df["date"].dt.to_period("Q")

# frontier model per org x category x quarter (max rating)
g = df.groupby(["organization", "category", "quarter"])["rating"].max().reset_index()

# keep firms with reasonable coverage
firm_counts = g.groupby("organization")["quarter"].nunique()
keep_firms = firm_counts[firm_counts >= 6].index
g = g[g["organization"].isin(keep_firms)]

# two-way FE residual, computed WITHIN each quarter (cross-sectional demeaning)
def demean(group):
    grand = group["rating"].mean()
    firm_mean = group.groupby("organization")["rating"].transform("mean")
    cat_mean = group.groupby("category")["rating"].transform("mean")
    group["resid"] = group["rating"] - firm_mean - cat_mean + grand
    return group

g = g.groupby("quarter", group_keys=False).apply(demean)

# --- 1. Lag-1 autocorrelation of residuals, pooled across firm x category series ---
g = g.sort_values(["organization", "category", "quarter"])
g["resid_lag1"] = g.groupby(["organization", "category"])["resid"].shift(1)
pairs = g.dropna(subset=["resid_lag1"])
corr = pairs["resid"].corr(pairs["resid_lag1"])
n = len(pairs)
print(f"Lag-1 autocorrelation of firm-category residuals: r = {corr:.3f}  (n={n} quarter-pairs)")

# --- 2. Rank stability: early half vs late half within each category ---
quarters_sorted = sorted(g["quarter"].unique())
mid = len(quarters_sorted) // 2
early_q = set(quarters_sorted[:mid])
late_q = set(quarters_sorted[mid:])

early = g[g["quarter"].isin(early_q)].groupby(["organization", "category"])["resid"].mean().reset_index()
late = g[g["quarter"].isin(late_q)].groupby(["organization", "category"])["resid"].mean().reset_index()

rank_corrs = []
for cat, sub_e in early.groupby("category"):
    sub_l = late[late["category"] == cat]
    m = sub_e.merge(sub_l, on="organization", suffixes=("_early", "_late"))
    if len(m) >= 4:
        rc = m["resid_early"].rank().corr(m["resid_late"].rank(), method="spearman")
        rank_corrs.append((cat, rc, len(m)))

rank_df = pd.DataFrame(rank_corrs, columns=["category", "spearman_rank_corr", "n_firms"]).sort_values("spearman_rank_corr", ascending=False)
print()
print("Rank stability (early-half vs late-half firm ranking within category):")
print(rank_df.to_string(index=False))
print()
print(f"Mean Spearman rank correlation across categories: {rank_df['spearman_rank_corr'].mean():.3f}")

# --- 3. Plot residual trajectories for a few firms x categories ---
firms_to_plot = ["openai", "google", "anthropic", "meta", "mistral"]
cats_to_plot = ["coding", "math", "creative_writing"]

fig, axes = plt.subplots(1, len(cats_to_plot), figsize=(18, 5), sharey=True)
for ax, cat in zip(axes, cats_to_plot):
    sub = g[(g["category"] == cat) & (g["organization"].isin(firms_to_plot))]
    for firm, s in sub.groupby("organization"):
        s = s.sort_values("quarter")
        ax.plot(s["quarter"].astype(str), s["resid"], marker="o", label=firm)
    ax.set_title(cat)
    ax.axhline(0, color="gray", lw=0.5)
    ax.tick_params(axis="x", rotation=90)
axes[0].set_ylabel("residual (firm advantage after removing\noverall firm strength & category difficulty)")
axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.savefig("experiments/scratch_persistence/firm_category_residuals.png", dpi=130, bbox_inches="tight")
print("\nSaved plot to experiments/scratch_persistence/firm_category_residuals.png")
