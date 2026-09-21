import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_parquet("data/leaderboard/text/full-00000-of-00001.parquet")
df = df[df["organization"].str.strip() != ""]
df = df[df["category"] != "overall"]
df = df[~df["category"].isin(["exclude_ties", "hard_prompts", "hard_prompts_english", "longer_query"])]

df["date"] = pd.to_datetime(df["leaderboard_publish_date"])
df["quarter"] = df["date"].dt.to_period("Q")

g = df.groupby(["organization", "category", "quarter"])["rating"].max().reset_index()

firm_counts = g.groupby("organization")["quarter"].nunique()
keep_firms = firm_counts[firm_counts >= 6].index
g = g[g["organization"].isin(keep_firms)]

def demean(group):
    grand = group["rating"].mean()
    firm_mean = group.groupby("organization")["rating"].transform("mean")
    cat_mean = group.groupby("category")["rating"].transform("mean")
    group = group.copy()
    group["resid"] = group["rating"] - firm_mean - cat_mean + grand
    return group

g = g.groupby("quarter", group_keys=False).apply(demean)
g = g.sort_values(["organization", "category", "quarter"])
g["resid_lag1"] = g.groupby(["organization", "category"])["resid"].shift(1)

results = []
for cat, sub in g.dropna(subset=["resid_lag1"]).groupby("category"):
    if len(sub) < 10:
        continue
    r = sub["resid"].corr(sub["resid_lag1"])
    results.append((cat, r, len(sub)))

res_df = pd.DataFrame(results, columns=["category", "corr_h1", "n_pairs"]).sort_values("corr_h1", ascending=False)
print(res_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(res_df["category"], res_df["corr_h1"], color="steelblue")
ax.axvline(0, color="gray", lw=0.8)
ax.set_xlabel("lag-1 autocorrelation: corr(residual_t, residual_{t+1})")
ax.set_title("Persistence of firm advantage (h=1) by category")
ax.invert_yaxis()
for i, row in enumerate(res_df.itertuples()):
    ax.annotate(f"n={row.n_pairs}", (row.corr_h1, i), textcoords="offset points", xytext=(5, 0), fontsize=8, va="center")
plt.tight_layout()
plt.savefig("experiments/scratch_persistence/autocorr_h1_by_category.png", dpi=130, bbox_inches="tight")
print("\nSaved plot to experiments/scratch_persistence/autocorr_h1_by_category.png")
