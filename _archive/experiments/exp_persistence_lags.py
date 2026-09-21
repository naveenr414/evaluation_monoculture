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

results = []
for h in range(1, 13):
    g[f"resid_lead{h}"] = g.groupby(["organization", "category"])["resid"].shift(-h)
    pairs = g.dropna(subset=[f"resid_lead{h}"])
    if len(pairs) < 10:
        continue
    r = pairs["resid"].corr(pairs[f"resid_lead{h}"])
    results.append((h, r, len(pairs)))

res_df = pd.DataFrame(results, columns=["h", "corr", "n_pairs"])
print(res_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(res_df["h"], res_df["corr"], marker="o")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("lag h (quarters)")
ax.set_ylabel("corr(residual_t, residual_{t+h})")
ax.set_title("Persistence of firm x category residual advantage across lags")
ax.set_xticks(range(1, 13))
for _, row in res_df.iterrows():
    ax.annotate(f"n={int(row['n_pairs'])}", (row["h"], row["corr"]), textcoords="offset points", xytext=(0, 8), fontsize=7, ha="center", color="gray")
plt.tight_layout()
plt.savefig("experiments/scratch_persistence/autocorr_by_lag.png", dpi=130, bbox_inches="tight")
print("\nSaved plot to experiments/scratch_persistence/autocorr_by_lag.png")
