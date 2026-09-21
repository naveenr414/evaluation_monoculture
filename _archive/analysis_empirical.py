"""
Four empirical analyses for the competition model:
  1. Win-rate decay: shape and rate at which a frontier model's Elo advantage erodes
  2. Training compute (B) over time for frontier models
  3. Scaling law: benchmark performance vs. compute → what functional form is f?
  4. w (evaluator weights) vs v (user weights) by task category
"""

import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.special import expit

plt.rcParams.update({"font.size": 11, "figure.dpi": 130})

# ── Load data ──────────────────────────────────────────────────────────────────
arena    = pd.read_parquet("data/arena_leaderboard.parquet")
lb       = pd.read_parquet("data/leaderboard/text/full-00000-of-00001.parquet")
convs    = pd.read_parquet("data/140k")          # individual Arena conversations
or_daily = pd.read_parquet("data/openrouter_daily.parquet")
frontier = pd.read_csv("data/kaggle_llm_tracker/frontier_models1.csv")

arena["leaderboard_publish_date"] = pd.to_datetime(arena["leaderboard_publish_date"])
lb["leaderboard_publish_date"]    = pd.to_datetime(lb["leaderboard_publish_date"])
frontier["release_date"]          = pd.to_datetime(frontier["release_date"], errors="coerce")
frontier["year_frac"]             = (frontier["release_date"].dt.year +
                                     frontier["release_date"].dt.dayofyear / 365)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. WIN-RATE DECAY
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. WIN-RATE DECAY OF LANDMARK MODELS")
print("=" * 60)

def elo_to_winrate(gap, scale=400.0):
    """Win-rate of model with this Elo advantage over an average competitor."""
    return expit(gap / scale)

medians = arena.groupby("leaderboard_publish_date")["rating"].median().rename("median_elo")

landmarks = {
    "gpt-4-0314"           : "GPT-4 (Mar 2023)",
    "claude-3-opus-20240229": "Claude 3 Opus (Mar 2024)",
    "gpt-4o-2024-05-13"    : "GPT-4o (May 2024)",
    "o1-2024-12-17"        : "o1 (Dec 2024)",
}

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
axes = axes.flatten()
decay_rows = []

for ax, (model_name, label) in zip(axes, landmarks.items()):
    hist = (arena[arena["model_name"] == model_name]
            .sort_values("leaderboard_publish_date")
            .join(medians, on="leaderboard_publish_date")
            .copy())
    hist["gap"] = hist["rating"] - hist["median_elo"]
    hist["wr"]  = elo_to_winrate(hist["gap"])
    hist["t"]   = (hist["leaderboard_publish_date"] - hist["leaderboard_publish_date"].iloc[0]).dt.days

    gap0  = hist["gap"].iloc[0]
    t     = hist["t"].values
    g     = hist["gap"].values

    # Linear fit
    sl, ic, r_lin, *_ = linregress(t, g)
    zero_cross = -gap0 / sl if sl < 0 else np.inf   # days until gap = 0

    # Exponential decay fit (while gap > 0)
    exp_ok = False
    try:
        pos = g > 5
        if pos.sum() >= 5:
            popt, _ = curve_fit(lambda t, A, tau: A * np.exp(-t / tau),
                                t[pos], g[pos], p0=[gap0, 400], maxfev=8000,
                                bounds=([0, 10], [2000, 10000]))
            A_exp, tau_exp = popt
            exp_ok = True
    except Exception:
        pass

    decay_rows.append({
        "model"       : label,
        "gap_debut"   : gap0,
        "gap_now"     : hist["gap"].iloc[-1],
        "wr_debut"    : hist["wr"].iloc[0],
        "wr_now"      : hist["wr"].iloc[-1],
        "slope_elo_day": sl,
        "zero_days"   : zero_cross,
        "tau_days"    : tau_exp if exp_ok else np.nan,
    })

    print(f"  {label}:")
    print(f"    Elo gap:   debut={gap0:.0f}  peak={g.max():.0f}  now={hist['gap'].iloc[-1]:.0f}")
    print(f"    Win-rate:  debut={hist['wr'].iloc[0]:.3f}  now={hist['wr'].iloc[-1]:.3f}")
    print(f"    Linear:    {sl:+.2f} Elo/day  → 0-crossing in {zero_cross:.0f} d ({zero_cross/365:.1f} yr)")
    if exp_ok:
        print(f"    Exp decay: τ = {tau_exp:.0f} d ({tau_exp/365:.1f} yr)")
    print()

    # Plot gap + fits
    ax.plot(hist["leaderboard_publish_date"], g, "o-", ms=3, lw=1.5, label="Elo gap")
    t_fit = np.linspace(0, t.max(), 300)
    d_fit = hist["leaderboard_publish_date"].iloc[0] + pd.to_timedelta(t_fit, "D")
    ax.plot(d_fit, ic + sl * t_fit, "k:", lw=1.2, alpha=0.7, label="linear")
    if exp_ok:
        ax.plot(d_fit, A_exp * np.exp(-t_fit / tau_exp), "r--", lw=1.5,
                label=f"exp τ={tau_exp:.0f}d")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(label)
    ax.set_ylabel("Elo gap above median")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=28, fontsize=8)

fig.suptitle("Win-rate decay: each model's Elo advantage above the field median", fontsize=12)
fig.tight_layout()
fig.savefig("plot_winrate_decay.png", bbox_inches="tight")
print("Decay summary:")
print(pd.DataFrame(decay_rows).to_string(index=False))
print("→ saved plot_winrate_decay.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TRAINING COMPUTE (B) OVER TIME
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("2. TRAINING COMPUTE (B) OVER TIME")
print("=" * 60)

# (A) Parameter growth — use only large (≥50B) frontier models
fm = (frontier.dropna(subset=["parameters_b", "release_date"])
              .query("fine_tuned_variant == 'No' and parameters_b >= 50")
              .sort_values("release_date"))

sl_p, ic_p, r_p, *_ = linregress(fm["year_frac"], np.log10(fm["parameters_b"]))
dt_p = np.log10(2) / sl_p
print(f"  Parameter growth (large frontier ≥50B):   ×2 every {dt_p:.1f} yr  (r²={r_p**2:.2f})")

# (B) Frontier Elo trajectory — proxy for benefit of increasing compute
best_elo = (arena.groupby("leaderboard_publish_date")["rating"].max()
               .reset_index().sort_values("leaderboard_publish_date"))
best_elo["year_frac"] = (best_elo["leaderboard_publish_date"].dt.year +
                         best_elo["leaderboard_publish_date"].dt.dayofyear / 365)
sl_e, ic_e, r_e, *_ = linregress(best_elo["year_frac"], best_elo["rating"])
print(f"  Frontier Elo growth:                       +{sl_e:.0f} Elo/yr  (r²={r_e**2:.2f})")
print(f"  Elo range: {best_elo['rating'].min():.0f} (May 2023) → {best_elo['rating'].max():.0f} (latest)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: parameter count (large frontier only)
ax = axes[0]
ax.scatter(fm["release_date"], fm["parameters_b"], s=60, alpha=0.8, color="steelblue")
for _, row in fm.iterrows():
    ax.annotate(row["model_name"][:16], (row["release_date"], row["parameters_b"]),
                fontsize=6.5, xytext=(4, 3), textcoords="offset points")
t_dt = pd.date_range(fm["release_date"].min(), fm["release_date"].max(), periods=200)
t_yr = t_dt.year + t_dt.dayofyear / 365
ax.plot(t_dt, 10**(ic_p + sl_p * t_yr), "r--", lw=2,
        label=f"trend: ×2 / {dt_p:.1f} yr")
ax.set_yscale("log")
ax.set_ylabel("Parameters (B)  [training budget proxy]")
ax.set_title("Frontier model size (B ≥ 50B, non-fine-tuned)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# Right: frontier Elo over time
ax2 = axes[1]
ax2.plot(best_elo["leaderboard_publish_date"], best_elo["rating"], "o-", ms=3, lw=1)
t_dt2 = pd.date_range(best_elo["leaderboard_publish_date"].min(),
                      best_elo["leaderboard_publish_date"].max(), periods=200)
t_yr2 = t_dt2.year + t_dt2.dayofyear / 365
ax2.plot(t_dt2, ic_e + sl_e * t_yr2, "r--", lw=2,
         label=f"+{sl_e:.0f} Elo/yr  r²={r_e**2:.2f}")
ax2.set_ylabel("Arena Elo (best model at each date)")
ax2.set_title("Frontier capability over time\n(consequence of growing training compute B)")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)

fig.tight_layout()
fig.savefig("plot_compute_over_time.png", bbox_inches="tight")
print("→ saved plot_compute_over_time.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SCALING LAW: f(compute) → performance
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("3. SCALING LAW — what is f?")
print("=" * 60)

# Use frontier models + Arena Elo to estimate f(N_params)
fm_all = (frontier.dropna(subset=["parameters_b", "release_date"])
                  .query("fine_tuned_variant == 'No' and parameters_b >= 1"))

print("\n(A) Benchmark score ~ f(N_params):  cross-sectional fit across all non-fine-tuned models")
results_A = []
for bench, col in [("MMLU","mmlu_score"), ("Arena Elo","arena_elo"),
                    ("GPQA","gpqa_score"), ("MATH","math_score")]:
    sub = fm_all.dropna(subset=[col, "parameters_b"]).copy()
    sub = sub[sub["parameters_b"] >= 1]
    if len(sub) < 6:
        continue
    x, y = sub["parameters_b"].values, sub[col].values
    lx = np.log10(x)
    try:
        popt, _ = curve_fit(lambda t, a, b: a * t + b, lx, y, p0=[5, 30], maxfev=5000)
        y_hat = popt[0] * lx + popt[1]
        r2 = 1 - np.sum((y - y_hat)**2) / np.sum((y - y.mean())**2)
        results_A.append({"bench": bench, "a": popt[0], "b": popt[1], "r2": r2, "n": len(sub)})
        print(f"  {bench:10s}: y = {popt[0]:.2f}·log₁₀(N) + {popt[1]:.2f}   r²={r2:.3f}  n={len(sub)}")
    except Exception as e:
        print(f"  {bench}: {e}")

# (B) Frontier Elo vs. year — log fit suggests f ≈ log(B(t))
print("\n(B) Frontier Elo ~ f(time):")
be = best_elo.copy()
# Elo vs. log(time since start), where time is a proxy for cumulative compute
t0 = be["year_frac"].min()
be["log_dt"] = np.log10(np.maximum(be["year_frac"] - t0 + 0.1, 0.01))
sl_log, ic_log, r_log, *_ = linregress(be["log_dt"], be["rating"])
sl_lin, ic_lin, r_lin, *_ = linregress(be["year_frac"], be["rating"])
print(f"  Linear in time:         Elo = {sl_lin:.1f}·t + {ic_lin:.1f}   r²={r_lin**2:.3f}")
print(f"  Linear in log(time):    Elo = {sl_log:.1f}·log(t) + {ic_log:.1f}   r²={r_log**2:.3f}")
print(f"  → Both good; linear-in-time slightly better, consistent with log(B) if B ~ exp(t)")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: MMLU vs N
sub = fm_all.dropna(subset=["mmlu_score", "parameters_b"]).query("parameters_b >= 1").copy()
years = sub["year_frac"].fillna(2023)
sc = axes[0].scatter(sub["parameters_b"], sub["mmlu_score"], c=years,
                      cmap="viridis", s=55, alpha=0.85)
plt.colorbar(sc, ax=axes[0], label="release yr")
lx_fit = np.linspace(np.log10(sub["parameters_b"].min()), np.log10(sub["parameters_b"].max()), 200)
axes[0].plot(10**lx_fit, results_A[0]["a"] * lx_fit + results_A[0]["b"], "r-", lw=2,
             label=f"log fit: {results_A[0]['a']:.1f}·log₁₀N + {results_A[0]['b']:.1f}\nr²={results_A[0]['r2']:.2f}")
axes[0].set_xscale("log")
axes[0].set_xlabel("Parameters (B)  [proxy for compute]")
axes[0].set_ylabel("MMLU score")
axes[0].set_title("MMLU vs. model size (cross-sectional)")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3, which="both")

# Plot 2: Arena Elo vs N (frontier + non-frontier, coloured by year)
sub2 = fm_all.dropna(subset=["arena_elo", "parameters_b"]).query("parameters_b >= 1")
years2 = sub2["year_frac"].fillna(2023)
sc2 = axes[1].scatter(sub2["parameters_b"], sub2["arena_elo"], c=years2,
                       cmap="viridis", s=55, alpha=0.85)
plt.colorbar(sc2, ax=axes[1], label="release yr")
r2_elo = [r for r in results_A if r["bench"] == "Arena Elo"]
if r2_elo:
    axes[1].plot(10**lx_fit, r2_elo[0]["a"] * lx_fit + r2_elo[0]["b"], "r-", lw=2,
                 label=f"log fit r²={r2_elo[0]['r2']:.2f}")
axes[1].set_xscale("log")
axes[1].set_xlabel("Parameters (B)")
axes[1].set_ylabel("Arena Elo")
axes[1].set_title("Arena Elo vs. model size")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3, which="both")

# Plot 3: Frontier Elo vs time
axes[2].plot(best_elo["leaderboard_publish_date"], best_elo["rating"], "o", ms=4, alpha=0.7)
t_dt3 = pd.date_range(best_elo["leaderboard_publish_date"].min(),
                       best_elo["leaderboard_publish_date"].max(), periods=200)
t_yr3 = t_dt3.year + t_dt3.dayofyear / 365
axes[2].plot(t_dt3, ic_lin + sl_lin * t_yr3, "r-", lw=2,
             label=f"linear: +{sl_lin:.0f}/yr  r²={r_lin**2:.2f}")
# Add log-time fit
log_dt3 = np.log10(np.maximum(t_yr3 - t0 + 0.1, 0.01))
axes[2].plot(t_dt3, ic_log + sl_log * log_dt3, "b--", lw=1.5,
             label=f"log-time r²={r_log**2:.2f}")
axes[2].set_ylabel("Best Arena Elo")
axes[2].set_title("Frontier Elo vs. time\n(reflects f(B(t)) along compute frontier)")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=28, fontsize=8)

fig.suptitle("Scaling law: f(compute) → performance", fontsize=12)
fig.tight_layout()
fig.savefig("plot_scaling_law.png", bbox_inches="tight")
print("→ saved plot_scaling_law.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. w vs v
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("4. w (EVALUATOR) vs v (USER) BY TASK CATEGORY")
print("=" * 60)

# ── w: Arena category-leaderboard vote shares ──────────────────────────────────
latest_date = lb["leaderboard_publish_date"].max()
lb_latest   = lb[lb["leaderboard_publish_date"] == latest_date]

cat_map = {
    "coding"               : "coding",
    "industry_software_and_it_services": "coding",
    "math"                 : "math",
    "industry_mathematical": "math",
    "industry_life_and_physical_and_social_science": "math",
    "creative_writing"     : "creative_writing",
    "industry_entertainment_and_sports_and_media": "creative_writing",
    "instruction_following": "general_instruction",
    "multi_turn"           : "general_instruction",
    "hard_prompts_english" : "general_instruction",
    "industry_writing_and_literature_and_language": "general_instruction",
    "industry_business_and_management_and_financial_operations": "general_instruction",
    "industry_legal_and_government": "general_instruction",
    "industry_medicine_and_healthcare": "general_instruction",
}

raw_votes = (lb_latest[lb_latest["category"].isin(cat_map)]
             .groupby("category")["vote_count"]
             .mean()
             .rename(cat_map)
             .groupby(level=0).sum())
w_ev = raw_votes / raw_votes.sum()

# ── v: Arena organic conversation categories (140k dataset) ───────────────────
def parse_tag(s):
    if pd.isna(s): return {}
    try: return ast.literal_eval(s) if isinstance(s, str) else s
    except: return {}

tags = convs["category_tag"].apply(parse_tag)
convs["is_math"]     = tags.apply(lambda t: bool(t.get("math_v0.1",{}).get("math", False)))
convs["is_creative"] = tags.apply(lambda t: bool(t.get("creative_writing_v0.1",{}).get("creative_writing",False)))

def assign_task(row):
    if row["is_code"]:     return "coding"
    if row["is_math"]:     return "math"
    if row["is_creative"]: return "creative_writing"
    return "general_instruction"

convs["task"] = convs.apply(assign_task, axis=1)
v_user = convs["task"].value_counts() / len(convs)

# ── OpenRouter: token-weighted usage by model capability ─────────────────────
or_total = or_daily.groupby("model_permaslug")["total_tokens"].sum()

def tag_or(slug):
    s = slug.lower()
    if any(k in s for k in ["code","coder","codestral","grok-code","qwen-coder","cursor"]):
        return "coding"
    if any(k in s for k in ["/r1","-r1","-r2","deepseek-r","o1","o3","qwq","reasoning"]):
        return "math"
    if any(k in s for k in ["muse","story","creative","novelist"]):
        return "creative_writing"
    return "general_instruction"

or_cats = or_total.to_frame("tokens").reset_index()
or_cats["task"] = or_cats["model_permaslug"].apply(tag_or)
v_or = or_cats.groupby("task")["tokens"].sum()
v_or = v_or / v_or.sum()

# ── Print table ───────────────────────────────────────────────────────────────
buckets = ["coding", "math", "creative_writing", "general_instruction"]
print(f"\n  {'Category':25s}  {'w (Arena LB)':>12}  {'v (Arena conv)':>15}  {'v (OpenRouter)':>14}  {'w−v_conv':>9}")
for b in buckets:
    ww = w_ev.get(b, 0)
    vu = v_user.get(b, 0)
    vo = v_or.get(b, 0)
    print(f"  {b:25s}  {ww:12.3f}  {vu:15.3f}  {vo:14.3f}  {ww-vu:+9.3f}")

# Fine-grained w from leaderboard
print(f"\nFine-grained Arena leaderboard w ({latest_date.date()}):")
for cat, val in (lb_latest[lb_latest["category"].isin(cat_map)]
                  .groupby("category")["vote_count"].mean()
                  .rename(cat_map)
                  .sort_values(ascending=False).items()):
    print(f"  {cat:50s}: {val/raw_votes.sum():.3f}")

print(f"\nArena conversation (n=116k, Apr–Jul 2025):")
for b, v in v_user.sort_values(ascending=False).items():
    print(f"  {b:30s}: {v:.3f}")

# ── Category-rank heatmap: top-10 overall vs core task categories ─────────────
core_cats   = ["coding", "math", "creative_writing", "instruction_following", "multi_turn"]
overall_top10 = (lb_latest.query("category=='overall'")
                 .dropna(subset=["rating"])
                 .nlargest(10,"rating")["model_name"].tolist())

rank_mat = {}
for cat in core_cats:
    sub = (lb_latest[lb_latest["category"] == cat]
           .dropna(subset=["rating"])
           .sort_values("rating", ascending=False)
           .reset_index(drop=True))
    sub["rk"] = sub.index + 1
    rd = sub.set_index("model_name")["rk"].to_dict()
    rank_mat[cat] = [rd.get(m, np.nan) for m in overall_top10]

rm_df = pd.DataFrame(rank_mat, index=[m[:22] for m in overall_top10])

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left: three-way bar chart (w_leaderboard, v_conversation, v_openrouter)
ax = axes[0]
xp = np.arange(len(buckets))
width = 0.25
b1 = ax.bar(xp - width,    [w_ev.get(b,0) for b in buckets], width,
            label="w — Arena leaderboard", color="steelblue", alpha=0.85)
b2 = ax.bar(xp,            [v_user.get(b,0) for b in buckets], width,
            label="v — Arena conversations", color="seagreen", alpha=0.85)
b3 = ax.bar(xp + width,    [v_or.get(b,0) for b in buckets], width,
            label="v — OpenRouter API", color="coral", alpha=0.85)
ax.set_xticks(xp)
ax.set_xticklabels(["Coding","Math","Creative writing","General\ninstruction"],
                   rotation=10, ha="right")
ax.set_ylabel("Weight (fraction)")
ax.set_title("Evaluator weights (w) vs User weights (v)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8)

# Right: heatmap of category ranks
ax2 = axes[1]
data = rm_df.values.astype(float)
im = ax2.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=15)
ax2.set_xticks(range(len(rm_df.columns)))
ax2.set_xticklabels(rm_df.columns, rotation=28, ha="right", fontsize=9)
ax2.set_yticks(range(len(rm_df.index)))
ax2.set_yticklabels(rm_df.index, fontsize=8)
ax2.set_title("Category rank of the top-10 overall models\n(green=1st, red=15+)")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = data[i, j]
        if not np.isnan(v):
            ax2.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=8,
                     color="white" if v > 8 else "black")
plt.colorbar(im, ax=ax2, label="category rank")

fig.tight_layout()
fig.savefig("plot_w_vs_v.png", bbox_inches="tight")
print("\n→ saved plot_w_vs_v.png")

print("\nAll done.")
plt.close("all")
