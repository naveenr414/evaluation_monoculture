"""
D2 Analysis: Does bias increase over time under iterative training?

Two approaches:
  A) Empirical: style bonus (raw ELO - SC ELO) trend for top-20 frontier models,
     Aug 2024–Apr 2025 (pre-methodology-break window). A growing bonus means frontier
     models are increasingly rewarded for verbosity in the raw signal.

  B) Simulation: numerically integrate the discrete-time RLHF bias-contamination model.
     Show that below Δ*, quality gap collapses and bias fraction grows. Above Δ*, stable.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ===========================================================================
# PART A: Empirical style-bonus trend
# ===========================================================================

print("=" * 68)
print("PART A: Empirical style bonus trend (raw ELO − SC ELO), top-20 frontier")
print("=" * 68)

sc = pd.read_parquet("data/style_control/text_style_control/full-00000-of-00001.parquet")
sc["leaderboard_publish_date"] = pd.to_datetime(sc["leaderboard_publish_date"])
arena = pd.read_parquet("data/arena_leaderboard.parquet")
arena["leaderboard_publish_date"] = pd.to_datetime(arena["leaderboard_publish_date"])

sc_ov = sc[sc["category"] == "overall"].copy()
raw_cols = arena[["model_name", "rating", "leaderboard_publish_date"]].rename(columns={"rating": "raw_elo"})
sc_cols = sc_ov[["model_name", "rating", "leaderboard_publish_date"]].rename(columns={"rating": "sc_elo"})
joined = raw_cols.merge(sc_cols, on=["model_name", "leaderboard_publish_date"])
joined["style_bonus"] = joined["raw_elo"] - joined["sc_elo"]

# Restrict to pre-methodology-break window (Aug 2024 – May 11 2025)
# After May 19, 2025 the SC methodology was revised (sign flipped discontinuously)
pre_break = joined[
    (joined["leaderboard_publish_date"] >= "2024-08-28") &
    (joined["leaderboard_publish_date"] <= "2025-05-11")
].copy()

results_a = []
for date, grp in pre_break.groupby("leaderboard_publish_date"):
    top20 = grp.nlargest(20, "raw_elo")
    results_a.append({
        "date": date,
        "mean_bonus": top20["style_bonus"].mean(),
        "median_bonus": top20["style_bonus"].median(),
        "std_raw": top20["raw_elo"].std(),
        "std_sc": top20["sc_elo"].std(),
        "n": len(top20),
    })
res_a = pd.DataFrame(results_a).sort_values("date")
res_a["days"] = (res_a["date"] - res_a["date"].min()).dt.days

slope_a, intercept_a, r_a, p_a, _ = stats.linregress(res_a["days"], res_a["mean_bonus"])

print(f"\nWindow: {res_a['date'].min().date()} to {res_a['date'].max().date()}")
print(f"n snapshots: {len(res_a)}")
print()
print(f"  Style bonus grew from {res_a['mean_bonus'].iloc[0]:.1f} to {res_a['mean_bonus'].iloc[-1]:.1f} ELO points")
print(f"  Linear trend: {slope_a:.3f} ELO pts/day = {slope_a*30:.1f} pts/month")
print(f"  R² = {r_a**2:.3f},  p = {p_a:.2e}")
print()
print(f"  Monthly snapshots:")
monthly = res_a.resample("ME", on="date").mean(numeric_only=True).dropna()
print(f"  {'Month':<12}  {'Mean bonus':>12}  {'Cumulative change':>18}")
first_bonus = res_a["mean_bonus"].iloc[0]
for idx, row in res_a.set_index("date").resample("ME").mean(numeric_only=True).iterrows():
    if pd.isna(row["mean_bonus"]): continue
    print(f"  {str(idx.date())[:7]:<12}  {row['mean_bonus']:>12.1f}  {row['mean_bonus']-first_bonus:>+18.1f}")

print()
print("NOTE: Confound — reasoning models (DeepSeek-R1, QwQ-32B) entered the top-20")
print("in Jan 2025. The pre-Jan surge (Aug → Dec 2024, +8 ELO pts) is cleaner.")

# Pre-reasoning window (Aug–Dec 2024)
pre_reasoning = res_a[res_a["date"] < "2025-01-01"].copy()
sl2, ic2, r2, p2, _ = stats.linregress(pre_reasoning["days"], pre_reasoning["mean_bonus"])
print(f"\n  Aug–Dec 2024 only: {sl2:.3f} ELO pts/day = {sl2*30:.1f} pts/month,")
print(f"  R² = {r2**2:.3f},  p = {p2:.2e}")
print(f"  Start: {pre_reasoning['mean_bonus'].iloc[0]:.1f} → End: {pre_reasoning['mean_bonus'].iloc[-1]:.1f}")

# ===========================================================================
# PART B: Simulation
# ===========================================================================

print()
print("=" * 68)
print("PART B: Simulation — discrete-time RLHF bias-contamination dynamics")
print("=" * 68)

def sigma(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def alpha(delta, tau):
    """Annotator reliability: how well quality predicts the outcome."""
    return sigma(delta / tau)

def simulate(delta_0, tau, gamma, eta, C_q, n_rounds):
    """
    Simulate iterated RLHF under noisy annotation.

    State: delta_t = quality gap between models (we track the gap, not individual models)
    alpha_t = sigma(delta_t / tau) = quality-based annotation weight
    beta_t  = 1 - alpha_t = bias weight

    Quality update (from gradient decomposition):
      signal term   : alpha * (sigma(delta) - 0.5) * C_q  [reward model learns quality]
      drag term     : (1-alpha) * gamma                     [biased policy update]

    delta_{t+1} = delta_t + eta * [alpha_t * (sigma(delta_t) - 0.5) * C_q - (1-alpha_t) * gamma]

    Bias contamination fraction: beta_t = 1 - alpha_t (driven by delta)
    """
    deltas = [delta_0]
    betas  = [1 - alpha(delta_0, tau)]

    for _ in range(n_rounds):
        d = deltas[-1]
        a = alpha(d, tau)
        signal = a * (sigma(d) - 0.5) * C_q
        drag   = (1 - a) * gamma
        d_new  = max(0.0, d + eta * (signal - drag))
        deltas.append(d_new)
        betas.append(1 - alpha(d_new, tau))

    return np.array(deltas), np.array(betas)

# Find Δ* analytically: F(Δ*) = 0
# F(Δ) = alpha(Δ) * (sigma(Δ) - 0.5) * C_q - (1-alpha(Δ)) * gamma
# Work in logit scale: delta = BT logit = ELO / 173.7
# (sigma(delta) is then the win probability directly)
# Δ = 11 ELO pts ≈ 0.063 logits; 30 ELO ≈ 0.173 logits; 75 ELO ≈ 0.432 logits
LOGIT_PER_ELO = 1.0 / 173.7

tau   = 0.30   # annotator discrimination threshold (logit scale)
gamma = 0.010  # bias drag per step
C_q   = 0.50   # quality gradient scale
eta   = 0.50   # learning rate

from scipy.optimize import brentq

def F(d):
    a = alpha(d, tau)
    return a * (sigma(d) - 0.5) * C_q - (1 - a) * gamma

delta_star = brentq(F, 0.001, 5.0)
delta_star_elo = delta_star / LOGIT_PER_ELO

print(f"\n  Parameters: τ={tau} logits ({tau/LOGIT_PER_ELO:.0f} ELO), γ={gamma}, C_q={C_q}")
print(f"  Theoretical Δ* = {delta_star:.3f} logits ≈ {delta_star_elo:.1f} ELO points")
print(f"  (Empirical phase transition from B3: ~20-30 ELO points — order-of-magnitude match)")
print(f"  F(0) = {F(0):.4f} < 0 → near-zero always collapses")
print(f"  F(2) = {F(2):.4f} > 0 → large gap expands (stable above Δ*)")

n_rounds = 120

# Trajectories in logit scale; display as ELO
scenarios_elo = [3, 8, delta_star_elo, 25, 60]
scenarios = [
    (f"Δ₀ = {int(e)} ELO (below Δ*)" if e < delta_star_elo else
     (f"Δ₀ = Δ* ≈ {delta_star_elo:.0f} ELO" if abs(e - delta_star_elo) < 1 else
      f"Δ₀ = {int(e)} ELO (above Δ*)"),
     e * LOGIT_PER_ELO,
     col)
    for e, col in zip(scenarios_elo, ["red", "orange", "grey", "steelblue", "darkblue"])
]

print()
print(f"  {'Scenario':<36}  {'Δ_final (ELO)':>14}  {'β_init':>8}  {'β_final':>8}  {'Outcome':>12}")
print("  " + "-" * 86)
for label, d0, color in scenarios:
    deltas, betas = simulate(d0, tau, gamma, eta, C_q, n_rounds)
    d_final_elo = deltas[-1] / LOGIT_PER_ELO
    outcome = ("collapse" if deltas[-1] < 0.001 else
               ("unstable" if abs(deltas[-1] - delta_star) < 0.005 else "stable↑"))
    print(f"  {label:<36}  {d_final_elo:>14.1f}  {betas[0]:>8.3f}  {betas[-1]:>8.3f}  {outcome:>12}")

# Key: below Δ*, bias fraction grows as quality gap collapses
print()
print(f"  Below Δ* ({delta_star_elo:.0f} ELO): Δ_t → 0, β_t → 0.5 (annotators stuck at coin-flip)")
print(f"  Above Δ* ({delta_star_elo:.0f} ELO): Δ_t grows, β_t stays low (annotators track quality)")
print()
print("  Bias fraction β = 1 - α = 1 - σ(Δ/τ)")
print(f"  At Δ=0:    β={1-alpha(0,tau):.3f}  (half of annotation signal is bias)")
print(f"  At Δ=Δ*:   β={1-alpha(delta_star,tau):.3f}  (critical bias level)")
print(f"  At Δ=60 ELO: β={1-alpha(60*LOGIT_PER_ELO,tau):.3f}  (low bias, quality dominates)")

# Show step-by-step for a below-Δ* case to illustrate D2
d0_ex = 8 * LOGIT_PER_ELO  # 8 ELO points below Δ*
print()
print(f"  Step-by-step: Δ₀≈8 ELO (below Δ*) — bias grows as gap closes")
deltas_d2, betas_d2 = simulate(d0_ex, tau, gamma, eta, C_q, n_rounds)
print(f"  {'Round':<8}  {'Δ_t':>8}  {'α_t':>8}  {'β_t (bias)':>12}  {'Signal':>10}  {'Drag':>10}")
print("  " + "-" * 68)
checkpoints = [0, 5, 10, 20, 40, 70, 100, 119]
print(f"  {'Round':<8}  {'Δ(ELO)':>9}  {'α_t':>8}  {'β_t(bias)':>12}  {'Signal':>10}  {'Drag':>10}")
print("  " + "-" * 68)
for t in checkpoints:
    d = deltas_d2[t]
    a = alpha(d, tau)
    signal = a * (sigma(d) - 0.5) * C_q
    drag   = (1 - a) * gamma
    d_elo = d / LOGIT_PER_ELO
    print(f"  {t:<8}  {d_elo:>9.2f}  {a:>8.3f}  {1-a:>12.3f}  {signal:>10.5f}  {drag:>10.5f}")

print()
print("  As Δ_t decreases: α decreases, β=1-α increases,")
print("  signal shrinks faster than drag → inevitable collapse once below Δ*.")

# ===========================================================================
# FIGURE
# ===========================================================================

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# --- A1: Style bonus trend ---
ax1 = fig.add_subplot(gs[0, :])
ax1.scatter(res_a["date"], res_a["mean_bonus"], color="steelblue", s=12, alpha=0.6, label="Weekly snapshots")
x_fit = pd.date_range(res_a["date"].min(), res_a["date"].max(), periods=100)
x_days = (x_fit - res_a["date"].min()).days
ax1.plot(x_fit, intercept_a + slope_a * x_days, "steelblue", lw=2, label=f"Trend ({slope_a*30:.1f} ELO/month, R²={r_a**2:.2f}, p={p_a:.0e})")
ax1.axvline(pd.Timestamp("2025-01-01"), color="orange", lw=1.5, ls="--", label="DeepSeek-R1 release (Jan 2025)")
ax1.axhline(0, color="black", lw=0.8, ls=":")
ax1.set_ylabel("Style bonus (raw ELO − SC ELO), top-20 models")
ax1.set_title("A) Verbosity premium in Arena ELO grows over time (Aug 2024 – May 2025)\n"
              "Larger bonus = frontier models increasingly rewarded for length/formatting")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# --- B1: Quality gap Δ_t ---
ax2 = fig.add_subplot(gs[1, 0])
t_axis = np.arange(n_rounds + 1)
for label, d0, color in scenarios:
    deltas, betas = simulate(d0, tau, gamma, eta, C_q, n_rounds)
    deltas_elo = np.clip(deltas / LOGIT_PER_ELO, 0, 80)
    ax2.plot(t_axis, deltas_elo, color=color, lw=2, label=label)
ax2.axhline(delta_star_elo, color="grey", lw=1.5, ls="--", label=f"Δ* ≈ {delta_star_elo:.0f} ELO")
ax2.set_xlabel("Training round")
ax2.set_ylabel("Quality gap Δₜ (ELO points)")
ax2.set_title("B) Quality gap: collapse below Δ*, grows above")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-2, 85)

# --- B2: Bias fraction β_t ---
ax3 = fig.add_subplot(gs[1, 1])
for label, d0, color in scenarios:
    deltas, betas = simulate(d0, tau, gamma, eta, C_q, n_rounds)
    ax3.plot(t_axis, betas, color=color, lw=2, label=label)
ax3.axhline(0.5, color="red", lw=1, ls=":", label="β=0.5 (pure noise)")
ax3.axhline(1 - alpha(delta_star, tau), color="grey", lw=1.5, ls="--", label=f"β* at Δ*")
ax3.set_xlabel("Training round")
ax3.set_ylabel("Bias fraction βₜ = 1 − α(Δₜ)")
ax3.set_title("C) Bias accumulation: below Δ*, β grows toward 0.5")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0.3, 0.55)

plt.savefig("plot_d2_analysis.png", dpi=150, bbox_inches="tight")
print(f"\nFigure saved to plot_d2_analysis.png")

# ===========================================================================
# PART C: 2023 vs 2025 bias effect in coin-flip pairs (quick check)
# ===========================================================================

print()
print("=" * 68)
print("PART C: 2023 vs 2025 — does the 'weaker-but-longer wins' rate differ?")
print("=" * 68)

# 2023 frontier coin-flip pairs: need ELO from a 2023 leaderboard snapshot
arena2023 = arena[arena["leaderboard_publish_date"] < "2024-01-01"]
if len(arena2023) > 0:
    # Get the last 2023 snapshot
    snap23_date = arena2023["leaderboard_publish_date"].max()
    snap23 = arena2023[arena2023["leaderboard_publish_date"] == snap23_date].set_index("model_name")["rating"]
    print(f"\n  Using 2023 leaderboard snapshot: {snap23_date.date()}")
    print(f"  n models: {len(snap23)}, ELO range: {snap23.min():.0f}–{snap23.max():.0f}")
    print()

    # 2023 battle data — compute response lengths from text
    df23 = pd.read_parquet("data/data/train-00000-of-00001-cced8514c7ed782a.parquet")
    df23 = df23[df23["winner"].isin(["model_a", "model_b"])].copy()
    df23["a_wins"] = (df23["winner"] == "model_a").astype(int)

    def get_len_23(conv):
        if conv is None: return 0
        for turn in (conv if hasattr(conv, '__iter__') else []):
            if isinstance(turn, dict) and turn.get("role") == "assistant":
                return len(str(turn.get("content", "")))
        return 0

    df23["len_a"] = df23["conversation_a"].apply(get_len_23)
    df23["len_b"] = df23["conversation_b"].apply(get_len_23)
    df23["log_len_ratio"] = np.log1p(df23["len_a"]) - np.log1p(df23["len_b"])

    import re
    snap23_norm = {re.sub(r"[^a-z0-9]", "", k.lower()): v for k, v in snap23.items()}

    def lookup_23(name):
        key = re.sub(r"[^a-z0-9]", "", str(name).lower())
        if key in snap23_norm: return snap23_norm[key]
        if len(key) >= 4:
            for k, v in snap23_norm.items():
                if key in k or k in key: return v
        return np.nan

    df23["elo_a"] = df23["model_a"].map(lookup_23)
    df23["elo_b"] = df23["model_b"].map(lookup_23)
    m23 = df23.dropna(subset=["elo_a", "elo_b"]).copy()
    m23["elo_gap"] = np.abs(m23["elo_a"] - m23["elo_b"])
    m23["elo_diff"] = m23["elo_a"] - m23["elo_b"]

    print(f"  2023 battles matched to ELO: {len(m23):,} / {len(df23):,}")
    if len(m23) > 50:
        print(f"  ELO gap: median={m23['elo_gap'].median():.1f}, max={m23['elo_gap'].max():.1f}")

    # Compare bias-wins rate for coin-flip pairs in 2023 vs 2025
    # Use same <30 ELO gap threshold (generous, since 2023 has fewer tight pairs)
    def bias_wins_rate(df_m, gap_thresh=30):
        sub = df_m[df_m["elo_gap"] < gap_thresh].copy()
        if len(sub) < 30:
            return None, 0
        conflict = sub[((sub["log_len_ratio"] > 0) & (sub["elo_diff"] < 0)) |
                       ((sub["log_len_ratio"] < 0) & (sub["elo_diff"] > 0))]
        if len(conflict) < 20:
            return None, len(sub)
        weaker_wins = np.concatenate([
            conflict[conflict["log_len_ratio"] > 0]["a_wins"].values,
            (1 - conflict[conflict["log_len_ratio"] < 0]["a_wins"]).values,
        ])
        return weaker_wins.mean(), len(conflict)

    print()
    print(f"  {'Period':<10}  {'Gap thresh':<12}  {'N conflict':>12}  "
          f"{'Weaker(longer) wins':>22}  {'z vs 0.5':>10}")
    print("  " + "-" * 72)
    for thresh, label in [(10, "<10 gap"), (30, "<30 gap")]:
        rate_23, n_23 = bias_wins_rate(m23, thresh)
        if rate_23 is None:
            print(f"  {'2023':<10}  {label:<12}  {'too few':>12}")
            continue
        se_23 = np.sqrt(rate_23*(1-rate_23)/n_23)
        z_23 = (rate_23-0.5)/se_23

        # 2025 equivalent (from analysis_b3 results)
        rate_25 = {10: 0.5713, 30: 0.561}[thresh]
        print(f"  {'2023':<10}  {label:<12}  {n_23:>12}  {rate_23:>22.4f}  {z_23:>+10.2f}")
        print(f"  {'2025':<10}  {label:<12}  {'(B3)':>12}  {rate_25:>22.4f}  {'(z≈+7.9)':>10}")
        print()
else:
    print("  No 2023 Arena leaderboard snapshots found in the data.")
