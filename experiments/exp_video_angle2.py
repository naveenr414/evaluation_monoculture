import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

CAT_X, CAT_Y = "chinese", "creative_writing"
FIRMS = ["google", "openai", "xai", "deepseek", "anthropic"]
COLORS = {"openai": "#10A37F", "anthropic": "#D97757", "google": "#4285F4",
          "deepseek": "#5865F2", "xai": "#6B6B6B"}

df = pd.read_parquet("data/leaderboard/text/full-00000-of-00001.parquet")
df = df[df["organization"].str.strip() != ""]
df["date"] = pd.to_datetime(df["leaderboard_publish_date"])
df["quarter"] = df["date"].dt.to_period("Q")

sub = df[df["category"].isin([CAT_X, CAT_Y])]
g = sub.groupby(["organization", "category", "quarter"])["rating"].max().reset_index()
piv = g.pivot_table(index=["organization", "quarter"], columns="category", values="rating").dropna().reset_index()
piv["x_norm"] = piv.groupby("quarter")[CAT_X].transform(lambda s: s / s.mean())
piv["y_norm"] = piv.groupby("quarter")[CAT_Y].transform(lambda s: s / s.mean())
piv["angle_deg"] = np.degrees(np.arctan2(piv["y_norm"], piv["x_norm"])) - 45.0

piv = piv[piv["organization"].isin(FIRMS)].sort_values(["organization", "quarter"])
quarters = sorted(piv["quarter"].unique())
qlabels = [str(q) for q in quarters]

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.axhline(0, color="gray", lw=1, linestyle="--")
ax.text(len(quarters)-1.05, 0.15, "balanced", fontsize=8, color="gray", ha="right")
ax.set_xlim(-0.3, len(quarters)-0.7)
ymax = max(3.5, piv["angle_deg"].abs().max()*1.15)
ax.set_ylim(-ymax, ymax)
ax.set_xticks(range(len(quarters)))
ax.set_xticklabels(qlabels, rotation=45)
ax.set_ylabel(f"degrees toward {CAT_Y}  (0 = balanced)  degrees toward {CAT_X}", fontsize=10)
ax.text(-0.25, ymax*0.92, f"↑ leans {CAT_Y}", fontsize=9, color="gray")
ax.text(-0.25, -ymax*0.96, f"↓ leans {CAT_X}", fontsize=9, color="gray")
title = ax.set_title("")

lines = {f: ax.plot([], [], "-", color=COLORS[f], lw=2, alpha=0.85)[0] for f in FIRMS}
dots = {f: ax.plot([], [], "o", color=COLORS[f], markersize=20, markeredgecolor="white",
                    markeredgewidth=2, zorder=5)[0] for f in FIRMS}
labels = {f: ax.text(0, 0, f[:4], fontsize=7, ha="center", va="center", zorder=6,
                      color="white", fontweight="bold") for f in FIRMS}

qidx = {q: i for i, q in enumerate(quarters)}
series = {f: piv[piv.organization == f].set_index("quarter")["angle_deg"] for f in FIRMS}

def init():
    for f in FIRMS:
        lines[f].set_data([], [])
        dots[f].set_data([], [])
    return list(lines.values()) + list(dots.values()) + list(labels.values())

def update(i):
    q = quarters[i]
    title.set_text(f"Category lean, unit-vector angle from balanced ({CAT_X} vs {CAT_Y})  —  {q}")
    for f in FIRMS:
        s = series[f]
        xs = [qidx[qq] for qq in s.index if qidx[qq] <= i]
        ys = [s[qq] for qq in s.index if qidx[qq] <= i]
        lines[f].set_data(xs, ys)
        if xs:
            dots[f].set_data([xs[-1]], [ys[-1]])
            labels[f].set_position((xs[-1], ys[-1]))
    return list(lines.values()) + list(dots.values()) + list(labels.values())

ax.legend(handles=[plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=COLORS[f], markersize=10, label=f) for f in FIRMS],
          loc="upper left", fontsize=9, ncol=5, bbox_to_anchor=(0, 1.12))

anim = animation.FuncAnimation(fig, update, frames=len(quarters), init_func=init, blit=False, interval=900)
writer = animation.FFMpegWriter(fps=1, bitrate=3000)
plt.tight_layout()
anim.save("experiments/scratch_persistence/firm_angle_video_chinese_cw.mp4", writer=writer, dpi=150)
print("saved experiments/scratch_persistence/firm_angle_video_chinese_cw.mp4")
print(quarters)
