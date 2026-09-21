import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

CAT_X, CAT_Y = "coding", "math"
FIRMS = ["openai", "anthropic", "google", "alibaba", "xai"]
COLORS = {"openai": "#10A37F", "anthropic": "#D97757", "google": "#4285F4",
          "alibaba": "#FF6A00", "xai": "#000000"}

df = pd.read_parquet("data/leaderboard/text/full-00000-of-00001.parquet")
df = df[df["organization"].str.strip() != ""]
df["date"] = pd.to_datetime(df["leaderboard_publish_date"])
df["quarter"] = df["date"].dt.to_period("Q")

sub = df[df["category"].isin([CAT_X, CAT_Y])]
g = sub.groupby(["organization", "category", "quarter"])["rating"].max().reset_index()
piv = g.pivot_table(index=["organization", "quarter"], columns="category", values="rating").dropna().reset_index()
piv["x_norm"] = piv.groupby("quarter")[CAT_X].transform(lambda s: s / s.mean())
piv["y_norm"] = piv.groupby("quarter")[CAT_Y].transform(lambda s: s / s.mean())

norm = np.sqrt(piv["x_norm"]**2 + piv["y_norm"]**2)
piv["ux"] = piv["x_norm"] / norm
piv["uy"] = piv["y_norm"] / norm

piv = piv[piv["organization"].isin(FIRMS)].sort_values(["organization", "quarter"])
quarters = sorted(piv["quarter"].unique())

fig, ax = plt.subplots(figsize=(8, 8))
theta = np.linspace(0, np.pi/2, 200)
ax.plot(np.cos(theta), np.sin(theta), color="lightgray", lw=1, zorder=0)
ax.plot([0, 1.05], [0, 0], color="gray", lw=0.5)
ax.plot([0, 0], [0, 1.05], color="gray", lw=0.5)
ax.plot([0, 0.7071], [0, 0.7071], color="gray", lw=0.5, linestyle="--")
ax.text(0.72, 0.68, "balanced", fontsize=8, color="gray", rotation=45)
ax.set_xlim(0.55, 0.85)
ax.set_ylim(0.55, 0.85)
ax.set_xlabel(f"unit-vector component: {CAT_X} (normalized)")
ax.set_ylabel(f"unit-vector component: {CAT_Y} (normalized)")
title = ax.set_title("")

trails = {f: ax.plot([], [], color=COLORS[f], lw=1.5, alpha=0.5)[0] for f in FIRMS}
dots = {f: ax.plot([], [], "o", color=COLORS[f], markersize=22, markeredgecolor="white",
                    markeredgewidth=2, zorder=5)[0] for f in FIRMS}
labels = {f: ax.text(0, 0, f, fontsize=8, ha="center", va="center", zorder=6,
                      color="white", fontweight="bold") for f in FIRMS}

history = {f: {"x": [], "y": []} for f in FIRMS}

def init():
    for f in FIRMS:
        trails[f].set_data([], [])
        dots[f].set_data([], [])
        labels[f].set_position((0, 0))
    return list(trails.values()) + list(dots.values()) + list(labels.values())

def update(i):
    q = quarters[i]
    title.set_text(f"Direction of firm advantage: {CAT_X} vs {CAT_Y}  —  {q}")
    for f in FIRMS:
        row = piv[(piv.organization == f) & (piv.quarter == q)]
        if len(row) == 0:
            continue
        x, y = row["ux"].values[0], row["uy"].values[0]
        history[f]["x"].append(x)
        history[f]["y"].append(y)
        trails[f].set_data(history[f]["x"], history[f]["y"])
        dots[f].set_data([x], [y])
        labels[f].set_position((x, y))
    return list(trails.values()) + list(dots.values()) + list(labels.values())

ax.legend(handles=[plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=COLORS[f], markersize=10, label=f) for f in FIRMS],
          loc="lower left", fontsize=9)

anim = animation.FuncAnimation(fig, update, frames=len(quarters), init_func=init, blit=False, interval=900)
writer = animation.FFMpegWriter(fps=1, bitrate=3000)
anim.save("experiments/scratch_persistence/firm_direction_video.mp4", writer=writer, dpi=150)
print("saved experiments/scratch_persistence/firm_direction_video.mp4")
print(quarters)
