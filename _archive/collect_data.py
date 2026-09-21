"""
Data collection for cyclic optimization study.
Saves three clean datasets to data/:
  arena_leaderboard.parquet   - historical Arena ELO snapshots (overall category)
  benchmarks.parquet          - Open LLM Leaderboard v2 benchmark scores
  matched.parquet             - joined dataset with Arena ELO + benchmarks + metadata
"""

import pandas as pd
import numpy as np
import re
import os
from huggingface_hub import hf_hub_download, HfApi

OUT = "/Users/naveenr/Downloads/cyclic_optimization/data"
os.makedirs(OUT, exist_ok=True)

# ── 1. Arena historical leaderboard ──────────────────────────────────────────
print("Loading Arena leaderboard...")
arena_path = f"{OUT}/leaderboard/text/full-00000-of-00001.parquet"
arena_raw = pd.read_parquet(arena_path)
arena_raw["leaderboard_publish_date"] = pd.to_datetime(arena_raw["leaderboard_publish_date"])

# Keep overall category only (the main English general leaderboard)
arena = arena_raw[arena_raw["category"] == "overall"].copy()
arena = arena[["model_name", "organization", "rating", "rating_lower", "rating_upper",
               "variance", "vote_count", "rank", "leaderboard_publish_date"]].copy()
arena = arena.sort_values(["leaderboard_publish_date", "rank"]).reset_index(drop=True)

print(f"  Arena: {len(arena)} rows, {arena['model_name'].nunique()} unique models, "
      f"{arena['leaderboard_publish_date'].nunique()} snapshots")
print(f"  Date range: {arena['leaderboard_publish_date'].min().date()} to "
      f"{arena['leaderboard_publish_date'].max().date()}")
arena.to_parquet(f"{OUT}/arena_leaderboard.parquet", index=False)
print(f"  Saved -> arena_leaderboard.parquet")

# ── 2. Open LLM Leaderboard benchmarks ───────────────────────────────────────
print("\nLoading Open LLM Leaderboard benchmarks...")
bench_path = f"{OUT}/benchmarks/data/train-00000-of-00001.parquet"
bench_raw = pd.read_parquet(bench_path)

# fullname is already the clean HF model ID (e.g. "meta-llama/Meta-Llama-3-70B-Instruct")
bench_raw["hf_model_id"] = bench_raw["fullname"].astype(str)
bench_raw["short_name"] = bench_raw["hf_model_id"].apply(
    lambda x: x.split("/")[-1].lower() if "/" in str(x) else str(x).lower()
)

# Model type: pretrained (base) vs chat/instruct
def classify_type(t):
    if pd.isna(t):
        return "unknown"
    t = str(t)
    if "pretrained" in t.lower() or "🟢" in t:
        return "base"
    if any(x in t.lower() for x in ["chat", "rlhf", "dpo", "ift", "💬"]):
        return "instruct"
    return "other"

bench_raw["model_type"] = bench_raw["Type"].apply(classify_type)

bench = bench_raw[[
    "hf_model_id", "short_name", "model_type", "Architecture",
    "#Params (B)", "Upload To Hub Date", "Base Model",
    "Average ⬆️", "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO",
    "Flagged", "Merged"
]].copy()
bench.columns = [
    "hf_model_id", "short_name", "model_type", "architecture",
    "params_b", "upload_date", "base_model",
    "avg_score", "ifeval", "bbh", "math", "gpqa", "musr", "mmlu_pro",
    "flagged", "merged"
]

# Drop flagged/merged models (contaminated)
bench = bench[~bench["flagged"] & ~bench["merged"]].copy()
bench["upload_date"] = pd.to_datetime(bench["upload_date"], errors="coerce")

print(f"  Benchmarks: {len(bench)} models after removing flagged/merged")
bench.to_parquet(f"{OUT}/benchmarks.parquet", index=False)
print(f"  Saved -> benchmarks.parquet")

# ── 3. Match Arena models to benchmark scores ─────────────────────────────────
print("\nMatching Arena models to benchmarks...")

arena_models = arena["model_name"].unique()
print(f"  Arena has {len(arena_models)} unique models")

def normalize(name):
    name = str(name).lower()
    name = re.sub(r"[_\-\.\s/]", "-", name)
    name = re.sub(r"-+", "-", name)
    return name.strip("-")

# Build lookup: normalized tokens from full HF ID and short name → bench row index
bench_lookup = {}  # norm_key -> list of bench indices
for idx, row in bench.iterrows():
    keys = set()
    if row["short_name"]:
        keys.add(normalize(row["short_name"]))
    if row["hf_model_id"]:
        keys.add(normalize(row["hf_model_id"]))
        keys.add(normalize(row["hf_model_id"].split("/")[-1]))
    for k in keys:
        bench_lookup.setdefault(k, []).append(idx)

def find_bench_match(arena_model):
    arena_n = normalize(arena_model)
    # 1. Exact match
    if arena_n in bench_lookup:
        return bench_lookup[arena_n]
    # 2. Arena name is substring of any bench key, or bench key is substring of arena name
    candidates = []
    for bk, idxs in bench_lookup.items():
        if len(arena_n) >= 6 and len(bk) >= 6:
            if arena_n in bk or bk in arena_n:
                candidates.extend(idxs)
    return list(set(candidates)) if candidates else []

matches = {}
unmatched = []
for arena_model in arena_models:
    idxs = find_bench_match(arena_model)
    if idxs:
        # Among candidates, pick highest avg_score
        best_row = bench.loc[idxs].sort_values("avg_score", ascending=False).iloc[0]
        matches[arena_model] = best_row.name
    else:
        unmatched.append(arena_model)

print(f"  Matched: {len(matches)} / {len(arena_models)}")
print(f"  Unmatched ({len(unmatched)}): {sorted(unmatched)[:30]} ...")

# Build match table
match_records = []
for arena_model, bench_idx in matches.items():
    bench_row = bench.loc[bench_idx]
    match_records.append({
        "arena_model": arena_model,
        **bench_row.to_dict()
    })

match_df = pd.DataFrame(match_records)

# Add base/instruct label from Arena model name heuristic for unmatched
def infer_type_from_name(name):
    name_l = str(name).lower()
    instruct_keywords = ["chat", "instruct", "it", "dpo", "rlhf", "sft",
                         "assistant", "turbo", "claude", "gpt-4", "gpt-3",
                         "gemini", "palm", "command"]
    base_keywords = ["base", "pretrain", "raw"]
    if any(k in name_l for k in base_keywords):
        return "base"
    if any(k in name_l for k in instruct_keywords):
        return "instruct"
    return "unknown"

# Merge match info into the arena time series
arena_with_bench = arena.merge(
    match_df[["arena_model", "model_type", "params_b", "upload_date",
              "avg_score", "ifeval", "bbh", "math", "gpqa", "musr", "mmlu_pro",
              "hf_model_id", "base_model"]],
    left_on="model_name", right_on="arena_model", how="left"
)
arena_with_bench.drop(columns=["arena_model"], inplace=True)

# Fill model_type for unmatched models using name heuristic
mask = arena_with_bench["model_type"].isna()
arena_with_bench.loc[mask, "model_type"] = \
    arena_with_bench.loc[mask, "model_name"].apply(infer_type_from_name)

matched_pct = arena_with_bench["avg_score"].notna().mean()
print(f"\n  Arena rows with benchmark match: {matched_pct:.1%}")
print(f"  Unique Arena models with benchmark data: "
      f"{arena_with_bench[arena_with_bench['avg_score'].notna()]['model_name'].nunique()}")

arena_with_bench.to_parquet(f"{OUT}/matched.parquet", index=False)
print(f"  Saved -> matched.parquet")

# ── 4. Summary ────────────────────────────────────────────────────────────────
print("\n=== Data collection complete ===")
print(f"  arena_leaderboard.parquet : {len(arena)} rows")
print(f"  benchmarks.parquet        : {len(bench)} rows")
print(f"  matched.parquet           : {len(arena_with_bench)} rows")
print(f"\nModel type breakdown in matched dataset (latest snapshot):")
latest = arena_with_bench[
    arena_with_bench["leaderboard_publish_date"] == arena_with_bench["leaderboard_publish_date"].max()
]
print(latest["model_type"].value_counts().to_string())
