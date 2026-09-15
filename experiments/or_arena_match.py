"""OpenRouter slug -> Arena model_name matching for Experiment H.

Two layers:
1. Safe automatic matching: exact token match, or exact match after dropping
   genuine 8-digit YYYYMMDD date tokens. Both require a UNIQUE Arena candidate.
   No fuzzy/trimming heuristics -- those were tried and silently produce wrong
   matches (e.g. collapsing "kimi-k2.5" into "kimi-k2" by treating the ".5" as
   part of a date to be stripped). Coverage: ~130 models / 57% of tokens.
2. Manual overrides: every remaining OpenRouter model with material volume,
   read and linked (or explicitly left unmatched) by hand against the Arena
   model list. Each entry is tagged 'high' (unambiguous, just formatting/order
   differences or a single obvious date-dropped candidate), 'medium' (a real
   judgment call -- e.g. picking a reasoning-effort variant, a checkpoint
   promoted from preview to GA, or a precision variant), or left out entirely
   when nothing in Arena is a defensible match (stealth OpenRouter codenames,
   embedding/vision/safety/base models outside the chat arena's scope, or a
   version/checkpoint genuinely absent from Arena).

Left OUT on purpose (not a coverage gap, just out of scope for a chat arena):
  - OpenRouter stealth/cloaked codenames: owl-alpha, hunter-alpha, elephant-alpha,
    healer-alpha, sonoma-sky-alpha, sonoma-dusk-alpha, quasar-alpha, horizon-alpha,
    horizon-beta, polaris-alpha, sherlock-think-alpha, sherlock-dash-alpha,
    pony-alpha, optimus-alpha, bert-nebulon-alpha, andromeda-alpha, cypher-alpha,
    aurora-alpha -- Arena never lists these; by the time a model ships publicly
    it has its real name, and there's no reliable public mapping.
  - embeddings (text-embedding-3-*, gemini-embedding-001, qwen3-embedding-*,
    pplx-embed-v1, multilingual-e5-large, all-minilm-l6-v2)
  - vision-only / safety / base / distill checkpoints not separately benchmarked
    in the CORE text categories (llama-guard, r1-distill-*, v3-base, *-vl-*,
    gpt-oss-safeguard, llama-3.2-11b-vision)
  - niche community fine-tunes never benchmarked on LMArena (sao10k/*, gryphe,
    thedrummer, neversleep, infermatic, anthracite-org, cognitivecomputations,
    wizardlm-2-* [Arena's wizardlm-13b/70b is the OLDER v1 line, not v2])
"""
import re

# slug (post "provider/", pre ":variant") -> (arena model_name, confidence)
OVERRIDES = {
    # --- Gemini ---
    'gemini-3-flash-preview-20251217': ('gemini-3-flash', 'medium'),
    'gemini-3-pro-preview-20251117': ('gemini-3-pro', 'high'),
    'gemini-3.1-flash-lite-20260507': ('gemini-3.1-flash-lite-preview', 'high'),
    'gemini-3.1-flash-lite-preview-20260303': ('gemini-3.1-flash-lite-preview', 'high'),
    'gemini-3.1-pro-preview-customtools-20260219': ('gemini-3.1-pro-preview', 'medium'),
    'gemini-2.5-flash-lite-preview-09-2025': ('gemini-2.5-flash-lite-preview-09-2025-no-thinking', 'high'),
    'gemini-flash-1.5': ('gemini-1.5-flash-002', 'high'),
    'gemini-flash-1.5-8b': ('gemini-1.5-flash-8b-001', 'high'),
    'gemini-pro-1.5': ('gemini-1.5-pro-002', 'high'),

    # --- OpenAI ---
    'gpt-4o-mini': ('gpt-4o-mini-2024-07-18', 'high'),
    'chatgpt-4o-latest': ('chatgpt-4o-latest-20250326', 'high'),
    'o3-mini-2025-01-31': ('o3-mini', 'high'),
    'o3-mini-high-2025-01-31': ('o3-mini-high', 'high'),
    'gpt-5-2025-08-07': ('gpt-5-high', 'medium'),
    'gpt-5-mini-2025-08-07': ('gpt-5-mini-high', 'medium'),
    'gpt-5-nano-2025-08-07': ('gpt-5-nano-high', 'medium'),
    'gpt-5.4-mini-20260317': ('gpt-5.4-mini-high', 'medium'),
    'gpt-5.4-nano-20260317': ('gpt-5.4-nano-high', 'medium'),
    'gpt-5.3-chat-20260303': ('gpt-5.3-chat-latest', 'high'),

    # --- Anthropic ---
    # bare, undated OR alias: Anthropic redirects it to the current default;
    # pick the more recent of the two dated Arena checkpoints.
    'claude-3.5-sonnet': ('claude-3-5-sonnet-20241022', 'medium'),

    # --- DeepSeek ---
    'deepseek-chat-v3': ('deepseek-v3', 'high'),
    'deepseek-chat-v3-0324': ('deepseek-v3-0324', 'high'),
    'deepseek-chat-v3.1': ('deepseek-v3.1', 'high'),

    # --- Qwen ---
    'qwen3-235b-a22b-04-28': ('qwen3-235b-a22b', 'high'),
    'qwen3-235b-a22b-07-25': ('qwen3-235b-a22b-instruct-2507', 'high'),
    'qwen3-30b-a3b-04-28': ('qwen3-30b-a3b', 'high'),
    'qwen3-32b-04-28': ('qwen3-32b', 'high'),
    'qwen3-coder-480b-a35b-07-25': ('qwen3-coder-480b-a35b-instruct', 'high'),
    'qwen3-next-80b-a3b-instruct-2509': ('qwen3-next-80b-a3b-instruct', 'high'),
    'qwen3-next-80b-a3b-thinking-2509': ('qwen3-next-80b-a3b-thinking', 'high'),
    'qwen3.6-plus-04-02': ('qwen3.6-plus', 'high'),
    'qwen3.6-plus-preview': ('qwen3.6-plus', 'high'),
    'qwen3.7-max-20260520': ('qwen3.7-max-preview', 'high'),
    'qwen-plus-2025-01-25': ('qwen-plus-0125', 'high'),

    # --- Gemma ---
    'gemma-4-31b-it-20260402': ('gemma-4-31b', 'high'),
    'gemma-4-26b-a4b-it-20260403': ('gemma-4-26b-a4b', 'high'),

    # --- Grok (xAI) ---
    'grok-4-07-09': ('grok-4-0709', 'high'),
    'grok-4.1-fast': ('grok-4-1-fast-reasoning', 'medium'),
    'grok-3-mini': ('grok-3-mini-beta', 'medium'),

    # --- Moonshot Kimi ---
    'kimi-k2.5-0127': ('kimi-k2.5-instant', 'medium'),

    # --- MiniMax ---
    'minimax-m2.1': ('minimax-m2.1-preview', 'high'),

    # --- Xiaomi MiMo ---
    'mimo-v2-flash-20251210': ('mimo-v2-flash (non-thinking)', 'medium'),
    'mimo-v2-omni-20260318': ('mimo-v2-omni', 'high'),
    'mimo-v2-pro-20260318': ('mimo-v2-pro', 'high'),
    'mimo-v2.5-20260422': ('mimo-v2.5', 'high'),
    'mimo-v2.5-pro-20260422': ('mimo-v2.5-pro', 'high'),

    # --- Zhipu GLM (z-ai) ---
    'glm-4.7-20251222': ('glm-4.7', 'high'),
    'glm-4.7-flash-20260119': ('glm-4.7-flash', 'high'),
    'glm-5-20260211': ('glm-5', 'high'),
    'glm-5.1-20260406': ('glm-5.1', 'high'),

    # --- Tencent Hunyuan ---
    'hy3-preview-20260421': ('hunyuan-hy3-preview', 'high'),

    # --- Amazon Nova ---
    'nova-lite-v1': ('amazon-nova-lite-v1.0', 'high'),
    'nova-micro-v1': ('amazon-nova-micro-v1.0', 'high'),
    'nova-pro-v1': ('amazon-nova-pro-v1.0', 'high'),

    # --- Mistral ---
    'mixtral-8x7b-instruct': ('mixtral-8x7b-instruct-v0.1', 'high'),
    'mixtral-8x22b-instruct': ('mixtral-8x22b-instruct-v0.1', 'high'),
    'ministral-8b': ('ministral-8b-2410', 'medium'),

    # --- NVIDIA Nemotron ---
    'nemotron-3-super-120b-a12b-20230311': ('nvidia-nemotron-3-super-120b-a12b', 'high'),
    'nemotron-3-nano-30b-a3b': ('nvidia-nemotron-3-nano-30b-a3b-bf16', 'high'),

    # --- Meta Llama ---
    # bf16/fp8 are two precision releases of the same weights; OR doesn't
    # distinguish, so pick the reference (bf16) one.
    'llama-3.1-405b-instruct': ('llama-3.1-405b-instruct-bf16', 'medium'),
}


def tokenize(s):
    s = s.lower().split('/')[-1].split(':')[0]
    return re.findall(r'\d+|[a-z]+', s)


def _mset(toks):
    return (tuple(sorted(t for t in toks if t.isalpha())),
            tuple(sorted(t for t in toks if t.isdigit())))


def _drop_8digit(toks):
    return [t for t in toks if not (t.isdigit() and len(t) == 8)]


def match(or_slugs, arena_models):
    """Return {slug: (arena_name, method)} for slugs in `or_slugs`.

    method is 'exact', 'drop8digit', or 'override:<confidence>'.
    """
    amap_raw, amap_d8 = {}, {}
    for m in arena_models:
        t = tokenize(m)
        amap_raw.setdefault(_mset(t), []).append(m)
        amap_d8.setdefault(_mset(_drop_8digit(t)), []).append(m)

    out = {}
    for slug in or_slugs:
        base_key = slug.split('/')[-1].split(':')[0]
        if base_key in OVERRIDES:
            name, conf = OVERRIDES[base_key]
            if name in arena_models:
                out[slug] = (name, f'override:{conf}')
            continue
        t = tokenize(slug)
        key = _mset(t)
        if key in amap_raw and len(amap_raw[key]) == 1:
            out[slug] = (amap_raw[key][0], 'exact')
            continue
        key2 = _mset(_drop_8digit(t))
        if key2 in amap_d8 and len(amap_d8[key2]) == 1:
            out[slug] = (amap_d8[key2][0], 'drop8digit')
    return out
