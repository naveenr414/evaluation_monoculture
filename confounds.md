# Known Confounds and Alternative Explanations

We document six known structural features of Chatbot Arena that represent potential alternative explanations for our findings, and assess how much of each finding each confound can explain.

---

## 1. Length and Verbosity Bias

**What it is.** Human evaluators systematically prefer longer, more formatted responses independent of content quality. LMSYS quantified this in August 2024, finding response length has a coefficient of 0.249–0.267 in a regression predicting voter preference, and introduced a "style-controlled" ELO as a corrective. Our analysis confirmed a significant verbosity-ELO correlation in the 2025 battle data (Spearman ρ = 0.578, p < 0.0001), concentrated in the top ELO tier (ρ = 0.714).

**Does it explain our main findings?** No. We replicated the C4 compression analysis using style-controlled ELO scores (available from August 2024 onward) and find the same pattern: style-controlled WR-variance trend slope = −0.000084/yr, R² = 0.431, p < 0.0001. The compression is not an artifact of lengthening responses mechanically inflating top model ELOs. Additionally, the gap between raw and style-controlled ELO for top-20 models has moved from +24 points (2024, raw > SC) to −15 points (2026, SC > raw), indicating that top models' length advantage has actually reversed as all frontier models converged on long outputs — if anything, failing to control for length slightly deflates top-model ELO in 2026 rather than inflating it.

**Does it explain the verbosity-ELO correlation?** Partially. The correlation is real and driven by known bias. However, it is heavily confounded by reasoning models (DeepSeek-R1, QwQ, Gemini 2.5 Flash with thinking mode) whose long outputs reflect chain-of-thought generation, not strategic padding. Disentangling these is not possible with the available data.

---

## 2. Private Testing and Selective Score Disclosure

**What it is.** "The Leaderboard Illusion" (Cohere Labs et al., arXiv 2504.20879, NeurIPS 2025 Datasets & Benchmarks) documents that select providers — Google, OpenAI, Meta, Anthropic — have privately tested multiple model variants before public release and selectively disclosed only the best-performing scores. Meta tested 27 private variants in the lead-up to the Llama-4 release. Providers receiving this access account for approximately 60% of all Arena battle data (Google ~19%, OpenAI ~20%), while 83 open-weight models collectively received ~30%.

**Does it explain C4 (frontier compression)?** Substantially and importantly. This is the strongest alternative explanation for our compression finding. If each major lab tests N variants privately and releases only the one with the highest Arena ELO, the published top models are order statistics — the maximum over N draws from each lab's capability distribution. As N grows and this practice spreads, published scores mechanically cluster near each lab's maximum achievable ELO. The gaps between labs' published bests will be small regardless of whether models genuinely converged or gamed Arena signals. This confound cannot be ruled out from our data and must be acknowledged prominently.

**Does it explain Option C (statistical unresolvability)?** Partially. If published top models are all near their labs' ceiling, they will be tightly clustered in ELO regardless of the true distribution of capability, producing the CI/gap pattern we observe.

**Mitigation.** LMSYS published a benchmarking policy in December 2024 committing to consistent sampling rates across models. However, historical data through 2024 is affected, and enforcement is unverified.

---

## 3. Vote Rigging and Adversarial Manipulation

**What it is.** "Improving Your Model Ranking on Chatbot Arena by Vote Rigging" (arXiv 2501.17858, ICML 2025) demonstrates that omnipresent rigging strategies — exploiting the global Bradley-Terry rating mechanism so any vote can influence any model's rating — can improve a target model's rank by 15 positions using only hundreds of additional votes. The attack does not require voting on battles involving the target model directly.

**Does it explain our findings?** This mechanism is a direct empirical instantiation of v-inflation: votes are cast to optimize a model's θ rather than to honestly report preference. If multiple labs or third parties use such strategies, they could inflate top model ELOs artificially and compress the frontier. However, the quantitative scale (hundreds of votes against 1.7 million total) makes it unlikely to drive the multi-year, platform-wide compression we observe. More likely a tail risk than a primary cause.

---

## 4. Prompt Distribution Shift

**What it is.** The distribution of prompts submitted to Arena has changed substantially since 2023. Early Arena featured short, simple prompts from a small pool of users. By 2025, Arena attracted harder prompts (dedicated hard-prompts and expert categories were added), more coding tasks, and direct-challenge battles with prior conversational context. The user population skews toward technically sophisticated users whose preferences may not match practitioners or end-users.

**Does it explain C4?** Potentially partially. If harder prompts better discriminate models, adding them should increase ELO spread among frontier models — the opposite of what we observe. Conversely, if prompt difficulty converged (everyone asking similar hard questions), it could reduce discriminability. The direction of this effect is ambiguous and cannot be cleanly estimated from available data.

**Does it explain the C1 negative result?** Yes, partially. As prompts became harder and more coding-focused, the capability dimensions rewarded by Arena may have shifted, making ELO harder to compare across time periods. The apparent stability of ELO-benchmark correlation could partly reflect two changes cancelling out.

---

## 5. Asymmetric Battle Sampling

**What it is.** Arena's active sampling prioritises battles between closely matched models — pairs where the outcome is uncertain and would most reduce ELO uncertainty. This means high-performing models receive more battles than lower-performing ones. Frontier models accumulate votes faster. This design choice is statistically sensible but creates a structural feature relevant to our Option C finding.

**Does it explain Option C (CI/gap ratio)?** In the direction that should help, not hurt. More battles → narrower CIs → lower CI/gap ratio. Yet we observe the ratio growing rapidly (doubling every 0.7–0.9 years). This means ELO gaps among frontier models are shrinking faster than the additional battles can compensate. The asymmetric sampling actually makes our Option C finding *more* conservative — the unresolvability problem is worse than it appears because the system is already trying to allocate votes optimally.

---

## 6. Style Control Not Applied to Historical Analysis

**What it is.** Arena introduced style-controlled ELO in August 2024. Our main analysis used the raw `overall` category ELO from May 2023 onward, which does not correct for response length or markdown formatting. The style-controlled dataset only provides coverage from August 2024 onward.

**Impact.** For the 2023–2024 period, raw ELO slightly overstates the advantage of verbose models. As shown in the comparison above, the raw-SC gap for top-20 models was +24 ELO points in 2024, meaning raw ELO was somewhat inflated for long-response models. By 2026 the gap reversed (SC > raw by 15 points), as universal long-output adoption changed the direction of the bias.

**Impact on our findings.** As demonstrated above, C4 compression holds in style-controlled ELO with comparable magnitude and significance. Our 2023 raw-ELO estimates are slightly more favourable to verbose models, but this does not create or amplify the compression trend.

---

## Summary Assessment

| Confound | C4 (compression) | Option C (unresolvable) | C1 (negative) | Verbosity-ELO |
|----------|-----------------|------------------------|---------------|--------------|
| Length bias | Does not explain (SC replication) | Does not explain | Minor | Major (confounded) |
| Private testing / selective disclosure | **Substantial alternative explanation** | Partial | Unclear | Not relevant |
| Vote rigging | Possible but small scale | Small | Not relevant | Not relevant |
| Prompt distribution shift | Ambiguous direction | Not directly | Partial | Not relevant |
| Asymmetric sampling | Makes finding more conservative | Makes finding more conservative | Not relevant | Not relevant |
| No style control in 2023 | Does not explain (SC replication) | Does not explain | Minor | Not relevant |

The most important confound is **private testing and selective disclosure**. Our compression finding (C4) and the statistical unresolvability finding (Option C) are consistent with a world where frontier labs all optimise their Arena submissions — testing many variants, publishing only the best — which mechanically clusters published scores near each lab's ceiling. Distinguishing this from genuine capability convergence or v-inflation would require access to the private test scores that labs do not disclose.

The finding that appears most robust to all confounds is **Analysis 3 (votes needed)**. Regardless of why frontier model ELOs are clustered — genuine convergence, selective disclosure, or v-inflation — the practical consequence is identical: the current vote count cannot statistically resolve frontier rankings, and the number of votes required is doubling faster than Arena can collect them. This finding holds under style-controlled ELO and is made more, not less, conservative by Arena's asymmetric sampling policy.
