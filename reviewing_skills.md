# Reviewing Skills: Pre-Submission Checklist and Failure Mode Library
*Synthesized from writing guidelines + reviews across EC (549A/B/C), MSOM (SIG-2026-0340 x2), ICML (DRS concept selection), April 2026.*

---

## The Core Principle

Every review failure traces back to one of three root causes:
1. **Design failure** — the claim, experiment, or theorem is wrong for the purpose
2. **Scope failure** — the proved setting doesn't cover the claimed setting
3. **Communication failure** — the right thing is in the paper but not findable

Design and scope failures cannot be caught at editing time. Communication failures can. The checklist below is ordered by which failures it catches and when.

---

## Scoping Stage (1 month before)

### The "So What" Test
Write one sentence a layperson could understand and find important. Not "we study X" — the actual punchline. If you can't write it, the framing isn't ready.

Evidence: Every paper in this portfolio that lost rebuttal capital to "why does this matter?" failed this test at scoping time. The papers that didn't (PeerCoPilot, RCT methodology) never faced significance challenges regardless of venue.

**The five scoping gates — if any fail, fix before committing:**
1. What is the specific claim, not just the topic? (Write it as one sentence.)
2. Is this claim open? (Search at claim level, not topic level.)
3. What is the one experiment or theorem that would prove the one-sentence summary? Would it convince a skeptic?
4. Who specifically changes their behavior if this result is true? Name people and institutions.
5. What are the three hardest questions a reviewer will ask, and do you have answers now?

### The Theory-Practice Gap Check
Before committing, sketch the main theorem. Ask:
- Does it apply to the setting that motivated the work, or a simplified version?
- Unit capacity vs. general capacity? Binary vs. continuous? i.i.d. vs. adversarial noise? Single provider vs. general M?
- If there's a gap: can you close it theoretically, bridge it empirically, or disclose it explicitly?
- If the gap cannot be closed theoretically, check whether your existing experiments already bridge it empirically. If so, surface that connection in the main text as a deliberate design choice rather than an implementation detail. A fact buried in an appendix ("concepts are discretized from continuous features") is not the same as a framing in the main text ("our framework handles continuous inputs via discretization, demonstrated in Environments X and Y").

**Evidence of gap cost:** The MSOM paper's unit capacity (c̄_j = 1) assumption was visible before execution but became the central rejection issue for both MSOM reviewers. One wrote "I do not see a reasonable path for the authors to achieve broader theoretical generalizations without fundamentally overhauling their approach." Discovering this at review is maximally expensive.

### The Benchmark Consistency Check
Fix one benchmark and use it throughout. Write down: "My benchmark is X, and I use it for every comparison." If different sections need different benchmarks, state explicitly why at the point of first use.

**Evidence:** EC 549B's core complaint was that the MSOM paper "zig zags from one approach and benchmark to a different one, without coherence or main takeaway." Theorem 3 compared Offer-One to OPT; Theorem 4 compared Offer-All to Offer-One (not OPT). Reviewers noticed immediately.

### The Literature Pre-Search
Before writing theorems, identify:
- The most natural prior work this could be framed as a special case of
- The closest paper from an adjacent community that reviews might expect

**Evidence:** Golrezaei et al. (2014) "Real-Time Optimization of Personalized Assortments" was the paper EC 549B most objected to not seeing. It's in a directly adjacent literature. Finding it before submission means positioning against it, not scrambling in rebuttal.

---

## First Edit: Story and Motivation (14 days before)

### Motivation Check
- [ ] What is the main research question? Is it stated in the first two paragraphs of the introduction?
- [ ] Is it clear why the problem is hard? Are the three main challenges stated before the algorithm?
- [ ] Have all obvious solutions been evaluated? For each baseline, is it clear why it fails and why that failure motivates your approach?
- [ ] Does the main contribution follow naturally from the problem statement, or does it appear to solve a different problem?

### The Repetition Audit
Read the abstract, then the first two introduction paragraphs, then the contributions list. If they say the same things in the same order, cut. Each should add information the previous didn't have.

**Evidence:** EC 549A: "the first two paragraphs on Page 2 largely restate ideas already presented on Page 1, while the contributions section essentially lists the same ideas again." This is the single most common introduction failure and easy to catch.

### Theorem Scope Audit
For each main theorem or proposition:
- [ ] Which setting does it apply to? (List all simplifying assumptions in the statement, not just the proof.)
- [ ] Is this the setting that motivated the paper, or a simplified version?
- [ ] If simplified: do the experiments test the practically relevant general case?
- [ ] Is tightness stated in the theorem statement if the bound is tight?

**Evidence:** EC 549A explicitly: "The tightness for the bound in Thm 4.3 should be included in the statement of the theorem, otherwise the result by itself does not really imply that the offer-one mechanism can perform poorly." Tightness in the text but not the statement reads as an afterthought.

### The Claim-Experiment Alignment Check
For each main claim:
- [ ] Write the claim as one specific sentence.
- [ ] Identify the experiment that most directly tests it.
- [ ] If the theorem requires assumption X (unit capacity, i.i.d. noise), do the experiments also use assumption X? If so, say why — or add an experiment relaxing X.

**Evidence:** MSOM Reviewer 2 on the assortment paper: "Currently, the semi-synthetic experiments set the capacity to 1 'for simplicity'. However, testing strictly with unit capacity creates an extremely competitive matching environment, which may inherently favor the SAM algorithm over naive baselines." Theory and experiments used same simplified assumption; neither covered the practical case.

### The Algorithm-Theory Identity Check
- [ ] Is the algorithm theoretically analyzed the same one implemented in experiments?
- [ ] If any constraint or modeling assumption was relaxed or changed between theory and experiments: is the gap disclosed in the main text, quantified, and justified?

Gaps found in appendix look like hidden weaknesses. Gaps disclosed in the main text look like honest characterization.

---

## Second Edit: Clear Communication (7 days before)

### Structure Check
- [ ] Can the main point be reached in two minutes of reading the introduction and Figure 1?
- [ ] Is each section's purpose clear from its first paragraph?
- [ ] What can move to the appendix without losing the logical flow? Move it.
- [ ] What is currently in the appendix that is load-bearing for understanding the main result? Move it to the main paper.

### Figure Audit (highest-leverage check)
For every figure:
- [ ] Do the axes have labels? Units?
- [ ] Does the caption explain the figure completely, without requiring the body text?
- [ ] Is there a one-sentence takeaway in the caption?
- [ ] Does the figure appear near where it is first referenced?
- [ ] Does the text accurately describe what the figure shows? (Numbers, directions, trends — check all of them.)
- [ ] Does anything in the figure appear to contradict a stated theorem? If so, is the discrepancy explained?

**Evidence of figure failures from EC 549A alone:**
- Figure 1 referenced on page 1, appears on page 3
- Figure 3 appeared to show Offer-One > Offer-All, which the reviewer read as contradicting Theorem 4.4 (actually consistent but unexplained)
- Figure 4: text says distance ∈ [1,30] but figure shows distance up to 100
- Figure 6: y-axis not explained; why omniscient and offer-one go to 5 while others are 0–0.4
- Figure 8: unexplained arrow on x-axis

Each of these generated a reviewer question. Every question is a credibility cost.

### Introduction and Abstract
- [ ] Abstract: no more than one sentence per contribution; no definitions that aren't needed
- [ ] Introduction paragraph 1: what is the problem and why does anyone care?
- [ ] Introduction paragraph 2: why is it hard? What are the obstacles?
- [ ] Introduction paragraph 3: what do you do, at high level?
- [ ] Contributions list: specific and falsifiable, not "we study" or "we investigate"
- [ ] A reader who reads only the introduction and figures should get the main message

### Theorem Presentation
- [ ] Is each theorem motivated before it is stated? (One paragraph of intuition before the formal statement.)
- [ ] Is there a proof sketch after each theorem statement?
- [ ] Is the setting (which covariates, which assumptions) named explicitly in the theorem statement?
- [ ] Are related theorems connected to each other? Does the paper explain the logical chain?
- [ ] Does the nomenclature match the difficulty? Propositions for structural observations, theorems for central results.

**Evidence:** MSOM Reviewer 2: "Some of the results currently labeled as 'Theorems' possess proofs that are relatively mathematically straightforward." Mislabeled theorems reduce perceived contribution weight.

### Experiment Presentation
For each experiment:
- [ ] What is the hypothesis being tested?
- [ ] Why is this the right experiment for that hypothesis?
- [ ] What are the results?
- [ ] What is the main takeaway, stated as a sentence?

If any of the four are missing, the experiment reads as filler.

---

## Third Edit: Details and Adversarial Reviewers (4 days before)

### The Adversarial Reviewer Simulation
Write out the three questions the most skeptical reviewer will ask. For each:
- Why can't you just use the most natural prior approach?
- What does this buy a practitioner that they couldn't do before, step by step?
- Why does the biggest assumption hold in practice?

If the answer to any of these is not in the paper body (not just the appendix), add it.

### The Adjacent Literature Check
For each adjacent community whose work could subsume yours:
- [ ] Is it cited in the related work?
- [ ] Is it explicitly distinguished from your contribution in the body of the paper, not just related work?

**Evidence:** The reason EC 549B rejected the MSOM paper is almost entirely Golrezaei et al. (2014). The reviewer wrote "The paper does not cite or engage with the highly relevant literature on online (one-sided) assortment optimization." If this had been in the paper, the review would have been different. The fix is one paragraph in the body explaining the distinction; the cost of not having it is a reject.

### Notation and Consistency Audit
- [ ] Make a list of every symbol defined. Check each is used only for one thing.
- [ ] Check every theorem for internal consistency: are the variables in the statement the same as those in the proof?
- [ ] Are there multiple words for the same concept? Pick one and use it throughout.

**Evidence:** MSOM Reviewer 2 on Theorem 1: "There is a notational contradiction between the statement of Theorem 1 and its proof that obscures a probabilistic gap." Notation errors make reviewers distrust the correctness of proofs regardless of whether the math is right.

### The Typo and Edge Case Pass
- [ ] Check every equation for index consistency (summation index in equation matches summation index used in bounds)
- [ ] Check all subscripts, superscripts, and variable ranges
- [ ] Check that set sizes are defined with the right variable (N for patients, M for providers — don't swap)

**Evidence:** MSOM Reviewer 2 found: "the authors state '...when k = N' but N denotes the number of patients, whereas the menu size should be bounded by M (providers)." And: "the summation is indexed from s=1 to S, but the summand is written as Z_k(X)" — should be Z_s(X). Both found in the same review. These generate distrust out of proportion to their size.

### Skimming Test
Read only: abstract → introduction → figure captions → conclusion. Ask:
- Is the main message clear?
- Is each figure self-explanatory?
- Is it obvious what someone should do differently as a result of this paper?

If not, revise the introduction and figure captions until it is.

---

## Recurring Failure Mode Library

Evidence-based failures across six reviewed papers. Each pattern is identified with venue evidence.

### 1. The Shifting Benchmark
**Pattern:** Different sections implicitly compare against different benchmarks. Readers notice even if it isn't stated.  
**Signature:** Reviewer says "the message is unclear" or "the paper zig-zags."  
**Fix:** State the benchmark once, use it everywhere, explain any deviation explicitly.  
**Evidence:** EC 549B on MSOM paper. EC 549C similarly.

### 2. The Tight-But-Unstated Bound
**Pattern:** The bound is tight and the tightness example is in the text, but the theorem statement doesn't say the bound is tight. The theorem by itself then seems weak.  
**Signature:** Reviewer asks "does this actually imply the mechanism performs poorly?"  
**Fix:** Add "Moreover, this bound is tight: [example]" as a second part of the theorem statement.  
**Evidence:** EC 549A on Theorem 4.3.

### 3. The Positive Result Framed as Negative
**Pattern:** A theorem shows policy A outperforms policy B in regime R. But policy B was also the "bad" policy. So the theorem looks like a contradiction.  
**Signature:** Reviewer says "I don't understand the message — isn't this a good thing?"  
**Fix:** State explicitly which regime each theorem applies to. Frame regime-specific results as phase diagrams, not failure lists.  
**Evidence:** EC 549A Q2 on Theorems 4 and 5 of the MSOM paper. Theorem 4 said Offer-All is bad at ε=0; Theorem 5 said Offer-All is good at large ε. Without the regime framing, they appeared contradictory.

### 4. Theory Without Algorithm Connection
**Pattern:** Section N characterizes failure modes of baselines. Section N+1 introduces an algorithm. No theorem explains why the algorithm addresses the specific failure modes.  
**Signature:** Reviewer says "the theory identifies symptoms but not a prescription."  
**Fix:** Add a proposition showing the algorithm reduces to the optimal in the regimes where each simple policy fails.  
**Evidence:** Both MSOM reviewers and EC 549C on the MSOM paper.

### 5. The Missing Adjacent Paper
**Pattern:** A closely related paper exists in an adjacent field that the submission doesn't cite. A reviewer from that community rejects based on this alone.  
**Signature:** Reviewer writes "this is essentially a special case of [paper]" or "this does not engage with [highly relevant literature]."  
**Fix:** Do the targeted literature search before writing, not after. For every community whose language your paper uses (assortment, matching, online algorithms), find the most relevant paper from that community.  
**Evidence:** EC 549B on Golrezaei et al. (2014). Became the primary rejection reason.

### 6. The Discrepancy Between Figure and Text
**Pattern:** The paper's text describes a figure with numbers, ranges, or trends that don't match what the figure actually shows.  
**Signature:** Reviewer quotes your text and the figure side by side and points to the contradiction.  
**Fix:** After each draft, re-read the paragraph describing each figure while looking at the figure. Check every numerical claim.  
**Evidence:** EC 549A Q7: "you say you take d̄ between 1 and 30 but in the figure the distance goes up to 100."

### 7. The Unit Capacity Trap
**Pattern:** Theoretical results are proved for unit capacity (c̄_j = 1) because it simplifies analysis. Experiments also use unit capacity "for simplicity." But the practical setting has general capacity. Neither theory nor experiments cover what the paper claims to study.  
**Signature:** Reviewer says the unit capacity setting is "a limited operational setting" or "creates an extremely competitive environment that may favor your algorithm."  
**Fix:** If theory requires unit capacity, say so explicitly in each theorem statement. Add at least one experiment with general capacity to demonstrate the algorithm still works.  
**Evidence:** Both MSOM reviewers. One wrote it was the central reason for uncertainty about publication.

### 8. The Overclaimed Scope
**Pattern:** The paper proves things for setting A (e.g., i.i.d. noise, M=1, unit capacity) but the abstract and introduction claim results for setting B (general noise, general M, general capacity).  
**Signature:** Reviewer says "the proved scope is narrower than the claimed scope."  
**Fix:** Align claims in the abstract and introduction to what is actually proved. Move broader claims to "discussion" or "future work."  
**Evidence:** research_skills.md §2: "The proved scope is narrower than the claimed scope" is one of three recurring execution failures across all six reviewed papers.

### 9. The Deterministic-Stochastic Conflation
**Pattern:** Paper models user behavior as deterministic given the platform's menu and capacities, but reviewers expect stochastic choice behavior (MNL, probit).  
**Signature:** Reviewer says "a vast literature demonstrates human decision-making is inherently stochastic" or "this conflates platform uncertainty with patient behavioral randomness."  
**Fix:** State explicitly why deterministic choice is appropriate for your setting (e.g., small choice sets, healthcare context). Add at least a sensitivity experiment with probabilistic choice.  
**Evidence:** MSOM Reviewer 2, major comment 1.

### 10. The Non-Self-Contained Figure
**Pattern:** A figure is understandable only after reading several paragraphs of body text. Reviewers who skim (most) miss the point.  
**Signature:** Multiple reviewer questions about what specific figures show, what the axes mean, or why certain lines behave a certain way.  
**Fix:** Every caption should answer: what is shown, what varies, what is the takeaway. One sentence each.  
**Evidence:** EC 549A had eight separate figure questions (Q4–Q9 and two unnumbered comments). Each question is a credibility cost.

---

## Theory-Specific Review Checklist

Before submitting a paper with theorems:

**For each theorem:**
- [ ] What setting does it apply to? All simplifying assumptions are in the statement.
- [ ] Is the bound tight? If so, state it in the theorem.
- [ ] What does this theorem tell a practitioner? Write one sentence.
- [ ] Does it apply to the setting that motivated the paper?
- [ ] Is the proof sketch in the main paper, not just the appendix?
- [ ] Is the theorem connected to the next theorem? ("This motivates X, which we prove in Theorem N+1.")

**For the theorem sequence overall:**
- [ ] Is there a logical chain from problem → failure modes → algorithm → guarantee?
- [ ] Does the guarantee theorem explain why the algorithm addresses the failure modes?
- [ ] Is every theorem labeled at the right level? (Proposition for structural observations; Theorem for central results.)
- [ ] Is the same benchmark used across all theorems?

**For approximation results:**
- [ ] Is the benchmark stated explicitly and justified?
- [ ] Is the ratio multiplicative or additive? Both? State which and when.
- [ ] Under what assumptions does the bound hold? Are those assumptions satisfied in the experimental setting?
- [ ] If the bound is vacuous at the parameter values used in experiments, is this disclosed?

---

## Experiment-Specific Review Checklist

**For each experiment:**
- [ ] What is the hypothesis?
- [ ] Is this the most direct test of that hypothesis? If not, why is this proxy valid?
- [ ] What are the results, stated as one sentence?
- [ ] What is the takeaway for a practitioner?
- [ ] Are confidence intervals / error bars shown?
- [ ] Are formal significance tests reported (t-tests, bootstrap confidence intervals, or Wilcoxon)? Error bars are necessary but not sufficient at most ML venues. With ≤10 seeds, the difference between "visually separable error bars" and "statistically significant" matters.
- [ ] Does the result appear consistent with the stated theorems? If there's an apparent contradiction, is it explained?

**For the experiment set overall:**
- [ ] Are all parameter combinations tested that are relevant to the theory?
- [ ] Is there at least one experiment that relaxes each major theoretical assumption?
- [ ] Is the algorithm that is tested the same as the algorithm that is theoretically analyzed?
- [ ] Is the baseline set complete? Does it include the "most natural prior approach" a reviewer would expect?
- [ ] If a stronger or more natural baseline exists anywhere in the paper and confirms your results, is it in the main results figure? A confirmatory baseline in the appendix is as invisible as a missing one — and creates the same reviewer complaint.
- [ ] Are all experiments reproducible from the stated setup?

---

## Writing-Specific Checks

**Introduction must answer:**
1. What is the problem? (One paragraph, lay terms.)
2. Why is it hard? (One paragraph, three specific obstacles.)
3. What do you do? (One paragraph, method at high level.)
4. What do you show? (Specific, falsifiable claims — not "we study" or "we investigate.")
5. How does it connect to prior work? (One paragraph, positioned not summarized.)

**Related work must:**
- Distinguish from each related line, not just describe it
- Include the closest adjacent-community paper even if it doesn't look related at first
- Explain in the body (not just related work) why the contribution isn't subsumed by the closest prior paper

**Each section's first paragraph must:**
- State what the section does
- Explain why this section comes here in the logical flow
- Preview the main result of the section

**Conclusion must:**
- State the main finding as one sentence
- Identify who should change their behavior and how
- Acknowledge the most important limitation explicitly

---

## The Single Most Useful Pre-Submission Test

Read only the introduction and figure captions. Imagine you are a reviewer who has thirty papers to review and will spend fifteen minutes on this one. Ask:

1. Is the main result clear and surprising?
2. Is it clear why it couldn't have been achieved before this paper?
3. Do the figures show what the text claims they show?
4. Is it obvious what a practitioner should do differently?

If any answer is no, fix the introduction and captions before touching anything else. A reviewer who gets the message from the introduction and figures will be generous about details. A reviewer who doesn't will be hostile about everything.
