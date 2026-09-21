# Research Skills: Frameworks and Recommendations
*Synthesized from conversation, April 2026. Updated with evidence from paper reviews, April 2026.*

---

## The Four Skills

1. **Crafting novel problems** — identifying what's worth working on
2. **Executing on novel problems** — proving it, building it, finishing it
3. **Marketing and presenting work** — framing, positioning, communicating impact
4. **Working and meeting people** — building relationships that lead to collaborations

Current ranking (honest): Execution > Marketing ≈ Networking > Problem Crafting

---

## 1. Crafting Novel Problems

### The core framework: world-problems vs. literature-problems

**Literature-problems** exist because the literature defined a framework with assumptions that can be relaxed. The problem disappears if the framework disappears. Examples from your portfolio: Whittle index approximation for specific RMAB variants, concept selection in RL. Audience is bounded to people who already committed to that framework.

**World-problems** exist independent of any technical framework. A non-researcher immediately understands and cares. Examples: developer burnout, patients not getting matched with the right providers, AI rankings being untrustworthy. Audience extends to multiple communities.

**Important correction:** this isn't about applied vs. theoretical work. You can do deeply theoretical work on world-problems (Arrow's impossibility, Deferred Acceptance, G-S theorem). The question isn't "is this technical?" — it's "does the problem exist outside the literature?"

**The better underlying principle:**
The real factor determining citation ceiling is: *if your result is true, how many people need to update their behavior?* SGD speedup affects everyone training ML models. RMAB bounds for a specific variant affects dozens of researchers. This is the audience breadth question, which is more precise than world vs. literature.

**What determines impact ceiling (jointly):**
1. How large is the community that uses the thing you're studying?
2. Has the framework become infrastructure, or is it still niche? (SGD = infrastructure; RMAB specific variant = not)
3. Does your result reveal a generalizable principle practitioners can use without knowing the framework?
4. Does the result have implications that extend beyond the specific setting you studied?

**How theory reaches practice — four pathways:**
1. **Theory is the algorithm** — the result is directly implementable and outperforms existing approaches (Deferred Acceptance, backpropagation, your assortment optimization)
2. **Theory sets a bound** — tells practitioners what's impossible so they don't waste effort; requires the bound to enter community vocabulary
3. **Theory explains an observed phenomenon** — practitioners see something puzzling, your theory gives it a name and mechanism (highest velocity once published)
4. **Theory provides conceptual infrastructure** — slow pathway through textbooks and courses, but durable

If you can't name which pathway your theory paper takes to reach practice, practical impact is unlikely regardless of technical depth.

### Your portfolio ranked on this metric

| Paper | Tier | Why |
|---|---|---|
| ICSE burnout | 1 — world-problem | Affects every software developer; principle generalizes to all knowledge workers |
| PeerCoPilot deployment | 1 — world-problem | Mental health access is immediately legible; real deployed system |
| Pluralistic RLHF (planned) | 1 — world-problem | RLHF is infrastructure; "scalar rewards are wrong" is legible to anyone in AI |
| Fairness in ride-pooling | 2 — world roots, technical execution | Hot keyword + legible application; fairness searched constantly |
| RCT methodology (planned) | 2 — world roots | "How do we know if AI helps people?" is universally understood |
| MSOM assortment paper | 2 — world roots | Provider turnover affecting patients is legible; managerial insights are actionable |
| LLM evaluation (Yiling) | 2-3 — depends on framing | "Can AI rankings be trusted?" is world-level; BT manipulation bounds are more literature-level |
| EC menu matching | 3 — mixed | Real application, but contribution pitched at mechanism design community |
| AIES concept uncertainty | 3 — mixed | Co-author effect inflated citations; CBMs are a literature framework |
| TMLR concept bottleneck | 4 — literature-problem | CBM locality exists because CBMs exist |
| NeurIPS bandits / RMAB | 4 — literature-problem | Specific variant of RMAB; audience bounded by framework adopters |
| ICML concept selection | 4 — literature-problem | Most niche paper in portfolio; problem created by prior literature |

### Evidence from paper reviews

Review evidence across six papers (concept selection, CBM locality, PeerCoPilot, RescueLens, two-stage RCT, MSOM assortment) directly confirms the Tier 4 significance skepticism pattern. The two Tier 4 papers (concept selection, CBM locality) faced consistent "significance: fair" scores and had to spend rebuttal capital defending why the question matters. The Tier 1-2 papers (PeerCoPilot, RescueLens, RCT, MSOM) had zero reviewers question the problem's importance — all rebuttal capital went to execution. The tier assignment predicts how hard the fight will be, not whether the paper gets through.

### Concrete habits to develop

**Keep a problem notebook, not a paper notebook.** For every paper you read, write down: (1) what did they assume away?, (2) what would change if that assumption is wrong? Review monthly. Problems that seem small when noticed often look important six months later.

**Write the specific claim before searching.** Not the topic — the actual result: "a λ-coalition achieves rank distortion Θ(λn/Δ)." Search at that level of specificity. Searching at topic level always finds prior work; searching at claim level tells you what's actually open.

**Meet one non-CS practitioner per quarter.** A clinician, social worker, food bank operator, PM at an AI company. Ask: "what's the hardest decision you make repeatedly that you think you're getting wrong?" Problems visible from the applied side often haven't been touched by theory.

**For every paper you read, write one "what did they assume away?" question.** Make this a reflex. The next paper often lives in the assumptions of the current one.

**Ask: "does my result reveal a principle practitioners can use without knowing the framework?"** The RMAB result is buried inside the framework. The assortment paper's result — "offering too much choice is as bad as too little when capacity is constrained" — is legible without knowing what assortment optimization is. That's the target.

---

## 2. Executing on Novel Problems

Currently your strongest skill. Publication record (NeurIPS, ICML x2, CHI, etc.) plus real deployments at 2.5 years is rare. The marginal gain here is smaller than elsewhere.

### The main execution gap: committing before cheap checks

The Arena paper invested significant execution before the literature confirmed the space was open. The fix is staged commitment — doing cheap checks before expensive execution.

**Stage-gate process:**

**Stage 1 (1-3 days) — before any formal model:**
1. Can you write the specific claim, not the topic?
2. Searching at that level of specificity, is it open?
3. Can you write the "so what" in two sentences a non-specialist would find important?
4. Can you name three communities who would cite it?
5. Is there a plausible path to results (sketch, not proof)?

If all five pass, proceed. If any fail, fix them or move on.

**Stage 2 (1-2 weeks) — before full commitment:**
- Sketch the formal model and one key result
- Run the quick empirical check if one exists
- If you can't sketch the main theorem, reconsider

**Stage 3 (commit):** 6-8 weeks of focused execution without second-guessing direction.

**Write the "so what" paragraph before any LaTeX.** One paragraph, plain English, aimed at a smart non-specialist. If you can't write it, the framing isn't right. This is the single most useful execution habit.

**Build empirical work in parallel, not after.** Theory and empirics should inform each other. The Arena paper would have looked different if cross-benchmark anomaly analysis had been run before the formal model was completed.

**Share sketches early.** With Fei and Yiling before full proofs, not after. A 20-minute conversation about a sketch can save a month of proving the wrong thing.

### Scoping checklist (use at Stage 2, before full commitment)

Five questions ordered by leverage. If any don't have a good answer, fix the scope before committing.

**1. Can you write the "so what" paragraph?**
One paragraph, plain English, for a smart non-specialist. If you can't write it compellingly, the problem or framing isn't ready. This is the highest-leverage question: it simultaneously catches weak problems (Skill 1), vague framing (Skill 3), and misaligned evaluation (Skill 2).

**2. What experiment would directly validate the core claim — and can you run it?**
Write the claim as one specific sentence. Ask: if the primary experiment came out positive, what would a skeptic still say is unproven? If the direct experiment is feasible, commit to running it. If genuinely infeasible (outcome data, longitudinal study), decide now what proxy you'll use and articulate why — so the paper names the limitation rather than hiding it. The evaluation gap across all six reviewed papers traces back to this question not being asked at the start.

**3. For theory papers: does the setting you can prove things about cover the setting that motivated the work?**
Sketch the main theorem before full commitment. Does it hold for general capacity, or only unit capacity? For arbitrary noise, or only i.i.d.? If there's a gap between the provable setting and the practical setting, decide now: close it theoretically, bridge it empirically, or explicitly scope the claims. Discovering this gap at review is expensive — the proofs are locked in by then. The MSOM unit capacity limitation and the DRS P1c gap were both visible before execution; they became major revision issues because they weren't addressed until reviewers raised them.

**4. Who specifically changes their behavior if this result is true?**
Not "the ML community" — name actual people. How large is that group, and does your result reveal a principle they can use without knowing your framework? This sets the significance ceiling before you start. If the audience is narrow, decide whether that's acceptable for this paper or whether the framing should be broadened.

**5. What are the three hardest questions a reviewer will ask, and what are the answers?**
Write them explicitly now. They are usually some form of: "why can't you just use [most natural prior approach]?", "what does this buy a practitioner that they couldn't do before?", and "why does [most restrictive assumption] hold in practice?" If any don't have a good answer, that's a scope problem to fix now — not in rebuttal.

### Recurring failure modes from paper reviews

Evidence from six reviewed papers shows three consistent weaknesses. These are design-layer failures, not communication-layer failures — they can't be caught at the editing stage.

**1. Evaluation doesn't directly test the core claim.**
The most consistent failure across all six papers. Papers measure something real but one step removed: interpretability claimed, performance measured; outcome improvement claimed, satisfaction surveys run; general-capacity algorithm claimed, unit-capacity experiments run. Experimental breadth is actually a consistent strength — the problem is targeting, not volume. The measurement gap is almost always visible before experiments are designed; it becomes fixed once results are in.

**2. The proved scope is narrower than the claimed scope.**
In theory papers, the setting where results hold is narrower than the setting that motivated the work, and this isn't disclosed until a reviewer finds it. When a theory-implementation gap exists (constraint dropped, assumption relaxed), it must be quantified and disclosed in the main text, not the appendix. If the gap is found in a rebuttal, it looks like a hidden weakness; if disclosed in the paper, it reads as an honest characterization.

**3. The practical "so what" is reconstructed by reviewers.**
Reviewers end up asking "who would use this, and how?" across theory and applied papers alike. The answer existed in every case — it just wasn't in the paper. Reviewers who have to construct the motivation themselves sometimes construct the wrong version (one TMLR reviewer read the abstract as criticizing concept-based explanations rather than evaluating their faithfulness). Write the practical workflow before the formal model, not after.

---

## 3. Marketing and Presenting Work

Strong but with a specific gap: positioning happens too late. Related work searches happen after drafts are written; framing gets revised after execution. This is expensive.

### The positioning-before-execution principle

The marketing framing should be established *before* committing to execution, not after. Specifically:
- Write the "so what" paragraph before the formal model
- Do the targeted literature search before writing theorems
- Know the three-community audience before the paper is drafted

### Submission checklist additions

These items are missing from standard editing checklists but surface consistently in reviews. Add to the relevant edit stage.

**First edit (story and motivation):**
- For each main theorem, does it apply to the setting that motivated the paper? If theoretical results require simplifying assumptions (unit capacity, binary concepts, i.i.d. noise), do the experiments also test the practically relevant general case?
- Write the main claim as a single sentence. If the primary experiment came out positive, what would a skeptic still say is unproven? If valid, either design a closing experiment, scope the claim down, or name the limitation explicitly in the paper.

**Second edit (main idea clearly communicated):**
- Is the algorithm theoretically analyzed identical to the one implemented? If any constraint or modeling choice was dropped or changed during development, is the gap quantified and disclosed in the main text — not the appendix?
- For deployment or applied papers: what is the behavioral outcome the system claims to produce? Is that outcome measured, or is a proxy (satisfaction ratings, quotes, survey preferences) measured instead? If proxy, does the paper explicitly state what it does and doesn't demonstrate?

**Third edit (details and adversarial reviewers):**
- What is the most natural prior work this contribution could be framed as a special case of? Does the paper explicitly explain why it isn't subsumed by that work, outside of the related work section?
- Can a practitioner in the target domain describe in two sentences a concrete workflow for using this result? If not, add it to the paper.

### What reviews consistently show is strong

These are patterns from actual reviews — things working well that should not be broken in the process of fixing weaknesses.

- **Writing clarity** — uniformly praised across all six papers and venues. Don't trade clarity for compression.
- **Figure 1** — reviewers specifically credit figures for carrying the core idea before finishing the introduction. Every paper should have a Figure 1 that conveys the main idea independently of the text.
- **Theoretical rigor within the proved scope** — proofs hold up; no reviewer found a core error. The issue is always scope, not correctness. Preserve this by not overclaiming.
- **Experimental breadth** — when experiments are designed, they cover multiple conditions and parameter settings systematically. The instinct is right; the targeting (whether the right experiment is included) needs work.
- **Problem motivation** — reviewers consistently understand why the question is being asked, even in Tier 4 papers. Don't lose this in pursuit of brevity.
- **Rebuttal quality** — papers consistently address reviewer concerns directly, often with new experiments. Don't let strong rebuttal ability substitute for catching issues before submission; it's better to spend that effort on the scoping checklist.

### Concrete habits

**Write a two-sentence plain-English summary before submitting.** The version you'd say at a conference dinner to someone from a different field. If you can't write it, the framing needs work. Also forces you to extract the generalizable principle rather than just the result.

**Give one talk per semester outside your immediate community.** A mechanism design seminar, HCI seminar, clinical informatics group. Each forces you to pitch to a different audience and reveals which parts of the framing don't travel. More useful feedback than another ML seminar.

**Submit to workshops 6 months before the main paper is done.** Workshop feedback is faster and cheaper than conference rejection. Gets your work in front of people before the final submission, building awareness.

**For fellowship applications:** practice the two-minute verbal pitch with someone who isn't in CS. Fellowship committees are often not specialists. If a non-CS person doesn't find it important by the end of two minutes, the pitch needs to change.

**The "bar conversation test":** can you explain the result in two sentences to someone outside your field, and do they find it surprising or important? Not interesting — *surprising* or *alarming*. Papers that pass this test have broader reach.

---

## 4. Working and Meeting People

Good foundations (Yiling, Wilder, Churchill connections), but the current pattern of "many chats, few conversions" has specific fixable causes.

### Why chats don't convert

Most networking conversations are **mutual interest conversations** ("we both find X interesting"), not **joint problem conversations** ("I'm stuck on Y and you have exactly what I need"). Collaborations start from the second type, not the first.

Mutual interest conversations don't convert to collaborations. Collaborations happen when:
- You have a tool/method they need for their problem, OR
- They have data/domain knowledge/proof technique you need for yours

Random chats rarely surface these asymmetries because nobody comes in with a specific problem they're trying to solve.

### Two types of conversations — different goals

**Exploratory conversations** (conference chats, seminar dinners, introductions): Goal is not collaboration. Goal is finding out enough about someone to know whether a focused conversation is worth having. Success = leaving knowing one specific thing about their work you didn't know before, and them knowing one about yours.

**Focused conversations** (scheduled calls, working meetings): These have a specific problem at the center. This is where collaborations actually start.

Most people have too many exploratory and not enough focused. The first type is how you find who deserves a second conversation of the second type.

### The conversion mechanism

The gap in your current pattern: exploratory chat → pleasant ending → nothing. The missing step is converting a promising exploratory conversation into a focused one.

During exploratory conversations: **listen for what they're stuck on**, not just what they work on. Ask "what's the hard part?" Most people will tell you.

**End with a specific next step when there's overlap.** Not "let's stay in touch" — "you mentioned you're trying to X, I've been thinking about that, can we do 30 minutes next week?" That's the conversion.

**If no overlap:** ask "who else do you think I should talk to about X?" Converts a dead-end into two new leads.

### Why sending papers gets low response

- "Here's my paper" is a broadcast, not an ask — people don't know what to do with it
- Finished papers are hard to engage with because all decisions are made
- Cold sends to people without prior relationship almost never work
- "Thought you'd find this interesting" is not a reason

**The version that works:**
- Send a draft or sketch, not a finished paper — asks for input when input is still useful
- Include a specific question tied to their work: "your paper on X relates directly to our result on Y — does this modeling assumption hold in your setting?"
- Make the connection explicit: not "thought you'd find this interesting" but "this specific result connects to your work in this specific way"
- Send after at least one prior conversation

### Concrete habits

**At each conference, schedule three specific conversations in advance.** Look at the attendee list and program beforehand. "I want to talk to X about Y" with a calendar entry. Cold emails sent a week before the conference get responses; same emails during the conference get lost.

**Follow up within 24 hours with something specific.** Not "great meeting you" — "you mentioned X, I read the paper and here's what I thought" or "the connection you drew made me think about Y." People remember conversations that continued.

**Send the paper before you need something.** If you want a letter or collaboration for 2027, send relevant work in fall 2026 with a note about the specific connection to their work. Build the relationship when there's no ask attached — the ask six months later lands in a completely different context.

**Convert the networking list in your plan to specific actions with triggers.** "Email Wei-Lin Chiang at LMSYS the week the paper is submitted, with these two specific questions" is a commitment. "Reach out when the paper is submitted" is not.

---

## Time Allocation: Idea Generation vs. Execution

**Steady state: ~20% idea generation, ~80% execution.**

The 20% runs continuously at low intensity — background processing, not focused sprints. It looks like:
- Reading one paper outside your immediate area weekly (an hour, regardless of current execution)
- Writing assumptions and open questions from papers you're already reading
- Problem notebook entries when something strikes you, not on a schedule
- Paying attention differently during seminars and conversations you'd attend anyway

**Focused bursts on idea selection only at genuine decision points:**
- Choosing the summer project
- Deciding what to write after a paper is submitted
- Responding to getting scooped
- These are bounded: 2-3 weeks maximum, then back to execution

**Why continuous low-intensity beats periodic bursts:** the best ideas come from noticing an assumption while reading, a conversation surfacing unexpected overlap, sitting with an execution problem until you realize the framing is wrong. These don't happen on a schedule. Dedicated brainstorming sessions produce ideas optimized for sounding good in brainstorming, not ideas that survive contact with the literature and execution.

**On when to commit:** once an idea clears the stage-gate checks and is better than what you've generated recently, commit. Don't wait for the perfect idea. Execution teaches you things about the problem that ideation never will.

---

## The Single Most Useful Habit Across All Four

**Write the "so what" paragraph before anything else.**

Before the formal model, before the literature search, before any LaTeX: one paragraph in plain English aimed at a smart non-specialist, explaining what would be surprising or important about the result if it's true.

This single habit simultaneously:
- Forces the world-problem framing (Skill 1)
- Catches wrong directions before expensive execution (Skill 2)
- Establishes the marketing framing early (Skill 3)
- Gives you something to share and discuss in conversations (Skill 4)

If you can write it compellingly, proceed. If you can't, the problem or framing isn't right yet.
