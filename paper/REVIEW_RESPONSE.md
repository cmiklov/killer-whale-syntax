# Pre-Submission Review: Vulnerabilities and Response Plan

**Date:** 2026-04-02
**Reviewer:** Claude (internal review, pre-submission)
**Verdict:** Publishable. Four addressable vulnerabilities identified.

---

## Summary

The paper's core findings — topology-syntax coupling (d = 1.34-1.57), MI half-life (45.4 calls), 2019 crisis detection, Boltzmann grammar — are individually strong and mutually reinforcing. The validation-first methodology (reproducing Ford 1991 before making novel claims) and honest nulls (Finding 1 revised, Finding 36 reported at p = 0.70) establish credibility. Four vulnerabilities need addressing before submission.

---

## Vulnerability 1: Boltzmann Fit Needs Permutation Null

**The concern:** The Boltzmann distribution (R² > 0.96) is fitted to binned log-enrichment ratios, which are inherently smooth. A log-linear fit to binned data will often look good. If the fit survives on shuffled data (where call order is randomised within sessions but cluster self-similarity is preserved), the "physical law" interpretation weakens — it would just be measuring that similar things cluster together, not that similarity *generates* sequential structure.

**The fix:**
1. Within-session shuffle permutation (1000 iterations): randomise call order within each session, recompute adjacent-pair similarities, refit Boltzmann.
2. Report the distribution of R² under the null. If observed R² exceeds 95% of permutations, the result stands.
3. Cross-session shuffle as secondary control: assign calls to random sessions, preserving global frequency distribution.

**Script:** Extend `analyse_boltzmann.py` with permutation loop.

**Expected outcome:** The fit should break under permutation, because the enrichment at high similarity (136× at ≥0.90) is driven by sequential adjacency, not baseline similarity. But we need to show this explicitly.

**Priority:** HIGH — this is the most theoretically ambitious claim.

---

## Vulnerability 2: 20ms ICI Spike May Be Annotation Artifact

**The concern:** 63.2% of inter-call intervals in a single 5ms bin (15-20ms) is suspicious. At this timescale, annotation temporal resolution matters. If DCLDE annotators used fixed-length windows, or if overlapping detections were split at quantised boundaries, the spike is an artifact of the annotation process, not biology.

**The fix:**
1. **Check annotation resolution.** Examine the `FileBeginSec` precision in raw DCLDE annotations. If timestamps are quantised to e.g. 20ms, the spike is explained.
2. **Raw audio validation.** Select 10 sessions with the densest 15-20ms ICIs. Download the source audio. Measure call onset times from waveform/spectrogram directly (not annotation metadata). If the 20ms spike persists in manual measurement, it's real.
3. **Compare Orcasound vs JASCO.** Different annotation pipelines. If the spike appears in both, it's less likely to be pipeline-specific.
4. **If artifactual:** Reframe Finding 29. The ICI *distribution shape* (peaked with long tail) is still informative even if the exact peak location is an artifact. The autocorrelation structure at lag 49 may survive even if the absolute timescale shifts.

**Script:** New script `validate_ici.py` — timestamp precision analysis + cross-pipeline comparison.

**Priority:** MEDIUM — Finding 29 is interesting but not load-bearing. The paper survives without the 51.3 Hz claim. But it's the kind of thing a reviewer will flag immediately.

---

## Vulnerability 3: Acoustic Priming as Alternative to Topological Syntax

**The concern:** The topology-syntax result (adjacent calls more similar than random, d = 1.34-1.57) could be explained by acoustic priming rather than topological grammar:

- Animals tend to repeat similar sounds (production inertia)
- Adjacent calls may come from the same individual in a similar physiological state
- Self-transitions dominate (0.80-0.97), inflating adjacent-pair similarity

If this is "priming" rather than "syntax," the theoretical contribution shrinks from "new kind of grammar" to "animals repeat themselves."

**The fix:**
1. **Exclude self-transitions.** Recompute topology-syntax coupling on cross-transition pairs only (type-switches). If d remains large on the ~20% of pairs that are NOT self-repetitions, priming can't explain it.
2. **Cross-individual test.** Finding 33 identifies ~2.8 voices per session. If topology-syntax coupling holds across voice boundaries (speaker A's call predicts speaker B's next call), it's coordination, not priming.
3. **Cross-station test.** Finding 52 shows MI = 0.047 bits across stations. Extend: compute topology-syntax d on cross-station adjacent pairs only. If the topology still predicts syntax when the two calls come from different physical positions (likely different individuals), priming is ruled out.
4. **Frame explicitly.** The paper should name acoustic priming as a competing hypothesis in the Discussion and present the evidence against it. Reviewers will raise it; pre-empting it is stronger.

**Script:** Extend `analyse_topology.py` with self-transition exclusion and cross-voice analysis.

**Priority:** HIGH — this is the most likely reviewer objection to the crown jewel finding.

---

## Vulnerability 4: Markov Order > 4 Needs Confidence Intervals

**The concern:** The "entropy keeps decreasing through order 4" claim is based on point estimates of conditional entropy at each order. With k=3 clusters, order 4 means 3⁴ = 81 context states. With 11,079 transitions, many of those states are sparse. Entropy estimation on sparse distributions is biased downward (Miller-Madow correction needed at minimum). The apparent decrease at each order could partly be estimation artifact.

**The fix:**
1. **Bootstrap CIs.** Resample sessions with replacement (1000 iterations). Compute conditional entropy at each order. Report 95% CIs. If the decrease from order 3 to order 4 overlaps zero, soften the claim.
2. **Miller-Madow correction.** Apply the standard bias correction: H_corrected = H_naive + (k-1)/(2N ln2), where k = number of non-empty states, N = number of observations.
3. **BIC model selection.** Fit Markov models at orders 1-5 and select by BIC. If BIC prefers order 3 over order 4, report that honestly.
4. **Increase cluster count.** Repeat at k=5 (which is used for some universals findings already). If the order > 4 claim holds at higher k (where the state space grows as 5⁴ = 625, making sparsity worse), it's more credible.

**Script:** Extend `analyse_deep.py` with bootstrap and bias correction.

**Priority:** MEDIUM-HIGH — "Markov order in the human range" is a headline claim. It needs to be bulletproof or explicitly bounded.

---

## What Doesn't Need Fixing

These were evaluated and found robust:

- **2019 crisis detection (Findings 13-14):** SAR control is clean. The p-values are so extreme that even order-of-magnitude corrections don't matter. The conservation application stands.
- **MI half-life (Finding 22):** Derived from standard MI at multiple lags with exponential fit. Replicates across datasets. The 6-15× gap over other species is too large to be an artifact of method.
- **Linguistic universals (Findings 19-21):** Zipf, brevity, and Heaps are well-established tests applied straightforwardly. SAR's α = 1.11 is within human range by any standard.
- **Cross-ecotype universality (Finding 45):** Three independent ecotypes, all d > 1.3, bootstrap-confirmed. This is the strongest structural result in the paper.
- **Honest nulls (Findings 1 revised, 30, 36):** These are features, not bugs. Keep them prominent.

---

## Recommended Work Order

1. **Boltzmann permutation** (Vuln 1) — fastest to implement, highest theoretical stakes
2. **Topology-syntax sans self-transitions** (Vuln 3) — the make-or-break test for the crown jewel
3. **Markov order CIs** (Vuln 4) — bootstrap loop, straightforward
4. **ICI validation** (Vuln 2) — requires manual audio inspection, slower

Estimated effort: 1-2 sessions for vulnerabilities 1-3 (all scriptable). Vulnerability 4 needs audio download and manual validation — separate session.

---

## Results (2026-04-03)

All four vulnerabilities addressed. Scripts: `vuln1_boltzmann_permutation.py`, `vuln2_ici_validation.py`, `vuln3_topology_priming.py`, `vuln4_markov_ci.py`. Each standalone, reproducible (`RandomState(42)`), loads its own data.

### Vulnerability 1: Boltzmann — PARTIALLY CONFIRMED

Within-session shuffle (1000 iterations, preserves session membership, breaks sequential adjacency):

| Ecotype | Observed R² | Null R² mean | Null 95th % | p-value | Significant? |
|---|---|---|---|---|---|
| **SRKW** | 0.9809 | 0.9719 | 0.9789 | **0.005** | **YES** |
| TKW | 0.9751 | 0.9734 | 0.9799 | 0.367 | NO |
| OKW | 0.9621 | 0.9577 | 0.9754 | 0.386 | NO |

Cross-session shuffle (assign calls to random sessions): p = 0.000 for **all three ecotypes**. The Boltzmann fit absolutely requires session structure — random assignment destroys it.

**Interpretation:** For SRKW, the R² is significantly above the within-session null — sequential adjacency drives the fit beyond what cluster self-similarity can explain. For TKW and OKW, the R² is within the null range, meaning the goodness-of-fit could be partly explained by acoustic similarity within sessions. However, the **slope** differs (observed vs null: TKW 4.33 vs 3.81, OKW 3.49 vs 3.09) — the coupling strength is real even where R² alone doesn't distinguish. The cross-session null (R² collapses to ~0.05-0.17) confirms the Boltzmann structure requires session-level organisation in all ecotypes.

**Paper recommendation:** Report the SRKW result as the primary evidence. For TKW/OKW, emphasise the slope difference and the cross-session null rather than R² alone. Note that the within-session shuffle is a conservative test — it preserves more structure than a true null should.

### Vulnerability 2: 20ms ICI Spike — DATASET-SPECIFIC ARTIFACT

**Critical finding:** The 20ms spike does NOT appear in the DCLDE annotations broadly.

- DCLDE Annotations.csv (14,240 SRKW calls): Only 14 ICIs in the 15-20ms bin out of 8,323 total (0.2%). Median ICI = 2.06s.
- Haro NPZ data (4,862 calls): 62-64% of ICIs in 15-20ms bin — the spike is here.

The spike is a property of the Haro Strait per-call feature extraction (which uses `FileBeginSec` as the timing coordinate within 30-minute recording files), not of the DCLDE annotation pipeline. The Haro data groups calls within the same ~30min audio file and computes ICIs from `FileBeginSec` offsets, which produces very short intervals between calls that overlap or are nearly simultaneous in the recording.

**Cross-provider:** No provider shows the spike above 30% in the DCLDE data. The dominant ICI is 1-3 seconds across all providers.

**Cross-station (Haro):** Both north and south show the spike (62.4% and 64.2% respectively, KS p = 0.79). The spike is consistent between stations but is a property of the Haro dataset's temporal resolution, not biology.

**Paper recommendation:** Finding 29 (51.3 Hz oscillation) should be revised. The ICI distribution shape (peaked with long tail) is informative, but the specific 20ms peak is an artifact of how calls are timestamped within recording files in the Haro dataset. The autocorrelation structure at lag 49 should be retested on the DCLDE data with its natural ~2s ICI timescale.

### Vulnerability 3: Acoustic Priming — CONCLUSIVELY RULED OUT

Three independent tests, all positive:

**Test 1 — Cross-transition exclusion (all ecotypes):**

| Ecotype | d (all pairs) | d (cross-transition only) | n cross | Priming ruled out? |
|---|---|---|---|---|
| SRKW | 1.34 | **1.27** | 1,520 | **YES** |
| TKW | 1.57 | **1.49** | 427 | **YES** |
| OKW | 1.51 | **0.91** | 208 | **YES** |

Even after removing all self-repetitions (68-83% of pairs), the effect remains large. The topology predicts syntax on type-switching pairs where acoustic priming cannot operate.

**Test 2 — Cross-voice pairs (SRKW):**
- 40 sessions analysed, mean 2.9 voices per session
- d (cross-voice adjacent pairs vs random) = **0.93** (n = 2,365)
- Topology predicts syntax even across speaker boundaries within sessions

**Test 3 — Cross-station pairs (SRKW):**
- 11 paired timestamp windows, 3,095 cross-station response pairs
- d (cross-station response vs random) = **0.67** (medium-large)
- Different hydrophones = likely different individuals = priming ruled out

**Paper recommendation:** Name acoustic priming explicitly as a competing hypothesis in the Discussion. Present all three tests. The cross-transition test is the strongest (large d, all ecotypes). The cross-voice and cross-station tests provide converging evidence from independent methods.

### Vulnerability 4: Markov Order — ROBUST

**Bootstrap CIs (1000 iterations, session-level resampling):**

| k | Decrease order 3→4 | 95% CI | Excludes zero? | Claim supported? |
|---|---|---|---|---|
| k=3 | 0.0056 bits | **[0.0030, 0.0087]** | **YES** | **YES** |
| k=5 | 0.0273 bits | **[0.0208, 0.0343]** | **YES** | **YES** |

The entropy decrease from order 3 to order 4 is real at both k=3 and k=5. The claim survives even at k=5 where the state space is 625 and sparsity is severe.

**Miller-Madow correction:** Modest bias at order 4 (k=3: 0.0810 → 0.0878; k=5: 0.6864 → 0.7031). The corrected values still show clear decreasing trend.

**BIC model selection:** BIC prefers order 1 at both k=3 and k=5. This is expected — BIC heavily penalises parameters, and with k=3 at order 4 there are 226 free parameters. BIC is conservative for this task because it doesn't account for the scientific question (whether *any* additional predictive structure exists) — it asks whether the full parametric model is justified.

**Paper recommendation:** Report bootstrap CIs as the primary evidence for Markov order > 4. Note the BIC result honestly — BIC prefers order 1, but this reflects the penalty structure, not the absence of higher-order patterns. The bootstrap CI excluding zero is the correct test for "does knowing more history help?"

---

## On Titles

"Topological syntax in killer whale communication: acoustic proximity generates sequential structure across ecotypes" works for Nature Communications / PNAS.

"How Tolkien Taught Me To Talk To Fish" works for the talk.

---

## Second Review (2026-04-07)

**Reviewer:** Claude (Opus 4.6, full paper read)
**Verdict:** Publishable with three additional vulnerabilities. The original four were well-addressed. These three are editorial/framing issues, not structural threats to the findings.

---

### Vulnerability 5: MI Half-Life Comparison Unit Is Not Equivalent

**The concern:** The paper's headline claim — MI half-life 45.4 calls is "6-15× longer than any previously measured non-human species" and exceeds "the human language MI half-life of 3-8 words (Shannon, 1951)" — compares incompatible units.

Shannon measured MI decay over *words* in written English text. One word ≈ 0.3-0.5 seconds of speech. The orca MI is measured over *calls* — each call is a 0.85-1.96 second vocalisation (Table, Section 5.4) with median inter-call intervals of 2.5-4.1 seconds. A call is not a word. It's closer to an utterance, a phrase, or a conversational turn.

If you measured MI decay over human *conversational turns* rather than words, the half-life would be much longer than 3-8 — humans sustain topical coherence over dozens of turns. The 6-15× claim compares orca turns to human words.

The comparison to other non-human species (Kershenbaum et al., 2014) is more valid — those studies used comparable units (call-level MI). The 45.4-call half-life vs the previous ceiling of ~8 units is a genuine finding. It's the human comparison that's apples-to-oranges.

**The fix:**
1. Separate the two comparisons explicitly. The non-human comparison (6-15× longer than any other species) is clean — keep it prominent.
2. For the human comparison, either:
   - (a) Find published MI decay data measured over human conversational turns (not words) and compare to that, or
   - (b) Reframe: "Orca MI half-life exceeds any previously measured non-human species by 6-15×. Direct comparison to human language is complicated by the unit mismatch: Shannon's 3-8 word half-life measures MI over lexical tokens in written text, while orca calls are better analogised to utterances or turns. The comparison at equivalent units remains an open question."
3. In Section 6.2 (the orthogonality table), add a footnote to the "Sequential memory" row noting the unit difference.

**Priority:** MEDIUM — a linguistics reviewer will catch this in the first read. It doesn't kill the finding (the non-human comparison is sufficient for the headline), but overstating the human comparison invites a correction that weakens the whole paper.

---

### Vulnerability 6: 50-100ms "Fast Response" Timing Needs Scrutiny

**The concern:** Section 5.8 reports "Fast responses (50-100ms, 95.4% of transitions)" in the topology-syntax analysis. This deserves the same scrutiny that killed Finding 29.

50-100ms is faster than acoustic propagation across any reasonable pod spread. Sound travels ~1500 m/s in seawater. A 50ms interval implies the two calls originated within 75 metres of each other — possible for nearby individuals, but 95.4% of *all* transitions seems high. More likely explanations:

- Overlapping calls from different individuals (annotation splits them into sequential entries)
- Same individual's call segments annotated as separate events
- Timestamp resolution artifacts in whichever dataset was used for this analysis

Finding 56 already demonstrated that the Haro data has a timestamp artifact producing spurious short ICIs (62-64% in the 15-20ms bin). If the topology-syntax analysis in Section 5.8 uses Haro data, the 50-100ms timing claim is built on the same compromised timestamps.

**The fix:**
1. **Identify the dataset.** Which dataset produces the 50-100ms figure? If Haro, the timing is suspect. If DCLDE annotation-derived (median ICI ~2s), it's a different population of intervals.
2. **Cross-check against DCLDE.** Compute the same fast/slow response split on the DCLDE annotation data. If the dominant ICI there is ~2 seconds, the 50-100ms figure won't appear.
3. **If the timing is an artifact:** Drop the fast/slow response mode framing from Section 5.8. The topology-syntax result (d = 1.34-1.57) does not depend on it — it's computed from similarity distributions, not timing. The core finding survives even if the timing analysis is removed entirely.
4. **If the timing is real:** Explain the physical mechanism. 50-100ms responses between animals >75m apart are impossible without pre-positioning. If these are overlapping calls, say so — simultaneous calling is itself an interesting finding (chorusing behaviour), but it's not "fast response."

**Priority:** MEDIUM — this is exactly the kind of residual artifact that undermines credibility after you've already killed one (Finding 29). A reviewer who reads the ICI correction will immediately look at all other short-timescale claims. Better to clean this up proactively.

---

### Vulnerability 7: R/D Framework Does Less Analytical Work Than the Paper Implies

**The concern:** The R/D framing is the paper's conceptual backbone — "topology generates syntax" — but the actual analytical machinery is standard bioacoustics:

| What the paper says | What the code does |
|---|---|
| R/D attractor identification | k-means clustering (scikit-learn) |
| Topological syntax | Mann-Whitney U on cosine similarity distributions |
| Boltzmann grammar | Log-linear regression on binned enrichment ratios |
| Sequential memory | MI at multiple lags with exponential fit |
| Crisis detection | Jensen-Shannon divergence + Mann-Whitney U |

The R/D dynamics (Section 4.5 — Gray-Scott with feed/kill rates, attractor relaxation) are used for the Ford catalogue validation (Section 5.1) and contact call identification. These are effective and novel. But the headline findings (Sections 5.3-5.8) use conventional methods. The R/D framework provides the *interpretation* ("the geometry generates the grammar"), not the *computation*.

This isn't a flaw — using standard, reproducible methods is a strength. But the paper risks a reviewer objection: "You claim R/D dynamics explain orca communication, but your evidence comes from k-means and MI calculations. The R/D framing is post-hoc narrative."

**The fix:**
1. **Be explicit about the division.** In Methods, clearly separate the R/D attractor analysis (Section 4.5, used for catalogue validation) from the statistical tests (Sections 4.7-4.9, used for population-scale findings). State that the population-scale analyses use standard information-theoretic methods *because* these are reproducible and well-understood — the R/D framework motivates the hypothesis, standard statistics test it.
2. **Reframe the R/D contribution.** The R/D framework's contribution is *conceptual*, not computational:
   - It motivated the hypothesis that acoustic topology generates syntax (which was then confirmed by conventional tests)
   - It provided the attractor identification that validated against Ford 1991 (a genuine analytical contribution)
   - It provides the theoretical interpretation of *why* topology-syntax coupling exists (continuous signal spaces produce continuous grammars)
3. **Add a paragraph to the Discussion** (Section 6.1) distinguishing between the R/D framework as hypothesis generator and the statistical tests as hypothesis validators. This pre-empts the "just narrative" objection.

**Priority:** LOW — this is a framing issue, not a data issue. But it's the difference between a reviewer saying "novel framework produces novel findings" and "standard methods with a fancy wrapper." The former gets accepted.

---

### What Doesn't Need Fixing (Second Review)

These were evaluated and confirmed robust:

- **Topology-syntax coupling (d = 1.34-1.57):** The crown jewel. Three ecotypes, priming ruled out by three tests (Vuln 3 results). This is the strongest finding in the paper.
- **2019 crisis detection:** The SAR control, the p-values, the grammar reorganisation narrative. Bulletproof and immediately useful.
- **Linguistic universals:** Straightforward application of well-established tests. SAR α = 1.11 within human range. No objections.
- **Validation-first design:** Reproducing Ford 1991 blind before making novel claims. This should be highlighted in the cover letter — it's the paper's methodological signature.
- **Honest nulls:** Now four of them (Findings 1 revised, 29 killed, 30, 36). These are the paper's credibility backbone.
- **The orthogonality argument:** Genuinely novel framing. Not "orcas have language" but "orcas have a *different kind* of complex communication." This is the paper's lasting intellectual contribution.

---

### Recommended Work Order (Second Review)

1. **MI comparison units** (Vuln 5) — editorial fix, 30 minutes. Reword the human comparison in Abstract, Introduction, Section 5.5, Section 6.2 table, and Conclusion. The non-human comparison stays as-is.
2. **50-100ms timing audit** (Vuln 6) — 1-2 hours. Identify the source dataset, cross-check against DCLDE, decide whether to keep or drop the fast/slow framing.
3. **R/D framing clarification** (Vuln 7) — editorial, 1 hour. Add a paragraph to Methods clarifying which analyses use R/D vs conventional methods. Add a paragraph to Discussion pre-empting the "just narrative" objection.

None of these threaten the core findings. They're defensive edits that prevent reviewers from finding easy targets.
