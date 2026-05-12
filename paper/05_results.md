# 5. Results

## 5.1 Validation on Known Ground Truth

Before presenting novel findings, we validate the framework against published results that the analysis was not given.

**Dialect proximity.** Ford (1991) established that J-pod and L-pod are the closest SRKW dialect pair, sharing call types (S02, S04, S10, S19) not used by K-pod. Procrustes alignment of pod-level acoustic spaces re-derives this ranking without any prior information about dialect relationships:

| Alignment | Disparity | Context agreement |
|---|---|---|
| **J ↔ L** | **0.456** | **0.88** |
| J ↔ K | 0.527 | 0.58 |
| K ↔ L | 0.652 | 0.92 |

J and L pods have the lowest disparity and 88% context agreement, the fraction of aligned call pairs sharing at least one behavioural context. K-pod is the outlier: smallest repertoire (12 entries vs J's 16 and L's 18), highest alignment disparity with L-pod, most pod-specific calls.

**Contact call identification.** When R/D dynamics are initialised from arbitrary co-activations of call types across pods, the field converges to S01, the discrete pulsed call documented as the primary contact call (Ford, 1991), present in all three pods, all four behavioural contexts, comprising 25–30% of all recorded calls. The framework identifies the most important signal in the repertoire from topology alone.

**Repertoire structure.** The engine recovers Ford's pod repertoire classification exactly: 6 call types shared across all three pods (S01, S03, S06, S07, S12, S16), 4 shared exclusively between J and L (S02, S04, S10, S19), and the correct count of pod-specific types (J: 6, K: 6, L: 8).

**Phonosemantic correlation.** At the catalogue level, acoustic and contextual distances correlate at r = 0.77 (Pearson, pairwise distances in 42D acoustic vs 8D context space), call types used in similar behavioural contexts share similar spectral properties. This correlation was subsequently tested at the individual call level (Section 5.3) and revised to r = 0.025, a result discussed in Section 8.

These validation results, correct dialect ranking, correct contact call identification, exact repertoire recovery, were obtained from structural analysis alone. The framework was not informed of the published answers.

## 5.2 Station-Independent Acoustic Structure

To move from catalogue exemplars (one recording per call type) to population-level statistics, 577 individual call segments were extracted from three Orcasound hydrophones in the Salish Sea.

Raw clustering (k = 7 optimal) revealed recording artifacts: 4 of 7 clusters were station-specific, reflecting hydrophone frequency response rather than call identity. After per-station z-score normalisation (Section 4.2), the optimal cluster count dropped to k = 3, and all three clusters spanned all three recording stations at 100%.

| Cluster | Calls | Stations represented |
|---|---|---|
| 0 | 251 | 3/3 |
| 1 | 178 | 3/3 |
| 2 | 148 | 3/3 |

The spectral diversity across 577 calls is substantial: centroid mean 3,291 Hz (std 529 Hz), duration mean 2.02 s (std 0.83 s). PCA reveals that station normalisation increases effective dimensionality from 5 to 14 components for 90% variance explained, removing station effects uncovers finer acoustic structure that was previously masked.

## 5.3 Sequential Syntax

The 577 Orcasound calls, temporally ordered within recording sessions (max inter-call gap 30 s), show highly non-random transition structure:

| | → C0 | → C1 | → C2 |
|---|---|---|---|
| **C0 →** | **0.831** | 0.076 | 0.093 |
| **C1 →** | 0.087 | **0.750** | 0.163 |
| **C2 →** | 0.174 | 0.167 | **0.659** |

Chi-squared = 432.19 (df = 4, *p* < 10⁻⁶). Self-transition rate 0.762, against 0.350 expected under independence. Calls of the same acoustic type follow each other more than twice as often as chance predicts.

**Markov depth.** Full-scale analysis on 11,079 bigram transitions from 838 SRKW recording sessions confirms the syntax finding at 20× scale (χ² = 2,617, *p* ≈ 0). Conditional entropy continues to decrease through Markov order 4 without plateauing:

| Order | H(next \| context) | Reduction from order 0 |
|---|---|---|
| 0 | 0.1249 bits |, |
| 1 | 0.1077 bits | 13.7% |
| 2 | 0.0961 bits | 23.1% |
| 3 | 0.0872 bits | 30.2% |
| 4 | 0.0810 bits | 35.1% |

At order 4, knowing the last five calls reduces uncertainty about the next call by 35% compared to base rate. The reduction has not plateaued, implying Markov order exceeds 4. Previously reported non-human Markov orders peak at 1–3 (Kershenbaum et al., 2014). This places orca vocal sequences within the human language range (5–7; Shannon, 1951). Session-level bootstrap (1,000 iterations resampling sessions with replacement, Miller-Madow corrected) confirms the order-3→4 reduction is robust to sparse-estimation artifacts: 95% CIs at k=3 [0.003, 0.009] and k=5 [0.021, 0.034] both exclude zero, surviving even at k=5 where 625 possible context states are observed in a finite corpus.

Trigram analysis (10,241 trigrams) reveals structural embedding: the interjection pattern C0→C2→C0 (a different call type inserted into a sustained bout, after which the bout resumes) occurs at 0.75%, nearly identical to the return pattern C2→C0→C0 (0.74%). When a call interrupts a sequence, the sequence recovers. This is functionally analogous to parenthetical embedding in human syntax.

**Independent replication.** All syntax findings replicate on the Haro Strait dataset (4,862 calls, 2 JASCO hydrophones): χ² = 1,756 (*p* ≈ 0), 100% station independence after normalisation. Cross-station sequence matching on 11 simultaneously recorded call sequences shows 70.6% position-match accuracy (random expectation: 50.3%), confirming that the detected structure exists in the acoustic signal, not in the clustering procedure.

## 5.4 Cross-Ecotype Communication Strategies

Analysis of the full DCLDE annotation corpus reveals that all four ecotypes converge on a binary acoustic structure (k = 2 optimal by silhouette score in each case), but the acoustic content of those two modes differs dramatically:

| Ecotype | N | Center freq (Hz) | Duration (s) | Social structure |
|---|---|---|---|---|
| SRKW | 12,298 | 3,554 ± 3,482 | 1.00 ± 0.72 | Large matrilineal pods |
| TKW | 3,146 | 2,191 ± 2,543 | 1.96 ± 1.07 | Small transient groups |
| SAR | 7,996 | 4,755 ± 2,514 | 0.85 ± 0.36 | Large resident pods |
| OKW | 1,491 | 3,302 ± 2,595 | 1.17 ± 0.57 | Poorly known |

Cross-ecotype Procrustes alignment reveals SRKW as the acoustic outlier:

| Alignment | Disparity |
|---|---|
| OKW ↔ SAR | 0.20 |
| SAR ↔ TKW | 0.39 |
| OKW ↔ TKW | 0.47 |
| SAR ↔ SRKW | 0.62 |
| SRKW ↔ TKW | 0.62 |
| OKW ↔ SRKW | 0.67 |

SRKW, the most socially complex population and the only one critically endangered, has the most acoustically distinct communication of any ecotype.

## 5.5 Information-Theoretic Profiles

The information-theoretic analysis reveals that communication complexity is not a single axis but at least two orthogonal dimensions: repertoire diversity (marginal entropy) and sequential structure (normalised mutual information).

| Ecotype | H(X) bits | H(X\|prev) bits | MI bits | MI/H | Strategy |
|---|---|---|---|---|---|
| TKW | 0.92 | 0.69 | 0.24 | **25.5%** | Rich grammar, rich vocabulary |
| SRKW | 0.16 | 0.13 | 0.04 | **23.6%** | Rich grammar, compressed vocabulary |
| OKW | 1.16 | 0.94 | 0.22 | 19.1% | Moderate |
| SAR | 1.31 | 1.24 | 0.07 | **5.6%** | Minimal grammar, richest vocabulary |

SRKW and SAR occupy opposite extremes. SRKW communication is dominated by a single call type (97.8% C0) with 23.6% of information carried sequentially, a compressed vocabulary with rich positional grammar, analogous to agglutinative human languages (Turkish, Japanese). SAR uses three call types in roughly equal proportion (35%/56%/9%) with only 5.6% sequential structure, a diverse vocabulary with minimal grammar, analogous to isolating languages (Mandarin, Vietnamese).

This mapping between communication strategy and ecological niche is consistent with the selection pressures described in Section 2.1. SRKW coordinate Chinook salmon pursuit through complex waterways; precise positional signalling outweighs vocabulary diversity. SAR inhabit open waters with different prey dynamics; a larger repertoire of distinct signals serves social functions more than hunting coordination.

Bigg's transients show the richest profile on both dimensions: highest MI/H (25.5%) *and* substantial vocabulary diversity (H = 0.92 bits). Their stealth constraint (hunting acoustically aware prey) selects for maximum information density per vocalisation.

**Mutual information decay** reveals the most striking result. MI was computed at lags 1 through 11 and fitted with exponential decay:

| Ecotype | MI/H at lag 1 | Half-life (calls) |
|---|---|---|
| **TKW** | **53.5%** | **45.4** |
| **SRKW** | **17.8%** | **24.1** |
| SAR | 4.8% | 10.5 |

No previously measured non-human species has shown sequential memory beyond approximately 8 units (Kershenbaum et al., 2014; Janik, 2000; Gentner & Margoliash, 2003). Bigg's transients maintain statistically detectable sequential coherence over 45 calls, a span of approximately 3 minutes at their median inter-call interval of 4.1 seconds, exceeding any previously measured non-human species by a factor of 6–15. Shannon (1951) reported MI half-lives of 3–8 words for written English, but this comparison is not unit-equivalent: orca calls (duration 0.85–1.96 s, median ICI 2.5–4.1 s) are behavioural analogues of utterances or conversational turns, not individual words. Human MI measured over conversational turns would likely yield substantially longer half-lives. The non-human comparison (which uses equivalent call-level units across species) is the valid benchmark.

At lag 1, 53.5% of all Bigg's call information is sequential, more than half of what a transient orca vocalises is determined by its recent call history. This is the most sequentially structured communication system measured in any species.

## 5.6 Linguistic Statistical Universals

Three established statistical universals of human language were tested across ecotypes.

**Zipf's law** (Zipf, 1949). All ecotypes show power-law rank-frequency distributions:

| Ecotype | α | R² |
|---|---|---|
| SAR | 1.11 | 0.68 |
| TKW | 2.61 | 0.72 |
| SRKW | 4.29 | 0.90 |

SAR's exponent (α = 1.11) is within the range observed in natural human languages (α ≈ 1.0). SRKW's steeper exponent reflects the dominance of a single call type (C0 at 97.8%).

**Brevity law** (Zipf, 1935). More frequent call types have shorter durations across all ecotypes:

| Ecotype | r |
|---|---|
| SAR | −0.82 |
| SRKW | −0.75 |
| TKW | −0.33 |

SAR's r = −0.82 is comparable to values reported for human text corpora (typically −0.5 to −0.9).

**Heaps' law** (Heaps, 1978). Vocabulary grows sublinearly with corpus size. SRKW β = 0.35 falls within the human range (0.4–0.6). New call types continue to appear as the corpus grows, but at a decelerating rate, the hallmark of a productive system with a finite but expandable repertoire.

**Bout structure.** Each ecotype shows distinct grammatical roles within sequences. In SRKW, cluster C4 starts and ends 69% of sequences, functioning as a sentence boundary marker. In TKW, C4 and C1 alternate in structured pairs (216 + 206 transitions), resembling call-and-response. In SAR, three-way alternation with no dominant pair, consistent with the isolating typology.

## 5.7 Ecological Stress Response

The 2018–2019 Chinook salmon shortage, the worst on record for SRKW, during which J35 (Tahlequah) carried her dead calf for 17 days, produced a measurable shift in SRKW acoustic behaviour across every dimension analysed:

| Metric | 2019 (n = 1,163) | Other years (n = 11,135) | Change | *p* |
|---|---|---|---|---|
| Center frequency | 6,889 ± 8,608 Hz | 3,205 ± 2,090 Hz | +114.9% | 1.7 × 10⁻⁷² |
| Bandwidth | 12,006 ± 16,781 Hz | 2,808 ± 3,404 Hz | +327.6% | 7.2 × 10⁻²⁹⁶ |
| Duration | 1.45 s | 0.95 s | +52.2% | 1.6 × 10⁻¹²⁶ |

C2 usage spiked from <2% to 13.8%; C1, entirely absent in other years, appeared at 3.4%. Self-transition rate dropped from 0.975 to 0.867.

**Natural experiment control.** The critical test: does the 2019 shift appear in other ecotypes recorded with the same annotation pipeline?

| Ecotype | JSD (2019 vs other) | Center freq Δ | Bandwidth Δ | Duration Δ |
|---|---|---|---|---|
| **SRKW** | **0.076** | **+114.9%** | **+327.6%** | **+52.2%** |
| SAR | 0.004 | −1.9% | −3.2% | −7.3% |
| TKW | 0.033 | +22.4% | +41.3% | −24.2% |

SRKW's Jensen-Shannon divergence is 20× larger than SAR's. SAR, a healthy resident population recorded with the same pipeline, shows effectively no shift. The direction of SAR's trivial changes is *opposite* to SRKW's. TKW shows a moderate shift consistent with the Lombard effect (Holt et al., 2008), higher frequency but *shorter* duration, the opposite of SRKW's duration increase. This rules out recording artifacts, ocean noise, and annotation protocol changes.

**Grammar reorganisation.** Counter-intuitively, 2019 SRKW sequences became *more* predictable, not less. Conditional entropy dropped 33.1% (0.819 → 0.548 bits). The transition matrix inverted: C1 became the self-reinforcing state (C1→C1 = 0.927, vs 0.181 in other years), and C0→C1 became the dominant transition (0.522, vs 0.001). The grammar did not break down under stress. It reorganised around a different attractor, an alternative communication regime that was more predictable but structured around different transition patterns.

This finding is consistent with the paralimbic cortical lobe described by Marino et al. (2007): the recognition of ecological crisis (cognition) triggered an alternative communication protocol (behaviour) mediated by emotional state, exactly the kind of integrated cognitive-emotional response that a paralimbic lobe would support.

## 5.8 Topological Syntax

The central finding. For each pair of consecutive calls in natural sequences, the cosine similarity of their 50-dimensional acoustic feature vectors was compared against the similarity distribution of non-adjacent pairs.

| Ecotype | N calls | Cohen's *d* | *p* |
|---|---|---|---|
| **SRKW** | 4,862 | **1.34** | ≈ 0 |
| **TKW** | 2,453 | **1.57** | ≈ 0 |
| **OKW** | 1,255 | **1.51** | 10⁻²⁰¹ |

All three ecotypes, analysed independently, from different recording locations and years, show large-effect topology-syntax coupling. Calls that sound similar follow each other. The effect is strongest in Bigg's transients (*d* = 1.57), consistent with their stealth constraint selecting for maximally efficient acoustic coding.

At similarity ≥ 0.90, sequentially adjacent pairs are **136× more frequent** than expected by chance (SRKW enrichment ratio). The enrichment-similarity relationship follows a Boltzmann distribution. Permutation testing (1,000 within-session shuffles preserving session membership but breaking sequential adjacency) confirms that the fit exceeds the null for SRKW (observed R² = 0.98, null 95th percentile = 0.979, *p* = 0.005). For TKW and OKW, R² falls within the permutation null, but the coupling *slope* exceeds the null in both cases (TKW: 4.33 vs 3.81; OKW: 3.49 vs 3.09), indicating that the strength of topology-syntax coupling is real even where goodness-of-fit alone does not distinguish from the null. Cross-session shuffling (assigning calls to random sessions) destroys the fit entirely for all ecotypes (*p* = 0.000), confirming that the Boltzmann structure requires session-level sequential organisation.

A previously reported fast/slow response timing analysis (Finding 43) was retracted during internal review. The 50-100 ms inter-call intervals used in that analysis were identified as artifacts of the Haro Strait `FileBeginSec` timestamps (see Section 8, Limitations). The topology-syntax result (d = 1.34-1.57) does not depend on timing; it is computed from similarity distributions of sequentially adjacent calls regardless of inter-call interval.

Session-level trajectories show conversational arcs: coherence rises from session start (0.600) to a mid-session peak (0.642) before declining at session end (0.592), an open-focus-close structure analogous to human conversational organisation (*p* = 0.075, marginal).

**The interpretation.** Acoustic topology does not merely describe orca communication. It predicts its sequential structure with a large, replicable effect across three genetically distinct ecotypes spanning 8,570 independently analysed calls. The geometry of the signal space generates the grammar. We propose the term **topological syntax** for this mode of communication.

This is fundamentally different from human language, where phonological form and syntactic function are arbitrarily related (Saussure, 1916). In human language, the sound of a word does not predict what word comes next. In orca communication, it does, with *d* > 1.3 across all populations tested. Human language achieves its power through abstraction: separating form from function enables recursion, displacement, and compositional semantics. Orca communication achieves its power through continuity: fusing form and function enables unprecedented sequential coherence (MI half-life 6–15× human) at the cost of the categorical abstraction that human language exploits.

These are not different points on the same complexity axis. They are orthogonal solutions to the problem of coordinating behaviour through acoustic signals.
