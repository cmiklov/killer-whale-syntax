# 6. Discussion

## 6.1 Topological Syntax as a Distinct Mode of Communication

The central finding of this study, that acoustic topology predicts sequential structure with Cohen's *d* = 1.34–1.57 across three independently analysed ecotypes, identifies a mode of communication not previously described. We term this *topological syntax*: a system in which the geometry of the signal space generates the grammar directly, without an intervening layer of categorical abstraction.

Human language is characterised by the *arbitrariness of the sign* (Saussure, 1916): the phonological form of a word bears no systematic relationship to its syntactic role or sequential position. "Cat" does not sound like "dog" despite occupying identical syntactic slots. This separation of form from function is what enables recursion, displacement, and compositional semantics, the defining features of human language (Chomsky, 1957; Hockett, 1960).

Orca communication, as measured here, operates on a different principle. Acoustic form *is* syntactic function. Calls that sound similar follow each other, not occasionally, but with a large, replicable effect verified across 8,570 calls from three genetically distinct populations. At the highest similarity thresholds (≥ 0.90), sequential adjacency is 136× more frequent than chance. The topology does not describe the grammar; it *produces* it.

This distinction has a precedent in Turing's (1952) framework. In reaction-diffusion systems, pattern emerges not from a template but from the interaction dynamics of the substrate. The activator-inhibitor dynamic does not encode the pattern, it generates it. Similarly, orca topological syntax does not encode sequential rules, the acoustic topology generates sequential structure as a consequence of the signal space geometry. The grammar is an emergent property of the medium, not a separate layer imposed upon it.

We note that the R/D framework's contribution to this study is primarily conceptual. It motivated the hypothesis that acoustic topology generates syntax, and it provided the attractor identification used for blind catalogue validation (Section 5.1). The population-scale findings, topology-syntax coupling, MI half-life, Markov order, crisis detection, linguistic universals, were all tested using standard information-theoretic and non-parametric statistical methods (see Section 4.5, "Analytical role of the R/D framework"). The R/D model is the hypothesis generator; conventional statistics are the hypothesis validators. The findings stand independently of whether one accepts the R/D interpretation.

### Alternative hypothesis: acoustic priming

The topology-syntax result (that acoustically similar calls tend to follow each other) admits an alternative explanation: acoustic priming. Animals may simply repeat similar sounds due to production inertia, articulatory constraints, or shared physiological state. If adjacent calls are produced by the same individual, self-repetition alone could generate the observed similarity enrichment. Self-transitions do dominate (0.80–0.97 across ecotypes), consistent with this concern.

Three independent tests rule out priming as the sole explanation:

1. **Cross-transition exclusion.** After removing all self-repetitions (68–83% of pairs), the topology-syntax effect persists with large effect sizes: *d* = 1.27 (SRKW), 1.49 (TKW), 0.91 (OKW). Priming cannot operate on pairs where the call type changes.

2. **Cross-voice pairs.** Within SRKW sessions containing an estimated 2.9 voices, topology-syntax coupling holds across speaker boundaries (*d* = 0.93, n = 2,365 cross-voice pairs). One individual's call predicts another individual's subsequent call; this is coordination, not self-repetition.

3. **Cross-station pairs.** Calls recorded at different hydrophone locations (likely different individuals) show medium-large topology-syntax coupling (*d* = 0.67, n = 3,095 cross-station response pairs).

The effect is reduced but not eliminated after controlling for priming, consistent with a model where both priming (self-repetition maintaining acoustic consistency) and topological syntax (acoustic proximity driving cross-individual sequential structure) contribute to the observed pattern. The cross-individual tests confirm that topological syntax operates beyond what priming alone can explain.

## 6.2 The Orthogonality of Human and Orca Communication

The five principal dimensions of communication complexity, vocabulary diversity, compression efficiency, sequential memory, syntactic depth, and form-syntax coupling, reveal that human language and orca communication occupy orthogonal regions of the space:

| Dimension | Human language | Orca (TKW) | Orca (SRKW) |
|---|---|---|---|
| Vocabulary (H₀) | ~4.7 bits | 0.9 bits | 0.2 bits |
| Compression (H₀/H₄) | ~4.7× | 1.8× | 1.9× |
| Sequential memory (MI t½) | 3–8 words† | **45.4 calls** | **24.1 calls** |
| Syntactic depth (Markov) | 5–7 | **>4** | **>4** |
| Form-syntax coupling (*d*) | ~0 | **1.57** | **1.34** |

†Human MI half-life is measured over lexical tokens in written text (Shannon, 1951); orca MI is measured over calls (duration 0.85–1.96 s). Calls are behavioural analogues of utterances or conversational turns, not individual words. The values are not unit-equivalent; the non-human comparison (6–15× longer than any other species, using call-level units) is the valid benchmark.

Human language dominates on vocabulary and compression. Orca communication dominates on sequential memory and form-syntax coupling. Neither system is "more complex", they are complex in *different dimensions*.

The analogy is structural, not metaphorical. A symphony and a novel are both complex artefacts. A symphony achieves complexity through sustained temporal coherence across multiple simultaneous channels, exactly the orca profile. A novel achieves complexity through referential depth and compositional abstraction, exactly the human profile. Asking which is "more complex" is a category error. They solve different problems.

## 6.3 Neuroanatomical Substrate

The findings are consistent with three features of cetacean neuroanatomy documented by Marino et al. (2004, 2007, 2016) and Hof and Van der Gucht (2005).

The **paralimbic lobe**, a cortical region absent in humans that integrates emotional and cognitive processing, provides a substrate for the ecological stress response documented in Section 5.7. SRKW did not merely increase call rate or amplitude during the 2019 prey crisis; they switched to an entirely different grammatical regime (33% lower conditional entropy, reorganised transition matrix, new dominant attractor). This is not a stress reflex. It is cognitive-emotional integration: recognising a crisis, selecting an alternative communication protocol, and maintaining that protocol coherently across the population.

The **elaborated insular cortex and temporal operculum** provide the processing capacity for the temporal structure documented throughout Section 5. The 45-call MI half-life (Section 5.5) requires maintaining a running model of the preceding ~3 minutes of communication. The cortical surface area of 3,745 cm² (the largest of any mammal) provides the working memory substrate for this sustained sequential coherence. The median inter-call interval of 2-4 seconds implies a conversational turn-taking timescale of 0.25-0.5 Hz, consistent with the processing demands of sustained sequential coherence rather than rapid neural oscillation.

The **thalamic audio-visual convergence** documented by Berns et al. (2015) reframes the entire communication system. In the cetacean brain, auditory and visual thalamic pathways overlap before cortical processing begins. Echolocation returns are processed through fused audio-visual hardware, the acoustic signal *is* a spatial percept at the neural level. Every pod member within acoustic range processes every other individual's sonar returns through this fused pathway, producing a distributed spatial model updated continuously by the entire group.

The three-channel architecture described in Section 2.4, syntactically structured calls for coordination, individually distinctive whistles for attribution, echolocation clicks for shared spatial perception, constitutes a distributed cognitive system. The pod is not a collection of individuals exchanging messages. It is a network of brains sharing a perceptual field, coordinating behaviour through topological syntax, and attributing signals to individuals through signature whistles.

## 6.4 Ecological Predictions

The information-theoretic profiles (Section 5.5) predict ecotype-specific communication strategies from ecological constraints:

**Bigg's transients** hunt acoustically aware prey. Every vocalisation risks alerting the target. Prediction: minimal acoustic output, maximum information per emission, longest sequential coherence. Observed: median ICI 4.1 s (slowest), MI/H = 53.5% (highest), MI half-life 45.4 calls (longest). The stealth protocol, maximum coherence, minimum exposure.

**SRKW** hunt Chinook salmon, which cannot detect killer whale calls. No stealth constraint. Prediction: faster call rate, compressed vocabulary optimised for positional coordination. Observed: median ICI 2.5 s, H₀ = 0.16 bits (most compressed vocabulary), MI/H = 23.6% (rich positional grammar).

**SAR** occupy open water with different social dynamics. Prediction: high vocabulary diversity for social signalling, low sequential structure. Observed: median ICI 1.4 s (fastest), H₀ = 1.31 bits (richest vocabulary), MI/H = 5.6% (minimal grammar).

These predictions are testable: changes in prey availability or social structure should produce measurable shifts along the vocabulary-grammar axes. The 2019 SRKW crisis (Section 5.7) provides the first natural experiment confirming this.
