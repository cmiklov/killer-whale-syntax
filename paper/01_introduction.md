# 1. Introduction

The quantitative study of non-human vocal communication has historically relied on categorical classification: assigning call types, measuring repertoire size, and testing for non-random sequential structure at low Markov orders (Kershenbaum et al., 2014). This approach has established that many species produce structured vocal sequences (Gentner & Margoliash, 2003; Ouattara et al., 2009; Suzuki et al., 2006), but has consistently found sequential dependencies limited to orders 1–3, with mutual information (MI) half-lives of 3–8 units across all non-human species measured (Kershenbaum et al., 2014; Janik, 2000).

Killer whales (*Orcinus orca*) present a compelling case for deeper analysis. They possess the largest brain of any odontocete (Marino et al., 2004), a paralimbic cortical lobe absent in humans that integrates emotional and cognitive processing (Marino et al., 2007), and an elaborated insular cortex and temporal operculum whose functional significance remains poorly understood (Marino et al., 2016). Their vocal repertoire has been extensively catalogued: Ford (1987, 1991) identified 16–18 discrete call types per pod in Southern Resident killer whales (SRKW), culturally transmitted across generations (Deecke et al., 2000). Yet the *sequential structure* of these calls (what follows what, and why) has received comparatively little quantitative attention at the population scale.

This paper introduces a topological approach to orca vocal analysis grounded in reaction-diffusion (R/D) dynamics. The analytical framework draws on Turing's (1952) foundational demonstration that two interacting substances (an activator and an inhibitor, diffusing at different rates) spontaneously generate stable spatial patterns from homogeneous initial conditions. In *The Chemical Basis of Morphogenesis*, Turing showed that pattern formation requires no blueprint: structure emerges from the interaction dynamics alone. The Gray-Scott model (Gray & Scott, 1984) parameterised this dynamic for continuous systems, producing the well-characterised zoo of spots, stripes, and labyrinthine patterns that have since been identified in contexts ranging from embryonic development to sand dune formation.

We apply this principle not to spatial patterns but to *acoustic topology*. Each call type occupies a position in a high-dimensional feature space defined by its spectral and temporal properties. R/D dynamics operating on this space identify basins of attraction, natural clusters the system converges to without imposed category boundaries. The critical test is *topological syntax*: whether acoustic proximity between call types predicts their sequential adjacency in natural vocal sequences.

We analyse approximately 40,000 killer whale calls across four ecotypes (SRKW, Bigg's/Transient, Southern Alaska Resident, and Offshore) spanning 18 years of hydrophone recordings from the DCLDE 2027 dataset (Myers et al., 2025), supplemented by Ford-Osborne call exemplars (Ford, 1987; Orcasound, 2018) and 4,862 individually extracted calls from Haro Strait.

The principal findings are:

1. Acoustic topology predicts sequential syntax with a large effect size (Cohen's *d* = 1.34–1.57) across three independently analysed ecotypes.
2. Sequential memory, measured by MI half-life, reaches 45.4 calls in Bigg's transients, 6–15× longer than any previously measured non-human species.
3. Markov order exceeds 4 for both SRKW and Bigg's, placing orca vocal sequences within the human language range (5–7).
4. During the 2019 Chinook salmon crisis, SRKW communication reorganised into an alternative grammatical regime (*p* = 10⁻²⁹⁶), confirmed as population-specific by a natural experiment control.
5. All tested linguistic statistical universals (Zipf's law, the brevity law, Heaps' law) hold for orca vocal sequences.

These results suggest that orca vocal communication is not a simpler version of human language on the same complexity axis, but an *orthogonal* system, one that trades vocabulary and abstraction for unprecedented sequential coherence and direct coupling between acoustic form and syntactic function. We propose the term *topological syntax* for this mode of communication, in which the geometry of the signal space generates the grammar.

The R/D framework used here follows Turing's original insight: we do not impose structure on the data. We provide the conditions (two interacting fields, acoustic and contextual) and observe what stabilises. The patterns that emerge are the findings.
