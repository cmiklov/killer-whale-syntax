# 9. Conclusion

We have presented a reaction-diffusion topological analysis of approximately 40,000 killer whale calls across four ecotypes and 18 years, grounded in Turing's (1952) foundational demonstration that stable pattern emerges from activator-inhibitor dynamics without external instruction.

The principal findings are:

1. **Topological syntax is universal in orca communication.** Acoustic proximity predicts sequential adjacency with Cohen's *d* = 1.34–1.57 across three independently analysed ecotypes (8,570 calls). This is a large effect, replicated across genetically distinct populations.

2. **Orca sequential memory is unprecedented among non-human animals.** MI half-life reaches 45.4 calls in Bigg's transients, 6–15× longer than any previously measured non-human species. Markov order exceeds 4. (Direct comparison to the human MI half-life of 3–8 words is complicated by unit mismatch, orca calls are closer to utterances than to words. The non-human comparison, which uses equivalent call-level units, is the valid benchmark.)

3. **Communication strategy tracks ecology.** Four ecotypes occupy different positions in a two-dimensional space of vocabulary diversity × sequential structure. Bigg's transients (stealth hunters) maximise both. SRKW (salmon hunters) compress vocabulary and enrich grammar. SAR (open-water residents) diversify vocabulary with minimal grammar.

4. **Ecological stress produces grammatical reorganisation.** During the 2019 Chinook salmon crisis, SRKW shifted to an alternative communication regime, lower entropy, reorganised transition matrix, different dominant attractor, confirmed as population-specific by a natural experiment control (*p* = 10⁻²⁹⁶).

5. **All tested linguistic statistical universals hold.** Zipf's law (SAR α = 1.11, within human range), the brevity law (r = −0.82), and Heaps' law (β = 0.35, within human range).

6. **Passive acoustic monitoring can detect population stress in real time**, using existing hydrophone infrastructure and the analytical framework presented here.

These results suggest that orca vocal communication is not a simpler version of human language but an orthogonal system, one that trades categorical abstraction for topological continuity, achieving unprecedented sequential coherence through direct coupling between acoustic form and syntactic function. The R/D framework, following Turing's original insight, imposed no structure on the data. The patterns that stabilised are the findings.

Turing demonstrated in 1952 that two interacting substances, diffusing at different rates, produce stable pattern from homogeneity, that structure requires no blueprint. Seventy-four years later, the same principle, applied to the acoustic topology of killer whale vocalisations, reveals that grammar requires no abstraction. The geometry of the signal space is sufficient. The pattern emerges from the interaction. It always did.

---

## Data Availability

The DCLDE 2027 dataset is publicly available from NOAA (DOI: 10.25921/15ey-mh50) under CC-BY 4.0. Ford-Osborne call exemplars are available from the Orcasound signals-srkw repository (github.com/orcasound/signals-srkw) under CC BY-NC-SA 4.0. Extracted feature datasets (Haro Strait, TKW, OKW) are included in the code repository.

## Code Availability

All analysis scripts, the orca-engine framework, and the 74-test validation suite are publicly available at https://github.com/cmiklov/killer-whale-syntax. Each script is self-contained and reproduces its findings from raw data with fixed random seeds (random_state=42 throughout). The pipeline requires Python 3.8+, numpy, scipy, scikit-learn, and librosa.

## AI Disclosure

Analysis scripts were developed with assistance from Claude (Anthropic, claude-opus-4-6). The paper text was drafted collaboratively with AI assistance and reviewed by the author. All scientific decisions, experimental design, interpretations, and claims are the author's. The internal pre-submission review (identifying and addressing four methodological vulnerabilities) was conducted with AI assistance and is documented in the repository.

## Competing Interests

The author declares no competing interests.

## Acknowledgements

The DCLDE 2027 dataset was compiled by Myers et al. (2025) and made available by NOAA. Ford-Osborne call exemplars were digitised and hosted by the Orcasound project. The Orcasound hydrophone network provides ongoing open-access acoustic monitoring of the Salish Sea. This work was conducted independently and received no external funding.
