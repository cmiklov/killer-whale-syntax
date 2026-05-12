# 4. Methods

The methodology evolved over the course of the study. Initial analyses (Section 5.1) used 42-dimensional features from catalogue exemplars; subsequent analyses (Sections 5.2–5.8) used 50-dimensional features extracted from individual call recordings. This section describes the final methodology as applied to the per-call datasets. Where the catalogue-level analysis differed, those differences are noted in the relevant results section.

## 4.1 Acoustic Feature Extraction

Individual call segments were extracted from DCLDE 2027 audio files using annotation bounding boxes (start time, end time, low frequency, high frequency). Audio was loaded at 22,050 Hz mono via librosa (McFee et al., 2015). For each call, a 50-dimensional feature vector was computed across six feature groups:

| Feature group | Dimensions | Extraction method |
|---|---|---|
| Spectral shape | 6 | Centroid, bandwidth, rolloff, flux, contrast, flatness (librosa spectral features, normalised by Nyquist frequency) |
| Temporal envelope | 5 | Soft classification into pulsed, tonal, mixed, burst, and silence modes from RMS energy and zero-crossing rate coefficient of variation |
| Click/tonal ratio | 2 | Onset strength ratio (onset envelope mean / max) as proxy for impulsive vs sustained energy |
| FM contour | 3 | pYIN pitch tracking (Mauch & Dixon, 2014): mean F0 (normalised by 10 kHz ceiling), modulation depth (F0 coefficient of variation), modulation rate (mean |ΔF0| / mean F0) |
| Structural/context | 8 | Duration (seconds) + annotation metadata (pod, behavioural context flags where available) |
| Spectral fingerprint | 26 | Mean log-mel energy across 26 frequency bands (librosa mel spectrogram, power-to-dB, min-max normalised per call) |

The first 42 dimensions (spectral, temporal, click/tonal, FM, fingerprint) constitute the acoustic features; the remaining 8 are structural/contextual. This separation enables independent analysis of acoustic form and communicative function.

For the annotation-only analyses (Sections 5.4–5.6), where individual audio was not extracted, a reduced 3-dimensional feature vector was derived from annotation metadata: center frequency, bandwidth, and duration. These were normalised to [0, 1] range per ecotype.

## 4.2 Station Normalisation

Different hydrophone installations have different frequency responses, environmental noise floors, and propagation characteristics. To isolate call-level acoustic structure from recording artifacts, per-station z-score normalisation was applied: for each feature dimension, the station mean was subtracted and the result divided by the station standard deviation. This removes systematic station-specific biases while preserving within-station and between-call variation.

Validation criterion: after normalisation, acoustic clusters must span all recording stations. If clusters remain station-specific after normalisation, the features are capturing recording conditions rather than call structure. This criterion was met at 100% for both the Orcasound (3 stations) and Haro Strait (2 stations) datasets.

## 4.3 Clustering

Unsupervised k-means clustering (scikit-learn; random_state=42, n_init=10) was applied at multiple granularities:

- **k = 2**: Optimal for cross-ecotype annotation metadata (silhouette score maximisation). Used for ecotype comparison (Section 5.4).
- **k = 3**: Optimal for station-normalised Orcasound per-call features. Used for syntax, temporal, information-theoretic, and crisis analyses (Sections 5.2–5.7).
- **k = 5**: Used for linguistic universal analyses (Section 5.6), where finer granularity is required for meaningful Zipf and Heaps statistics.

No cluster count was imposed *a priori*. In each case, silhouette scores were computed for k = 2 through k = 10 and the maximum selected. The relative ranking of ecotypes on information-theoretic measures (Section 5.5) is stable across k = 2, 3, and 5.

## 4.4 Sequence Construction

Temporal sequences were constructed from calls within recording sessions. Calls were ordered by start time within each session. A maximum inter-call gap of 30 seconds was imposed: consecutive calls separated by more than 30 seconds were treated as belonging to separate sequences. This threshold balances sensitivity (shorter gaps miss slow exchanges) against specificity (longer gaps conflate independent bouts).

## 4.5 Reaction-Diffusion Attractor Identification

Acoustic feature vectors were projected to 64 dimensions via seeded Johnson-Lindenstrauss random projection (Johnson & Lindenstrauss, 1984) and normalised to unit length on the hypersphere. For the dual-field analysis (catalogue-level, Section 5.1), two independent projections were computed, one for the 42 acoustic dimensions and one for the 8 contextual dimensions, with the combined field formed as a weighted blend (0.6 acoustic, 0.4 context), re-normalised.

Attractor identification used Gray-Scott reaction-diffusion dynamics (Gray & Scott, 1984) operating on the projected space. The R/D model treats each call-type vector as a potential attractor. Given an initial activation (a point in the 64-dimensional space), the dynamics iteratively apply:

1. **Reaction**, the field state is pulled toward the nearest attractor (diffusion_rate = 0.1)
2. **Inhibition**, competing attractors exert a weaker pull weighted by their similarity to the current state
3. **Feed**, amplitude-dependent growth (feed_rate = 0.04)
4. **Kill**, amplitude-dependent decay (kill_rate = 0.06)
5. **Normalisation**, the state is re-projected to unit length after each step

Convergence was defined as ‖Δstate‖ < 10⁻⁴ or 200 steps, whichever came first. The parameters (dt = 0.01, diffusion_rate = 0.1, feed_rate = 0.04, kill_rate = 0.06) are identical to those used in the parent framework and were not tuned for this dataset.

### Analytical role of the R/D framework

The R/D attractor identification (Section 4.5) serves two functions in this study: (1) it provides the blind catalogue validation against Ford (1991) reported in Section 5.1, and (2) it motivates the central hypothesis that acoustic topology generates sequential structure. The population-scale findings (Sections 5.2–5.8) are tested using standard information-theoretic and non-parametric statistical methods (Sections 4.7–4.9), k-means clustering, mutual information, Mann-Whitney U, Jensen-Shannon divergence, chi-squared tests, and exponential regression. These methods were chosen precisely because they are well-understood, reproducible, and independent of the R/D framework. The R/D dynamics generate the hypothesis; conventional statistics test it. This separation ensures that the findings do not depend on the validity of the R/D model itself.

## 4.6 Cross-Population Alignment

Orthogonal Procrustes analysis (Schönemann, 1966) was used to align acoustic spaces across pods and ecotypes. For two populations A and B with call-type vectors in 64-dimensional space, the optimal orthogonal rotation R minimising ‖AR − B‖² was computed via SVD of B^T A. Because pod repertoires (12–18 call types) are underdetermined in 64 dimensions, vectors were first reduced to 5 dimensions via PCA. Disparity was computed as the Frobenius norm of the residual after alignment, normalised by the number of aligned points.

Validation: for each aligned pair, context agreement was measured, the fraction of aligned call-type pairs sharing at least one behavioural context tag (foraging, socialising, travelling, resting). High context agreement indicates that the rotation maps functionally equivalent calls to each other, not just acoustically similar ones.

## 4.7 Information-Theoretic Measures

Following Shannon (1951) and Cover and Thomas (2006):

- **Marginal entropy** H(X) = −Σ p(x) log₂ p(x), computed from cluster frequency distributions. Measures repertoire diversity in bits per call.
- **Conditional entropy** H(X_{t+1} | X_t, ..., X_{t−k}) at Markov orders k = 0 through k = 4, computed from n-gram transition counts. Measures residual uncertainty about the next call given k previous calls.
- **Mutual information** MI = H(X) − H(X_{t+1} | X_t). Measures how much the previous call reduces uncertainty about the next.
- **Normalised MI** (MI/H): the fraction of total information that is sequential. Enables cross-ecotype and cross-species comparison independent of repertoire size.
- **MI decay**: MI computed at lags 1 through 11. Fitted with an exponential decay model MI(lag) = a · exp(−b · lag). Half-life = ln(2) / b.

Non-random sequential structure was tested via chi-squared test on the transition matrix against the null hypothesis of independence (expected counts from marginal probabilities).

## 4.8 Topology-Syntax Coupling

The central hypothesis (that acoustic topology predicts sequential adjacency) was tested by comparing the distribution of pairwise acoustic similarities for calls that are *sequentially adjacent* in natural sequences against calls that are *not adjacent*.

For each pair of consecutive calls (i, i+1) in the corpus, the cosine similarity between their 50-dimensional feature vectors was computed. The distribution of these *adjacent-pair similarities* was compared against the distribution of similarities for all non-adjacent pairs via Cohen's *d* (standardised mean difference) and the Mann-Whitney U test.

To characterise the relationship at different similarity thresholds, the enrichment ratio was computed: (fraction of adjacent pairs above threshold) / (fraction of all pairs above threshold). An enrichment ratio of 1.0 indicates no relationship between topology and syntax; values above 1.0 indicate that acoustically similar calls are over-represented among sequential pairs.

This analysis was performed independently for SRKW (Haro Strait, 4,862 calls), TKW (2,453 calls), and OKW (1,255 calls).

## 4.9 Statistical Tests

All p-values are two-sided unless otherwise noted. Non-parametric tests (Mann-Whitney U, chi-squared) were preferred throughout due to the non-normal distributions typical of acoustic data. Effect sizes (Cohen's *d*, Pearson's *r*) are reported alongside p-values. Jensen-Shannon divergence (symmetric, bounded [0, 1]) was used for distribution comparisons across time periods and ecotypes. Bonferroni correction was not applied to the ecotype comparisons, as each ecotype analysis addresses an independent biological question rather than multiple tests of a single hypothesis; however, the reported p-values (10⁻⁷² to 10⁻²⁹⁶) are robust to any reasonable correction.

## 4.10 Reproducibility

All analysis scripts are publicly available. Each script is self-contained and reproduces the findings from raw data with fixed random seeds (random_state=42 throughout). The analytical pipeline requires Python 3.8+, numpy, scipy, scikit-learn, and librosa. Feature extraction from audio additionally requires soundfile. The 74-test suite validates all engine components on synthetic data. Data access: DCLDE 2027 via NOAA (DOI: 10.25921/15ey-mh50); Ford-Osborne exemplars via Orcasound signals-srkw (github.com/orcasound/signals-srkw).
