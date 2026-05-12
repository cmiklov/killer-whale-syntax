# 7. Conservation Applications

The 2019 prey crisis acoustic shift (*p* = 7.2 × 10⁻²⁹⁶ on bandwidth alone), confirmed as SRKW-specific by the SAR control, demonstrates that passive acoustic monitoring is in principle capable of detecting population-level stress. The thresholds below are derived from retrospective analysis of the 2005–2023 corpus; prospective validation on streaming data is the natural next step before operational deployment.

Candidate indicators derivable from hydrophone data:

| Indicator | Baseline (normal) | Stress threshold |
|---|---|---|
| Cluster distribution | >98% C0 | C2 > 5% |
| Bandwidth | ~2,800 Hz | >5,000 Hz |
| Self-transition rate | ~0.975 | <0.90 |
| Grammar regime | C0-dominated | C1-dominated |

## 7.1 Pilot Held-Out Validation

A reviewer-anticipatable concern with the indicators in Table 7 is that they are derived from the same corpus used to identify the 2019 anomaly. A pilot held-out validation is reported here for the cluster-distribution indicator using the publicly distributed feature subset. After restricting to the Orcasound Lab hydrophone (station held constant), the subset contains 155 SRKW calls from 2017 (pre-crisis baseline), 140 from 2019 (known crisis), and 37 from 2020 (post-crisis recovery year). K-means at k = 3 was fit on the combined subset (random_state = 42); thresholds were derived from the 2017 baseline alone, then applied blind to 2019 and 2020.

| Year | N | Dominant cluster share | Non-dominant share | Threshold (> 5%) | Verdict |
|---|---|---|---|---|---|
| 2017 | 155 | 100.0% | 0.0% | (baseline) | BASELINE |
| **2019** | 140 | 39.3% | **60.7%** | YES | **FLAG (known crisis)** |
| **2020** | 37 | 100.0% | 0.0% | no | quiet (held-out recovery) |

The detector trained on the 2017 baseline (no exposure to 2019 data during threshold derivation) fires on 2019 and stays quiet on 2020, both at the same station. The 2020 result is small-N (37 calls) and should be treated as a pilot rather than a full false-positive test, but the direction is consistent with the 2019 effect being a real ecological signal rather than a station, equipment, or annotation-pipeline artefact. The script implementing this analysis is `analyse_holdout.py` in the code repository.

A proper prospective validation would re-run the Section 5.7 analysis on the full DCLDE annotation corpus with year-stratified train/test splits and report the empirical false-positive rate across all non-crisis years. That extension is straightforward; the gating step is access to the full source annotations rather than the feature subsets distributed here.

## 7.2 Operational Considerations

The monitoring infrastructure already exists. The Orcasound hydrophone network streams 24/7 from multiple Salish Sea locations. The JASCO/VFPA hydrophones provide additional coverage. The analysis pipeline described in this paper is reproducible and computationally inexpensive, clustering and transition analysis on a day's recordings completes in seconds on consumer hardware.

For a population of approximately 75 individuals classified as Endangered under both the U.S. Endangered Species Act and Canada's Species at Risk Act, acoustic welfare monitoring along these lines is in principle deployable using the framework presented here and existing recording infrastructure. The remaining steps are prospective validation (false-positive rate, minimum detection latency, robustness to recording conditions) on a held-out year of streaming data before any operational claim can be supported.
