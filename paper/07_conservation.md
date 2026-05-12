# 7. Conservation Applications

The 2019 prey crisis acoustic shift (*p* = 7.2 × 10⁻²⁹⁶ on bandwidth alone), confirmed as SRKW-specific by the SAR control, demonstrates that passive acoustic monitoring is in principle capable of detecting population-level stress. The thresholds below are derived from retrospective analysis of the 2005–2023 corpus; prospective validation on streaming data is the natural next step before operational deployment.

Candidate indicators derivable from hydrophone data:

| Indicator | Baseline (normal) | Stress threshold |
|---|---|---|
| Cluster distribution | >98% C0 | C2 > 5% |
| Bandwidth | ~2,800 Hz | >5,000 Hz |
| Self-transition rate | ~0.975 | <0.90 |
| Grammar regime | C0-dominated | C1-dominated |

The monitoring infrastructure already exists. The Orcasound hydrophone network streams 24/7 from multiple Salish Sea locations. The JASCO/VFPA hydrophones provide additional coverage. The analysis pipeline described in this paper is reproducible and computationally inexpensive, clustering and transition analysis on a day's recordings completes in seconds on consumer hardware.

For a population of approximately 75 individuals classified as Endangered under both the U.S. Endangered Species Act and Canada's Species at Risk Act, acoustic welfare monitoring along these lines is in principle deployable using the framework presented here and existing recording infrastructure. The remaining steps are prospective validation (false-positive rate, minimum detection latency, robustness to recording conditions) on a held-out year of streaming data before any operational claim can be supported.
