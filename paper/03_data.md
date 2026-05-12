# 3. Data

Three datasets were used, each serving a distinct role in the analysis. The study proceeded iteratively, initial findings on catalogue exemplars motivated extraction of per-call features from progressively larger and more independent datasets.

## 3.1 Ford-Osborne Call Catalogue

Ford (1987, 1991) classified SRKW discrete pulsed calls into types S01–S19, with pod-specific repertoires for J, K, and L pods. Digitised audio exemplars of 29 call types were obtained from the Orcasound signals-srkw repository (CC BY-NC-SA 4.0), which hosts the Ford-Osborne 2018 recordings. Seven whistle exemplars (SW01–SW07) from the SRKW Whistle Catalog (Day, 2024) were used for multi-channel analysis.

This dataset provides one high-quality recording per call type. It enabled the initial topology construction and cross-pod alignment (Section 5.1) but cannot support per-call statistical analysis. Its role in the study is foundational: it established that the R/D framework detects known structure, motivating the transition to per-call data.

## 3.2 DCLDE 2027 Dataset

The Detection, Classification, Localisation, and Density Estimation (DCLDE) 2027 dataset (Myers et al., 2025; DOI: 10.25921/15ey-mh50) contains 207,574 call-level annotations from 23 hydrophone locations across the northeastern Pacific, spanning 2005–2023. Annotations include bounding boxes in time-frequency space (start time, end time, low frequency, high frequency), ecotype labels (SRKW, TKW, NRKW, SAR, OKW), and recording metadata. The dataset is publicly available under CC-BY 4.0.

From the DCLDE annotations, three subsets were extracted:

**Orcasound stations (577 calls).** Individual call segments extracted from three simultaneously operating hydrophones in the Salish Sea (Orcasound Lab: 332 calls; Bush Point: 211 calls; Port Townsend: 34 calls). These provided the first per-call acoustic features and enabled station-independence validation (Section 5.2).

**Haro Strait (4,862 calls).** SRKW calls extracted from two simultaneous JASCO hydrophones in Haro Strait, core J/K/L pod territory. This is the primary independent validation dataset. Its role is to test whether findings from the Orcasound subset replicate on a 9× larger sample from different recording infrastructure (Section 5.3).

**Full annotation corpus.** Call-level annotations (without audio extraction) for all ecotypes: SRKW (14,240), TKW (4,121), SAR (8,078), OKW (1,495). Annotation metadata (center frequency, bandwidth, duration) were used for cross-ecotype information-theoretic analysis (Sections 5.4–5.6). An additional 2,453 TKW and 1,255 OKW calls were extracted with full acoustic features for the topology-syntax analysis (Section 5.8).

Total calls analysed across all subsets: approximately 40,000 across four ecotypes and 18 years.

## 3.3 Data Limitations

NRKW annotations in the DCLDE are detection-level only (8,266 presence/absence tags) and lack the individual call annotations required for sequential or information-theoretic analysis. Northern Residents are therefore absent from this study despite their relevance as a comparison population.

Behavioural metadata in the DCLDE is limited to annotation bounding boxes (low frequency, high frequency, duration). Per-call activity tags (foraging, socialising, travelling, resting) are not available. This constrains the phonosemantic analysis reported in Section 5.1: the type-level correlation between acoustic form and behavioural function (r = 0.77 on Ford-Osborne exemplars) could not be replicated at the call level using available metadata (r = 0.025 on Haro Strait data). The limitation is in the metadata resolution, not the acoustic features, a point discussed in Section 8.

The DORI dataset (Nestor et al., 2026; 919 hours of curated SRKW recordings) was identified during the study but not integrated. It may contain richer per-call metadata suitable for resolving the phonosemantic question at the call level.
