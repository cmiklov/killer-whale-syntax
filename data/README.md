# Orca-Engine Data Sources

## Call-Type Catalogues

The orca-engine loads call-type catalogues from CSV files in `data/catalogues/`.

### CSV Format

```csv
id,call_type,pod,description,contexts,frequency
1,S1,J,"Discrete pulsed call, two-component",foraging|socializing,0.15
2,S2i,J,"Rising upcall variant",travel|socializing,0.08
```

**Columns:**
- `id` — unique integer ID
- `call_type` — call type label (e.g. S1, S2i, S10)
- `pod` — pod identifier (e.g. J, K, L)
- `description` — human description of the call
- `contexts` — pipe-separated behavioural contexts
- `frequency` — normalised occurrence rate (0-1)

### Real Data: srkw_calls.csv

`data/catalogues/srkw_calls.csv` contains the real SRKW call-type catalogue:
46 entries across J, K, L pods. Call types S01-S46 from Ford's classification
system plus post-Ford additions from the Orcasound expanded taxonomy.

Acoustic features are NOT included in the CSV — they require audio extraction.
The kernel loads the CSV for topology analysis; audio features come separately.

### Published Sources

- **Ford (1987)** — "A catalogue of underwater calls produced by killer whales
  (Orcinus orca) in British Columbia." DFO Canada Technical Report No. 633.
  The original SRKW call catalogue with spectrograms.

- **Ford (1991)** — "Vocal traditions among resident killer whales in coastal
  waters of British Columbia." Can. J. Zool. 69:1454-1483. DOI: 10.1139/z91-206.
  Defines call types S1-S19 for SRKW. The foundational reference.

- **Orcasound signals-srkw** — Open versioned archive of SRKW signal types.
  42 S-type calls, 7 whistle types, click categories. FLAC + spectrograms.
  https://github.com/orcasound/signals-srkw (CC BY-NC-SA 4.0)

- **DCLDE 2026 (NOAA)** — 225,000+ call-level annotations from 23 locations
  (2005-2023). Three ecotypes: Resident, Bigg's, Offshore. CSV annotations.
  Google Cloud: gs://noaa-passive-bioacoustic/dclde/2026
  https://catalog.data.gov/dataset/dclde-2026-killer-whale-orcinus-orca-ecotype-and-other-species-annotations-for-the-detecti-2026

- **DORI (Nestor et al. 2026)** — "Positive-Unlabelled Active Learning to Curate
  a Dataset for Orca Resident Interpretation." arXiv:2602.09295.
  919 hours of SRKW data. Largest curation to date. CC-BY 4.0.

- **Nature Scientific Data (2025)** — "A Public Dataset of Annotated Orcinus orca
  Acoustic Signals for Detection and Ecotype Classification."
  https://www.nature.com/articles/s41597-025-05281-5

- **Orcasound AWS** — Raw hydrophone archives (2018-present).
  s3://streaming-orcasound-net/
  https://registry.opendata.aws/orcasound/

- **Kaggle: Orca Data for Animal-Spot** — Pre-processed ML-ready dataset.
  https://www.kaggle.com/datasets/davidkebert/orca-data-for-animal-spot

### Mock Data

The test suite uses synthetic call types based on real SRKW catalogue structure
but with generated acoustic features. See `tests/conftest.py` for the mock catalogue.

## Audio Features

When real audio data is available, features can be extracted using librosa
and stored alongside the catalogue. The feature extraction pipeline is in
`orca/features.py`. Feature vectors are 50-dimensional:

- Spectral shape (6D)
- Temporal envelope (5D)
- Click/tonal components (2D)
- Frequency modulation contour (3D)
- Structural/context (8D)
- Spectral fingerprint (26D)
