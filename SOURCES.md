# Sources and References

## Primary Data Sources

### DCLDE 2027 (NOAA)
- **Citation:** Myers, S. et al. (2025). "A Public Dataset of Annotated Orcinus orca Acoustic Signals for Detection and Ecotype Classification." *Nature Scientific Data.*
- **DOI:** 10.25921/15ey-mh50
- **Access:** gs://noaa-passive-bioacoustic/dclde/2027/dclde_2027_killer_whales/
- **Contents:** 207,574 annotations, 4 ecotypes (SRKW, TKW, NRKW, SAR, OKW), 23 locations, 2005-2023
- **License:** CC-BY 4.0
- **Used in:** Findings 6-39

### Ford-Osborne Call Catalogue
- **Citation:** Ford, J.K.B. (1987). "A catalogue of underwater calls produced by killer whales (Orcinus orca) in British Columbia." Canadian Data Report of Fisheries and Aquatic Sciences No. 633.
- **Citation:** Ford, J.K.B. (1991). "Vocal traditions among resident killer whales (Orcinus orca) in coastal waters of British Columbia." *Canadian Journal of Zoology*, 69(6), 1454-1483.
- **DOI:** 10.1139/z91-206
- **Access:** Via Orcasound signals-srkw repo (digitised audio)
- **Contents:** SRKW call types S01-S19, spectrograms, pod repertoire tables
- **Used in:** Findings 1-5

### Orcasound Signals-SRKW Repository
- **URL:** https://github.com/orcasound/signals-srkw
- **Contents:** 29 Ford-Osborne digitised audio exemplars (MP3), 7 whistle exemplars (SW01-SW07), click catalogue (ICI categories defined, audio being collected), labels taxonomy
- **License:** CC BY-NC-SA 4.0
- **Used in:** Findings 1-5, 39

### Orcasound SRKW Click Catalog
- **Citation:** Bolen, M. (2024). SRKW Click Catalog. Orcasound.
- **URL:** https://www.orcasound.net/portfolio/srkw-click-catalog/
- **Contents:** 5 ICI categories (very fast <50ms, fast 50-100ms, medium 100-250ms, slow 250-400ms, very slow >400ms) + 2 special cases (buzz, sweep)
- **Used in:** Discussion (future work)

### Orcasound SRKW Whistle Catalog
- **Citation:** Day, L. (2024). SRKW Whistle Catalog. Orcasound.
- **Contents:** 7 whistle types (SW01-SW07) with audio exemplars
- **Used in:** Finding 39

### DORI Dataset
- **Citation:** Nestor, B., Yao, B., Moore, J., Kanes, J. (2026). "Positive-Unlabelled Active Learning to Curate a Dataset for Orca Resident Interpretation." arXiv:2602.09295.
- **Contents:** 919 hours SRKW data (largest curation to date)
- **License:** CC-BY 4.0
- **Status:** Not yet used; identified for future work

### Orcasound Open Data (AWS)
- **URL:** https://registry.opendata.aws/orcasound/
- **Contents:** Live-streamed and archived hydrophone audio (2018-present), Salish Sea
- **License:** CC BY-NC-SA 4.0

---

## Neuroanatomy

### Orca Brain Structure
- Marino, L., Sherwood, C.C., Delman, B.N., Tang, C.Y., Naidich, T.P., Hof, P.R. (2004). "Neuroanatomy of the killer whale (Orcinus orca) from magnetic resonance images." *The Anatomical Record Part A*, 281A(2), 1256-1263. DOI: 10.1002/ar.a.20075
- Marino, L. et al. (2016). "Neuroanatomy of the killer whale (Orcinus orca): a magnetic resonance imaging investigation of structure with insights on function and evolution." *Anatomical Record.*

### Cetacean Cortical Complexity
- Hof, P.R., Van der Gucht, E. (2005). "Cortical complexity in cetacean brains." *The Anatomical Record Part A*, 287A(1). DOI: 10.1002/ar.a.20258

### Cetacean Cognition
- Marino, L., Connor, R.C., Fordyce, R.E., Herman, L.M., Hof, P.R., Lefebvre, L., Lusseau, D., McCowan, B., Nimchinsky, E.A., Pack, A.A., Rendell, L., Reidenberg, J.S., Reiss, D., Uhen, M.D., Van der Gucht, E., Whitehead, H. (2007). "Cetaceans Have Complex Brains for Complex Cognition." *PLOS Biology*, 5(5), e139.

### Auditory Cortex
- Hof, P.R., Glezer, I.I., Nimchinsky, E.A., Erwin, J.M. (1992). "The primary auditory cortex in cetacean and human brain: a comparative analysis of neurofilament protein-containing pyramidal neurons." *Neuroscience Letters*, 138(1).

---

## Cross-Species Communication Comparisons

### Information Theory Foundations
- Shannon, C.E. (1951). "Prediction and entropy of printed English." *Bell System Technical Journal*, 30(1), 50-64.
- Cover, T.M., Thomas, J.A. (2006). *Elements of Information Theory.* 2nd edition. Wiley-Interscience.

### Dolphin Communication
- McCowan, B., Hanser, S.F., Doyle, L.R. (1999). "Quantitative tools for comparing animal communication systems: information theory applied to bottlenose dolphin whistle repertoires." *Animal Behaviour*, 57(2), 409-419.
- McCowan, B., Doyle, L.R., Hanser, S.F. (2002). "Using information theory to assess the diversity, complexity, and development of communicative repertoires." *Journal of Comparative Psychology*, 116(2), 166-172.
- Janik, V.M. (2000). "Whistle matching in wild bottlenose dolphins (Tursiops truncatus)." *Science*, 289(5483), 1355-1357.

### Whale Communication
- Suzuki, R., Buck, J.R., Tyack, P.L. (2006). "Information entropy of humpback whale songs." *Journal of the Acoustical Society of America*, 119(3), 1849-1866.
- Rendell, L., Whitehead, H. (2003). "Vocal clans in sperm whales (Physeter macrocephalus)." *Proceedings of the Royal Society B*, 270(1512), 225-231.
- Sharma, G. et al. (2024). "Contextual and combinatorial structure in sperm whale vocalisations." *Nature Communications.*

### Birdsong
- Gentner, T.Q., Margoliash, D. (2003). "Neuronal populations and single cells representing learned auditory objects." *Nature*, 424(6949), 669-674.
- Sainburg, T., Thielk, M., Gentner, T.Q. (2019). "Finding, visualizing, and quantifying latent structure across diverse animal vocal repertoires." *PLOS Computational Biology.*
- Kershenbaum, A. et al. (2014). "Acoustic sequences in non-human animals: a tutorial review and prospectus." *Biological Reviews*, 89(1), 135-152.

### Primate Communication
- Ouattara, K., Lemasson, A., Zuberbühler, K. (2009). "Campbell's monkeys concatenate vocalizations into context-specific call sequences." *Proceedings of the National Academy of Sciences*, 106(51), 22026-22031.
- Gustison, M.L., le Roux, A., Bergman, T.J. (2012). "Derived vocalizations of geladas (Theropithecus gelada) and the evolution of vocal complexity in primates." *Philosophical Transactions of the Royal Society B*, 367(1597), 1847-1859.
- Bergman, T.J. (2013). "Speech-like vocalized lip-smacking in geladas." *Current Biology*, 23(7), R268-R269.

---

## Orca Ecology and Conservation

### SRKW Vocal Behaviour
- Ford, J.K.B. (1989). "Acoustic behaviour of resident killer whales (Orcinus orca) off Vancouver Island, British Columbia." *Canadian Journal of Zoology*, 67(3), 727-745.
- Deecke, V.B., Ford, J.K.B., Spong, P. (2000). "Dialect change in resident killer whales: implications for vocal learning and cultural transmission." *Animal Behaviour*, 60(5), 629-638.
- Holt, M.M. et al. (2008). "Speaking up: Killer whales (Orcinus orca) increase their call amplitude in response to vessel noise." *Journal of the Acoustical Society of America*, 125(1).
- Holt, M.M. et al. (2013). "Vocal performance affects metabolic rate in dolphins: implications for animals communicating in noisy environments." *Journal of Experimental Biology.*

### Orca ML/Detection
- Bergler, C. et al. (2019). "ORCA-SPOT: An Automatic Killer Whale Sound Detection Toolkit Using Deep Learning." *Scientific Reports*, 9, 10997.

---

## Linguistic Universals

### Zipf's Law
- Zipf, G.K. (1949). *Human Behavior and the Principle of Least Effort.* Addison-Wesley.

### Menzerath's Law
- Menzerath, P. (1954). *Die Architektonik des deutschen Wortschatzes.* Dümmler.

### Heaps' Law
- Heaps, H.S. (1978). *Information Retrieval: Computational and Theoretical Aspects.* Academic Press.

### Brevity Law (Zipf's Law of Abbreviation)
- Zipf, G.K. (1935). *The Psycho-Biology of Language.* Houghton Mifflin.

---

## Methodology

### Reaction-Diffusion
- Turing, A.M. (1952). "The Chemical Basis of Morphogenesis." *Philosophical Transactions of the Royal Society B*, 237(641), 37-72.
- Gray, P., Scott, S.K. (1984). "Autocatalytic reactions in the isothermal, continuous stirred tank reactor." *Chemical Engineering Science*, 39(6), 1087-1097.

### Johnson-Lindenstrauss Lemma
- Johnson, W.B., Lindenstrauss, J. (1984). "Extensions of Lipschitz mappings into a Hilbert space." *Contemporary Mathematics*, 26, 189-206.

### Procrustes Analysis
- Gower, J.C. (1975). "Generalized procrustes analysis." *Psychometrika*, 40(1), 33-51.
- Schönemann, P.H. (1966). "A generalized solution of the orthogonal procrustes problem." *Psychometrika*, 31(1), 1-10.

---

## Data Summary

| Source | Calls/Annotations | Ecotypes | Years | Used for |
|---|---|---|---|---|
| DCLDE 2027 | 207,574 | SRKW, TKW, SAR, OKW, NRKW | 2005-2023 | Findings 6-45 |
| Ford-Osborne 2018 | 29 exemplars | SRKW | 1987 | Findings 1-5 |
| Orcasound (audio extracted) | 577 calls | SRKW | 2017-2020 | Finding 6 |
| Haro Strait (audio extracted) | 4,862 calls | SRKW | 2017 | Findings 24-32, 40-42 |
| TKW audio (extracted) | 2,453 calls | TKW | 2011-2020 | Findings 44-45 |
| OKW audio (extracted) | 1,255 calls | OKW | 2013-2023 | Finding 45 |
| Whistle exemplars | 7 types | SRKW | 2022-2024 | Finding 39 |
| **Total analysed** | **approximately 40,000 calls** | **4 ecotypes** | **18 years** | **58 findings (52 standing, 2 retracted, 4 stress-tested)** |
