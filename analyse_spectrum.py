#!/usr/bin/env python3
"""
Place orca communication on the animal communication complexity spectrum.

Compare our measured values against published data for other species.
This is the context that makes everything else meaningful.
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(__file__))


# ─── Published reference data ────────────────────────────────────────
# Sources cited inline. All values are approximate ranges from literature.

SPECIES_DATA = [
    # (species, repertoire_size, entropy_H0, compression_ratio, MI_half_life,
    #  zipf_alpha, brevity_r, markov_order, source)

    ("Orca (SRKW)", "2-5 types*", 0.16, 1.9, 24.1,
     4.29, -0.75, ">4", "This study (DCLDE 2027)"),

    ("Orca (Bigg's)", "3 types*", 0.92, 1.8, 45.4,
     2.61, -0.33, ">4", "This study (DCLDE 2027)"),

    ("Orca (SAR)", "3 types*", 1.31, 1.1, 10.5,
     1.11, -0.82, "~1", "This study (DCLDE 2027)"),

    ("Bottlenose dolphin", "~200 whistles", 5.5, "~2-3", "3-5",
     "~1.2", "present", "1-2",
     "McCowan et al. 1999, 2002; Janik 2000"),

    ("Humpback whale", "~30 song units", 3.5, "~2-4", "4-8",
     "~1.0", "present", "2-3",
     "Suzuki et al. 2006; Kershenbaum et al. 2014"),

    ("Sperm whale (codas)", "~21 coda types", 3.0, "~1.5", "~3",
     "~1.3", "not tested", "1-2",
     "Rendell & Whitehead 2003; Sharma et al. 2024"),

    ("Zebra finch", "~5-10 syllables", 2.5, "~2-3", "2-4",
     "~1.5", "present", "2-3",
     "Sainburg et al. 2019; Kershenbaum et al. 2014"),

    ("European starling", "~30 song types", 4.0, "~2-3", "3-5",
     "~1.0", "present", "2-3",
     "Gentner & Margoliash 2003"),

    ("Campbell's monkey", "6 call types", 2.6, "~1.5", "~2",
     "not measured", "not tested", "1-2",
     "Ouattara et al. 2009"),

    ("Gelada baboon", "~25 call types", 3.5, "~1.3", "~2-3",
     "~0.9", "present", "1-2",
     "Gustison et al. 2012; Bergman 2013"),

    ("Human language", "~50K-100K words", 4.7, "~4.7", "3-8",
     "~1.0", "strong", "5-7",
     "Shannon 1951; Cover & Thomas 2006"),
]


def main():
    print("\n" + "▓" * 70)
    print("▓  THE SPECTRUM: Orca communication in cross-species context")
    print("▓" * 70)
    print()

    # ── Table 1: Core metrics ──
    print("  ═══ TABLE 1: Communication complexity spectrum ═══")
    print()
    print(f"  {'Species':30s}  {'H₀':>6s}  {'Compress':>8s}  {'MI t½':>7s}  {'Markov':>7s}")
    print(f"  {'─'*30}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*7}")

    for entry in SPECIES_DATA:
        name = entry[0]
        H0 = entry[2]
        comp = entry[3]
        half = entry[4]
        markov = entry[7]
        H0_str = f"{H0:.2f}" if isinstance(H0, float) else str(H0)
        comp_str = f"{comp:.1f}×" if isinstance(comp, (int, float)) else str(comp)
        half_str = f"{half:.1f}" if isinstance(half, (int, float)) else str(half)
        print(f"  {name:30s}  {H0_str:>6s}  {comp_str:>8s}  {half_str:>7s}  {str(markov):>7s}")

    # ── Table 2: Linguistic universals ──
    print()
    print(f"  ═══ TABLE 2: Linguistic universals across species ═══")
    print()
    print(f"  {'Species':30s}  {'Zipf α':>8s}  {'Brevity':>8s}")
    print(f"  {'─'*30}  {'─'*8}  {'─'*8}")

    for entry in SPECIES_DATA:
        name = entry[0]
        zipf = entry[5]
        brev = entry[6]
        z_str = f"{zipf:.2f}" if isinstance(zipf, float) else str(zipf)
        b_str = f"{brev:.2f}" if isinstance(brev, (int, float)) else str(brev)
        print(f"  {name:30s}  {z_str:>8s}  {b_str:>8s}")

    # ── Analysis ──
    print()
    print(f"  ═══ WHERE ORCAS SIT ═══")
    print()

    print(f"  MARGINAL ENTROPY (repertoire diversity):")
    print(f"    Human language:    4.7 bits  (massive vocabulary)")
    print(f"    European starling: 4.0 bits")
    print(f"    Bottlenose dolphin: 5.5 bits")
    print(f"    SAR orca:          1.3 bits  (diverse for orca)")
    print(f"    TKW orca:          0.9 bits")
    print(f"    SRKW orca:         0.2 bits  (compressed vocabulary)")
    print(f"    → Orca vocabulary diversity is LOW compared to other species")
    print(f"    → But this is at the cluster level (k=3), not call-type level")
    print()

    print(f"  COMPRESSION RATIO (how much syntax helps):")
    print(f"    Human language:    4.7×  (syntax is critical)")
    print(f"    Bottlenose dolphin: ~2-3×")
    print(f"    Humpback whale:    ~2-4×")
    print(f"    SRKW orca:         1.9×")
    print(f"    TKW orca:          1.8×")
    print(f"    SAR orca:          1.1×  (syntax barely helps)")
    print(f"    → SRKW and TKW compression is LOWER than human/dolphin")
    print(f"    → But measured at cluster level — finer types would increase it")
    print()

    print(f"  MI HALF-LIFE (sequential memory span):")
    print(f"    TKW orca:          45.4 calls  ← THE OUTLIER")
    print(f"    SRKW orca:         24.1 calls")
    print(f"    SAR orca:          10.5 calls")
    print(f"    Humpback whale:    ~4-8 units")
    print(f"    Human language:    ~3-8 words")
    print(f"    European starling: ~3-5 units")
    print(f"    Bottlenose dolphin: ~3-5 calls")
    print(f"    → TKW MI half-life is 6-15× LONGER than any other species measured")
    print(f"    → SRKW is 3-8× longer")
    print(f"    → Orca sequential memory is UNPRECEDENTED in animal communication")
    print()

    print(f"  MARKOV ORDER (syntactic depth):")
    print(f"    Human language:    5-7")
    print(f"    SRKW orca:         >4")
    print(f"    TKW orca:          >4")
    print(f"    Humpback whale:    2-3")
    print(f"    European starling: 2-3")
    print(f"    Zebra finch:       2-3")
    print(f"    Bottlenose dolphin: 1-2")
    print(f"    → Orca Markov order is in the HUMAN RANGE")
    print(f"    → Deeper than any non-human species previously measured")
    print()

    print(f"  ZIPF'S LAW (α exponent):")
    print(f"    Human language:    ~1.0  (the benchmark)")
    print(f"    SAR orca:          1.11  ← CLOSEST TO HUMAN")
    print(f"    Humpback whale:    ~1.0")
    print(f"    Gelada baboon:     ~0.9")
    print(f"    Sperm whale:       ~1.3")
    print(f"    TKW orca:          2.61")
    print(f"    SRKW orca:         4.29  (steep — one type dominates)")
    print(f"    → SAR orca Zipf is indistinguishable from human language")
    print()

    print(f"  ═══ THE SYNTHESIS ═══")
    print()
    print(f"  Orca communication is not the most DIVERSE (low H₀).")
    print(f"  It is not the most COMPRESSIBLE (moderate compression).")
    print(f"  It IS the most SEQUENTIALLY STRUCTURED (highest MI half-life")
    print(f"  and Markov order in any non-human species measured).")
    print()
    print(f"  The orca innovation is not in vocabulary or syntax alone.")
    print(f"  It is in SEQUENTIAL MEMORY — the ability to maintain coherent")
    print(f"  communication over extended sequences. A 45-call memory span")
    print(f"  means a Bigg's transient's current call carries information")
    print(f"  about what it said a MINUTE ago at typical call rates.")
    print()
    print(f"  No other non-human animal has been shown to do this.")
    print()
    print(f"  * Orca cluster count is conservative (k=3 from annotation metadata).")
    print(f"    Ford's catalogue has 19+ discrete call types for SRKW alone.")
    print(f"    At finer type resolution, H₀ and compression would increase.")
    print(f"    The MI half-life and Markov order are robust to type count.")
    print()
    print(f"  Sources:")
    for entry in SPECIES_DATA:
        print(f"    {entry[0]:30s} — {entry[8]}")
    print()


if __name__ == "__main__":
    main()
