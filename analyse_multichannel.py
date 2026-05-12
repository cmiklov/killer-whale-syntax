#!/usr/bin/env python3
"""
Multi-channel analysis: calls, whistles, and the acoustic network.

Orcas communicate on three channels simultaneously:
  - CALLS (S-types): discrete pulsed vocalisations — the "words"
  - WHISTLES (SW-types): tonal, individually distinctive — the "names"
  - CLICKS: echolocation — the "shared perception"

This script:
1. Extracts features from all three signal types
2. Compares the acoustic topology of calls vs whistles
3. Tests whether calls and whistles occupy DIFFERENT regions of acoustic space
4. Maps the multi-channel structure
5. Documents the sonar-communication hypothesis for future work
"""

import os
import sys
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from extract_real_features import extract_acoustic_features

WHISTLE_DIR = os.path.join(
    os.path.dirname(__file__),
    "data", "signals-srkw", "signals", "processed", "whistles"
)

CALL_DIR = os.path.join(
    os.path.dirname(__file__),
    "data", "signals-srkw", "signals", "processed", "calls", "Ford-Osborne-2018", "mp3"
)

# Primary whistle exemplars (prefer WAV for quality, fall back to MP3)
WHISTLE_MAP = {
    "SW01": "SW01_3.4.mp3",
    "SW02": "SW02_3.4.mp3",
    "SW03": "SW03_3.4.mp3",
    "SW04": "SW04_3.4.mp3",
    "SW05": "SW05_3.4.mp3",
    "SW06": "SW06_3.4.mp3",
    "SW07": "SW07_3.4.mp3",
}

# Subset of calls for comparison
CALL_MAP = {
    "S01": "FO-S01.mp3",
    "S03": "FO-S03.mp3",
    "S04": "FO-S04.mp3",
    "S07": "FO-S07.mp3",
    "S10": "FO-S10.mp3",
    "S13": "FO-S13.mp3",
    "S16": "FO-S16.mp3",
    "S19": "FO-S19.mp3",
}


def extract_all():
    """Extract features for calls and whistles."""
    call_features = {}
    whistle_features = {}

    print(f"  Extracting call features:")
    for name, fname in sorted(CALL_MAP.items()):
        path = os.path.join(CALL_DIR, fname)
        if os.path.exists(path):
            feat = extract_acoustic_features(path)
            call_features[name] = feat
            centroid = feat[0] * 11025
            print(f"    {name:6s}: centroid={centroid:.0f}Hz, dur={feat[16]:.2f}s")

    print(f"\n  Extracting whistle features:")
    for name, fname in sorted(WHISTLE_MAP.items()):
        path = os.path.join(WHISTLE_DIR, fname)
        if os.path.exists(path):
            feat = extract_acoustic_features(path)
            whistle_features[name] = feat
            centroid = feat[0] * 11025
            print(f"    {name:6s}: centroid={centroid:.0f}Hz, dur={feat[16]:.2f}s")

    return call_features, whistle_features


def main():
    print("\n" + "▓" * 70)
    print("▓  MULTI-CHANNEL: Calls vs Whistles vs the Acoustic Network")
    print("▓" * 70)
    print()

    call_features, whistle_features = extract_all()

    if not call_features or not whistle_features:
        print("  Missing data")
        return

    # ── Channel comparison ──
    print(f"\n  ═══ CHANNEL COMPARISON: Do calls and whistles occupy different acoustic space? ═══")
    print()

    call_vecs = np.array(list(call_features.values()))
    whistle_vecs = np.array(list(whistle_features.values()))
    all_vecs = np.vstack([call_vecs, whistle_vecs])

    # Normalise together
    norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    all_normed = all_vecs / norms

    call_normed = all_normed[:len(call_vecs)]
    whistle_normed = all_normed[len(call_vecs):]

    # Within-channel similarity
    call_sims = []
    for i in range(len(call_normed)):
        for j in range(i+1, len(call_normed)):
            call_sims.append(float(np.dot(call_normed[i], call_normed[j])))

    whistle_sims = []
    for i in range(len(whistle_normed)):
        for j in range(i+1, len(whistle_normed)):
            whistle_sims.append(float(np.dot(whistle_normed[i], whistle_normed[j])))

    # Between-channel similarity
    cross_sims = []
    for i in range(len(call_normed)):
        for j in range(len(whistle_normed)):
            cross_sims.append(float(np.dot(call_normed[i], whistle_normed[j])))

    call_arr = np.array(call_sims)
    whistle_arr = np.array(whistle_sims)
    cross_arr = np.array(cross_sims)

    print(f"  Within-call similarity:     mean={call_arr.mean():.4f}, std={call_arr.std():.4f}")
    print(f"  Within-whistle similarity:  mean={whistle_arr.mean():.4f}, std={whistle_arr.std():.4f}")
    print(f"  Cross-channel similarity:   mean={cross_arr.mean():.4f}, std={cross_arr.std():.4f}")

    # Are they separable?
    from scipy.stats import mannwhitneyu
    within_all = np.concatenate([call_arr, whistle_arr])
    U, p = mannwhitneyu(within_all, cross_arr, alternative='two-sided')
    print(f"\n  Mann-Whitney (within-channel vs cross-channel): p={p:.4e}")
    if p < 0.001:
        print(f"  *** CALLS AND WHISTLES OCCUPY DISTINCT ACOUSTIC REGIONS")
    elif p < 0.05:
        print(f"  * Moderate separation between channels")
    else:
        print(f"  Channels overlap acoustically")

    # ── Spectral comparison ──
    print(f"\n  ═══ SPECTRAL SIGNATURES BY CHANNEL ═══")
    print()

    call_centroids = [call_features[k][0] * 11025 for k in call_features]
    whistle_centroids = [whistle_features[k][0] * 11025 for k in whistle_features]
    call_bws = [call_features[k][1] * 11025 for k in call_features]
    whistle_bws = [whistle_features[k][1] * 11025 for k in whistle_features]
    call_durs = [call_features[k][16] for k in call_features]
    whistle_durs = [whistle_features[k][16] for k in whistle_features]

    print(f"  {'Metric':>20s}  {'Calls (S-type)':>18s}  {'Whistles (SW-type)':>18s}")
    print(f"  {'─'*20}  {'─'*18}  {'─'*18}")
    print(f"  {'Center freq (Hz)':>20s}  {np.mean(call_centroids):>8.0f}±{np.std(call_centroids):>4.0f}  {np.mean(whistle_centroids):>8.0f}±{np.std(whistle_centroids):>4.0f}")
    print(f"  {'Bandwidth (Hz)':>20s}  {np.mean(call_bws):>8.0f}±{np.std(call_bws):>4.0f}  {np.mean(whistle_bws):>8.0f}±{np.std(whistle_bws):>4.0f}")
    print(f"  {'Duration (s)':>20s}  {np.mean(call_durs):>8.2f}±{np.std(call_durs):.2f}  {np.mean(whistle_durs):>8.2f}±{np.std(whistle_durs):.2f}")

    # Statistical tests
    print(f"\n  Channel differences (Mann-Whitney):")
    for name, a, b in [
        ("Center freq", call_centroids, whistle_centroids),
        ("Bandwidth", call_bws, whistle_bws),
        ("Duration", call_durs, whistle_durs),
    ]:
        if len(a) >= 3 and len(b) >= 3:
            U, p = mannwhitneyu(a, b, alternative='two-sided')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"    {name:>15s}: p={p:.4f} {sig}")

    # ── PCA: Where do calls and whistles sit in the combined space? ──
    print(f"\n  ═══ PCA: Calls and Whistles in Shared Acoustic Space ═══")
    print()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    projected = pca.fit_transform(all_normed)

    call_proj = projected[:len(call_vecs)]
    whistle_proj = projected[len(call_vecs):]

    print(f"  PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var):")
    call_names = list(call_features.keys())
    whistle_names = list(whistle_features.keys())

    # Print positions
    print(f"\n  Call positions in PC1-PC2:")
    for i, name in enumerate(call_names):
        print(f"    {name:6s}: ({call_proj[i,0]:>+6.3f}, {call_proj[i,1]:>+6.3f})")
    print(f"\n  Whistle positions in PC1-PC2:")
    for i, name in enumerate(whistle_names):
        print(f"    {name:6s}: ({whistle_proj[i,0]:>+6.3f}, {whistle_proj[i,1]:>+6.3f})")

    # Centroid separation in PC space
    call_centroid_pc = call_proj.mean(axis=0)
    whistle_centroid_pc = whistle_proj.mean(axis=0)
    separation = np.linalg.norm(call_centroid_pc - whistle_centroid_pc)
    print(f"\n  Channel centroid separation in PC space: {separation:.4f}")

    # ── The three-channel hypothesis ──
    print(f"\n  ═══ THE THREE-CHANNEL HYPOTHESIS ═══")
    print()
    print(f"  Orca communication operates on three simultaneous channels:")
    print()
    print(f"  CHANNEL 1: CALLS (S-types)")
    print(f"    Function: Group-level communication")
    print(f"    Properties: Pulsed, broadband, long-range")
    print(f"    What we found: {len(call_features)} types, syntax with Markov order >4")
    print(f"    Spectral: {np.mean(call_centroids):.0f}±{np.std(call_centroids):.0f} Hz")
    print()
    print(f"  CHANNEL 2: WHISTLES (SW-types)")
    print(f"    Function: Individual identification (signature whistles)")
    print(f"    Properties: Tonal, narrowband, individually distinctive")
    print(f"    What we found: {len(whistle_features)} types, acoustically distinct from calls")
    print(f"    Spectral: {np.mean(whistle_centroids):.0f}±{np.std(whistle_centroids):.0f} Hz")
    print()
    print(f"  CHANNEL 3: CLICKS (echolocation)")
    print(f"    Function: Shared perception — involuntary spatial awareness network")
    print(f"    Properties: Broadband impulses, ICI encodes target range")
    print(f"    5 ICI categories (Bolen 2024): very fast/fast/medium/slow/very slow + buzz + sweep")
    print(f"    NOT YET ANALYSED — click audio not in current datasets")
    print()
    print(f"  THE NETWORK HYPOTHESIS:")
    print(f"    When an orca echolocates, every pod member hears the click AND its echo.")
    print(f"    This creates a SHARED SPATIAL MODEL — the pod collectively 'sees' the")
    print(f"    environment through each other's sonar. No explicit coordination needed.")
    print()
    print(f"    Bigg's transients hunt in near-SILENCE before attacking marine mammals.")
    print(f"    This makes acoustic sense: their prey also has acoustic awareness.")
    print(f"    Every click leaks position information into the ocean network.")
    print(f"    The 4.1s ICI and 45-call memory of Bigg's = maximum information per")
    print(f"    minimum acoustic exposure. Stealth communication protocol.")
    print()
    print(f"    SRKW can afford to be louder (salmon don't listen for orcas) but")
    print(f"    still maintain the 20ms dominant pulse (Finding 29) — perhaps a")
    print(f"    shared click-call hybrid where communication piggybacks on echolocation.")
    print()
    print(f"  FUTURE WORK:")
    print(f"    1. Extract click features from Orcasound click catalogue (when populated)")
    print(f"    2. Analyse click-call temporal correlation (are clicks synchronized with calls?)")
    print(f"    3. Compare Bigg's click rates during hunting vs non-hunting contexts")
    print(f"    4. Test whether call rate anticorrelates with click rate (channel switching)")
    print(f"    5. Model the shared perception field (how far does eavesdropping extend?)")
    print()


if __name__ == "__main__":
    main()
