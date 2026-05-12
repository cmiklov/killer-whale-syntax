#!/usr/bin/env python3
"""
Finding 54: Which frequency bands carry the most information?

The 26-band mel fingerprint is 52% of the feature space.
Which bands carry syntactic information (predict the next call)?
Which are noise? Band-specific MI tells us which frequencies matter.
"""

import os, sys, ast, numpy as np, math
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=" * 70)
    print("  MEL-BAND FORENSICS: Which frequencies carry information?")
    print("=" * 70)
    print()

    data = np.load("data/haro_srkw_features.npz", allow_pickle=True)
    features = data["features"].copy()
    metadata = [ast.literal_eval(m) for m in data["metadata"]]

    # The mel bands are the last 26 features (indices 24-49)
    mel_start = 24  # after spectral(6)+temporal(5)+component(2)+modulation(3)+structural(8)
    mel_bands = features[:, mel_start:]
    n_bands = mel_bands.shape[1]

    print(f"  Calls: {len(features)}")
    print(f"  Mel bands: {n_bands}")

    # Build adjacent pairs
    by_file = defaultdict(list)
    for i, m in enumerate(metadata):
        by_file[(m['filename'], m['station'])].append(
            (float(m.get('duration', 0)), i))

    adjacent_pairs = []
    for key, entries in by_file.items():
        entries.sort()
        for j in range(len(entries) - 1):
            adjacent_pairs.append((entries[j][1], entries[j+1][1]))

    print(f"  Adjacent pairs: {len(adjacent_pairs)}")

    # For each mel band: compute correlation between current and next call
    print(f"\n  Per-band sequential correlation (current → next):")
    print(f"  {'Band':>6s}  {'Freq range':>14s}  {'Corr':>8s}  {'|Corr|':>8s}  {'Bar':>30s}")
    print(f"  {'─'*6}  {'─'*14}  {'─'*8}  {'─'*8}  {'─'*30}")

    # Approximate mel band frequency ranges (22050 Hz sr, 26 bands)
    mel_freqs = np.linspace(0, 22050/2, n_bands + 1)

    band_corrs = []
    for b in range(n_bands):
        current = np.array([mel_bands[i, b] for i, j in adjacent_pairs])
        next_val = np.array([mel_bands[j, b] for i, j in adjacent_pairs])

        if current.std() > 1e-8 and next_val.std() > 1e-8:
            r = np.corrcoef(current, next_val)[0, 1]
        else:
            r = 0

        band_corrs.append(r)
        freq_lo = mel_freqs[b]
        freq_hi = mel_freqs[b + 1]
        bar = "█" * int(abs(r) * 60)
        print(f"  {b:>6d}  {freq_lo:>6.0f}-{freq_hi:>5.0f}Hz  {r:>+8.4f}  {abs(r):>8.4f}  {bar}")

    band_corrs = np.array(band_corrs)

    # Which bands carry the most sequential information?
    top_bands = np.argsort(np.abs(band_corrs))[::-1]
    print(f"\n  Top 5 most informative bands:")
    for rank, b in enumerate(top_bands[:5]):
        print(f"    #{rank+1}: Band {b} ({mel_freqs[b]:.0f}-{mel_freqs[b+1]:.0f} Hz), r={band_corrs[b]:+.4f}")

    bottom_bands = np.argsort(np.abs(band_corrs))
    print(f"\n  Bottom 5 (least informative):")
    for rank, b in enumerate(bottom_bands[:5]):
        print(f"    #{rank+1}: Band {b} ({mel_freqs[b]:.0f}-{mel_freqs[b+1]:.0f} Hz), r={band_corrs[b]:+.4f}")

    # Overall: how much of the mel spectrum carries sequential info?
    significant_bands = (np.abs(band_corrs) > 0.1).sum()
    print(f"\n  Bands with |r| > 0.1: {significant_bands}/{n_bands}")
    print(f"  Mean |r|: {np.abs(band_corrs).mean():.4f}")
    print(f"  Max |r|:  {np.abs(band_corrs).max():.4f} (band {np.argmax(np.abs(band_corrs))})")

    # Is the information concentrated in low, mid, or high frequencies?
    low = np.abs(band_corrs[:9]).mean()
    mid = np.abs(band_corrs[9:18]).mean()
    high = np.abs(band_corrs[18:]).mean()
    print(f"\n  Information by frequency region:")
    print(f"    Low  (0-{mel_freqs[9]:.0f} Hz):     mean |r| = {low:.4f}")
    print(f"    Mid  ({mel_freqs[9]:.0f}-{mel_freqs[18]:.0f} Hz): mean |r| = {mid:.4f}")
    print(f"    High ({mel_freqs[18]:.0f}-{mel_freqs[26]:.0f} Hz):  mean |r| = {high:.4f}")

    if low > mid and low > high:
        print(f"    *** Low frequencies carry the most sequential information")
    elif mid > low and mid > high:
        print(f"    *** Mid frequencies carry the most sequential information")
    elif high > low and high > mid:
        print(f"    *** High frequencies carry the most sequential information")

    print()


if __name__ == "__main__":
    main()
