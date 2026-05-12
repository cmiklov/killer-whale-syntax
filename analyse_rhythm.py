#!/usr/bin/env python3
"""
Diggy diggy hole.

1. Rhythm analysis: is the 20ms ICI a phase-locked oscillation?
   If the ICI distribution is sharply peaked (not exponential), calls
   are rhythmically entrained — the pod is a coupled oscillator.

2. Call-response detection: when North station records a call, does
   South station record an ANSWER? Turn-taking between individuals
   at different positions.

3. Spectral drift within bouts: do calls change acoustically over
   the course of a sequence? Warming up, building intensity, fading.

4. The September 5 2017 event: 500+ calls from both stations.
   A complete pod passage. The deepest dive into a single event.

5. Rhythm-break analysis: when the steady 20ms rhythm breaks,
   what happens acoustically? Is the break itself structured?
"""

import os
import sys
import ast
import math
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))


def load_haro():
    data = np.load("data/haro_srkw_features.npz", allow_pickle=True)
    features = data["features"].copy()
    metadata = [ast.literal_eval(m) for m in data["metadata"]]
    return features, metadata


def normalise(features, metadata):
    by_station = defaultdict(list)
    for i, m in enumerate(metadata):
        by_station[m['station']].append(i)
    for station, indices in by_station.items():
        sf = features[indices]
        mean, std = sf.mean(axis=0), sf.std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        for i in indices:
            features[i] = (features[i] - mean) / std
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.where(norms > 0, norms, 1.0)


def cluster(normed, k=2):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(normed), km


# ─────────────────────────────────────────────────────────────────────
# 1. RHYTHM ANALYSIS
# ─────────────────────────────────────────────────────────────────────

def analyse_rhythm(metadata, labels):
    print("=" * 70)
    print("  1. RHYTHM: Is the pod a coupled oscillator?")
    print("=" * 70)
    print()

    # Collect ICIs within sessions
    by_file = defaultdict(list)
    for i, m in enumerate(metadata):
        by_file[(m['filename'], m['station'])].append(
            (float(m.get('duration', 0)), labels[i], i))

    all_icis = []
    bout_icis = defaultdict(list)  # ICI within same-type bouts

    for key, entries in by_file.items():
        entries.sort()
        for j in range(len(entries) - 1):
            ici = entries[j+1][0] - entries[j][0]
            if 0.001 < ici < 5.0:
                all_icis.append(ici)
                if entries[j][1] == entries[j+1][1]:
                    bout_icis[entries[j][1]].append(ici)

    if not all_icis:
        print("  No ICI data")
        return

    ici_arr = np.array(all_icis)

    # Distribution analysis
    print(f"  Total ICIs: {len(ici_arr)}")
    print(f"  Mean: {ici_arr.mean():.4f}s ({1/ici_arr.mean():.1f} Hz)")
    print(f"  Median: {np.median(ici_arr):.4f}s")
    print(f"  Std: {ici_arr.std():.4f}s")
    print(f"  CV (std/mean): {ici_arr.std()/ici_arr.mean():.3f}")

    # Is the distribution peaked or exponential?
    # A coupled oscillator has a PEAKED distribution (low CV)
    # Random firing has an EXPONENTIAL distribution (CV ≈ 1.0)
    cv = ici_arr.std() / ici_arr.mean()
    print(f"\n  Coefficient of variation: {cv:.3f}")
    if cv < 0.5:
        print(f"  *** STRONGLY RHYTHMIC: CV < 0.5 indicates phase-locked oscillation")
        print(f"  *** The pod is functioning as a COUPLED OSCILLATOR NETWORK")
    elif cv < 0.8:
        print(f"  ** Moderately rhythmic: some regularity in timing")
    else:
        print(f"  Random/Poisson-like timing (CV ≈ 1.0)")

    # Histogram of ICIs
    bins = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0]
    hist, _ = np.histogram(ici_arr, bins=bins)
    print(f"\n  ICI distribution:")
    for i in range(len(hist)):
        bar = "█" * (hist[i] * 60 // max(hist))
        pct = hist[i] / len(ici_arr) * 100
        print(f"    {bins[i]:>6.3f}-{bins[i+1]:>5.3f}s: {hist[i]:>5d} ({pct:>5.1f}%) {bar}")

    # Peak detection: is there a dominant ICI?
    fine_bins = np.linspace(0.001, 0.1, 100)
    fine_hist, _ = np.histogram(ici_arr[ici_arr < 0.1], bins=fine_bins)
    peak_idx = np.argmax(fine_hist)
    peak_ici = (fine_bins[peak_idx] + fine_bins[peak_idx + 1]) / 2
    peak_freq = 1.0 / peak_ici if peak_ici > 0 else 0
    print(f"\n  Peak ICI: {peak_ici:.4f}s ({peak_freq:.1f} Hz)")
    print(f"  Peak bin count: {fine_hist[peak_idx]} ({fine_hist[peak_idx]/len(ici_arr)*100:.1f}%)")

    # Per-bout rhythm regularity
    print(f"\n  Within-bout rhythm regularity:")
    for label in sorted(bout_icis.keys()):
        icis = np.array(bout_icis[label])
        if len(icis) >= 10:
            cv_bout = icis.std() / icis.mean()
            print(f"    C{label}: mean={icis.mean():.4f}s, CV={cv_bout:.3f}, n={len(icis)}")

    # Autocorrelation of ICI series (test for periodic structure)
    if len(all_icis) > 100:
        ici_series = np.array(all_icis[:1000])
        ici_centered = ici_series - ici_series.mean()
        autocorr = np.correlate(ici_centered, ici_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]

        # Find first peak after lag 0
        peaks = []
        for i in range(2, min(50, len(autocorr) - 1)):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.1:
                peaks.append((i, autocorr[i]))

        print(f"\n  ICI autocorrelation peaks:")
        if peaks:
            for lag, val in peaks[:5]:
                print(f"    Lag {lag}: r={val:.4f} (period = {lag} calls)")
            print(f"  *** Periodic structure detected in ICI series")
        else:
            print(f"    No significant peaks found")

    print()
    return ici_arr


# ─────────────────────────────────────────────────────────────────────
# 2. SPECTRAL DRIFT WITHIN BOUTS
# ─────────────────────────────────────────────────────────────────────

def analyse_spectral_drift(features, metadata, labels):
    print("=" * 70)
    print("  2. SPECTRAL DRIFT: Do calls change over a bout?")
    print("=" * 70)
    print()

    # Build bouts: consecutive same-type calls
    by_file = defaultdict(list)
    for i, m in enumerate(metadata):
        by_file[(m['filename'], m['station'])].append(
            (float(m.get('duration', 0)), labels[i], i))

    bouts = []
    for key, entries in by_file.items():
        entries.sort()
        current_label = entries[0][1]
        bout = [entries[0]]
        for j in range(1, len(entries)):
            if entries[j][1] == current_label and entries[j][0] - entries[j-1][0] < 5.0:
                bout.append(entries[j])
            else:
                if len(bout) >= 5:
                    bouts.append(bout)
                current_label = entries[j][1]
                bout = [entries[j]]
        if len(bout) >= 5:
            bouts.append(bout)

    print(f"  Bouts with 5+ calls: {len(bouts)}")
    print(f"  Mean bout length: {np.mean([len(b) for b in bouts]):.1f}")
    print(f"  Max bout length: {max(len(b) for b in bouts)}")

    # For each bout, track how the spectral centroid changes
    # (centroid is feature[0] * 11025)
    drift_first_half = []
    drift_second_half = []
    drift_correlations = []

    for bout in bouts:
        indices = [entry[2] for entry in bout]
        centroids = [features[idx][0] for idx in indices]  # normalised centroid

        n = len(centroids)
        positions = np.arange(n)
        centroid_arr = np.array(centroids)

        # Correlation between position and centroid (drift direction)
        if centroid_arr.std() > 1e-8:
            r = np.corrcoef(positions, centroid_arr)[0, 1]
            drift_correlations.append(r)

        # First half vs second half
        mid = n // 2
        first = centroid_arr[:mid].mean()
        second = centroid_arr[mid:].mean()
        drift_first_half.append(first)
        drift_second_half.append(second)

    if drift_correlations:
        dc = np.array(drift_correlations)
        print(f"\n  Position-centroid correlation across bouts:")
        print(f"    Mean r: {dc.mean():.4f}")
        print(f"    Std: {dc.std():.4f}")
        print(f"    Positive (frequency rises): {(dc > 0.1).sum()}/{len(dc)}")
        print(f"    Negative (frequency falls): {(dc < -0.1).sum()}/{len(dc)}")
        print(f"    Neutral: {((dc >= -0.1) & (dc <= 0.1)).sum()}/{len(dc)}")

        if abs(dc.mean()) > 0.05:
            direction = "RISES" if dc.mean() > 0 else "FALLS"
            print(f"\n  *** Systematic drift detected: frequency {direction} over bouts")
        else:
            print(f"\n  No systematic drift — frequency is stable within bouts")

    if drift_first_half and drift_second_half:
        f_arr = np.array(drift_first_half)
        s_arr = np.array(drift_second_half)
        from scipy.stats import wilcoxon
        try:
            stat, p = wilcoxon(f_arr, s_arr)
            print(f"\n  First half vs second half (Wilcoxon):")
            print(f"    First half mean centroid: {f_arr.mean():.4f}")
            print(f"    Second half mean centroid: {s_arr.mean():.4f}")
            print(f"    p={p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'}")
        except:
            pass

    # Amplitude/energy drift
    print(f"\n  Energy drift within bouts:")
    energy_drifts = []
    for bout in bouts:
        indices = [entry[2] for entry in bout]
        # Use mel-band energy sum as proxy for amplitude
        energies = [features[idx][-26:].sum() for idx in indices]  # last 26 = mel bands
        positions = np.arange(len(energies))
        e_arr = np.array(energies)
        if e_arr.std() > 1e-8:
            r = np.corrcoef(positions, e_arr)[0, 1]
            energy_drifts.append(r)

    if energy_drifts:
        ed = np.array(energy_drifts)
        print(f"    Mean energy-position r: {ed.mean():.4f}")
        rising = (ed > 0.1).sum()
        falling = (ed < -0.1).sum()
        print(f"    Energy rises: {rising}/{len(ed)}, falls: {falling}/{len(ed)}")
        if abs(ed.mean()) > 0.05:
            print(f"    *** {'Energy BUILDS' if ed.mean() > 0 else 'Energy FADES'} over bouts")

    print()


# ─────────────────────────────────────────────────────────────────────
# 3. THE SEPTEMBER 5 2017 EVENT
# ─────────────────────────────────────────────────────────────────────

def analyse_sept5_event(features, metadata, labels, normed):
    print("=" * 70)
    print("  3. THE SEPTEMBER 5 2017 EVENT: A complete pod passage")
    print("=" * 70)
    print()

    # Find all calls from Sept 5 2017
    sept5 = [(i, m) for i, m in enumerate(metadata) if '20170905' in m.get('filename', '')]

    if not sept5:
        print("  No Sept 5 data found")
        return

    print(f"  Total calls from Sept 5: {len(sept5)}")

    by_station = defaultdict(list)
    for i, m in sept5:
        by_station[m['station']].append((i, m))

    for station in sorted(by_station.keys()):
        entries = by_station[station]
        n = len(entries)
        call_types = Counter(labels[i] for i, _ in entries)
        print(f"\n  {station.upper()} station: {n} calls")
        print(f"    Call types: {dict(call_types)}")

        # Sequence of call types over time
        sorted_entries = sorted(entries, key=lambda x: float(x[1].get('duration', 0)))
        seq = [labels[i] for i, _ in sorted_entries]

        # Show first 100 calls as a visual sequence
        seq_str = "".join(str(c) for c in seq[:100])
        print(f"    First 100 calls: {seq_str}")

        # Bout analysis for this event
        bout_lengths = []
        current = seq[0]
        length = 1
        for j in range(1, len(seq)):
            if seq[j] == current:
                length += 1
            else:
                bout_lengths.append((current, length))
                current = seq[j]
                length = 1
        bout_lengths.append((current, length))

        print(f"    Bouts: {len(bout_lengths)}")
        top_bouts = sorted(bout_lengths, key=lambda x: -x[1])[:5]
        print(f"    Longest bouts: {', '.join(f'C{t}×{l}' for t, l in top_bouts)}")

        # Transition rate
        transitions = sum(1 for j in range(len(seq)-1) if seq[j] != seq[j+1])
        print(f"    Transitions: {transitions}/{len(seq)-1} ({transitions/(len(seq)-1)*100:.1f}%)")

    # Cross-station alignment for Sept 5
    north_calls = [i for i, m in sept5 if m['station'] == 'north']
    south_calls = [i for i, m in sept5 if m['station'] == 'south']

    if north_calls and south_calls:
        n_seq = [labels[i] for i in sorted(north_calls)]
        s_seq = [labels[i] for i in sorted(south_calls)]
        min_len = min(len(n_seq), len(s_seq))
        matches = sum(1 for a, b in zip(n_seq[:min_len], s_seq[:min_len]) if a == b)
        score = matches / min_len
        print(f"\n  Cross-station match for Sept 5:")
        print(f"    North: {len(n_seq)} calls, South: {len(s_seq)} calls")
        print(f"    Position-matched: {matches}/{min_len} ({score*100:.1f}%)")

        # Acoustic similarity between stations for this event
        north_vecs = normed[north_calls]
        south_vecs = normed[south_calls]
        n_centroid = north_vecs.mean(axis=0)
        s_centroid = south_vecs.mean(axis=0)
        n_centroid /= (np.linalg.norm(n_centroid) + 1e-12)
        s_centroid /= (np.linalg.norm(s_centroid) + 1e-12)
        cross_sim = float(np.dot(n_centroid, s_centroid))
        print(f"    Acoustic centroid similarity: {cross_sim:.4f}")

    print()


# ─────────────────────────────────────────────────────────────────────
# 4. RHYTHM-BREAK ANALYSIS
# ─────────────────────────────────────────────────────────────────────

def analyse_rhythm_breaks(features, metadata, labels, normed):
    print("=" * 70)
    print("  4. RHYTHM-BREAK: What happens when the pattern breaks?")
    print("=" * 70)
    print()

    by_file = defaultdict(list)
    for i, m in enumerate(metadata):
        by_file[(m['filename'], m['station'])].append(
            (float(m.get('duration', 0)), labels[i], i))

    # Find rhythm breaks: transitions where the call type changes
    break_features = []    # acoustic features at the break point
    nonbreak_features = [] # acoustic features in the middle of a bout
    pre_break = []         # features of the last call before a break
    post_break = []        # features of the first call after a break

    for key, entries in by_file.items():
        entries.sort()
        for j in range(1, len(entries) - 1):
            prev_label = entries[j-1][1]
            curr_label = entries[j][1]
            next_label = entries[j+1][1]
            idx = entries[j][2]

            if prev_label == curr_label and curr_label != next_label:
                # This is the LAST call before a break
                pre_break.append(features[idx])
            elif prev_label != curr_label and curr_label == next_label:
                # This is the FIRST call after a break
                post_break.append(features[idx])
            elif prev_label == curr_label and curr_label == next_label:
                # Middle of a bout
                nonbreak_features.append(features[idx])

    if not pre_break or not post_break or not nonbreak_features:
        print("  Insufficient break data")
        return

    pre_arr = np.array(pre_break)
    post_arr = np.array(post_break)
    mid_arr = np.array(nonbreak_features)

    # Compare: do pre-break calls differ from mid-bout calls?
    # Spectral centroid
    pre_centroid = pre_arr[:, 0].mean()
    post_centroid = post_arr[:, 0].mean()
    mid_centroid = mid_arr[:, 0].mean()

    print(f"  Calls analysed:")
    print(f"    Pre-break (last before switch):  n={len(pre_arr)}")
    print(f"    Post-break (first after switch): n={len(post_arr)}")
    print(f"    Mid-bout (stable position):      n={len(mid_arr)}")

    # Normalised centroid comparison
    print(f"\n  Spectral centroid (normalised):")
    print(f"    Mid-bout:   {mid_centroid:.4f}")
    print(f"    Pre-break:  {pre_centroid:.4f} (diff: {pre_centroid - mid_centroid:+.4f})")
    print(f"    Post-break: {post_centroid:.4f} (diff: {post_centroid - mid_centroid:+.4f})")

    # Energy comparison
    pre_energy = pre_arr[:, -26:].sum(axis=1).mean()
    post_energy = post_arr[:, -26:].sum(axis=1).mean()
    mid_energy = mid_arr[:, -26:].sum(axis=1).mean()

    print(f"\n  Spectral energy (mel-band sum):")
    print(f"    Mid-bout:   {mid_energy:.4f}")
    print(f"    Pre-break:  {pre_energy:.4f} (diff: {pre_energy - mid_energy:+.4f})")
    print(f"    Post-break: {post_energy:.4f} (diff: {post_energy - mid_energy:+.4f})")

    # Duration comparison
    pre_dur = pre_arr[:, 16].mean()  # structural[0] = duration
    post_dur = post_arr[:, 16].mean()
    mid_dur = mid_arr[:, 16].mean()

    print(f"\n  Duration:")
    print(f"    Mid-bout:   {mid_dur:.4f}s")
    print(f"    Pre-break:  {pre_dur:.4f}s (diff: {pre_dur - mid_dur:+.4f}s)")
    print(f"    Post-break: {post_dur:.4f}s (diff: {post_dur - mid_dur:+.4f}s)")

    # Statistical test
    from scipy.stats import mannwhitneyu

    print(f"\n  Mann-Whitney U (pre-break vs mid-bout):")
    for name, pre, mid in [
        ("Centroid", pre_arr[:, 0], mid_arr[:, 0]),
        ("Energy", pre_arr[:, -26:].sum(axis=1), mid_arr[:, -26:].sum(axis=1)),
        ("Duration", pre_arr[:, 16], mid_arr[:, 16]),
    ]:
        if len(pre) >= 10 and len(mid) >= 10:
            U, p = mannwhitneyu(pre, mid, alternative='two-sided')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"    {name:>12s}: p={p:.4f} {sig}")

    print()


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "▓" * 70)
    print("▓  DIGGY DIGGY HOLE")
    print("▓  Rhythm, drift, the Sept 5 event, and the breaks")
    print("▓" * 70)
    print()

    features, metadata = load_haro()
    normed = normalise(features.copy(), metadata)
    labels, km = cluster(normed, k=2)

    ici_arr = analyse_rhythm(metadata, labels)
    analyse_spectral_drift(features, metadata, labels)
    analyse_sept5_event(features, metadata, labels, normed)
    analyse_rhythm_breaks(features, metadata, labels, normed)

    print("▓" * 70)
    print("▓  THE HOLE GOES DEEPER")
    print("▓" * 70)
    print()


if __name__ == "__main__":
    main()
