#!/usr/bin/env python3
"""
The silence and the structure.

1. Inter-call intervals: does the TIMING between calls carry information?
2. Cross-station sequence matching: can we find the same sequence on both hydrophones?
3. Acoustic dimensionality: how many real dimensions does orca communication use?
4. Combinatorial productivity: how much of the possible sequence space is used?
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


def normalise_by_station(features, metadata):
    by_station = defaultdict(list)
    for i, m in enumerate(metadata):
        by_station[m['station']].append(i)
    for station, indices in by_station.items():
        sf = features[indices]
        mean = sf.mean(axis=0)
        std = np.where(sf.std(axis=0) > 1e-8, sf.std(axis=0), 1.0)
        for i in indices:
            features[i] = (features[i] - mean) / std
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return features / norms


def cluster(normed, k=2):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(normed), km


def build_file_sequences(metadata, labels):
    """Build sequences grouped by file, sorted by begin time."""
    by_file = defaultdict(list)
    for i, m in enumerate(metadata):
        by_file[m['filename']].append({
            'idx': i,
            'label': labels[i],
            'begin': float(m.get('duration', 0)),  # FileBeginSec stored as duration during extraction
            'station': m['station'],
            'utc': m['utc'],
        })

    sequences = {}
    for fname, entries in by_file.items():
        entries.sort(key=lambda x: x['begin'])
        sequences[fname] = entries
    return sequences


# ─────────────────────────────────────────────────────────────────────
# 1. INTER-CALL INTERVALS
# ─────────────────────────────────────────────────────────────────────

def analyse_intervals(metadata, labels, features):
    print("=" * 70)
    print("  1. THE SILENCE: Inter-call intervals as information")
    print("=" * 70)
    print()

    # Build sequences with timing
    by_file = defaultdict(list)
    for i, m in enumerate(metadata):
        try:
            by_file[m['filename']].append({
                'idx': i, 'label': labels[i],
                'begin': float(m.get('duration', 0)),
                'dur_call': float(m.get('duration', 0)),
            })
        except:
            pass

    # Compute ICIs per transition type
    ici_by_transition = defaultdict(list)
    ici_by_position = defaultdict(list)
    all_icis = []

    for fname, entries in by_file.items():
        entries.sort(key=lambda x: x['begin'])
        for j in range(len(entries) - 1):
            # ICI = start of next call - end of this call
            # We don't have exact end times, so use begin-to-begin as proxy
            ici = entries[j+1]['begin'] - entries[j]['begin']
            if 0 < ici < 30:
                a = entries[j]['label']
                b = entries[j+1]['label']
                ici_by_transition[(a, b)].append(ici)
                ici_by_position[j if j < 20 else 20].append(ici)
                all_icis.append(ici)

    if not all_icis:
        print("  No ICI data available")
        return

    ici_arr = np.array(all_icis)
    print(f"  Total ICIs: {len(ici_arr)}")
    print(f"  Mean: {ici_arr.mean():.3f}s, Median: {np.median(ici_arr):.3f}s, Std: {ici_arr.std():.3f}s")

    # ICI by transition type
    n_clusters = max(labels) + 1
    print(f"\n  ICI by transition type (seconds):")
    print(f"  {'Transition':>15s}  {'Mean':>8s}  {'Median':>8s}  {'Std':>8s}  {'N':>6s}")
    print(f"  {'─'*15}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}")

    for a in range(n_clusters):
        for b in range(n_clusters):
            icis = ici_by_transition.get((a, b), [])
            if len(icis) >= 10:
                arr = np.array(icis)
                trans_name = f"C{a}→C{b}"
                marker = " ◄ self" if a == b else ""
                print(f"  {trans_name:>15s}  {arr.mean():>8.3f}  {np.median(arr):>8.3f}  {arr.std():>8.3f}  {len(arr):>6d}{marker}")

    # Key test: do self-transitions have different ICIs than cross-transitions?
    self_icis = []
    cross_icis = []
    for (a, b), icis in ici_by_transition.items():
        if a == b:
            self_icis.extend(icis)
        else:
            cross_icis.extend(icis)

    if self_icis and cross_icis:
        self_arr = np.array(self_icis)
        cross_arr = np.array(cross_icis)
        print(f"\n  Self-transition ICI:  mean={self_arr.mean():.3f}s, median={np.median(self_arr):.3f}s (n={len(self_arr)})")
        print(f"  Cross-transition ICI: mean={cross_arr.mean():.3f}s, median={np.median(cross_arr):.3f}s (n={len(cross_arr)})")

        from scipy.stats import mannwhitneyu
        U, p = mannwhitneyu(self_arr, cross_arr, alternative='two-sided')
        print(f"  Mann-Whitney U: p={p:.2e} {'***' if p < 0.001 else 'n.s.'}")

        if p < 0.001:
            if self_arr.mean() < cross_arr.mean():
                print(f"  *** Calls of the SAME type come faster — bouts are tight")
                print(f"  *** Switching to a different type takes LONGER — the pause is the punctuation")
            else:
                print(f"  *** Cross-transitions are FASTER than self-transitions")

    # ICI carries mutual information?
    # Discretise ICIs into bins and compute MI with next call type
    print(f"\n  Does silence duration predict the next call type?")
    ici_bins = np.digitize(ici_arr, bins=[0.2, 0.5, 1.0, 2.0, 5.0])  # 6 bins

    # Collect (ici_bin, next_label) pairs
    pairs = []
    for fname, entries in by_file.items():
        entries.sort(key=lambda x: x['begin'])
        for j in range(len(entries) - 1):
            ici = entries[j+1]['begin'] - entries[j]['begin']
            if 0 < ici < 30:
                bin_idx = np.digitize(ici, bins=[0.2, 0.5, 1.0, 2.0, 5.0])
                pairs.append((bin_idx, entries[j+1]['label']))

    if len(pairs) > 100:
        # MI between ICI bin and next call type
        joint = Counter(pairs)
        total = len(pairs)
        ici_counts = Counter(p[0] for p in pairs)
        label_counts = Counter(p[1] for p in pairs)

        H_ici = -sum((c/total) * math.log2(c/total) for c in ici_counts.values() if c > 0)
        H_label = -sum((c/total) * math.log2(c/total) for c in label_counts.values() if c > 0)
        H_joint = -sum((c/total) * math.log2(c/total) for c in joint.values() if c > 0)
        MI = H_ici + H_label - H_joint

        print(f"  MI(silence_duration, next_call_type) = {MI:.4f} bits")
        print(f"  H(silence) = {H_ici:.4f}, H(next_call) = {H_label:.4f}")
        mi_ratio = MI / H_label if H_label > 0 else 0
        print(f"  MI / H(next_call) = {mi_ratio:.1%}")
        if mi_ratio > 0.05:
            print(f"  *** THE SILENCE CARRIES INFORMATION: {mi_ratio:.1%} of next-call uncertainty")
            print(f"  *** is resolved by how long the orca paused")
    print()


# ─────────────────────────────────────────────────────────────────────
# 2. CROSS-STATION SEQUENCE MATCHING
# ─────────────────────────────────────────────────────────────────────

def analyse_cross_station(metadata, labels):
    print("=" * 70)
    print("  2. CROSS-STATION SEQUENCE MATCHING")
    print("  Can we find the same sequence on both hydrophones?")
    print("=" * 70)
    print()

    # Group by station and filename pattern (same timestamp = same event)
    # JASCO files: AMAR{unit}.1.{timestamp}.wav
    # Same timestamp from different units = same acoustic event

    by_timestamp = defaultdict(lambda: {'north': [], 'south': []})
    for i, m in enumerate(metadata):
        fname = m['filename']
        # Extract timestamp from filename: AMAR###.1.YYYYMMDDTHHMMSSZ.wav
        parts = fname.split('.')
        if len(parts) >= 3:
            timestamp = parts[2].replace('.wav', '')
            by_timestamp[timestamp][m['station']].append({
                'idx': i, 'label': labels[i],
            })

    # Find timestamps where both stations have data
    paired = {ts: data for ts, data in by_timestamp.items()
              if data['north'] and data['south']}

    print(f"  Timestamps with data from both stations: {len(paired)}")

    if not paired:
        print("  No cross-station pairs found by timestamp")
        # Try matching by UTC minute instead
        by_minute = defaultdict(lambda: {'north': [], 'south': []})
        for i, m in enumerate(metadata):
            minute = m['utc'][:16]  # YYYY-MM-DD HH:MM
            by_minute[minute][m['station']].append({
                'idx': i, 'label': labels[i],
            })
        paired = {ts: data for ts, data in by_minute.items()
                  if len(data['north']) >= 3 and len(data['south']) >= 3}
        print(f"  Minutes with 3+ calls from both stations: {len(paired)}")

    if not paired:
        print("  Insufficient cross-station data for sequence matching")
        return

    # Compare sequences: for each paired timestamp/minute, extract the
    # sequence of call types from each station
    match_scores = []
    for ts, data in sorted(paired.items()):
        n_seq = [e['label'] for e in data['north']]
        s_seq = [e['label'] for e in data['south']]

        # Simple matching: what fraction of call types agree at each position?
        min_len = min(len(n_seq), len(s_seq))
        if min_len < 3:
            continue

        # Use edit distance normalised by length
        matches = sum(1 for a, b in zip(n_seq[:min_len], s_seq[:min_len]) if a == b)
        score = matches / min_len
        match_scores.append({
            'timestamp': ts,
            'north_len': len(n_seq),
            'south_len': len(s_seq),
            'compared': min_len,
            'matches': matches,
            'score': score,
            'north_seq': n_seq[:10],
            'south_seq': s_seq[:10],
        })

    if not match_scores:
        print("  No comparable sequences found")
        return

    scores = np.array([m['score'] for m in match_scores])
    print(f"\n  Cross-station sequence comparison:")
    print(f"  Pairs compared: {len(match_scores)}")
    print(f"  Mean match score: {scores.mean():.4f}")
    print(f"  Median: {np.median(scores):.4f}")

    # What would random matching give?
    n_clusters = max(labels) + 1
    cluster_probs = np.array([(labels == c).sum() for c in range(n_clusters)]) / len(labels)
    expected_random = sum(p**2 for p in cluster_probs)
    print(f"  Expected if random: {expected_random:.4f}")

    if scores.mean() > expected_random * 1.1:
        excess = (scores.mean() - expected_random) / expected_random * 100
        print(f"  *** Sequences match {excess:.0f}% better than random")
        print(f"  *** The same call sequence was heard from BOTH hydrophones")
    print()

    # Show top matching sequences
    match_scores.sort(key=lambda x: -x['score'])
    print(f"  Top 5 matching sequences:")
    for m in match_scores[:5]:
        n_str = "".join(f"C{x}" for x in m['north_seq'])
        s_str = "".join(f"C{x}" for x in m['south_seq'])
        print(f"    {m['timestamp']}: score={m['score']:.2f}")
        print(f"      North: {n_str}")
        print(f"      South: {s_str}")


# ─────────────────────────────────────────────────────────────────────
# 3. ACOUSTIC DIMENSIONALITY
# ─────────────────────────────────────────────────────────────────────

def analyse_dimensionality(features, metadata):
    print("=" * 70)
    print("  3. ACOUSTIC DIMENSIONALITY: How many real dimensions?")
    print("=" * 70)
    print()

    # PCA on raw features
    centered = features - features.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    cumulative = np.cumsum(explained)

    print(f"  PCA on {features.shape[0]} calls × {features.shape[1]} features:")
    print(f"\n  {'PC':>5s}  {'Var%':>8s}  {'Cumul%':>8s}  {'Bar'}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*30}")
    for i in range(min(15, len(explained))):
        bar = "█" * int(explained[i] * 200)
        print(f"  PC{i+1:>2d}  {explained[i]*100:>7.2f}%  {cumulative[i]*100:>7.2f}%  {bar}")

    dims_80 = np.searchsorted(cumulative, 0.8) + 1
    dims_90 = np.searchsorted(cumulative, 0.9) + 1
    dims_95 = np.searchsorted(cumulative, 0.95) + 1
    dims_99 = np.searchsorted(cumulative, 0.99) + 1

    print(f"\n  Effective dimensionality:")
    print(f"    80% variance: {dims_80} dimensions")
    print(f"    90% variance: {dims_90} dimensions")
    print(f"    95% variance: {dims_95} dimensions")
    print(f"    99% variance: {dims_99} dimensions")

    # What do the top PCs correspond to?
    print(f"\n  Top PC loadings (what each dimension captures):")
    feature_names = (
        ['centroid', 'bandwidth', 'rolloff', 'flux', 'contrast', 'flatness'] +  # spectral 6
        ['pulsed', 'tonal', 'mixed', 'burst', 'silence'] +  # temporal 5
        ['click_ratio', 'tonal_ratio'] +  # component 2
        ['mean_freq', 'mod_depth', 'mod_rate'] +  # modulation 3
        ['duration', 'rep_count', 'ici', 'frequency', 'pod', 'social', 'feeding', 'travel'] +  # structural 8
        [f'mel_{i}' for i in range(26)]  # fingerprint 26
    )

    for pc in range(min(5, len(Vt))):
        loadings = Vt[pc]
        top_idx = np.argsort(np.abs(loadings))[-5:][::-1]
        top_features = [(feature_names[i] if i < len(feature_names) else f'feat_{i}',
                         loadings[i]) for i in top_idx]
        features_str = ", ".join(f"{name}({val:+.3f})" for name, val in top_features)
        print(f"    PC{pc+1}: {features_str}")

    print()
    return dims_90


# ─────────────────────────────────────────────────────────────────────
# 4. COMBINATORIAL PRODUCTIVITY
# ─────────────────────────────────────────────────────────────────────

def analyse_productivity(metadata, labels):
    print("=" * 70)
    print("  4. COMBINATORIAL PRODUCTIVITY")
    print("  How much of the possible sequence space is used?")
    print("=" * 70)
    print()

    n_clusters = max(labels) + 1

    # Build sequences
    by_file = defaultdict(list)
    for i, m in enumerate(metadata):
        by_file[m['filename']].append((float(m.get('duration', 0)), labels[i]))

    sequences = []
    for fname, entries in by_file.items():
        entries.sort()
        seq = [entries[0][1]]
        for j in range(1, len(entries)):
            if entries[j][0] - entries[j-1][0] <= 30:
                seq.append(entries[j][1])
            else:
                if len(seq) >= 2:
                    sequences.append(seq)
                seq = [entries[j][1]]
        if len(seq) >= 2:
            sequences.append(seq)

    # Count unique n-grams at each length
    print(f"  {len(sequences)} sequences, {sum(len(s) for s in sequences)} total calls")
    print(f"  Call types: {n_clusters}")
    print()

    print(f"  {'N-gram':>8s}  {'Possible':>10s}  {'Observed':>10s}  {'Used%':>8s}  {'Entropy':>10s}  {'Max entropy':>12s}  {'H/Hmax':>8s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*8}")

    for n in range(1, 7):
        possible = n_clusters ** n

        # Count observed n-grams
        observed = Counter()
        for seq in sequences:
            for i in range(len(seq) - n + 1):
                ngram = tuple(seq[i:i+n])
                observed[ngram] += 1

        n_observed = len(observed)
        total = sum(observed.values())
        used_pct = n_observed / possible * 100

        # Entropy
        H = -sum((c/total) * math.log2(c/total) for c in observed.values() if c > 0) if total > 0 else 0
        H_max = math.log2(possible) if possible > 0 else 0
        h_ratio = H / H_max if H_max > 0 else 0

        print(f"  {n:>8d}  {possible:>10d}  {n_observed:>10d}  {used_pct:>7.1f}%  {H:>10.4f}  {H_max:>12.4f}  {h_ratio:>7.1%}")

    # Hapax legomena: n-grams occurring only once
    print(f"\n  Hapax legomena (n-grams occurring only once):")
    for n in [2, 3, 4]:
        observed = Counter()
        for seq in sequences:
            for i in range(len(seq) - n + 1):
                observed[tuple(seq[i:i+n])] += 1
        hapax = sum(1 for c in observed.values() if c == 1)
        total_types = len(observed)
        print(f"    {n}-grams: {hapax}/{total_types} ({hapax/total_types*100:.1f}% are unique)")

    print()


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "▓" * 70)
    print("▓  THE SILENCE AND THE STRUCTURE")
    print("▓  Intervals, cross-station matching, dimensionality, productivity")
    print("▓" * 70)
    print()

    features, metadata = load_haro()
    normed = normalise_by_station(features.copy(), metadata)
    labels, km = cluster(normed, k=2)

    print(f"  Loaded {len(features)} Haro Strait calls")
    print(f"  Clusters: {Counter(labels)}")
    print()

    analyse_intervals(metadata, labels, normed)
    analyse_cross_station(metadata, labels)
    analyse_dimensionality(features, metadata)
    analyse_productivity(metadata, labels)

    print("▓" * 70)
    print("▓  COMPLETE")
    print("▓" * 70)
    print()


if __name__ == "__main__":
    main()
