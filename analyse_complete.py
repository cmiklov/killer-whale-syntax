#!/usr/bin/env python3
"""
Complete SRKW analysis pipeline:
  1. Label clusters against Ford call types
  2. Temporal sequence analysis (transition syntax)
  3. Scale to full DCLDE SRKW dataset
  4. Three-ecotype comparison (SRKW vs Bigg's vs Offshore)
  5. Cross-ecotype Procrustes alignment

This is the paper.
"""

import os
import sys
import ast
import csv
import math
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from orca.features import N_FEATURES
from extract_real_features import extract_acoustic_features, EXEMPLAR_MAP, AUDIO_DIR


# ─────────────────────────────────────────────────────────────────────
# STEP 1: Label clusters against Ford call types
# ─────────────────────────────────────────────────────────────────────

def step1_label_clusters():
    print("=" * 70)
    print("  STEP 1: Label clusters against Ford call types")
    print("=" * 70)
    print()

    # Load DCLDE per-call features
    data = np.load("data/dclde_srkw_features.npz", allow_pickle=True)
    dclde_features = data["features"].copy()
    dclde_metadata = [ast.literal_eval(m) for m in data["metadata"]]

    # Load Ford-Osborne exemplar features
    ford_data = np.load("data/srkw_acoustic_features.npz", allow_pickle=True)
    ford_types = list(ford_data["call_types"])
    ford_features = ford_data["features"].copy()

    # Per-station normalise the DCLDE features (same as analyse_dclde_normalised)
    by_station = defaultdict(list)
    for i, m in enumerate(dclde_metadata):
        by_station[m['dataset']].append(i)

    station_means = {}
    station_stds = {}
    for ds, indices in by_station.items():
        sf = dclde_features[indices]
        station_means[ds] = sf.mean(axis=0)
        station_stds[ds] = np.where(sf.std(axis=0) > 1e-8, sf.std(axis=0), 1.0)
        for i in indices:
            dclde_features[i] = (dclde_features[i] - station_means[ds]) / station_stds[ds]

    norms = np.linalg.norm(dclde_features, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    dclde_normed = dclde_features / norms

    # Cluster
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(dclde_normed)

    # Now normalise Ford exemplars using the global DCLDE statistics
    # Use orcasound_lab stats as the reference (largest station)
    ref_mean = station_means['orcasound_lab']
    ref_std = station_stds['orcasound_lab']
    ford_normed = (ford_features - ref_mean) / ref_std
    ford_norms = np.linalg.norm(ford_normed, axis=1, keepdims=True)
    ford_norms = np.where(ford_norms > 0, ford_norms, 1.0)
    ford_normed = ford_normed / ford_norms

    # For each Ford exemplar, find nearest cluster centroid
    centroids = km.cluster_centers_
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroid_norms = np.where(centroid_norms > 0, centroid_norms, 1.0)
    centroids_normed = centroids / centroid_norms

    print(f"  Mapping {len(ford_types)} Ford call types to 3 clusters:\n")

    cluster_names = {}
    cluster_calls = defaultdict(list)
    for i, ct in enumerate(ford_types):
        sims = ford_normed[i] @ centroids_normed.T
        best_cluster = int(np.argmax(sims))
        best_sim = float(sims[best_cluster])
        cluster_calls[best_cluster].append((ct, best_sim))

    for c in range(3):
        calls = sorted(cluster_calls[c], key=lambda x: -x[1])
        n_dclde = (labels == c).sum()
        call_list = ", ".join(f"{ct}({sim:.2f})" for ct, sim in calls[:10])
        print(f"  Cluster {c} ({n_dclde} DCLDE calls):")
        print(f"    Ford types: {call_list}")
        if len(calls) > 10:
            print(f"    ... and {len(calls) - 10} more")
        print()

    return labels, dclde_normed, dclde_metadata, km


# ─────────────────────────────────────────────────────────────────────
# STEP 2: Temporal sequence analysis
# ─────────────────────────────────────────────────────────────────────

def step2_sequence_analysis(labels, metadata):
    print("=" * 70)
    print("  STEP 2: Temporal sequence analysis (transition syntax)")
    print("=" * 70)
    print()

    # Group calls by recording session (same file) and sort by time
    by_file = defaultdict(list)
    for i, m in enumerate(metadata):
        by_file[m['filename']].append((i, m))

    # Build transition matrix between clusters
    n_clusters = 3
    transitions = np.zeros((n_clusters, n_clusters), dtype=int)
    total_transitions = 0
    sequences = []

    for fname, entries in by_file.items():
        if len(entries) < 2:
            continue
        # Sort by timestamp within file
        sorted_entries = sorted(entries, key=lambda x: x[1].get('utc', ''))
        seq = [labels[idx] for idx, _ in sorted_entries]
        sequences.append(seq)

        for j in range(len(seq) - 1):
            transitions[seq[j], seq[j + 1]] += 1
            total_transitions += 1

    print(f"  Sessions with 2+ calls: {len(sequences)}")
    print(f"  Total transitions: {total_transitions}")

    # Transition probabilities
    print(f"\n  Transition matrix (counts):")
    print(f"          → C0    → C1    → C2")
    for i in range(n_clusters):
        row = "    ".join(f"{transitions[i, j]:5d}" for j in range(n_clusters))
        total = transitions[i].sum()
        print(f"    C{i} →  {row}   (total={total})")

    # Normalise to probabilities
    print(f"\n  Transition probabilities:")
    print(f"          → C0     → C1     → C2")
    for i in range(n_clusters):
        total = transitions[i].sum()
        if total > 0:
            probs = transitions[i] / total
            row = "    ".join(f"{probs[j]:.3f}" for j in range(n_clusters))
            print(f"    C{i} →  {row}")

    # Test for non-randomness: chi-squared
    # Under null hypothesis, transitions are proportional to cluster sizes
    cluster_sizes = np.array([(labels == c).sum() for c in range(n_clusters)])
    expected_probs = cluster_sizes / cluster_sizes.sum()

    chi2 = 0
    for i in range(n_clusters):
        row_total = transitions[i].sum()
        if row_total == 0:
            continue
        for j in range(n_clusters):
            expected = row_total * expected_probs[j]
            if expected > 0:
                chi2 += (transitions[i, j] - expected) ** 2 / expected

    df = (n_clusters - 1) ** 2
    # Rough p-value from chi2 distribution
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df)

    print(f"\n  Non-randomness test:")
    print(f"    Chi-squared: {chi2:.2f} (df={df})")
    print(f"    p-value: {p_value:.6f}")
    if p_value < 0.001:
        print(f"    *** HIGHLY SIGNIFICANT: Transitions are non-random (p < 0.001)")
        print(f"    *** This is evidence of SYNTAX in orca communication")
    elif p_value < 0.05:
        print(f"    * Significant: Transitions are non-random (p < 0.05)")
    else:
        print(f"    Not significant: Transitions appear random")

    # Self-transition rate (does a call type tend to repeat?)
    self_trans = sum(transitions[i, i] for i in range(n_clusters))
    self_rate = self_trans / total_transitions if total_transitions > 0 else 0
    expected_self = sum(expected_probs[i] ** 2 for i in range(n_clusters))
    print(f"\n  Self-transition rate: {self_rate:.3f} (expected if random: {expected_self:.3f})")
    if self_rate > expected_self * 1.2:
        print(f"    Calls tend to REPEAT (same type follows same type)")
    elif self_rate < expected_self * 0.8:
        print(f"    Calls tend to ALTERNATE (different type follows)")
    print()


# ─────────────────────────────────────────────────────────────────────
# STEP 3: Scale up — extract features from more DCLDE data
# ─────────────────────────────────────────────────────────────────────

def step3_scale_up():
    """Use the full annotation set for statistics (no audio download needed)."""
    print("=" * 70)
    print("  STEP 3: Full DCLDE SRKW annotation analysis (14,240 calls)")
    print("=" * 70)
    print()

    with open("data/dclde/Annotations.csv", 'r') as f:
        rows = list(csv.DictReader(f))

    srkw = [r for r in rows if r['KW'] == '1' and r['AnnotationLevel'] == 'Call'
            and r['Ecotype'] == 'SRKW']

    print(f"  Total SRKW call-level annotations: {len(srkw)}")

    # Temporal analysis on the FULL dataset
    by_file = defaultdict(list)
    for r in srkw:
        try:
            by_file[r['Soundfile']].append({
                'utc': r['UTC'],
                'begin': float(r['FileBeginSec']),
                'end': float(r['FileEndSec']),
                'low_freq': float(r['LowFreqHz']) if r['LowFreqHz'] != 'NA' else None,
                'high_freq': float(r['HighFreqHz']) if r['HighFreqHz'] != 'NA' else None,
                'dataset': r['Dataset'],
            })
        except (ValueError, KeyError):
            pass

    # Frequency-based clustering on the full annotation set
    # Use LowFreqHz and HighFreqHz as proxy features
    freq_calls = [c for calls in by_file.values() for c in calls
                  if c['low_freq'] is not None and c['high_freq'] is not None
                  and c['low_freq'] > 0 and c['high_freq'] > 0]

    print(f"  Calls with frequency data: {len(freq_calls)}")

    lows = np.array([c['low_freq'] for c in freq_calls])
    highs = np.array([c['high_freq'] for c in freq_calls])
    bws = highs - lows
    centers = (highs + lows) / 2
    durs = np.array([c['end'] - c['begin'] for c in freq_calls])

    print(f"\n  Frequency statistics (full SRKW dataset):")
    print(f"    Center freq: mean={centers.mean():.0f} Hz, std={centers.std():.0f} Hz")
    print(f"    Bandwidth:   mean={bws.mean():.0f} Hz, std={bws.std():.0f} Hz")
    print(f"    Duration:    mean={durs.mean():.2f}s, std={durs.std():.2f}s")

    # Quick clustering on frequency features
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = np.column_stack([
        centers / centers.max(),
        bws / bws.max(),
        durs / durs.max(),
    ])

    print(f"\n  K-means on frequency features ({len(X)} calls):")
    best_k, best_sil = 0, -1
    for k in range(2, 12):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lab = km.fit_predict(X)
        sil = silhouette_score(X, lab, sample_size=min(2000, len(X)))
        if sil > best_sil:
            best_sil = sil
            best_k = k
        if k <= 8 or k == best_k:
            print(f"    k={k:2d}: silhouette={sil:.4f}{'  ◀ BEST' if k == best_k else ''}")

    print(f"\n  Best k={best_k} on full annotation set")

    # Inter-call intervals for sequence analysis
    session_transitions = 0
    session_count = 0
    ici_values = []

    for fname, calls in by_file.items():
        if len(calls) < 2:
            continue
        sorted_calls = sorted(calls, key=lambda c: c['begin'])
        session_count += 1
        for j in range(len(sorted_calls) - 1):
            ici = sorted_calls[j + 1]['begin'] - sorted_calls[j]['end']
            if 0 < ici < 30:  # reasonable ICI range
                ici_values.append(ici)
                session_transitions += 1

    ici_arr = np.array(ici_values) if ici_values else np.array([0])
    print(f"\n  Sequence statistics:")
    print(f"    Sessions with 2+ calls: {session_count}")
    print(f"    Total transitions: {session_transitions}")
    print(f"    Inter-call interval: mean={ici_arr.mean():.2f}s, median={np.median(ici_arr):.2f}s, std={ici_arr.std():.2f}s")
    print()

    return best_k


# ─────────────────────────────────────────────────────────────────────
# STEP 4: Three-ecotype comparison
# ─────────────────────────────────────────────────────────────────────

def step4_ecotype_comparison():
    print("=" * 70)
    print("  STEP 4: Three-ecotype comparison (SRKW vs Bigg's vs Offshore)")
    print("=" * 70)
    print()

    with open("data/dclde/Annotations.csv", 'r') as f:
        rows = list(csv.DictReader(f))

    kw_calls = [r for r in rows if r['KW'] == '1' and r['AnnotationLevel'] == 'Call']

    ecotypes = {}
    for r in kw_calls:
        eco = r['Ecotype']
        if eco in ('SRKW', 'TKW', 'OKW', 'NRKW', 'SAR'):
            if eco not in ecotypes:
                ecotypes[eco] = []
            try:
                low = float(r['LowFreqHz']) if r['LowFreqHz'] != 'NA' else None
                high = float(r['HighFreqHz']) if r['HighFreqHz'] != 'NA' else None
                dur = float(r['FileEndSec']) - float(r['FileBeginSec'])
                if low and high and low > 0 and high > 0 and dur > 0:
                    ecotypes[eco].append({
                        'low': low, 'high': high, 'dur': dur,
                        'center': (low + high) / 2,
                        'bw': high - low,
                        'dataset': r['Dataset'],
                    })
            except (ValueError, KeyError):
                pass

    print(f"  Ecotype call counts (with frequency data):")
    for eco in sorted(ecotypes.keys()):
        print(f"    {eco:6s}: {len(ecotypes[eco]):>6d} calls")

    # Compare acoustic properties across ecotypes
    print(f"\n  Acoustic properties by ecotype:")
    print(f"  {'Ecotype':8s} {'Center Hz':>10s} {'Bandwidth':>10s} {'Duration':>10s}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    eco_features = {}
    for eco in sorted(ecotypes.keys()):
        calls = ecotypes[eco]
        if len(calls) < 10:
            continue
        centers = np.array([c['center'] for c in calls])
        bws = np.array([c['bw'] for c in calls])
        durs = np.array([c['dur'] for c in calls])
        print(f"  {eco:8s} {centers.mean():>8.0f}±{centers.std():>4.0f} {bws.mean():>8.0f}±{bws.std():>4.0f} {durs.mean():>8.2f}±{durs.std():.2f}")

        eco_features[eco] = np.column_stack([
            centers / 10000,
            bws / 10000,
            durs / durs.max(),
        ])

    # Cluster each ecotype separately
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    print(f"\n  Optimal cluster count per ecotype:")
    eco_best_k = {}
    for eco in sorted(eco_features.keys()):
        X = eco_features[eco]
        if len(X) < 30:
            continue
        best_k, best_sil = 2, -1
        for k in range(2, min(15, len(X) // 5)):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lab = km.fit_predict(X)
            sil = silhouette_score(X, lab, sample_size=min(1000, len(X)))
            if sil > best_sil:
                best_sil = sil
                best_k = k
        eco_best_k[eco] = best_k
        print(f"    {eco:6s}: k={best_k} (silhouette={best_sil:.4f}, n={len(X)})")

    # KEY FINDING: Does communication complexity correlate with social complexity?
    print(f"\n  ═══ KEY FINDING: Communication complexity vs social structure ═══")
    print()
    social_complexity = {
        'SRKW': 'Large matrilineal pods (J/K/L), complex social bonds',
        'NRKW': 'Large matrilineal pods, similar to SRKW',
        'SAR': 'Southern Alaska residents, large pods',
        'TKW': 'Small transient groups (2-6), loose social bonds',
        'OKW': 'Poorly known, deep-water, rare encounters',
    }
    for eco in sorted(eco_best_k.keys()):
        desc = social_complexity.get(eco, 'Unknown')
        print(f"    {eco:6s}: {eco_best_k[eco]} acoustic types — {desc}")

    return ecotypes, eco_features, eco_best_k


# ─────────────────────────────────────────────────────────────────────
# STEP 5: Cross-ecotype Procrustes alignment
# ─────────────────────────────────────────────────────────────────────

def step5_cross_ecotype_alignment(eco_features, eco_best_k):
    print("\n" + "=" * 70)
    print("  STEP 5: Cross-ecotype Procrustes alignment")
    print("=" * 70)
    print()

    from sklearn.cluster import KMeans
    from scipy.linalg import orthogonal_procrustes

    # Build cluster centroids for each ecotype
    eco_centroids = {}
    for eco, X in eco_features.items():
        k = eco_best_k.get(eco)
        if k is None or len(X) < 30:
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        eco_centroids[eco] = km.cluster_centers_

    # Pairwise alignment between ecotypes
    eco_list = sorted(eco_centroids.keys())
    print(f"  Ecotypes with centroids: {', '.join(eco_list)}")
    print()

    for i, e1 in enumerate(eco_list):
        for e2 in eco_list[i+1:]:
            c1 = eco_centroids[e1]
            c2 = eco_centroids[e2]

            # Pad to same number of centroids (use smaller set)
            n = min(len(c1), len(c2))
            if n < 2:
                continue

            # Find best matching between centroids (greedy by distance)
            # Compute distance matrix
            dists = np.zeros((len(c1), len(c2)))
            for a in range(len(c1)):
                for b in range(len(c2)):
                    dists[a, b] = np.linalg.norm(c1[a] - c2[b])

            # Greedy matching
            matched_1 = []
            matched_2 = []
            used_1 = set()
            used_2 = set()
            flat = [(dists[a, b], a, b) for a in range(len(c1)) for b in range(len(c2))]
            flat.sort()
            for d, a, b in flat:
                if a in used_1 or b in used_2:
                    continue
                matched_1.append(a)
                matched_2.append(b)
                used_1.add(a)
                used_2.add(b)
                if len(matched_1) >= n:
                    break

            S = c1[matched_1]
            T = c2[matched_2]

            try:
                R, scale = orthogonal_procrustes(S, T)
                aligned = S @ R
                disparity = float(np.linalg.norm(aligned - T) / (np.linalg.norm(T) + 1e-12))
            except:
                disparity = float('inf')

            # Mean distance between matched centroids
            mean_dist = np.mean([np.linalg.norm(c1[matched_1[j]] - c2[matched_2[j]])
                                 for j in range(n)])

            print(f"  {e1} ↔ {e2}:")
            print(f"    Disparity: {disparity:.4f}")
            print(f"    Mean centroid distance: {mean_dist:.4f}")
            print(f"    Matched centroids: {n}")
            if disparity < 0.3:
                print(f"    → CLOSE: Similar acoustic structure")
            elif disparity < 0.6:
                print(f"    → MODERATE: Overlapping but distinct")
            else:
                print(f"    → DISTANT: Very different acoustic structure")
            print()


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "▓" * 70)
    print("▓  ORCA-ENGINE: COMPLETE SRKW ANALYSIS PIPELINE")
    print("▓  Five steps from raw data to cross-ecotype alignment")
    print("▓" * 70)
    print()

    # Step 1
    labels, normed, metadata, km = step1_label_clusters()

    # Step 2
    step2_sequence_analysis(labels, metadata)

    # Step 3
    full_best_k = step3_scale_up()

    # Step 4
    ecotypes, eco_features, eco_best_k = step4_ecotype_comparison()

    # Step 5
    step5_cross_ecotype_alignment(eco_features, eco_best_k)

    # Final summary
    print("▓" * 70)
    print("▓  COMPLETE ANALYSIS SUMMARY")
    print("▓" * 70)
    print()
    print(f"  Step 1: 3 acoustic clusters mapped to Ford call types")
    print(f"  Step 2: Transition syntax analysis on temporal sequences")
    print(f"  Step 3: Full DCLDE statistics (14,240 SRKW calls, best k={full_best_k})")
    print(f"  Step 4: {len(eco_best_k)} ecotypes compared:")
    for eco, k in sorted(eco_best_k.items()):
        print(f"          {eco}: k={k}")
    print(f"  Step 5: Cross-ecotype Procrustes alignment complete")
    print()


if __name__ == "__main__":
    main()
