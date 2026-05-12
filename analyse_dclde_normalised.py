#!/usr/bin/env python3
"""
Re-analyse DCLDE data with per-station normalisation.

The raw analysis showed clusters correlated with recording station.
By subtracting the per-station mean (removing the hydrophone's spectral
signature), we isolate call-level acoustic structure from recording artifacts.

This is standard practice in bioacoustics: different hydrophones have
different frequency responses, and environmental noise varies by location.
"""

import os
import sys
import ast
import math
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))


def main():
    print("=" * 60)
    print("  DCLDE SRKW: Station-Normalised Topology Analysis")
    print("  Removing hydrophone signatures to isolate call structure")
    print("=" * 60)
    print()

    data = np.load(os.path.join(os.path.dirname(__file__), "data", "dclde_srkw_features.npz"),
                   allow_pickle=True)
    features = data["features"].copy()
    metadata = [ast.literal_eval(m) for m in data["metadata"]]

    n_calls = len(features)
    print(f"  {n_calls} calls loaded")

    # ─── Per-station normalisation ──────────────────────────────────
    by_station = {}
    for i, m in enumerate(metadata):
        ds = m['dataset']
        if ds not in by_station:
            by_station[ds] = []
        by_station[ds].append(i)

    print(f"\n  Normalising per station:")
    for ds, indices in sorted(by_station.items()):
        station_features = features[indices]
        station_mean = station_features.mean(axis=0)
        station_std = station_features.std(axis=0)
        station_std = np.where(station_std > 1e-8, station_std, 1.0)

        # Z-score normalisation within each station
        for i in indices:
            features[i] = (features[i] - station_mean) / station_std

        print(f"    {ds:25s}: {len(indices)} calls normalised")

    # Re-normalise to unit sphere
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normed = features / norms

    # ─── Post-normalisation station clustering ──────────────────────
    print(f"\n  ═══ POST-NORMALISATION: Station Effects Removed? ═══")
    print()

    stations = sorted(by_station.keys())
    station_centroids = {}
    for ds in stations:
        indices = by_station[ds]
        vecs = normed[indices]
        centroid = np.mean(vecs, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        station_centroids[ds] = centroid

        sim_matrix = vecs @ vecs.T
        upper = sim_matrix[np.triu_indices(len(vecs), k=1)]
        print(f"  {ds}:")
        print(f"    intra-station similarity: mean={upper.mean():.4f}, std={upper.std():.4f}")

    print(f"\n  Inter-station centroid similarities:")
    for i, s1 in enumerate(stations):
        for s2 in stations[i+1:]:
            sim = float(np.dot(station_centroids[s1], station_centroids[s2]))
            print(f"    {s1} ↔ {s2}: {sim:.4f}")

    # ─── Temporal analysis post-normalisation ───────────────────────
    by_file = {}
    for i, m in enumerate(metadata):
        fname = m['filename']
        if fname not in by_file:
            by_file[fname] = []
        by_file[fname].append(i)

    within = []
    between = []
    rng = np.random.RandomState(42)
    files_list = list(by_file.values())

    for indices in files_list:
        if len(indices) < 2:
            continue
        vecs = normed[indices]
        sim = vecs @ vecs.T
        upper = sim[np.triu_indices(len(vecs), k=1)]
        within.extend(upper.tolist())

    for _ in range(min(10000, len(within))):
        f1, f2 = rng.choice(len(files_list), 2, replace=False)
        i1 = rng.choice(files_list[f1])
        i2 = rng.choice(files_list[f2])
        between.append(float(np.dot(normed[i1], normed[i2])))

    within_arr = np.array(within)
    between_arr = np.array(between)
    diff = within_arr.mean() - between_arr.mean()

    print(f"\n  Within-session similarity:  mean={within_arr.mean():.4f}")
    print(f"  Between-session similarity: mean={between_arr.mean():.4f}")
    print(f"  Difference: {diff:.4f} (was 0.0411 before normalisation)")

    # ─── Unsupervised clustering post-normalisation ─────────────────
    print(f"\n  ═══ POST-NORMALISATION: Unsupervised Call Type Discovery ═══")
    print()

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_k = 0
    best_sil = -1
    results = []
    for k in range(3, 25):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(normed)
        sil = silhouette_score(normed, labels, sample_size=min(500, len(normed)))
        results.append((k, sil))
        if sil > best_sil:
            best_sil = sil
            best_k = k

    print(f"  K-means silhouette scores:")
    for k, sil in results:
        marker = " ◀ BEST" if k == best_k else ""
        if k <= 15 or k == best_k:
            print(f"    k={k:2d}: {sil:.4f}{marker}")

    print(f"\n  Best k={best_k} (silhouette={best_sil:.4f})")

    # Analyse best clustering
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(normed)

    print(f"\n  Cluster analysis:")
    mixed_clusters = 0
    for cluster_id in range(best_k):
        mask = labels == cluster_id
        count = mask.sum()
        cluster_feat = features[mask]

        # Station distribution
        station_dist = Counter(metadata[i]['dataset']
                               for i in range(n_calls) if labels[i] == cluster_id)
        n_stations = len(station_dist)
        total = sum(station_dist.values())

        # Shannon entropy of station distribution
        entropy = -sum((c/total) * math.log2(c/total + 1e-12)
                       for c in station_dist.values())
        max_entropy = math.log2(n_stations) if n_stations > 1 else 0

        if n_stations >= 2:
            mixed_clusters += 1

        stations_str = ", ".join(f"{s.split('_')[0]}:{c}" for s, c in station_dist.most_common(3))
        print(f"    Cluster {cluster_id:2d}: n={count:3d}  stations={n_stations}  [{stations_str}]")

    print(f"\n  Clusters spanning multiple stations: {mixed_clusters}/{best_k}")
    pct = mixed_clusters / best_k * 100
    print(f"  Station independence: {pct:.0f}%")

    if pct > 70:
        print(f"  ✓ STRONG: Clusters reflect CALL TYPES, not recording artifacts")
    elif pct > 50:
        print(f"  ~ MODERATE: Mix of call structure and recording effects")
    else:
        print(f"  ✗ WEAK: Recording conditions still dominate")

    # ─── PCA post-normalisation ─────────────────────────────────────
    centered = features - features.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    cumulative = np.cumsum(explained)

    dims_90 = np.searchsorted(cumulative, 0.9) + 1
    dims_95 = np.searchsorted(cumulative, 0.95) + 1
    print(f"\n  Post-normalisation dimensionality:")
    print(f"    90% variance in {dims_90} dimensions (was 5)")
    print(f"    95% variance in {dims_95} dimensions (was 8)")

    # ─── Summary ────────────────────────────────────────────────────
    print(f"\n  {'═' * 56}")
    print(f"  SUMMARY")
    print(f"  {'═' * 56}")
    print(f"\n  577 real SRKW calls, station-normalised")
    print(f"  Session clustering reduced from 0.0411 to {diff:.4f}")
    print(f"  Best k={best_k} acoustic clusters, {mixed_clusters}/{best_k} span multiple stations")
    print(f"  Station independence: {pct:.0f}%")
    print()


if __name__ == "__main__":
    main()
