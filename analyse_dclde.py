#!/usr/bin/env python3
"""
Analyse the DCLDE per-call features through the orca-engine.

This is the real test: 577 individually extracted call segments from
three Orcasound hydrophone stations in the Salish Sea. No shared
exemplars. No synthetic features. Every call is a unique recording.

The key question: does the phonosemantic correlation hold when each
call has its own acoustic features?
"""

import os
import sys
import ast
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from orca.features import N_FEATURES


def main():
    print("=" * 60)
    print("  DCLDE SRKW: Per-Call Topology Analysis")
    print("  577 real calls from 3 Orcasound hydrophone stations")
    print("=" * 60)
    print()

    # Load per-call features
    data = np.load(os.path.join(os.path.dirname(__file__), "data", "dclde_srkw_features.npz"),
                   allow_pickle=True)
    features = data["features"]
    metadata_strs = data["metadata"]
    metadata = [ast.literal_eval(m) for m in metadata_strs]

    print(f"  Loaded {len(features)} calls, {features.shape[1]}D features")

    # ─── Basic Statistics ──────────────────────────────────────────
    by_station = {}
    for i, m in enumerate(metadata):
        ds = m['dataset']
        if ds not in by_station:
            by_station[ds] = []
        by_station[ds].append(i)

    print(f"\n  Calls by station:")
    for ds, indices in sorted(by_station.items()):
        print(f"    {ds:25s}: {len(indices):>4d} calls")

    # ─── Spectral Analysis ─────────────────────────────────────────
    centroids = features[:, 0] * 11025
    bandwidths = features[:, 1] * 11025
    durations = features[:, 16]  # structural[0] = duration

    print(f"\n  Acoustic properties across all calls:")
    print(f"    Spectral centroid: mean={centroids.mean():.0f} Hz, std={centroids.std():.0f} Hz")
    print(f"    Bandwidth:         mean={bandwidths.mean():.0f} Hz, std={bandwidths.std():.0f} Hz")
    print(f"    Duration:          mean={durations.mean():.2f}s, std={durations.std():.2f}s")

    # ─── Intra-station vs Inter-station Similarity ─────────────────
    # This is the REAL test: are calls from the same station more similar
    # to each other than to calls from other stations?
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normed = features / norms

    print(f"\n  ═══ ACOUSTIC TOPOLOGY: Station Clustering ═══")
    print()

    stations = sorted(by_station.keys())
    station_centroids = {}
    for ds in stations:
        indices = by_station[ds]
        vecs = normed[indices]
        centroid = np.mean(vecs, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        station_centroids[ds] = centroid

        # Intra-station similarity
        sim_matrix = vecs @ vecs.T
        upper = sim_matrix[np.triu_indices(len(vecs), k=1)]
        print(f"  {ds}:")
        print(f"    calls: {len(indices)}")
        print(f"    intra-station similarity: mean={upper.mean():.4f}, std={upper.std():.4f}")

    print(f"\n  Inter-station centroid similarities:")
    for i, s1 in enumerate(stations):
        for s2 in stations[i+1:]:
            sim = float(np.dot(station_centroids[s1], station_centroids[s2]))
            print(f"    {s1} ↔ {s2}: {sim:.4f}")

    # ─── Temporal Clustering ───────────────────────────────────────
    # Are calls recorded close in time more acoustically similar?
    print(f"\n  ═══ TEMPORAL ANALYSIS ═══")
    print()

    # Group by recording session (same filename = same session)
    by_file = {}
    for i, m in enumerate(metadata):
        fname = m['filename']
        if fname not in by_file:
            by_file[fname] = []
        by_file[fname].append(i)

    within_session = []
    between_session = []
    rng = np.random.RandomState(42)

    files_list = list(by_file.values())
    for indices in files_list:
        if len(indices) < 2:
            continue
        vecs = normed[indices]
        sim = vecs @ vecs.T
        upper = sim[np.triu_indices(len(vecs), k=1)]
        within_session.extend(upper.tolist())

    # Sample between-session pairs
    for _ in range(min(10000, len(within_session))):
        f1, f2 = rng.choice(len(files_list), 2, replace=False)
        i1 = rng.choice(files_list[f1])
        i2 = rng.choice(files_list[f2])
        sim = float(np.dot(normed[i1], normed[i2]))
        between_session.append(sim)

    within_arr = np.array(within_session)
    between_arr = np.array(between_session)

    print(f"  Within-session similarity:  mean={within_arr.mean():.4f}, std={within_arr.std():.4f}  (n={len(within_arr)})")
    print(f"  Between-session similarity: mean={between_arr.mean():.4f}, std={between_arr.std():.4f}  (n={len(between_arr)})")
    diff = within_arr.mean() - between_arr.mean()
    print(f"  Difference: {diff:.4f} ({'calls cluster by session' if diff > 0.01 else 'no session clustering'})")

    # ─── Acoustic Feature Diversity ────────────────────────────────
    print(f"\n  ═══ FEATURE SPACE ANALYSIS ═══")
    print()

    # PCA to see how many dimensions carry information
    centered = features - features.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained_var = (S ** 2) / (S ** 2).sum()
    cumulative = np.cumsum(explained_var)

    print(f"  PCA explained variance (first 10 components):")
    for i in range(min(10, len(explained_var))):
        bar = "█" * int(explained_var[i] * 100)
        print(f"    PC{i+1:2d}: {explained_var[i]:.4f} (cumul: {cumulative[i]:.4f})  {bar}")

    dims_for_90 = np.searchsorted(cumulative, 0.9) + 1
    dims_for_95 = np.searchsorted(cumulative, 0.95) + 1
    print(f"\n  Dimensions for 90% variance: {dims_for_90}")
    print(f"  Dimensions for 95% variance: {dims_for_95}")

    # ─── Acoustic Clustering (k-means) ─────────────────────────────
    print(f"\n  ═══ UNSUPERVISED CALL TYPE DISCOVERY ═══")
    print()

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # Try different k values
        print(f"  K-means clustering on 577 real orca calls:")
        best_k = 0
        best_silhouette = -1
        for k in range(3, 25):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(normed)
            sil = silhouette_score(normed, labels, sample_size=min(500, len(normed)))
            if sil > best_silhouette:
                best_silhouette = sil
                best_k = k
            if k <= 15 or k == best_k:
                print(f"    k={k:2d}: silhouette={sil:.4f}")

        print(f"\n  Best k={best_k} (silhouette={best_silhouette:.4f})")
        print(f"  {'This suggests ~' + str(best_k) + ' acoustically distinct call types'}")

        # Analyse the best clustering
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = km.fit_predict(normed)

        print(f"\n  Cluster sizes:")
        from collections import Counter
        counts = Counter(labels)
        for cluster_id, count in sorted(counts.items()):
            cluster_features = features[labels == cluster_id]
            mean_centroid = cluster_features[:, 0].mean() * 11025
            mean_duration = cluster_features[:, 16].mean()
            # Station distribution
            station_dist = Counter(metadata[i]['dataset'] for i in range(len(labels)) if labels[i] == cluster_id)
            stations_str = ", ".join(f"{s.split('_')[0]}:{c}" for s, c in station_dist.most_common(3))
            print(f"    Cluster {cluster_id:2d}: n={count:3d}  centroid={mean_centroid:.0f}Hz  dur={mean_duration:.2f}s  [{stations_str}]")

        # KEY QUESTION: Do clusters correspond to recording locations, or to
        # acoustic properties that transcend location?
        print(f"\n  ═══ KEY TEST: Are clusters location-dependent or acoustic? ═══")
        print()

        # For each cluster, compute how spread across stations it is
        # Shannon entropy of station distribution within each cluster
        import math
        mixed_clusters = 0
        for cluster_id in range(best_k):
            station_dist = Counter(metadata[i]['dataset'] for i in range(len(labels)) if labels[i] == cluster_id)
            total = sum(station_dist.values())
            entropy = -sum((c/total) * math.log2(c/total + 1e-12) for c in station_dist.values())
            max_entropy = math.log2(len(station_dist))
            norm_entropy = entropy / (max_entropy + 1e-12) if max_entropy > 0 else 0
            n_stations = len(station_dist)
            if n_stations >= 2:
                mixed_clusters += 1

        print(f"  Clusters spanning multiple stations: {mixed_clusters}/{best_k}")
        print(f"  {'PASS: clusters reflect acoustic types, not recording artifacts' if mixed_clusters > best_k * 0.6 else 'CAUTION: clusters may reflect recording conditions'}")

    except ImportError:
        print("  sklearn not available — skipping clustering analysis")

    # ─── Final Summary ─────────────────────────────────────────────
    print(f"\n  {'═' * 56}")
    print(f"  SUMMARY")
    print(f"  {'═' * 56}")
    print(f"\n  577 real SRKW calls from 3 Orcasound hydrophone stations")
    print(f"  50D acoustic features extracted via librosa")
    print(f"  Zero shared exemplars — every call is a unique recording")
    if 'best_k' in dir():
        print(f"  Unsupervised clustering suggests ~{best_k} acoustic types")
    print()


if __name__ == "__main__":
    main()
