#!/usr/bin/env python3
"""
Finding 50: Do different Bigg's transient groups have different grammars?

The Scripps Cape Elizabeth recordings label distinct transient encounters:
"Transients", "BCTransients" (British Columbia), "CAtransients" (California).
These are different hunting groups, possibly targeting different prey.

If their syntax differs, communication strategy varies WITHIN an ecotype.
"""

import os
import sys
import ast
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(__file__))


def main():
    print("=" * 70)
    print("  TKW SUB-GROUPS: Different hunters, different grammars?")
    print("=" * 70)
    print()

    data = np.load("data/tkw_features.npz", allow_pickle=True)
    features = data["features"].copy()
    metadata = [ast.literal_eval(m) for m in data["metadata"]]

    # Classify by filename pattern
    groups = defaultdict(list)
    for i, m in enumerate(metadata):
        fname = m['filename']
        if 'BCtransient' in fname or 'BCTransient' in fname:
            groups['BC_Transients'].append(i)
        elif 'CAtransient' in fname or 'CATransient' in fname:
            groups['CA_Transients'].append(i)
        elif 'Transient' in fname or 'transient' in fname:
            groups['Generic_Transients'].append(i)
        else:
            groups['Other'].append(i)

    print(f"  Sub-groups by filename pattern:")
    for name, indices in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"    {name}: {len(indices)} calls")

    # Per-group normalisation
    for name, indices in groups.items():
        sf = features[indices]
        mean, std = sf.mean(axis=0), sf.std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        for i in indices:
            features[i] = (features[i] - mean) / std

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normed = features / np.where(norms > 0, norms, 1.0)

    # Per-group analysis
    from sklearn.cluster import KMeans
    from scipy.stats import mannwhitneyu
    import math

    for name, indices in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(indices) < 50:
            continue

        print(f"\n  ═══ {name} ({len(indices)} calls) ═══")

        g_normed = normed[indices]
        g_meta = [metadata[i] for i in indices]

        # Topology → syntax
        by_file = defaultdict(list)
        for j, idx in enumerate(indices):
            by_file[g_meta[j]['filename']].append((g_meta[j].get('duration', 0), j))

        adj_sims, rand_sims = [], []
        rng = np.random.RandomState(42)
        for key, entries in by_file.items():
            entries.sort()
            for k in range(len(entries) - 1):
                adj_sims.append(float(np.dot(g_normed[entries[k][1]], g_normed[entries[k+1][1]])))
        for _ in range(len(adj_sims)):
            i1, i2 = rng.choice(len(g_normed), 2, replace=False)
            rand_sims.append(float(np.dot(g_normed[i1], g_normed[i2])))

        if len(adj_sims) < 20:
            print(f"    Insufficient transitions ({len(adj_sims)})")
            continue

        adj, rand = np.array(adj_sims), np.array(rand_sims)
        diff = adj.mean() - rand.mean()
        d = diff / np.sqrt((adj.std()**2 + rand.std()**2) / 2)
        U, p = mannwhitneyu(adj, rand, alternative='greater')

        print(f"    Topology→syntax: d = {d:.4f}, p = {p:.4e}")

        # Self-transition rate
        sequences = []
        for key, entries in by_file.items():
            entries.sort()
            seq = [entries[0]]
            for k in range(1, len(entries)):
                if entries[k][0] - entries[k-1][0] <= 30:
                    seq.append(entries[k])
                else:
                    if len(seq) >= 2:
                        sequences.append(seq)
                    seq = [entries[k]]
            if len(seq) >= 2:
                sequences.append(seq)

        # Cluster
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = km.fit_predict(g_normed)

        total_trans = 0
        self_trans = 0
        for seq in sequences:
            for k in range(len(seq) - 1):
                l1 = labels[seq[k][1]]
                l2 = labels[seq[k+1][1]]
                total_trans += 1
                if l1 == l2:
                    self_trans += 1

        if total_trans > 0:
            sr = self_trans / total_trans
            print(f"    Self-transition rate: {sr:.4f} ({total_trans} transitions)")

        # Spectral centroid
        centroids = [features[i][0] * 11025 for i in indices]
        durs = [g_meta[j].get('duration', 0) for j in range(len(indices))]
        print(f"    Mean centroid: {np.mean(centroids):.0f} Hz")
        print(f"    Mean duration: {np.mean(durs):.2f}s")

    # Cross-group comparison
    group_names = [n for n, idx in groups.items() if len(idx) >= 50]
    if len(group_names) >= 2:
        print(f"\n  ═══ CROSS-GROUP ACOUSTIC DISTANCE ═══")
        for i, g1 in enumerate(group_names):
            for g2 in group_names[i+1:]:
                c1 = normed[groups[g1]].mean(axis=0)
                c2 = normed[groups[g2]].mean(axis=0)
                c1 /= (np.linalg.norm(c1) + 1e-12)
                c2 /= (np.linalg.norm(c2) + 1e-12)
                sim = float(np.dot(c1, c2))
                print(f"    {g1} ↔ {g2}: similarity = {sim:.4f}")

    print()


if __name__ == "__main__":
    main()
