#!/usr/bin/env python3
"""
Finding 52: Cross-station mutual information — distributed perception test.

Does what North station hears predict what South station hears a moment later?
If yes, information is flowing through the pod — the distributed perception
model has a measurable signature.
"""

import os, sys, ast, numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=" * 70)
    print("  CROSS-STATION MI: Does one station predict the other?")
    print("=" * 70)
    print()

    data = np.load("data/haro_srkw_features.npz", allow_pickle=True)
    features = data["features"].copy()
    metadata = [ast.literal_eval(m) for m in data["metadata"]]

    # Normalise per station
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
    normed = features / np.where(norms > 0, norms, 1.0)

    # Cluster
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = km.fit_predict(normed)

    # Find paired timestamps (same filename base = same time window)
    import re
    by_timestamp = defaultdict(lambda: {'north': [], 'south': []})
    for i, m in enumerate(metadata):
        match = re.search(r'(\d{8}T\d{6}Z)', m['filename'])
        if match:
            ts = match.group(1)
            by_timestamp[ts][m['station']].append((float(m.get('duration', 0)), labels[i], i))

    paired = {ts: d for ts, d in by_timestamp.items() if d['north'] and d['south']}
    print(f"  Paired timestamps: {len(paired)}")

    if len(paired) < 3:
        print("  Insufficient paired data")
        return

    # For each paired timestamp, build interleaved sequence
    # Then compute: does North's call type predict South's next call type?
    from collections import Counter
    import math

    north_predicts_south = Counter()  # (north_label, south_label_after)
    south_predicts_north = Counter()
    north_marginal = Counter()
    south_marginal = Counter()

    total_pairs = 0

    for ts, data_ts in paired.items():
        north = sorted(data_ts['north'])
        south = sorted(data_ts['south'])

        # For each north call, find next south call (within 2s)
        for n_time, n_label, n_idx in north:
            for s_time, s_label, s_idx in south:
                delta = s_time - n_time
                if 0.01 < delta < 2.0:
                    north_predicts_south[(n_label, s_label)] += 1
                    north_marginal[n_label] += 1
                    south_marginal[s_label] += 1
                    total_pairs += 1
                    break

    print(f"  North→South pairs (within 2s): {total_pairs}")

    if total_pairs < 50:
        print("  Insufficient cross-station pairs")
        return

    # Compute MI(North, South_after)
    H_north = -sum((c/total_pairs) * math.log2(c/total_pairs)
                    for c in north_marginal.values() if c > 0)
    H_south = -sum((c/total_pairs) * math.log2(c/total_pairs)
                    for c in south_marginal.values() if c > 0)

    H_joint = -sum((c/total_pairs) * math.log2(c/total_pairs)
                    for c in north_predicts_south.values() if c > 0)

    MI = H_north + H_south - H_joint

    print(f"\n  H(North call type):  {H_north:.4f} bits")
    print(f"  H(South next type):  {H_south:.4f} bits")
    print(f"  H(North, South):     {H_joint:.4f} bits")
    print(f"  MI(North → South):   {MI:.4f} bits")

    if H_north > 0:
        print(f"  MI / H(North):       {MI/H_north:.1%}")

    if MI > 0.01:
        print(f"\n  *** What one station hears PREDICTS what the other hears next")
        print(f"  *** Information flows between positions in the pod")
        print(f"  *** Consistent with distributed perception model")
    elif MI > 0.001:
        print(f"\n  * Weak cross-station predictability")
    else:
        print(f"\n  No cross-station predictability")

    # Joint distribution
    print(f"\n  Joint distribution P(North, South_after):")
    nc = 2
    for n in range(nc):
        for s in range(nc):
            count = north_predicts_south.get((n, s), 0)
            p = count / total_pairs if total_pairs > 0 else 0
            print(f"    North=C{n}, South=C{s}: {count:>5d} ({p:.3f})")

    print()


if __name__ == "__main__":
    main()
