#!/usr/bin/env python3
"""
Finding 53: Session onset analysis — does the first call predict everything?

Is the opening call always the same type? Does it predict session length,
diversity, or coherence? The "first word" of an orca conversation.
"""

import os, sys, ast, numpy as np
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=" * 70)
    print("  SESSION ONSET: Does the first call predict the conversation?")
    print("=" * 70)
    print()

    data = np.load("data/haro_srkw_features.npz", allow_pickle=True)
    features = data["features"].copy()
    metadata = [ast.literal_eval(m) for m in data["metadata"]]

    # Normalise
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

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = km.fit_predict(normed)

    # Build sessions
    by_file = defaultdict(list)
    for i, m in enumerate(metadata):
        by_file[(m['filename'], m['station'])].append(
            (float(m.get('duration', 0)), i))

    sessions = []
    for key, entries in by_file.items():
        entries.sort()
        if len(entries) >= 10:
            sessions.append([idx for _, idx in entries])

    print(f"  Sessions (10+ calls): {len(sessions)}")

    # Analyse onset
    onset_types = Counter()
    onset_length = defaultdict(list)   # onset type → session lengths
    onset_diversity = defaultdict(list)  # onset type → session diversities

    for seq in sessions:
        onset = labels[seq[0]]
        onset_types[onset] += 1
        onset_length[onset].append(len(seq))

        # Diversity: number of unique types / total types possible
        type_counts = Counter(labels[i] for i in seq)
        diversity = len(type_counts) / 2  # k=2
        onset_diversity[onset].append(diversity)

    print(f"\n  Onset call type distribution:")
    total = sum(onset_types.values())
    for t, count in onset_types.most_common():
        pct = count / total * 100
        mean_len = np.mean(onset_length[t])
        mean_div = np.mean(onset_diversity[t])
        print(f"    C{t}: {count} sessions ({pct:.1f}%), mean length={mean_len:.0f}, mean diversity={mean_div:.2f}")

    # Does onset predict session length?
    if len(onset_length) >= 2:
        from scipy.stats import mannwhitneyu
        types = sorted(onset_length.keys())
        if len(types) >= 2:
            a = np.array(onset_length[types[0]])
            b = np.array(onset_length[types[1]])
            if len(a) >= 5 and len(b) >= 5:
                U, p = mannwhitneyu(a, b, alternative='two-sided')
                print(f"\n  Does onset type predict session length?")
                print(f"    C{types[0]} sessions: mean={a.mean():.0f} calls")
                print(f"    C{types[1]} sessions: mean={b.mean():.0f} calls")
                print(f"    Mann-Whitney: p={p:.4f}")
                if p < 0.05:
                    print(f"    *** YES — onset type predicts session length")
                else:
                    print(f"    No — onset type doesn't predict session length")

            # Does onset predict diversity?
            a_div = np.array(onset_diversity[types[0]])
            b_div = np.array(onset_diversity[types[1]])
            if len(a_div) >= 5 and len(b_div) >= 5:
                U, p = mannwhitneyu(a_div, b_div, alternative='two-sided')
                print(f"\n  Does onset type predict session diversity?")
                print(f"    C{types[0]} sessions: mean diversity={a_div.mean():.3f}")
                print(f"    C{types[1]} sessions: mean diversity={b_div.mean():.3f}")
                print(f"    Mann-Whitney: p={p:.4f}")
                if p < 0.05:
                    print(f"    *** YES — onset type predicts session diversity")

    # Onset vs ending — are they the same type?
    print(f"\n  Onset vs ending type:")
    same = 0
    diff = 0
    for seq in sessions:
        if labels[seq[0]] == labels[seq[-1]]:
            same += 1
        else:
            diff += 1
    print(f"    Same type: {same}/{len(sessions)} ({same/len(sessions)*100:.1f}%)")
    print(f"    Different: {diff}/{len(sessions)} ({diff/len(sessions)*100:.1f}%)")

    # Acoustic similarity: first call vs last call
    first_last_sims = []
    first_mid_sims = []
    for seq in sessions:
        if len(seq) >= 10:
            first_last_sims.append(float(np.dot(normed[seq[0]], normed[seq[-1]])))
            mid = len(seq) // 2
            first_mid_sims.append(float(np.dot(normed[seq[0]], normed[seq[mid]])))

    if first_last_sims:
        fl = np.array(first_last_sims)
        fm = np.array(first_mid_sims)
        print(f"\n  Acoustic similarity:")
        print(f"    First ↔ Last:   mean={fl.mean():.4f}")
        print(f"    First ↔ Middle: mean={fm.mean():.4f}")
        if fl.mean() > fm.mean() + 0.02:
            print(f"    *** Sessions RETURN to their acoustic starting point")
            print(f"    *** Circular trajectory: the conversation comes home")

    print()


if __name__ == "__main__":
    main()
