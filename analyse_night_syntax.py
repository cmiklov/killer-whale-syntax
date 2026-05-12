#!/usr/bin/env python3
"""
Finding 51: Does the grammar change at night?

Finding 16 showed diel acoustic patterns. But does the SYNTAX change?
Different transition matrix? Different Markov order? Different Boltzmann T?
"""

import os
import sys
import csv
import math
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(__file__))


def main():
    print("=" * 70)
    print("  NIGHT SYNTAX: Does the grammar change after dark?")
    print("=" * 70)
    print()

    with open("data/dclde/Annotations.csv", 'r') as f:
        rows = list(csv.DictReader(f))

    srkw = [r for r in rows if r['KW'] == '1' and r['AnnotationLevel'] == 'Call'
            and r['Ecotype'] == 'SRKW']

    # Classify calls by time of day (Pacific ≈ UTC-7 in summer)
    from sklearn.cluster import KMeans

    day_calls = []  # 14:00-06:00 UTC = ~07:00-23:00 Pacific (day)
    night_calls = []  # 07:00-13:00 UTC = ~00:00-06:00 Pacific (night)

    for r in srkw:
        try:
            hour = int(r['UTC'][11:13])
            if 14 <= hour <= 23 or 0 <= hour <= 6:
                day_calls.append(r)
            else:
                night_calls.append(r)
        except:
            pass

    print(f"  Day calls (Pacific ~07:00-23:00):   {len(day_calls)}")
    print(f"  Night calls (Pacific ~00:00-06:00): {len(night_calls)}")

    # Cluster each period
    def cluster_and_analyse(calls, label):
        features, valid = [], []
        for i, c in enumerate(calls):
            try:
                lo = float(c['LowFreqHz']) if c['LowFreqHz'] != 'NA' else None
                hi = float(c['HighFreqHz']) if c['HighFreqHz'] != 'NA' else None
                dur = float(c['FileEndSec']) - float(c['FileBeginSec'])
                if lo and hi and lo > 0 and hi > 0 and dur > 0.05:
                    features.append([(lo+hi)/2/10000, (hi-lo)/10000, min(dur,10)/10])
                    valid.append(i)
            except:
                pass

        if len(features) < 30:
            print(f"  {label}: insufficient data ({len(features)})")
            return

        X = np.array(features)
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        label_map = {valid[j]: labels[j] for j in range(len(valid))}

        # Build sequences
        by_file = defaultdict(list)
        for j, c in enumerate(calls):
            if j in label_map:
                try:
                    by_file[c['Soundfile']].append((float(c['FileBeginSec']), label_map[j]))
                except:
                    pass

        sequences = []
        for fname, entries in by_file.items():
            se = sorted(entries)
            seq = [se[0]]
            for k in range(1, len(se)):
                if se[k][0] - se[k-1][0] <= 30:
                    seq.append(se[k])
                else:
                    if len(seq) >= 2:
                        sequences.append([lab for _, lab in seq])
                    seq = [se[k]]
            if len(seq) >= 2:
                sequences.append([lab for _, lab in seq])

        # Transition matrix
        nc = 3
        M = np.zeros((nc, nc), dtype=int)
        for seq in sequences:
            for k in range(len(seq) - 1):
                M[seq[k], seq[k+1]] += 1

        total = M.sum()
        if total < 20:
            print(f"  {label}: insufficient transitions ({total})")
            return

        # Stats
        cluster_dist = Counter(labels)
        total_calls = sum(cluster_dist.values())
        H0 = -sum((c/total_calls) * math.log2(c/total_calls) for c in cluster_dist.values() if c > 0)

        H1 = 0
        for i in range(nc):
            rt = M[i].sum()
            if rt == 0:
                continue
            for j in range(nc):
                if M[i, j] > 0:
                    p = M[i, j] / rt
                    H1 -= (rt / total) * p * math.log2(p)

        self_rate = sum(M[i, i] for i in range(nc)) / total
        MI = H0 - H1
        MI_H = MI / H0 if H0 > 0 else 0

        print(f"\n  ═══ {label} ═══")
        print(f"    Calls (with freq): {len(features)}")
        print(f"    Sequences: {len(sequences)}, transitions: {total}")
        print(f"    Cluster dist: {dict(cluster_dist)}")
        print(f"    H₀: {H0:.4f} bits")
        print(f"    H₁: {H1:.4f} bits")
        print(f"    MI: {MI:.4f} bits")
        print(f"    MI/H: {MI_H:.1%}")
        print(f"    Self-transition: {self_rate:.4f}")

        # Transition matrix
        print(f"    Transitions:")
        print(f"              " + "  ".join(f"→C{j}" for j in range(nc)))
        for i in range(nc):
            rt = M[i].sum()
            if rt > 0:
                probs = M[i] / rt
                row = "  ".join(f"{probs[j]:.3f}" for j in range(nc))
                print(f"        C{i} →  {row}")

        return {
            'H0': H0, 'H1': H1, 'MI': MI, 'MI_H': MI_H,
            'self_rate': self_rate, 'n_trans': total, 'dist': dict(cluster_dist),
        }

    day_stats = cluster_and_analyse(day_calls, "DAY (Pacific)")
    night_stats = cluster_and_analyse(night_calls, "NIGHT (Pacific)")

    if day_stats and night_stats:
        print(f"\n  ═══ COMPARISON ═══")
        print(f"  {'Metric':>20s}  {'Day':>10s}  {'Night':>10s}  {'Diff':>10s}")
        print(f"  {'─'*20}  {'─'*10}  {'─'*10}  {'─'*10}")
        for key, label in [('H0', 'H₀ (diversity)'), ('H1', 'H₁ (conditional)'),
                           ('MI', 'MI'), ('MI_H', 'MI/H'),
                           ('self_rate', 'Self-transition')]:
            d = day_stats[key]
            n = night_stats[key]
            diff = n - d
            fmt = ".4f" if key != 'MI_H' else ".1%"
            if key == 'MI_H':
                print(f"  {label:>20s}  {d:>10.1%}  {n:>10.1%}  {diff:>+10.1%}")
            else:
                print(f"  {label:>20s}  {d:>10.4f}  {n:>10.4f}  {diff:>+10.4f}")

        # Interpretation
        print()
        if day_stats['MI_H'] > night_stats['MI_H'] + 0.05:
            print(f"  *** Day grammar has MORE sequential structure")
        elif night_stats['MI_H'] > day_stats['MI_H'] + 0.05:
            print(f"  *** Night grammar has MORE sequential structure")
        else:
            print(f"  Day and night grammar are SIMILAR in sequential structure")

        if day_stats['self_rate'] > night_stats['self_rate'] + 0.05:
            print(f"  *** Day calls are MORE repetitive (longer bouts)")
        elif night_stats['self_rate'] > day_stats['self_rate'] + 0.05:
            print(f"  *** Night calls are MORE repetitive (longer bouts)")

    print()


if __name__ == "__main__":
    main()
