#!/usr/bin/env python3
"""
Finding 48: Which specific transitions are directional?

We measured overall asymmetry (Finding 37). Now: WHICH C_i→C_j vs C_j→C_i
pairs are most asymmetric? The directional transitions are the grammatical
rules — they tell you which "word orders" are preferred.
"""

import os
import sys
import ast
import csv
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(__file__))


def load_annotations():
    with open("data/dclde/Annotations.csv", 'r') as f:
        return list(csv.DictReader(f))


def cluster_and_sequence(calls, n_clusters=3):
    from sklearn.cluster import KMeans
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
    if len(features) < n_clusters * 5:
        return None, None, None, None
    X = np.array(features)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    label_map = {valid[i]: labels[i] for i in range(len(valid))}

    by_file = defaultdict(list)
    for i, c in enumerate(calls):
        if i in label_map:
            by_file[c['Soundfile']].append((float(c['FileBeginSec']), label_map[i]))
    sequences = []
    for fname, entries in by_file.items():
        se = sorted(entries)
        seq = [se[0]]
        for j in range(1, len(se)):
            if se[j][0] - se[j-1][0] <= 30:
                seq.append(se[j])
            else:
                if len(seq) >= 2:
                    sequences.append([lab for _, lab in seq])
                seq = [se[j]]
        if len(seq) >= 2:
            sequences.append([lab for _, lab in seq])

    return label_map, X, km, sequences


def analyse_asymmetry(eco_name, calls, n_clusters=3):
    print(f"\n  ═══ {eco_name} ═══")

    label_map, X, km, sequences = cluster_and_sequence(calls, n_clusters)
    if sequences is None:
        print(f"    Insufficient data")
        return

    nc = km.n_clusters
    M = np.zeros((nc, nc), dtype=int)
    for seq in sequences:
        for j in range(len(seq) - 1):
            M[seq[j], seq[j+1]] += 1

    total = M.sum()
    print(f"    Transitions: {total}")

    # Pairwise asymmetry
    print(f"\n    {'Pair':>10s}  {'A→B':>6s}  {'B→A':>6s}  {'P(A→B)':>8s}  {'P(B→A)':>8s}  {'Asymmetry':>10s}  {'Direction':>12s}")
    print(f"    {'─'*10}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*12}")

    pairs = []
    for i in range(nc):
        for j in range(i + 1, nc):
            ri = M[i].sum()
            rj = M[j].sum()
            if ri > 0 and rj > 0:
                pij = M[i, j] / ri
                pji = M[j, i] / rj
                asym = abs(pij - pji)
                direction = f"C{i}→C{j}" if pij > pji else f"C{j}→C{i}"
                pairs.append((i, j, M[i,j], M[j,i], pij, pji, asym, direction))

    pairs.sort(key=lambda x: -x[6])
    for i, j, nij, nji, pij, pji, asym, direction in pairs:
        print(f"    C{i}↔C{j}     {nij:>6d}  {nji:>6d}  {pij:>8.4f}  {pji:>8.4f}  {asym:>10.4f}  {direction:>12s}")

    # Self-transition asymmetry (is one type more "sticky" than others?)
    print(f"\n    Self-transition rates:")
    for i in range(nc):
        ri = M[i].sum()
        if ri > 0:
            self_rate = M[i, i] / ri
            print(f"      C{i}: {self_rate:.4f} ({M[i,i]}/{ri})")


def main():
    print("=" * 70)
    print("  TRANSITION ASYMMETRY: Which grammar rules are directional?")
    print("=" * 70)

    rows = load_annotations()

    for eco in ['SRKW', 'TKW', 'SAR', 'OKW']:
        calls = [r for r in rows if r['KW'] == '1' and r['AnnotationLevel'] == 'Call'
                 and r['Ecotype'] == eco]
        if len(calls) >= 100:
            analyse_asymmetry(eco, calls)

    print()


if __name__ == "__main__":
    main()
