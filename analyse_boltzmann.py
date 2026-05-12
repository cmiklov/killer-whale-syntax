#!/usr/bin/env python3
"""
Finding 49: Is the topology-syntax coupling Boltzmann-distributed?

If P(adjacent) as a function of acoustic similarity follows
exp(similarity / T), the grammar has a "temperature" — and T
tells you how tightly the topology constrains the syntax.

Low T = rigid coupling (only very similar calls follow each other).
High T = loose coupling (similarity matters less).
"""

import os
import sys
import ast
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))


def load_and_normalise(path, group_key='station'):
    data = np.load(path, allow_pickle=True)
    features = data["features"].copy()
    metadata = [ast.literal_eval(m) for m in data["metadata"]]

    by_group = defaultdict(list)
    for i, m in enumerate(metadata):
        by_group[m.get(group_key, 'default')].append(i)
    for g, indices in by_group.items():
        sf = features[indices]
        mean, std = sf.mean(axis=0), sf.std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        for i in indices:
            features[i] = (features[i] - mean) / std

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.where(norms > 0, norms, 1.0), metadata


def analyse_boltzmann(name, normed, metadata):
    print(f"\n  ═══ {name} ═══")

    by_file = defaultdict(list)
    for i, m in enumerate(metadata):
        by_file[m.get('filename', str(i))].append(
            (m.get('duration', 0), i))

    adj_sims, rand_sims = [], []
    rng = np.random.RandomState(42)
    for key, entries in by_file.items():
        entries.sort()
        for j in range(len(entries) - 1):
            adj_sims.append(float(np.dot(normed[entries[j][1]], normed[entries[j+1][1]])))
    for _ in range(len(adj_sims)):
        i1, i2 = rng.choice(len(normed), 2, replace=False)
        rand_sims.append(float(np.dot(normed[i1], normed[i2])))

    adj = np.array(adj_sims)
    rand = np.array(rand_sims)

    # Bin by similarity, compute P(adjacent) / P(random) = enrichment ratio
    bins = np.linspace(-1, 1, 21)
    adj_hist, _ = np.histogram(adj, bins=bins)
    rand_hist, _ = np.histogram(rand, bins=bins)

    print(f"    {'Similarity':>12s}  {'P(adj)':>8s}  {'P(rand)':>8s}  {'Ratio':>8s}  {'log(R)':>8s}")
    print(f"    {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    sim_centers = []
    log_ratios = []

    for i in range(len(bins) - 1):
        if adj_hist[i] > 5 and rand_hist[i] > 5:
            center = (bins[i] + bins[i+1]) / 2
            p_adj = adj_hist[i] / len(adj)
            p_rand = rand_hist[i] / len(rand)
            ratio = p_adj / p_rand
            log_r = np.log(ratio)
            sim_centers.append(center)
            log_ratios.append(log_r)
            print(f"    {center:>12.2f}  {p_adj:>8.4f}  {p_rand:>8.4f}  {ratio:>8.2f}  {log_r:>8.4f}")

    # Fit: log(ratio) = a * similarity + b
    # If linear, ratio = exp(a * sim + b) → Boltzmann with T = 1/a
    if len(sim_centers) >= 4:
        x = np.array(sim_centers)
        y = np.array(log_ratios)
        A = np.column_stack([x, np.ones(len(x))])
        result = np.linalg.lstsq(A, y, rcond=None)
        slope = result[0][0]
        intercept = result[0][1]

        # R²
        predicted = slope * x + intercept
        ss_res = np.sum((y - predicted) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        temperature = 1.0 / slope if abs(slope) > 0.01 else float('inf')

        print(f"\n    Boltzmann fit: log(ratio) = {slope:.4f} * sim + {intercept:.4f}")
        print(f"    R² = {r_squared:.4f}")
        print(f"    Slope = {slope:.4f}")
        if slope > 0:
            print(f"    Temperature T = 1/slope = {temperature:.4f}")
            if r_squared > 0.8:
                print(f"    *** STRONG Boltzmann fit — the grammar has a temperature")
            elif r_squared > 0.5:
                print(f"    ** Moderate Boltzmann fit")

    return slope if 'slope' in dir() else 0


def main():
    print("=" * 70)
    print("  BOLTZMANN: Does the grammar have a temperature?")
    print("=" * 70)

    results = {}
    for name, path, key in [
        ("SRKW", "data/haro_srkw_features.npz", "station"),
        ("TKW", "data/tkw_features.npz", "provider"),
        ("OKW", "data/okw_features.npz", "provider"),
    ]:
        normed, metadata = load_and_normalise(path, key)
        slope = analyse_boltzmann(name, normed, metadata)
        results[name] = slope

    print(f"\n  ═══ TEMPERATURE COMPARISON ═══")
    print(f"  {'Ecotype':>8s}  {'Slope':>8s}  {'T = 1/slope':>12s}  {'Interpretation':>20s}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*12}  {'─'*20}")
    for name, slope in sorted(results.items(), key=lambda x: -x[1]):
        T = 1.0 / slope if abs(slope) > 0.01 else float('inf')
        interp = "tightest coupling" if T < 1 else "moderate" if T < 5 else "loose"
        T_str = f"{T:.4f}" if T < 100 else "∞"
        print(f"  {name:>8s}  {slope:>8.4f}  {T_str:>12s}  {interp:>20s}")

    print()


if __name__ == "__main__":
    main()
