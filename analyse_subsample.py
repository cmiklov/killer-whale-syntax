#!/usr/bin/env python3
"""
Finding 47: Is TKW d=1.57 > SRKW d=1.34 real, or a sample size artifact?

Subsample SRKW to match TKW's n=2,453. Re-run topology→syntax.
Bootstrap 100 times to get confidence intervals.
"""

import os
import sys
import ast
import numpy as np
from collections import defaultdict
from scipy.stats import mannwhitneyu

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


def compute_d(normed, metadata):
    """Compute topology→syntax Cohen's d."""
    by_file = defaultdict(list)
    for i, m in enumerate(metadata):
        by_file[m.get('filename', str(i))].append(
            (m.get('duration', 0), i))

    adj, rand = [], []
    rng = np.random.RandomState(42)
    for key, entries in by_file.items():
        entries.sort()
        for j in range(len(entries) - 1):
            adj.append(float(np.dot(normed[entries[j][1]], normed[entries[j+1][1]])))
    for _ in range(len(adj)):
        i1, i2 = rng.choice(len(normed), 2, replace=False)
        rand.append(float(np.dot(normed[i1], normed[i2])))

    a, r = np.array(adj), np.array(rand)
    diff = a.mean() - r.mean()
    d = diff / np.sqrt((a.std()**2 + r.std()**2) / 2)
    return d, len(adj)


def main():
    print("=" * 70)
    print("  SUBSAMPLE TEST: Is TKW > SRKW a real difference?")
    print("=" * 70)
    print()

    # Load both
    srkw_normed, srkw_meta = load_and_normalise("data/haro_srkw_features.npz", "station")
    tkw_normed, tkw_meta = load_and_normalise("data/tkw_features.npz", "provider")

    print(f"  SRKW: {len(srkw_normed)} calls")
    print(f"  TKW:  {len(tkw_normed)} calls")

    # Full d values
    d_srkw_full, n_srkw = compute_d(srkw_normed, srkw_meta)
    d_tkw_full, n_tkw = compute_d(tkw_normed, tkw_meta)

    print(f"\n  Full samples:")
    print(f"    SRKW: d = {d_srkw_full:.4f} (n_adj = {n_srkw})")
    print(f"    TKW:  d = {d_tkw_full:.4f} (n_adj = {n_tkw})")
    print(f"    Difference: {d_tkw_full - d_srkw_full:+.4f}")

    # Bootstrap: subsample SRKW to TKW's size, 100 times
    target_n = len(tkw_normed)
    rng = np.random.RandomState(42)
    bootstrap_d = []

    print(f"\n  Bootstrap: subsampling SRKW to n={target_n}, 100 iterations...")

    for iteration in range(100):
        # Random subsample of SRKW files (not individual calls — preserve session structure)
        files = list(set(m['filename'] for m in srkw_meta))
        rng.shuffle(files)

        # Take files until we have enough calls
        selected = []
        for f in files:
            indices = [i for i, m in enumerate(srkw_meta) if m['filename'] == f]
            selected.extend(indices)
            if len(selected) >= target_n:
                break

        selected = selected[:target_n]
        sub_normed = srkw_normed[selected]
        sub_meta = [srkw_meta[i] for i in selected]

        d, _ = compute_d(sub_normed, sub_meta)
        bootstrap_d.append(d)

    bd = np.array(bootstrap_d)
    print(f"\n  SRKW subsampled d (n=100 bootstraps):")
    print(f"    Mean:   {bd.mean():.4f}")
    print(f"    Std:    {bd.std():.4f}")
    print(f"    95% CI: [{np.percentile(bd, 2.5):.4f}, {np.percentile(bd, 97.5):.4f}]")
    print(f"    Min:    {bd.min():.4f}")
    print(f"    Max:    {bd.max():.4f}")

    print(f"\n  TKW full d: {d_tkw_full:.4f}")
    print(f"  SRKW subsampled mean d: {bd.mean():.4f}")

    # How often does subsampled SRKW reach TKW's level?
    exceeds = (bd >= d_tkw_full).sum()
    print(f"\n  Times SRKW subsample d >= TKW d: {exceeds}/100")

    if exceeds < 5:
        print(f"  *** TKW > SRKW is REAL (not a sample size artifact)")
        print(f"  *** TKW topology-syntax coupling is genuinely tighter")
    elif exceeds < 25:
        print(f"  ** Likely real, but sample size contributes")
    else:
        print(f"  The difference may be a sample size artifact")

    # Also test: does SRKW d change with sample size?
    print(f"\n  SRKW d vs sample size (convergence test):")
    for frac in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        n = int(len(srkw_normed) * frac)
        sub = rng.choice(len(srkw_normed), n, replace=False)
        sub_normed = srkw_normed[sub]
        sub_meta = [srkw_meta[i] for i in sub]
        d, _ = compute_d(sub_normed, sub_meta)
        print(f"    n={n:>5d} ({frac*100:>3.0f}%): d = {d:.4f}")

    print()


if __name__ == "__main__":
    main()
