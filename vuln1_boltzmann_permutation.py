#!/usr/bin/env python3
"""
Vulnerability 1: Boltzmann Fit Permutation Null

The Boltzmann distribution (R² > 0.96) is fitted to binned log-enrichment
ratios, which are inherently smooth. A log-linear fit to binned data will
often look good. This script proves the R² is real by showing it breaks
under permutation.

Two tests:
1. Within-session shuffle: randomise call ORDER within each session
   (preserves session membership, breaks sequential adjacency).
2. Cross-session shuffle: reassign calls to random sessions
   (preserves global frequency distribution, breaks session membership).

If observed R² exceeds 95% of permuted R² values, the Boltzmann fit
is not an artifact of binning smoothness.
"""

import os
import sys
import ast
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────────
# Data loading (verbatim from analyse_boltzmann.py)
# ─────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────
# Session building
# ─────────────────────────────────────────────────────────────────────

def build_sessions(metadata):
    """Build session dict: filename -> [(duration, index), ...] sorted."""
    by_file = defaultdict(list)
    for i, m in enumerate(metadata):
        by_file[m.get('filename', str(i))].append(
            (m.get('duration', 0), i))
    for key in by_file:
        by_file[key].sort()
    return by_file


# ─────────────────────────────────────────────────────────────────────
# Adjacent similarities from session order
# ─────────────────────────────────────────────────────────────────────

def compute_adjacent_sims(normed, by_file):
    """Compute cosine similarities for all consecutive call pairs."""
    sims = []
    for key, entries in by_file.items():
        for j in range(len(entries) - 1):
            sims.append(float(np.dot(
                normed[entries[j][1]], normed[entries[j + 1][1]])))
    return np.array(sims)


def compute_random_sims(normed, n, rng):
    """Compute cosine similarities for n random pairs."""
    sims = []
    for _ in range(n):
        i1, i2 = rng.choice(len(normed), 2, replace=False)
        sims.append(float(np.dot(normed[i1], normed[i2])))
    return np.array(sims)


# ─────────────────────────────────────────────────────────────────────
# Boltzmann fit
# ─────────────────────────────────────────────────────────────────────

def fit_boltzmann(adj_sims, rand_sims):
    """
    Bin similarities, compute log-enrichment ratios, fit linear regression.
    Returns (slope, intercept, r_squared, n_bins_used).
    """
    bins = np.linspace(-1, 1, 21)
    adj_hist, _ = np.histogram(adj_sims, bins=bins)
    rand_hist, _ = np.histogram(rand_sims, bins=bins)

    sim_centers = []
    log_ratios = []

    for i in range(len(bins) - 1):
        if adj_hist[i] > 5 and rand_hist[i] > 5:
            center = (bins[i] + bins[i + 1]) / 2
            p_adj = adj_hist[i] / len(adj_sims)
            p_rand = rand_hist[i] / len(rand_sims)
            ratio = p_adj / p_rand
            log_ratios.append(np.log(ratio))
            sim_centers.append(center)

    if len(sim_centers) < 4:
        return 0.0, 0.0, 0.0, len(sim_centers)

    x = np.array(sim_centers)
    y = np.array(log_ratios)
    A = np.column_stack([x, np.ones(len(x))])
    result = np.linalg.lstsq(A, y, rcond=None)
    slope = result[0][0]
    intercept = result[0][1]

    predicted = slope * x + intercept
    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return slope, intercept, r_squared, len(sim_centers)


# ─────────────────────────────────────────────────────────────────────
# Permutation tests
# ─────────────────────────────────────────────────────────────────────

def shuffle_within_sessions(by_file, rng):
    """Shuffle call indices within each session, preserving session boundaries."""
    shuffled = {}
    for key, entries in by_file.items():
        indices = [idx for _, idx in entries]
        rng.shuffle(indices)
        # Pair shuffled indices with original durations (order doesn't matter,
        # we just need (something, shuffled_index) pairs for compute_adjacent_sims)
        shuffled[key] = [(0, idx) for idx in indices]
    return shuffled


def shuffle_across_sessions(by_file, rng):
    """Reassign calls to random sessions, preserving session sizes."""
    # Pool all indices
    all_indices = []
    session_sizes = []
    session_keys = []
    for key, entries in by_file.items():
        session_keys.append(key)
        session_sizes.append(len(entries))
        all_indices.extend(idx for _, idx in entries)

    # Shuffle the pool
    rng.shuffle(all_indices)

    # Redistribute into sessions of original sizes
    shuffled = {}
    offset = 0
    for key, size in zip(session_keys, session_sizes):
        shuffled[key] = [(0, all_indices[offset + j]) for j in range(size)]
        offset += size
    return shuffled


def run_permutation_test(name, normed, metadata, group_key, n_iterations=1000):
    """Run both permutation tests for one ecotype."""
    print(f"\n  ═══ {name} ═══")

    rng = np.random.RandomState(42)
    by_file = build_sessions(metadata)

    # Observed fit
    adj_sims = compute_adjacent_sims(normed, by_file)
    rand_sims = compute_random_sims(normed, len(adj_sims), rng)
    obs_slope, obs_intercept, obs_r2, obs_bins = fit_boltzmann(adj_sims, rand_sims)

    print(f"    Observed Boltzmann fit:")
    print(f"      Slope: {obs_slope:.4f}    Intercept: {obs_intercept:.4f}    R²: {obs_r2:.4f}")
    print(f"      Bins used: {obs_bins}    Adjacent pairs: {len(adj_sims)}")

    # ── Within-session shuffle ──
    print(f"\n    Within-session permutation ({n_iterations} iterations):")
    null_r2_within = np.zeros(n_iterations)
    null_slopes_within = np.zeros(n_iterations)

    for it in range(n_iterations):
        shuffled = shuffle_within_sessions(by_file, rng)
        shuffled_adj = compute_adjacent_sims(normed, shuffled)
        slope, intercept, r2, n_bins = fit_boltzmann(shuffled_adj, rand_sims)
        null_r2_within[it] = r2
        null_slopes_within[it] = slope

        if (it + 1) % 200 == 0:
            print(f"      ... {it + 1}/{n_iterations}")

    p_within = (null_r2_within >= obs_r2).sum() / n_iterations

    print(f"      Null R² distribution:")
    print(f"        Mean:    {null_r2_within.mean():.4f}")
    print(f"        Std:     {null_r2_within.std():.4f}")
    print(f"        Median:  {np.median(null_r2_within):.4f}")
    print(f"        95th %%:  {np.percentile(null_r2_within, 95):.4f}")
    print(f"        Max:     {null_r2_within.max():.4f}")
    print(f"      Null slope distribution:")
    print(f"        Mean:    {null_slopes_within.mean():.4f}")
    print(f"        Std:     {null_slopes_within.std():.4f}")
    print(f"      Observed R² = {obs_r2:.4f} vs 95th percentile = {np.percentile(null_r2_within, 95):.4f}")
    print(f"      p-value: {p_within:.4f} (fraction of null >= observed)")

    if p_within < 0.05:
        print(f"      *** SIGNIFICANT (p < 0.05): Boltzmann fit is NOT an artifact of binning")
    elif p_within < 0.10:
        print(f"      * Marginal (p < 0.10)")
    else:
        print(f"      NOT SIGNIFICANT: Boltzmann fit may be a binning artifact")

    # ── Cross-session shuffle ──
    print(f"\n    Cross-session permutation ({n_iterations} iterations):")
    null_r2_cross = np.zeros(n_iterations)

    for it in range(n_iterations):
        shuffled = shuffle_across_sessions(by_file, rng)
        shuffled_adj = compute_adjacent_sims(normed, shuffled)
        slope, intercept, r2, n_bins = fit_boltzmann(shuffled_adj, rand_sims)
        null_r2_cross[it] = r2

        if (it + 1) % 200 == 0:
            print(f"      ... {it + 1}/{n_iterations}")

    p_cross = (null_r2_cross >= obs_r2).sum() / n_iterations

    print(f"      Null R² distribution:")
    print(f"        Mean:    {null_r2_cross.mean():.4f}")
    print(f"        Std:     {null_r2_cross.std():.4f}")
    print(f"        95th %%:  {np.percentile(null_r2_cross, 95):.4f}")
    print(f"        Max:     {null_r2_cross.max():.4f}")
    print(f"      p-value: {p_cross:.4f}")

    if p_cross < 0.05:
        print(f"      *** SIGNIFICANT: Boltzmann fit requires session structure")
    else:
        print(f"      NOT SIGNIFICANT: session structure not required for fit")

    return {
        'observed_r2': obs_r2,
        'observed_slope': obs_slope,
        'p_within': p_within,
        'p_cross': p_cross,
        'null_r2_within_mean': null_r2_within.mean(),
        'null_r2_within_95': np.percentile(null_r2_within, 95),
        'null_r2_cross_mean': null_r2_cross.mean(),
        'null_r2_cross_95': np.percentile(null_r2_cross, 95),
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  VULNERABILITY 1: Boltzmann Permutation Null")
    print("  Does the R² > 0.96 survive shuffling?")
    print("=" * 70)

    results = {}
    for name, path, group_key in [
        ("SRKW", "data/haro_srkw_features.npz", "station"),
        ("TKW", "data/tkw_features.npz", "provider"),
        ("OKW", "data/okw_features.npz", "provider"),
    ]:
        normed, metadata = load_and_normalise(path, group_key)
        results[name] = run_permutation_test(name, normed, metadata, group_key)

    # Summary
    print(f"\n  {'═' * 60}")
    print(f"  SUMMARY")
    print(f"  {'═' * 60}")
    print(f"  {'Ecotype':>8s}  {'Obs R²':>8s}  {'Null R² (within)':>18s}  {'p(within)':>10s}  {'p(cross)':>10s}")
    print(f"  {'─' * 8}  {'─' * 8}  {'─' * 18}  {'─' * 10}  {'─' * 10}")
    for name in ['SRKW', 'TKW', 'OKW']:
        r = results[name]
        null_str = f"{r['null_r2_within_mean']:.4f} ± {r['null_r2_within_95']:.4f}"
        print(f"  {name:>8s}  {r['observed_r2']:>8.4f}  {null_str:>18s}  {r['p_within']:>10.4f}  {r['p_cross']:>10.4f}")

    all_sig = all(r['p_within'] < 0.05 for r in results.values())
    if all_sig:
        print(f"\n  *** ALL ECOTYPES: Boltzmann fit is NOT an artifact of binning")
        print(f"  *** The R² requires sequential adjacency — it reflects real structure")
    else:
        failed = [n for n, r in results.items() if r['p_within'] >= 0.05]
        print(f"\n  WARNING: Boltzmann fit NOT significant for: {', '.join(failed)}")

    print()


if __name__ == "__main__":
    main()
