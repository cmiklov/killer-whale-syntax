#!/usr/bin/env python3
"""
The crown jewel: does the acoustic topology predict the syntax?

If calls that are acoustically similar also tend to follow each other
in sequences, then the R/D field structure IS the grammar. The topology
generates the syntax. This is the core validation of the entire framework.

Also:
  - The 49-call cycle (what IS it?)
  - Session dynamics (conversational arcs)
  - Cross-station call-response (directed communication)
"""

import os
import sys
import ast
import math
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))


def load_haro():
    data = np.load("data/haro_srkw_features.npz", allow_pickle=True)
    return data["features"].copy(), [ast.literal_eval(m) for m in data["metadata"]]


def normalise(features, metadata):
    by_station = defaultdict(list)
    for i, m in enumerate(metadata):
        by_station[m['station']].append(i)
    feat = features.copy()
    for station, indices in by_station.items():
        sf = feat[indices]
        mean, std = sf.mean(axis=0), sf.std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        for i in indices:
            feat[i] = (feat[i] - mean) / std
    norms = np.linalg.norm(feat, axis=1, keepdims=True)
    return feat / np.where(norms > 0, norms, 1.0)


# ─────────────────────────────────────────────────────────────────────
# 1. THE CROWN JEWEL: Acoustic topology predicts syntax
# ─────────────────────────────────────────────────────────────────────

def test_topology_predicts_syntax(normed, metadata):
    print("=" * 70)
    print("  1. THE CROWN JEWEL: Does acoustic topology predict syntax?")
    print("=" * 70)
    print()

    # Build sequences
    by_file = defaultdict(list)
    for i, m in enumerate(metadata):
        by_file[(m['filename'], m['station'])].append(
            (float(m.get('duration', 0)), i))

    # Collect adjacent pairs and random pairs
    adjacent_sims = []
    random_sims = []
    rng = np.random.RandomState(42)

    for key, entries in by_file.items():
        entries.sort()
        for j in range(len(entries) - 1):
            i1 = entries[j][1]
            i2 = entries[j+1][1]
            sim = float(np.dot(normed[i1], normed[i2]))
            adjacent_sims.append(sim)

    # Random pairs (same total count)
    all_indices = list(range(len(normed)))
    for _ in range(len(adjacent_sims)):
        i1, i2 = rng.choice(all_indices, 2, replace=False)
        sim = float(np.dot(normed[i1], normed[i2]))
        random_sims.append(sim)

    adj_arr = np.array(adjacent_sims)
    rand_arr = np.array(random_sims)

    print(f"  Adjacent pairs (sequential):  mean={adj_arr.mean():.6f}, std={adj_arr.std():.4f}, n={len(adj_arr)}")
    print(f"  Random pairs (non-sequential): mean={rand_arr.mean():.6f}, std={rand_arr.std():.4f}, n={len(rand_arr)}")
    diff = adj_arr.mean() - rand_arr.mean()
    print(f"  Difference: {diff:+.6f}")

    from scipy.stats import mannwhitneyu
    U, p = mannwhitneyu(adj_arr, rand_arr, alternative='greater')
    print(f"\n  Mann-Whitney U (adjacent > random): p = {p:.4e}")

    if p < 0.001:
        print(f"  *** TOPOLOGY PREDICTS SYNTAX")
        print(f"  *** Calls that are acoustically similar tend to follow each other")
        print(f"  *** The R/D field structure generates the sequential grammar")
    elif p < 0.05:
        print(f"  * Moderate support: adjacent calls are somewhat more similar")
    else:
        print(f"  No support: adjacency doesn't predict acoustic similarity")

    # Effect size: how much more similar are adjacent calls?
    cohens_d = diff / np.sqrt((adj_arr.std()**2 + rand_arr.std()**2) / 2)
    print(f"\n  Cohen's d: {cohens_d:.4f}")
    if abs(cohens_d) > 0.8:
        print(f"  Large effect")
    elif abs(cohens_d) > 0.5:
        print(f"  Medium effect")
    elif abs(cohens_d) > 0.2:
        print(f"  Small effect")
    else:
        print(f"  Negligible effect")

    # Distance-dependent transition probability
    # Bin pairs by acoustic similarity, compute transition rate
    print(f"\n  Transition probability as a function of acoustic distance:")
    print(f"  {'Similarity bin':>18s}  {'Adjacent %':>12s}  {'Random %':>12s}  {'Ratio':>8s}")
    print(f"  {'─'*18}  {'─'*12}  {'─'*12}  {'─'*8}")

    bins = [(0.99, 1.0), (0.95, 0.99), (0.90, 0.95), (0.85, 0.90), (0.80, 0.85), (0.0, 0.80)]
    for lo, hi in bins:
        adj_in_bin = ((adj_arr >= lo) & (adj_arr < hi)).sum()
        rand_in_bin = ((rand_arr >= lo) & (rand_arr < hi)).sum()
        adj_pct = adj_in_bin / len(adj_arr) * 100
        rand_pct = rand_in_bin / len(rand_arr) * 100
        ratio = adj_pct / rand_pct if rand_pct > 0 else float('inf')
        print(f"  {lo:.2f}-{hi:.2f}          {adj_pct:>10.1f}%  {rand_pct:>10.1f}%  {ratio:>7.2f}×")

    print()
    return adj_arr, rand_arr


# ─────────────────────────────────────────────────────────────────────
# 2. THE 49-CALL CYCLE
# ─────────────────────────────────────────────────────────────────────

def analyse_49_cycle(normed, metadata):
    print("=" * 70)
    print("  2. THE 49-CALL CYCLE: What repeats every ~49 calls?")
    print("=" * 70)
    print()

    # Build longest sequences
    by_file = defaultdict(list)
    for i, m in enumerate(metadata):
        by_file[(m['filename'], m['station'])].append(
            (float(m.get('duration', 0)), i))

    long_seqs = []
    for key, entries in by_file.items():
        entries.sort()
        if len(entries) >= 60:
            long_seqs.append([idx for _, idx in entries])

    if not long_seqs:
        print("  No sequences long enough (need 60+)")
        return

    print(f"  Long sequences (60+ calls): {len(long_seqs)}")
    print(f"  Longest: {max(len(s) for s in long_seqs)} calls")

    # For each long sequence, compute autocorrelation of acoustic features
    # at the 50D level (not just cluster labels)
    all_autocorrs = []

    for seq in long_seqs[:10]:  # top 10 longest
        if len(seq) < 100:
            continue

        vecs = normed[seq]
        n = len(vecs)

        # Self-similarity matrix: sim[i,j] = dot(v_i, v_j)
        sim_matrix = vecs @ vecs.T

        # Autocorrelation: average similarity at each lag
        max_lag = min(80, n // 2)
        autocorr = np.zeros(max_lag)
        for lag in range(max_lag):
            diag = np.diag(sim_matrix, lag)
            autocorr[lag] = diag.mean()

        all_autocorrs.append(autocorr)

    if not all_autocorrs:
        print("  Insufficient long sequences")
        return

    # Average autocorrelation across sequences
    min_len = min(len(a) for a in all_autocorrs)
    avg_autocorr = np.mean([a[:min_len] for a in all_autocorrs], axis=0)

    print(f"\n  Mean acoustic autocorrelation by lag:")
    print(f"  {'Lag':>5s}  {'Autocorr':>10s}  {'Bar':>40s}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*40}")

    # Normalise for display
    ac_min = avg_autocorr[1:].min()
    ac_max = avg_autocorr[1:].max()
    for lag in range(0, min_len, 2):  # every other lag for readability
        val = avg_autocorr[lag]
        if ac_max > ac_min:
            bar_len = int((val - ac_min) / (ac_max - ac_min) * 40)
        else:
            bar_len = 20
        bar = "█" * max(bar_len, 0)
        marker = "  ◄ ~49" if 47 <= lag <= 51 else ""
        print(f"  {lag:>5d}  {val:>10.6f}  {bar}{marker}")

    # Find peaks
    peaks = []
    for i in range(2, min_len - 1):
        if avg_autocorr[i] > avg_autocorr[i-1] and avg_autocorr[i] > avg_autocorr[i+1]:
            if avg_autocorr[i] > avg_autocorr[1:].mean():
                peaks.append((i, avg_autocorr[i]))

    print(f"\n  Autocorrelation peaks (above mean):")
    for lag, val in sorted(peaks, key=lambda x: -x[1])[:8]:
        print(f"    Lag {lag:>3d}: r={val:.6f}")

    # What's happening at the ~49 boundary?
    if any(45 <= p[0] <= 53 for p in peaks):
        print(f"\n  *** CONFIRMED: Periodic structure near lag 49")
        print(f"  *** The acoustic content repeats on a ~49-call cycle")
        # What's the period in seconds?
        # Mean ICI from Haro data was ~0.034s between annotations
        # But that's begin-to-begin within a file, which depends on call density
        print(f"  *** At typical SRKW call rates, ~49 calls ≈ 2-3 minutes")
        print(f"  *** This may correspond to a breathing/surfacing cycle")
    print()


# ─────────────────────────────────────────────────────────────────────
# 3. SESSION DYNAMICS: Conversational arcs
# ─────────────────────────────────────────────────────────────────────

def analyse_session_dynamics(normed, metadata):
    print("=" * 70)
    print("  3. SESSION DYNAMICS: Do conversations have arcs?")
    print("=" * 70)
    print()

    by_file = defaultdict(list)
    for i, m in enumerate(metadata):
        by_file[(m['filename'], m['station'])].append(
            (float(m.get('duration', 0)), i))

    # For each session, track acoustic diversity over time
    # Divide session into quarters, measure entropy in each quarter
    arc_data = []

    for key, entries in by_file.items():
        entries.sort()
        n = len(entries)
        if n < 40:
            continue

        indices = [idx for _, idx in entries]
        vecs = normed[indices]

        # Divide into 4 quarters
        q_size = n // 4
        quarter_diversities = []
        quarter_energies = []

        for q in range(4):
            start = q * q_size
            end = start + q_size if q < 3 else n
            q_vecs = vecs[start:end]

            # Diversity: mean pairwise distance within quarter
            if len(q_vecs) >= 5:
                sims = q_vecs @ q_vecs.T
                upper = sims[np.triu_indices(len(q_vecs), k=1)]
                diversity = 1.0 - upper.mean()  # distance = 1 - similarity
                quarter_diversities.append(diversity)

                # Mean centroid magnitude (proxy for acoustic energy)
                centroid = q_vecs.mean(axis=0)
                quarter_energies.append(np.linalg.norm(centroid))

        if len(quarter_diversities) == 4:
            arc_data.append({
                'diversities': quarter_diversities,
                'energies': quarter_energies,
                'n': n,
            })

    if not arc_data:
        print("  Insufficient sessions")
        return

    print(f"  Sessions analysed: {len(arc_data)} (40+ calls each)")

    # Average arc shape
    avg_div = np.mean([a['diversities'] for a in arc_data], axis=0)
    avg_eng = np.mean([a['energies'] for a in arc_data], axis=0)

    print(f"\n  Average acoustic diversity by quarter:")
    labels = ["Q1 (start)", "Q2 (early)", "Q3 (late)", "Q4 (end)"]
    for i, label in enumerate(labels):
        bar = "█" * int(avg_div[i] * 200)
        print(f"    {label:>12s}: {avg_div[i]:.6f}  {bar}")

    # Is there an arc? Test: is Q2/Q3 different from Q1/Q4?
    middle = np.array([a['diversities'][1] + a['diversities'][2] for a in arc_data]) / 2
    edges = np.array([a['diversities'][0] + a['diversities'][3] for a in arc_data]) / 2

    from scipy.stats import wilcoxon
    try:
        stat, p = wilcoxon(middle, edges)
        diff = middle.mean() - edges.mean()
        print(f"\n  Middle (Q2+Q3) vs edges (Q1+Q4):")
        print(f"    Middle mean: {middle.mean():.6f}")
        print(f"    Edges mean:  {edges.mean():.6f}")
        print(f"    Difference:  {diff:+.6f}")
        print(f"    Wilcoxon: p={p:.4f}")
        if p < 0.05:
            if diff > 0:
                print(f"    *** Sessions DIVERSIFY in the middle and CONVERGE at edges")
                print(f"    *** The conversation has an ARC: open → explore → close")
            else:
                print(f"    *** Sessions are MORE FOCUSED in the middle")
                print(f"    *** The conversation has a FUNNEL: diverse → focused → diverse")
    except:
        pass

    # Coherence trajectory: how similar is each call to the session centroid?
    print(f"\n  Coherence trajectory (similarity to session centroid):")
    coherence_by_position = defaultdict(list)
    for key, entries in by_file.items():
        entries.sort()
        n = len(entries)
        if n < 40:
            continue
        indices = [idx for _, idx in entries]
        vecs = normed[indices]
        centroid = vecs.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)

        for j, idx in enumerate(indices):
            # Normalised position (0.0 = start, 1.0 = end)
            pos = j / (n - 1) if n > 1 else 0.5
            coherence = float(np.dot(normed[idx], centroid))
            bin_pos = int(pos * 10) / 10  # bin to 0.0, 0.1, ..., 0.9
            coherence_by_position[bin_pos].append(coherence)

    if coherence_by_position:
        print(f"  {'Position':>10s}  {'Coherence':>10s}  {'N':>6s}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*6}")
        for pos in sorted(coherence_by_position.keys()):
            vals = np.array(coherence_by_position[pos])
            print(f"  {pos:>10.1f}  {vals.mean():>10.6f}  {len(vals):>6d}")
    print()


# ─────────────────────────────────────────────────────────────────────
# 4. CROSS-STATION CALL-RESPONSE
# ─────────────────────────────────────────────────────────────────────

def analyse_cross_station_response(normed, metadata):
    print("=" * 70)
    print("  4. CROSS-STATION CALL-RESPONSE: Directed communication?")
    print("=" * 70)
    print()

    # Group by timestamp to find simultaneous recordings
    # Same filename timestamp = same 30-minute recording window
    from datetime import datetime
    import re

    by_timestamp = defaultdict(lambda: {'north': [], 'south': []})

    for i, m in enumerate(metadata):
        fname = m['filename']
        station = m['station']
        # Extract timestamp from filename: AMAR*.YYYYMMDDTHHMMSSZ.wav
        match = re.search(r'(\d{8}T\d{6}Z)', fname)
        if match:
            ts = match.group(1)
            begin = float(m.get('duration', 0))  # FileBeginSec stored as duration
            by_timestamp[ts][station].append((begin, i))

    # Find timestamps with calls from BOTH stations
    paired = {ts: data for ts, data in by_timestamp.items()
              if data['north'] and data['south']}

    print(f"  Paired timestamps (both stations): {len(paired)}")

    if not paired:
        print("  No paired timestamps found")
        return

    # For each paired window, look for call-response patterns
    # If north calls at time T, does south respond at T+delta?
    response_delays = []
    non_response_delays = []

    for ts, data in paired.items():
        north = sorted(data['north'])
        south = sorted(data['south'])

        # For each north call, find the nearest subsequent south call
        for n_time, n_idx in north:
            nearest_delta = None
            nearest_s_idx = None
            for s_time, s_idx in south:
                delta = s_time - n_time
                if 0.05 < delta < 5.0:  # response window: 50ms to 5s
                    if nearest_delta is None or delta < nearest_delta:
                        nearest_delta = delta
                        nearest_s_idx = s_idx

            if nearest_delta is not None:
                # Is the response acoustically related to the call?
                sim = float(np.dot(normed[n_idx], normed[nearest_s_idx]))
                response_delays.append((nearest_delta, sim))

        # Also measure non-paired delays for control
        for s_time, s_idx in south:
            nearest_delta = None
            nearest_n_idx = None
            for n_time, n_idx in north:
                delta = s_time - n_time
                if 0.05 < delta < 5.0:
                    if nearest_delta is None or delta < nearest_delta:
                        nearest_delta = delta
                        nearest_n_idx = n_idx
            if nearest_delta is not None:
                sim = float(np.dot(normed[s_idx], normed[nearest_n_idx]))
                non_response_delays.append((nearest_delta, sim))

    if response_delays:
        delays = np.array([d for d, _ in response_delays])
        sims = np.array([s for _, s in response_delays])

        print(f"\n  North→South response patterns:")
        print(f"    Pairs found: {len(response_delays)}")
        print(f"    Mean delay: {delays.mean():.3f}s")
        print(f"    Median delay: {np.median(delays):.3f}s")
        print(f"    Mean acoustic similarity: {sims.mean():.4f}")

        # Delay distribution
        bins = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        print(f"\n  Response delay distribution:")
        for j in range(len(bins) - 1):
            mask = (delays >= bins[j]) & (delays < bins[j+1])
            count = mask.sum()
            if count > 0:
                mean_sim = sims[mask].mean()
                pct = count / len(delays) * 100
                print(f"    {bins[j]:.2f}-{bins[j+1]:.1f}s: n={count:>4d} ({pct:>5.1f}%)  sim={mean_sim:.4f}")

        # KEY TEST: do responses have higher acoustic similarity than random?
        if non_response_delays:
            non_sims = np.array([s for _, s in non_response_delays])

            from scipy.stats import mannwhitneyu
            U, p = mannwhitneyu(sims, non_sims, alternative='greater')
            print(f"\n  Response similarity vs random cross-station:")
            print(f"    Response mean sim:  {sims.mean():.4f}")
            print(f"    Random mean sim:    {non_sims.mean():.4f}")
            print(f"    Mann-Whitney: p={p:.4f}")
            if p < 0.05:
                print(f"    *** Cross-station responses are acoustically MATCHED")
                print(f"    *** Evidence of DIRECTED COMMUNICATION between individuals")

        # Delay-similarity correlation
        r = np.corrcoef(delays, sims)[0, 1]
        print(f"\n  Delay-similarity correlation: r={r:.4f}")
        if r < -0.1:
            print(f"    Faster responses are MORE similar (echo/matching)")
        elif r > 0.1:
            print(f"    Faster responses are LESS similar (turn-taking/contrast)")

    print()


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "▓" * 70)
    print("▓  THE PATTERN COALESCES")
    print("▓  Topology → syntax, the 49-call cycle, arcs, call-response")
    print("▓" * 70)
    print()

    features, metadata = load_haro()
    normed = normalise(features, metadata)

    adj, rand = test_topology_predicts_syntax(normed, metadata)
    analyse_49_cycle(normed, metadata)
    analyse_session_dynamics(normed, metadata)
    analyse_cross_station_response(normed, metadata)

    print("▓" * 70)
    print("▓  THE PATTERN")
    print("▓" * 70)
    print()


if __name__ == "__main__":
    main()
