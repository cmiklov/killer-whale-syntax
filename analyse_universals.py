#!/usr/bin/env python3
"""
Linguistic universals + information-theoretic deep analysis.

Tests whether orca communication follows the same statistical laws as
human language:

  1. Zipf's law: frequency ∝ 1/rank (power law distribution)
  2. Brevity law: more frequent signals are shorter (Zipf's law of abbreviation)
  3. Menzerath's law: longer sequences have shorter constituents
  4. Heaps' law: vocabulary grows sublinearly with corpus size
  5. Mutual information decay: how far does sequential memory reach?
  6. Bout structure: what starts, sustains, and terminates a bout?
  7. Bigg's (TKW) deep dive: richest grammar + vocabulary

If orca calls follow these laws, they share the statistical backbone
of human language — regardless of whether they carry semantic content.
"""

import os
import sys
import csv
import math
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))


def load_annotations():
    with open("data/dclde/Annotations.csv", 'r') as f:
        return list(csv.DictReader(f))


def get_call_data(rows, ecotype):
    """Extract calls with frequency data for an ecotype."""
    calls = []
    for r in rows:
        if r['KW'] != '1' or r['AnnotationLevel'] != 'Call' or r['Ecotype'] != ecotype:
            continue
        try:
            lo = float(r['LowFreqHz']) if r['LowFreqHz'] != 'NA' else None
            hi = float(r['HighFreqHz']) if r['HighFreqHz'] != 'NA' else None
            dur = float(r['FileEndSec']) - float(r['FileBeginSec'])
            if lo and hi and lo > 0 and hi > 0 and dur > 0.05:
                calls.append({
                    'center': (lo + hi) / 2,
                    'bw': hi - lo,
                    'dur': dur,
                    'file': r['Soundfile'],
                    'begin': float(r['FileBeginSec']),
                    'utc': r.get('UTC', ''),
                })
        except:
            pass
    return calls


def cluster_and_sequence(calls, n_clusters=5):
    """Cluster calls and build temporal sequences. Use k=5 for finer granularity."""
    from sklearn.cluster import KMeans

    X = np.array([[c['center']/10000, c['bw']/10000, min(c['dur'],10)/10] for c in calls])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # Build sequences
    by_file = defaultdict(list)
    for i, c in enumerate(calls):
        by_file[c['file']].append((c['begin'], labels[i], c['dur']))

    sequences = []
    for fname, entries in by_file.items():
        se = sorted(entries, key=lambda x: x[0])
        seq = [se[0]]
        for j in range(1, len(se)):
            if se[j][0] - se[j-1][0] <= 30:
                seq.append(se[j])
            else:
                if len(seq) >= 2:
                    sequences.append(seq)
                seq = [se[j]]
        if len(seq) >= 2:
            sequences.append(seq)

    return labels, X, km, sequences


# ─────────────────────────────────────────────────────────────────────
# 1. ZIPF'S LAW
# ─────────────────────────────────────────────────────────────────────

def test_zipf(labels, name=""):
    print(f"\n  ─── Zipf's Law ({name}) ───")

    counts = Counter(labels)
    ranked = sorted(counts.values(), reverse=True)
    n = len(ranked)

    print(f"  Rank  Count  Freq    Expected (Zipf)")
    total = sum(ranked)
    C = ranked[0]  # Zipf constant
    for i, count in enumerate(ranked):
        rank = i + 1
        freq = count / total
        expected = C / rank / total
        print(f"  {rank:>4d}  {count:>5d}  {freq:.4f}  {expected:.4f}")

    # Fit power law: log(freq) = -alpha * log(rank) + const
    log_ranks = np.log(np.arange(1, n + 1))
    log_freqs = np.log(np.array(ranked) / total)

    # Linear regression
    A = np.column_stack([log_ranks, np.ones(n)])
    result = np.linalg.lstsq(A, log_freqs, rcond=None)
    alpha = -result[0][0]

    # R² goodness of fit
    predicted = result[0][0] * log_ranks + result[0][1]
    ss_res = np.sum((log_freqs - predicted) ** 2)
    ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"\n  Zipf exponent α = {alpha:.4f} (human language ≈ 1.0)")
    print(f"  R² = {r_squared:.4f}")
    if r_squared > 0.9:
        print(f"  *** STRONG Zipf's law fit")
    elif r_squared > 0.7:
        print(f"  ** Moderate Zipf's law fit")
    else:
        print(f"  Weak Zipf's law fit")

    return alpha, r_squared


# ─────────────────────────────────────────────────────────────────────
# 2. BREVITY LAW (Zipf's law of abbreviation)
# ─────────────────────────────────────────────────────────────────────

def test_brevity(labels, calls, name=""):
    print(f"\n  ─── Brevity Law ({name}) ───")
    print(f"  (More frequent call types should be shorter)")

    type_stats = defaultdict(lambda: {'count': 0, 'durations': []})
    for i, label in enumerate(labels):
        type_stats[label]['count'] += 1
        type_stats[label]['durations'].append(calls[i]['dur'])

    types = sorted(type_stats.keys(), key=lambda t: -type_stats[t]['count'])

    print(f"\n  {'Type':>6s}  {'Count':>7s}  {'Mean dur':>10s}  {'Std dur':>10s}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*10}  {'─'*10}")

    freqs = []
    mean_durs = []
    for t in types:
        s = type_stats[t]
        d = np.array(s['durations'])
        freqs.append(s['count'])
        mean_durs.append(d.mean())
        print(f"  C{t:>4d}  {s['count']:>7d}  {d.mean():>10.3f}s  {d.std():>10.3f}s")

    # Correlation: frequency vs duration (should be negative for brevity law)
    freqs = np.array(freqs)
    mean_durs = np.array(mean_durs)
    if len(freqs) > 2:
        r = np.corrcoef(np.log(freqs), mean_durs)[0, 1]
        print(f"\n  Correlation log(frequency) vs duration: r = {r:.4f}")
        if r < -0.3:
            print(f"  *** BREVITY LAW HOLDS: More frequent calls are shorter")
        elif r < 0:
            print(f"  * Weak brevity tendency")
        else:
            print(f"  Brevity law does NOT hold (r > 0)")
        return r
    return 0


# ─────────────────────────────────────────────────────────────────────
# 3. MENZERATH'S LAW
# ─────────────────────────────────────────────────────────────────────

def test_menzerath(sequences, name=""):
    print(f"\n  ─── Menzerath's Law ({name}) ───")
    print(f"  (Longer sequences should have shorter individual calls)")

    seq_data = []
    for seq in sequences:
        seq_len = len(seq)
        mean_dur = np.mean([dur for _, _, dur in seq])
        seq_data.append((seq_len, mean_dur))

    if len(seq_data) < 10:
        print(f"  Insufficient sequences ({len(seq_data)})")
        return 0

    lengths = np.array([s[0] for s in seq_data])
    mean_durs = np.array([s[1] for s in seq_data])

    # Bin by sequence length for cleaner visualization
    bins = defaultdict(list)
    for l, d in seq_data:
        bin_key = min(l // 5 * 5, 50)  # bin by 5s, cap at 50
        bins[bin_key].append(d)

    print(f"\n  {'Seq len':>10s}  {'N':>5s}  {'Mean call dur':>14s}")
    print(f"  {'─'*10}  {'─'*5}  {'─'*14}")
    for bin_key in sorted(bins.keys()):
        if len(bins[bin_key]) >= 3:
            d = np.array(bins[bin_key])
            label = f"{bin_key}-{bin_key+4}" if bin_key < 50 else "50+"
            print(f"  {label:>10s}  {len(d):>5d}  {d.mean():>14.3f}s")

    # Correlation
    r = np.corrcoef(np.log(lengths + 1), mean_durs)[0, 1]
    print(f"\n  Correlation log(seq_length) vs mean_call_duration: r = {r:.4f}")
    if r < -0.3:
        print(f"  *** MENZERATH'S LAW HOLDS: Longer sequences have shorter calls")
    elif r < 0:
        print(f"  * Weak Menzerath tendency")
    else:
        print(f"  Menzerath's law does NOT hold")
    return r


# ─────────────────────────────────────────────────────────────────────
# 4. HEAPS' LAW
# ─────────────────────────────────────────────────────────────────────

def test_heaps(labels, name=""):
    print(f"\n  ─── Heaps' Law ({name}) ───")
    print(f"  (Vocabulary should grow sublinearly with corpus size)")

    vocab_growth = []
    seen = set()
    for i, label in enumerate(labels):
        seen.add(label)
        if (i + 1) % max(1, len(labels) // 20) == 0 or i == len(labels) - 1:
            vocab_growth.append((i + 1, len(seen)))

    print(f"\n  {'Corpus size':>12s}  {'Vocabulary':>12s}  {'V/N':>8s}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*8}")
    for n, v in vocab_growth:
        print(f"  {n:>12d}  {v:>12d}  {v/n:>8.4f}")

    # Fit Heaps' law: V = k * N^beta (beta < 1 for sublinear growth)
    if len(vocab_growth) >= 3:
        ns = np.array([x[0] for x in vocab_growth], dtype=float)
        vs = np.array([x[1] for x in vocab_growth], dtype=float)
        log_ns = np.log(ns)
        log_vs = np.log(vs)

        A = np.column_stack([log_ns, np.ones(len(ns))])
        result = np.linalg.lstsq(A, log_vs, rcond=None)
        beta = result[0][0]

        print(f"\n  Heaps' exponent β = {beta:.4f} (human language ≈ 0.4-0.6)")
        if beta < 0.8:
            print(f"  *** HEAPS' LAW HOLDS: Vocabulary grows sublinearly")
        elif beta < 1.0:
            print(f"  * Weak sublinear growth")
        else:
            print(f"  Vocabulary grows linearly or faster (not Heaps)")
        return beta
    return 0


# ─────────────────────────────────────────────────────────────────────
# 5. MUTUAL INFORMATION DECAY
# ─────────────────────────────────────────────────────────────────────

def test_mi_decay(sequences, n_clusters, name=""):
    print(f"\n  ─── Mutual Information Decay ({name}) ───")
    print(f"  (How far does the 'memory' of a sequence reach?)")

    flat = []
    for seq in sequences:
        flat.extend([label for _, label, _ in seq])

    if len(flat) < 100:
        print(f"  Insufficient data ({len(flat)} calls)")
        return

    # Marginal entropy
    counts = Counter(flat)
    total = len(flat)
    H_marginal = -sum((c/total) * math.log2(c/total) for c in counts.values() if c > 0)

    # MI at different lags
    print(f"\n  {'Lag':>5s}  {'MI (bits)':>10s}  {'MI/H':>8s}  {'Decay':>8s}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*8}  {'─'*8}")

    prev_mi = None
    mi_values = []
    for lag in range(1, 12):
        # Joint distribution at lag
        joint = Counter()
        for i in range(len(flat) - lag):
            joint[(flat[i], flat[i + lag])] += 1

        total_pairs = sum(joint.values())
        if total_pairs < 50:
            break

        # MI = H(X) + H(Y) - H(X,Y)
        H_joint = -sum((c/total_pairs) * math.log2(c/total_pairs)
                       for c in joint.values() if c > 0)
        MI = 2 * H_marginal - H_joint  # since H(X) = H(Y) for stationary process
        MI = max(MI, 0)  # numerical floor

        mi_ratio = MI / H_marginal if H_marginal > 0 else 0
        decay = (prev_mi - MI) / prev_mi * 100 if prev_mi and prev_mi > 0 else 0
        mi_values.append((lag, MI))

        print(f"  {lag:>5d}  {MI:>10.4f}  {mi_ratio:>7.1%}  {decay:>7.1f}%")
        prev_mi = MI

    if len(mi_values) >= 3:
        # Fit exponential decay: MI(lag) = a * exp(-b * lag)
        lags = np.array([m[0] for m in mi_values], dtype=float)
        mis = np.array([m[1] for m in mi_values])
        mis_pos = np.where(mis > 0, mis, 1e-10)
        log_mis = np.log(mis_pos)

        A = np.column_stack([lags, np.ones(len(lags))])
        result = np.linalg.lstsq(A, log_mis, rcond=None)
        decay_rate = -result[0][0]
        half_life = math.log(2) / decay_rate if decay_rate > 0 else float('inf')

        print(f"\n  Decay rate: {decay_rate:.4f} per lag")
        print(f"  Half-life: {half_life:.1f} calls")
        print(f"  (Information about a call is halved after {half_life:.1f} subsequent calls)")

        if half_life > 5:
            print(f"  *** LONG MEMORY: Sequential information persists across many calls")
        elif half_life > 2:
            print(f"  ** Moderate memory")
        else:
            print(f"  Short memory (rapid decay)")


# ─────────────────────────────────────────────────────────────────────
# 6. BOUT STRUCTURE
# ─────────────────────────────────────────────────────────────────────

def analyse_bouts(sequences, name=""):
    print(f"\n  ─── Bout Structure ({name}) ───")

    # A bout is a run of the same call type within a sequence
    bout_lengths = defaultdict(list)
    bout_transitions = Counter()  # what call type follows a bout?
    bout_starters = Counter()     # what starts a sequence?
    bout_enders = Counter()       # what ends a sequence?

    for seq in sequences:
        if not seq:
            continue
        bout_starters[seq[0][1]] += 1
        bout_enders[seq[-1][1]] += 1

        current_type = seq[0][1]
        bout_len = 1
        for j in range(1, len(seq)):
            if seq[j][1] == current_type:
                bout_len += 1
            else:
                bout_lengths[current_type].append(bout_len)
                bout_transitions[(current_type, seq[j][1])] += 1
                current_type = seq[j][1]
                bout_len = 1
        bout_lengths[current_type].append(bout_len)

    print(f"\n  Bout lengths by call type:")
    print(f"  {'Type':>6s}  {'N bouts':>8s}  {'Mean':>8s}  {'Median':>8s}  {'Max':>6s}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}")

    for t in sorted(bout_lengths.keys()):
        bl = np.array(bout_lengths[t])
        print(f"  C{t:>4d}  {len(bl):>8d}  {bl.mean():>8.1f}  {np.median(bl):>8.0f}  {bl.max():>6d}")

    # What starts sequences?
    total_seqs = sum(bout_starters.values())
    print(f"\n  Sequence starters:")
    for t, count in bout_starters.most_common():
        print(f"    C{t}: {count} ({count/total_seqs*100:.1f}%)")

    # What ends sequences?
    print(f"\n  Sequence enders:")
    for t, count in bout_enders.most_common():
        print(f"    C{t}: {count} ({count/total_seqs*100:.1f}%)")

    # Bout-to-bout transitions
    print(f"\n  Top bout transitions (bout of X → bout of Y):")
    for (a, b), count in bout_transitions.most_common(10):
        print(f"    C{a} bout → C{b} bout: {count}")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "▓" * 70)
    print("▓  LINGUISTIC UNIVERSALS + INFORMATION THEORY")
    print("▓  Do orcas follow the same statistical laws as human language?")
    print("▓" * 70)
    print()

    all_rows = load_annotations()

    for ecotype in ['SRKW', 'TKW', 'SAR']:
        calls = get_call_data(all_rows, ecotype)
        if len(calls) < 100:
            continue

        print("\n" + "=" * 70)
        print(f"  ECOTYPE: {ecotype} ({len(calls)} calls)")
        print("=" * 70)

        # Use k=5 for finer clustering (more call types for Zipf/Heaps)
        labels, X, km, sequences = cluster_and_sequence(calls, n_clusters=5)
        nc = km.n_clusters

        # 1. Zipf
        alpha, r2 = test_zipf(labels, ecotype)

        # 2. Brevity
        brevity_r = test_brevity(labels, calls, ecotype)

        # 3. Menzerath
        menz_r = test_menzerath(sequences, ecotype)

        # 4. Heaps
        heaps_beta = test_heaps(labels, ecotype)

        # 5. MI decay
        test_mi_decay(sequences, nc, ecotype)

        # 6. Bout structure
        analyse_bouts(sequences, ecotype)

    # ── Summary table ──
    print("\n" + "▓" * 70)
    print("▓  LINGUISTIC UNIVERSALS SUMMARY")
    print("▓" * 70)
    print()
    print(f"  {'Law':>20s}  {'Human language':>20s}  {'Test':>30s}")
    print(f"  {'─'*20}  {'─'*20}  {'─'*30}")
    print(f"  {'Zipf':>20s}  {'α ≈ 1.0':>20s}  {'Rank-frequency power law'}")
    print(f"  {'Brevity':>20s}  {'r < 0':>20s}  {'Frequent signals shorter'}")
    print(f"  {'Menzerath':>20s}  {'r < 0':>20s}  {'Longer seqs, shorter calls'}")
    print(f"  {'Heaps':>20s}  {'β ≈ 0.4-0.6':>20s}  {'Sublinear vocab growth'}")
    print(f"  {'MI decay':>20s}  {'half-life 3-8':>20s}  {'Sequential memory span'}")
    print()


if __name__ == "__main__":
    main()
