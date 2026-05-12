#!/usr/bin/env python3
"""
Deep analysis: the findings that blow tits off.

1. Full-scale syntax on 14,240 SRKW calls (bigrams + trigrams)
2. Temporal evolution 2005-2023 (is the language changing?)
3. Information theory (entropy, mutual information, cross-ecotype)
4. Higher-order transition structure (Markov order estimation)
"""

import os
import sys
import csv
import math
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))


def load_annotations():
    with open("data/dclde/Annotations.csv", 'r') as f:
        return list(csv.DictReader(f))


def cluster_by_frequency(calls, n_clusters=3):
    """Cluster calls using frequency metadata (no audio needed)."""
    from sklearn.cluster import KMeans

    features = []
    valid_indices = []
    for i, c in enumerate(calls):
        try:
            lo = float(c['LowFreqHz']) if c['LowFreqHz'] != 'NA' else None
            hi = float(c['HighFreqHz']) if c['HighFreqHz'] != 'NA' else None
            dur = float(c['FileEndSec']) - float(c['FileBeginSec'])
            if lo and hi and lo > 0 and hi > 0 and dur > 0.05:
                features.append([
                    (lo + hi) / 2 / 10000,  # center freq, normalised
                    (hi - lo) / 10000,       # bandwidth, normalised
                    min(dur, 10) / 10,       # duration, capped and normalised
                ])
                valid_indices.append(i)
        except (ValueError, KeyError):
            pass

    X = np.array(features)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # Map back to full call list
    call_labels = {}
    for idx, label in zip(valid_indices, labels):
        call_labels[idx] = label

    return call_labels, X, km


def build_sequences(calls, call_labels, max_gap=30.0):
    """Build temporal sequences from annotated calls within recording sessions."""
    by_file = defaultdict(list)
    for i, c in enumerate(calls):
        if i not in call_labels:
            continue
        try:
            begin = float(c['FileBeginSec'])
            by_file[c['Soundfile']].append((begin, i, call_labels[i]))
        except (ValueError, KeyError):
            pass

    sequences = []
    for fname, entries in by_file.items():
        sorted_entries = sorted(entries, key=lambda x: x[0])
        # Split into sub-sequences by gaps
        seq = [sorted_entries[0]]
        for j in range(1, len(sorted_entries)):
            gap = sorted_entries[j][0] - sorted_entries[j-1][0]
            if gap <= max_gap:
                seq.append(sorted_entries[j])
            else:
                if len(seq) >= 2:
                    sequences.append([(label, begin) for begin, idx, label in seq])
                seq = [sorted_entries[j]]
        if len(seq) >= 2:
            sequences.append([(label, begin) for begin, idx, label in seq])

    return sequences


# ─────────────────────────────────────────────────────────────────────
# 1. FULL-SCALE SYNTAX
# ─────────────────────────────────────────────────────────────────────

def analyse_syntax(calls, call_labels, n_clusters=3):
    print("=" * 70)
    print("  1. FULL-SCALE SYNTAX: 14,240 SRKW calls")
    print("=" * 70)
    print()

    sequences = build_sequences(calls, call_labels)
    total_calls = sum(len(s) for s in sequences)
    print(f"  Sequences: {len(sequences)}")
    print(f"  Calls in sequences: {total_calls}")
    print(f"  Mean sequence length: {total_calls / len(sequences):.1f}")

    # ── Bigram analysis ──
    bigram_counts = np.zeros((n_clusters, n_clusters), dtype=int)
    for seq in sequences:
        for j in range(len(seq) - 1):
            bigram_counts[seq[j][0], seq[j+1][0]] += 1

    total_bigrams = bigram_counts.sum()
    print(f"\n  ─── Bigram Analysis ({total_bigrams} transitions) ───")

    # Transition probabilities
    print(f"\n  Transition probabilities:")
    header = "        " + "  ".join(f"→C{j:d}   " for j in range(n_clusters))
    print(header)
    for i in range(n_clusters):
        row_total = bigram_counts[i].sum()
        if row_total > 0:
            probs = bigram_counts[i] / row_total
            row = "  ".join(f"{probs[j]:.4f}" for j in range(n_clusters))
            print(f"    C{i} → {row}   (n={row_total})")

    # Chi-squared
    cluster_sizes = np.array([(np.array(list(call_labels.values())) == c).sum()
                               for c in range(n_clusters)])
    expected_probs = cluster_sizes / cluster_sizes.sum()

    chi2 = 0
    for i in range(n_clusters):
        row_total = bigram_counts[i].sum()
        for j in range(n_clusters):
            expected = row_total * expected_probs[j]
            if expected > 0:
                chi2 += (bigram_counts[i, j] - expected) ** 2 / expected

    df = (n_clusters - 1) ** 2
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df)

    print(f"\n  Chi-squared: {chi2:.1f} (df={df})")
    print(f"  p-value: {p_value:.2e}")
    if p_value < 1e-10:
        print(f"  *** OVERWHELMINGLY SIGNIFICANT (p < 10⁻¹⁰)")

    self_rate = sum(bigram_counts[i, i] for i in range(n_clusters)) / total_bigrams
    print(f"  Self-transition rate: {self_rate:.4f} (expected: {sum(expected_probs**2):.4f})")

    # ── Trigram analysis ──
    trigram_counts = Counter()
    for seq in sequences:
        for j in range(len(seq) - 2):
            trigram = (seq[j][0], seq[j+1][0], seq[j+2][0])
            trigram_counts[trigram] += 1

    total_trigrams = sum(trigram_counts.values())
    print(f"\n  ─── Trigram Analysis ({total_trigrams} transitions) ───")

    # Top 10 trigrams
    print(f"\n  Top 15 trigrams:")
    for (a, b, c), count in trigram_counts.most_common(15):
        freq = count / total_trigrams
        print(f"    C{a}→C{b}→C{c}: {count:>5d} ({freq:.4f})")

    # Test: does knowing the previous TWO calls help predict the next?
    # Compare bigram entropy vs trigram entropy
    bigram_entropy = 0
    for i in range(n_clusters):
        row_total = bigram_counts[i].sum()
        if row_total == 0:
            continue
        for j in range(n_clusters):
            if bigram_counts[i, j] > 0:
                p = bigram_counts[i, j] / row_total
                bigram_entropy -= (row_total / total_bigrams) * p * math.log2(p)

    # Conditional entropy given previous TWO calls
    trigram_contexts = defaultdict(lambda: Counter())
    for (a, b, c), count in trigram_counts.items():
        trigram_contexts[(a, b)][c] += count

    trigram_entropy = 0
    for (a, b), next_counts in trigram_contexts.items():
        context_total = sum(next_counts.values())
        for c, count in next_counts.items():
            if count > 0:
                p = count / context_total
                trigram_entropy -= (context_total / total_trigrams) * p * math.log2(p)

    print(f"\n  Entropy analysis:")
    print(f"    H(next | previous 1): {bigram_entropy:.4f} bits")
    print(f"    H(next | previous 2): {trigram_entropy:.4f} bits")
    reduction = bigram_entropy - trigram_entropy
    pct = reduction / bigram_entropy * 100 if bigram_entropy > 0 else 0
    print(f"    Reduction: {reduction:.4f} bits ({pct:.1f}%)")
    if pct > 5:
        print(f"    *** Second-order structure detected: knowing TWO previous calls")
        print(f"    *** reduces uncertainty by {pct:.1f}% beyond knowing just one")
        print(f"    *** This is evidence of HIGHER-ORDER SYNTAX")
    print()

    return sequences, bigram_counts


# ─────────────────────────────────────────────────────────────────────
# 2. TEMPORAL EVOLUTION
# ─────────────────────────────────────────────────────────────────────

def analyse_temporal_evolution(calls, call_labels):
    print("=" * 70)
    print("  2. TEMPORAL EVOLUTION: Is the language changing? (2005-2023)")
    print("=" * 70)
    print()

    # Extract year from UTC timestamp
    by_year = defaultdict(lambda: Counter())
    for i, c in enumerate(calls):
        if i not in call_labels:
            continue
        try:
            utc = c['UTC']
            year = int(utc[:4])
            if 2000 <= year <= 2025:
                by_year[year][call_labels[i]] += 1
        except (ValueError, IndexError):
            pass

    years = sorted(by_year.keys())
    n_clusters = max(max(counts.keys()) for counts in by_year.values()) + 1

    print(f"  Years with data: {min(years)}-{max(years)} ({len(years)} years)")
    print(f"\n  Cluster distribution by year:")
    print(f"  {'Year':>6s}  {'Total':>6s}  " + "  ".join(f"C{i}%" for i in range(n_clusters)))
    print(f"  {'─'*6}  {'─'*6}  " + "  ".join("───" for _ in range(n_clusters)))

    year_distributions = []
    for year in years:
        total = sum(by_year[year].values())
        if total < 10:
            continue
        dist = [by_year[year].get(c, 0) / total for c in range(n_clusters)]
        year_distributions.append((year, total, dist))
        pcts = "  ".join(f"{d*100:5.1f}" for d in dist)
        print(f"  {year:>6d}  {total:>6d}  {pcts}")

    # Test for temporal trend: correlation between year and cluster proportions
    if len(year_distributions) >= 5:
        print(f"\n  Temporal trends (Pearson r with year):")
        for c in range(n_clusters):
            years_arr = np.array([yd[0] for yd in year_distributions])
            props_arr = np.array([yd[2][c] for yd in year_distributions])
            if props_arr.std() > 0.001:
                r = np.corrcoef(years_arr, props_arr)[0, 1]
                # Significance test
                n = len(years_arr)
                if abs(r) > 0 and n > 3:
                    t_stat = r * math.sqrt((n - 2) / (1 - r**2 + 1e-12))
                    from scipy.stats import t as t_dist
                    p = 2 * (1 - t_dist.cdf(abs(t_stat), n - 2))
                else:
                    p = 1.0
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                direction = "increasing" if r > 0 else "decreasing"
                print(f"    C{c}: r={r:+.4f} (p={p:.4f}) {sig} {direction if sig else ''}")

    # Jensen-Shannon divergence between early and late periods
    if len(year_distributions) >= 4:
        mid = len(year_distributions) // 2
        early = year_distributions[:mid]
        late = year_distributions[mid:]

        early_dist = np.zeros(n_clusters)
        late_dist = np.zeros(n_clusters)
        for _, total, dist in early:
            early_dist += np.array(dist) * total
        for _, total, dist in late:
            late_dist += np.array(dist) * total
        early_dist /= early_dist.sum()
        late_dist /= late_dist.sum()

        # JSD
        m = (early_dist + late_dist) / 2
        jsd = 0
        for i in range(n_clusters):
            if early_dist[i] > 0:
                jsd += 0.5 * early_dist[i] * math.log2(early_dist[i] / (m[i] + 1e-12))
            if late_dist[i] > 0:
                jsd += 0.5 * late_dist[i] * math.log2(late_dist[i] / (m[i] + 1e-12))

        early_years = f"{early[0][0]}-{early[-1][0]}"
        late_years = f"{late[0][0]}-{late[-1][0]}"
        print(f"\n  Jensen-Shannon divergence ({early_years} vs {late_years}): {jsd:.6f}")
        if jsd > 0.1:
            print(f"  *** SIGNIFICANT SHIFT: The acoustic distribution has changed over time")
        elif jsd > 0.01:
            print(f"  * Moderate drift: some change in acoustic distribution")
        else:
            print(f"  No significant drift: acoustic distribution is stable")
    print()


# ─────────────────────────────────────────────────────────────────────
# 3. INFORMATION THEORY
# ─────────────────────────────────────────────────────────────────────

def analyse_information_theory(all_rows, n_clusters=3):
    print("=" * 70)
    print("  3. INFORMATION THEORY: Entropy across ecotypes")
    print("=" * 70)
    print()

    ecotypes_to_analyse = ['SRKW', 'TKW', 'NRKW', 'SAR', 'OKW']

    print(f"  {'Ecotype':>8s}  {'Calls':>7s}  {'Seqs':>5s}  {'H(call)':>8s}  {'H(next|prev)':>13s}  {'MI':>8s}  {'MI/H':>7s}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*5}  {'─'*8}  {'─'*13}  {'─'*8}  {'─'*7}")

    results = {}

    for eco in ecotypes_to_analyse:
        eco_calls = [r for r in all_rows
                     if r['KW'] == '1' and r['AnnotationLevel'] == 'Call'
                     and r['Ecotype'] == eco]
        if len(eco_calls) < 50:
            continue

        labels, X, km = cluster_by_frequency(eco_calls, n_clusters)
        sequences = build_sequences(eco_calls, labels)

        if not sequences:
            continue

        # Marginal entropy H(call type)
        cluster_counts = Counter(labels.values())
        total = sum(cluster_counts.values())
        marginal_entropy = -sum(
            (c / total) * math.log2(c / total)
            for c in cluster_counts.values() if c > 0
        )

        # Bigram transition matrix
        bigram_counts = np.zeros((n_clusters, n_clusters), dtype=int)
        for seq in sequences:
            for j in range(len(seq) - 1):
                bigram_counts[seq[j][0], seq[j+1][0]] += 1

        total_bigrams = bigram_counts.sum()
        if total_bigrams == 0:
            continue

        # Conditional entropy H(next | previous)
        cond_entropy = 0
        for i in range(n_clusters):
            row_total = bigram_counts[i].sum()
            if row_total == 0:
                continue
            for j in range(n_clusters):
                if bigram_counts[i, j] > 0:
                    p = bigram_counts[i, j] / row_total
                    cond_entropy -= (row_total / total_bigrams) * p * math.log2(p)

        # Mutual information MI = H(next) - H(next | previous)
        mi = marginal_entropy - cond_entropy
        mi_ratio = mi / marginal_entropy if marginal_entropy > 0 else 0

        n_seqs = len(sequences)
        print(f"  {eco:>8s}  {len(eco_calls):>7d}  {n_seqs:>5d}  {marginal_entropy:>8.4f}  {cond_entropy:>13.4f}  {mi:>8.4f}  {mi_ratio:>7.1%}")

        results[eco] = {
            'n_calls': len(eco_calls),
            'n_sequences': n_seqs,
            'marginal_entropy': marginal_entropy,
            'conditional_entropy': cond_entropy,
            'mutual_information': mi,
            'mi_ratio': mi_ratio,
        }

    # Analysis
    if len(results) >= 2:
        print(f"\n  Interpretation:")
        print(f"    H(call): marginal entropy — how diverse the call repertoire is")
        print(f"    H(next|prev): conditional entropy — uncertainty about next call given previous")
        print(f"    MI: mutual information — how much the previous call tells you about the next")
        print(f"    MI/H: normalised MI — fraction of total information that's sequential")
        print()

        # Which ecotype has the most sequential structure?
        by_mi = sorted(results.items(), key=lambda x: -x[1]['mi_ratio'])
        print(f"  Ranked by sequential structure (MI/H):")
        for eco, r in by_mi:
            print(f"    {eco}: {r['mi_ratio']:.1%} of information is sequential")

        top = by_mi[0]
        print(f"\n  *** {top[0]} has the strongest sequential structure")
        print(f"  *** {top[1]['mi_ratio']:.1%} of call information depends on what came before")
    print()

    return results


# ─────────────────────────────────────────────────────────────────────
# 4. MARKOV ORDER ESTIMATION
# ─────────────────────────────────────────────────────────────────────

def analyse_markov_order(calls, call_labels, n_clusters=3):
    print("=" * 70)
    print("  4. MARKOV ORDER: How deep is the syntax?")
    print("=" * 70)
    print()

    sequences = build_sequences(calls, call_labels)
    flat = []
    for seq in sequences:
        flat.extend([label for label, _ in seq])

    if len(flat) < 100:
        print("  Insufficient data")
        return

    # Compute conditional entropy at different orders
    print(f"  Total calls in sequences: {len(flat)}")
    print(f"\n  Conditional entropy by Markov order:")
    print(f"  {'Order':>7s}  {'H(next|context)':>15s}  {'Reduction':>10s}  {'% reduction':>12s}")
    print(f"  {'─'*7}  {'─'*15}  {'─'*10}  {'─'*12}")

    prev_entropy = None
    for order in range(0, 5):
        context_counts = defaultdict(lambda: Counter())
        for i in range(order, len(flat) - 1):
            context = tuple(flat[i - order:i + 1]) if order > 0 else (flat[i],)
            next_call = flat[i + 1]
            context_counts[context][next_call] += 1

        # Conditional entropy
        total = sum(sum(counts.values()) for counts in context_counts.values())
        entropy = 0
        for context, next_counts in context_counts.items():
            context_total = sum(next_counts.values())
            for c, count in next_counts.items():
                if count > 0:
                    p = count / context_total
                    entropy -= (context_total / total) * p * math.log2(p)

        if prev_entropy is not None:
            reduction = prev_entropy - entropy
            pct = reduction / prev_entropy * 100 if prev_entropy > 0 else 0
            print(f"  {order:>7d}  {entropy:>15.4f}  {reduction:>10.4f}  {pct:>11.1f}%")
        else:
            print(f"  {order:>7d}  {entropy:>15.4f}  {'─':>10s}  {'─':>12s}")

        prev_entropy = entropy

    print(f"\n  Interpretation:")
    print(f"    Order 0: H(next | current call)")
    print(f"    Order 1: H(next | current + previous)")
    print(f"    Order 2: H(next | current + previous 2)")
    print(f"    If entropy keeps decreasing, the syntax has depth > that order.")
    print(f"    When reduction plateaus, you've found the Markov order of the language.")
    print()


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "▓" * 70)
    print("▓  ORCA-ENGINE: DEEP ANALYSIS")
    print("▓  Syntax, evolution, information theory, Markov order")
    print("▓" * 70)
    print()

    all_rows = load_annotations()
    print(f"  Total annotations: {len(all_rows)}")

    # Get SRKW calls
    srkw = [r for r in all_rows if r['KW'] == '1'
            and r['AnnotationLevel'] == 'Call' and r['Ecotype'] == 'SRKW']
    print(f"  SRKW call-level: {len(srkw)}")

    # Cluster
    labels, X, km = cluster_by_frequency(srkw)
    n_valid = len(labels)
    print(f"  Clusterable (with freq data): {n_valid}")
    print()

    # 1. Full-scale syntax
    sequences, bigrams = analyse_syntax(srkw, labels)

    # 2. Temporal evolution
    analyse_temporal_evolution(srkw, labels)

    # 3. Information theory across ecotypes
    info_results = analyse_information_theory(all_rows)

    # 4. Markov order
    analyse_markov_order(srkw, labels)

    # ── Summary ──
    print("▓" * 70)
    print("▓  DEEP ANALYSIS SUMMARY")
    print("▓" * 70)
    print()
    print(f"  Syntax: {bigrams.sum()} bigram transitions analysed")
    print(f"  Temporal: {len(set(int(c['UTC'][:4]) for c in srkw if len(c.get('UTC','')) >= 4 and c['UTC'][:4].isdigit()))} years of data")
    print(f"  Information theory: {len(info_results)} ecotypes compared")
    print(f"  Markov order: conditional entropy at orders 0-4")
    print()


if __name__ == "__main__":
    main()
