#!/usr/bin/env python3
"""
Vulnerability 4: Markov Order > 4 Needs Confidence Intervals

The "entropy keeps decreasing through order 4" claim is based on point
estimates. With k=3 clusters, order 4 means 81 context states on 11,079
transitions — many states are sparse. Entropy estimation on sparse
distributions is biased downward (looks like more structure than exists).

Three fixes:
1. Bootstrap CIs: resample sessions with replacement (1000 iterations).
2. Miller-Madow correction: H_corrected = H_naive + (k-1)/(2N ln2).
3. BIC model selection: fit Markov models at orders 1-5, select by BIC.
4. Repeat all at k=5 to test robustness under worse sparsity.
"""

import os
import sys
import csv
import math
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────────
# Data loading (from analyse_deep.py)
# ─────────────────────────────────────────────────────────────────────

def load_annotations():
    with open(os.path.join(os.path.dirname(__file__),
              "data", "dclde", "Annotations.csv"), 'r') as f:
        return list(csv.DictReader(f))


def cluster_by_frequency(calls, n_clusters=3):
    """Cluster calls using frequency metadata."""
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
                    (lo + hi) / 2 / 10000,
                    (hi - lo) / 10000,
                    min(dur, 10) / 10,
                ])
                valid_indices.append(i)
        except (ValueError, KeyError):
            pass

    X = np.array(features)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

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
        seq = [sorted_entries[0]]
        for j in range(1, len(sorted_entries)):
            gap = sorted_entries[j][0] - sorted_entries[j - 1][0]
            if gap <= max_gap:
                seq.append(sorted_entries[j])
            else:
                if len(seq) >= 2:
                    sequences.append([label for _, _, label in seq])
                seq = [sorted_entries[j]]
        if len(seq) >= 2:
            sequences.append([label for _, _, label in seq])

    return sequences


# ─────────────────────────────────────────────────────────────────────
# Conditional entropy computation
# ─────────────────────────────────────────────────────────────────────

def compute_conditional_entropy(flat, n_clusters, max_order=4):
    """
    Compute conditional entropy at orders 0..max_order.
    Returns list of (order, entropy, n_nonempty_contexts, n_transitions).
    """
    results = []
    for order in range(0, max_order + 1):
        context_counts = defaultdict(lambda: Counter())
        for i in range(order, len(flat) - 1):
            if order > 0:
                context = tuple(flat[i - order:i + 1])
            else:
                context = (flat[i],)
            next_call = flat[i + 1]
            context_counts[context][next_call] += 1

        total = sum(sum(counts.values()) for counts in context_counts.values())
        if total == 0:
            results.append((order, 0.0, 0, 0))
            continue

        entropy = 0
        for context, next_counts in context_counts.items():
            context_total = sum(next_counts.values())
            for c, count in next_counts.items():
                if count > 0:
                    p = count / context_total
                    entropy -= (context_total / total) * p * math.log2(p)

        n_nonempty = len(context_counts)
        results.append((order, entropy, n_nonempty, total))

    return results


# ─────────────────────────────────────────────────────────────────────
# Miller-Madow correction
# ─────────────────────────────────────────────────────────────────────

def miller_madow(naive_entropy, n_nonempty_states, n_observations):
    """H_corrected = H_naive + (k - 1) / (2 * N * ln(2))"""
    if n_observations == 0:
        return naive_entropy
    correction = (n_nonempty_states - 1) / (2 * n_observations * math.log(2))
    return naive_entropy + correction


# ─────────────────────────────────────────────────────────────────────
# BIC model selection
# ─────────────────────────────────────────────────────────────────────

def compute_bic(flat, n_clusters, order):
    """
    Fit Markov model of given order. Return BIC, logL, n_params, n_obs.
    Uses Laplace (add-1) smoothing for zero counts.
    """
    context_counts = defaultdict(lambda: Counter())
    for i in range(order, len(flat) - 1):
        context = tuple(flat[i - order:i + 1]) if order > 0 else (flat[i],)
        next_call = flat[i + 1]
        context_counts[context][next_call] += 1

    total = sum(sum(counts.values()) for counts in context_counts.values())
    if total == 0:
        return 0, 0, 0, 0

    # Log-likelihood with Laplace smoothing
    log_likelihood = 0
    for context, next_counts in context_counts.items():
        context_total = sum(next_counts.values())
        for c in range(n_clusters):
            count = next_counts.get(c, 0)
            # Laplace smoothing: (count + 1) / (context_total + n_clusters)
            p = (count + 1) / (context_total + n_clusters)
            if count > 0:
                log_likelihood += count * math.log(p)

    # Free parameters: each non-empty context has (n_clusters - 1) free probs
    n_nonempty = len(context_counts)
    n_params = n_nonempty * (n_clusters - 1)

    # BIC = -2 * logL + k * ln(N)
    bic = -2 * log_likelihood + n_params * math.log(total)

    return bic, log_likelihood, n_params, total


# ─────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────

def bootstrap_entropy(sequences, n_clusters, n_iterations=1000, max_order=4):
    """
    Resample SESSIONS with replacement. Compute conditional entropy
    at each order. Return distributions for CIs.
    """
    rng = np.random.RandomState(42)
    n_sessions = len(sequences)

    # Storage: [order][iteration] = entropy
    naive_samples = {order: np.zeros(n_iterations) for order in range(max_order + 1)}
    corrected_samples = {order: np.zeros(n_iterations) for order in range(max_order + 1)}
    decrease_3_to_4 = np.zeros(n_iterations)

    for it in range(n_iterations):
        # Resample sessions with replacement
        session_indices = rng.choice(n_sessions, size=n_sessions, replace=True)
        flat = []
        for idx in session_indices:
            flat.extend(sequences[idx])

        if len(flat) < 10:
            continue

        results = compute_conditional_entropy(flat, n_clusters, max_order)
        for order, entropy, n_nonempty, n_trans in results:
            naive_samples[order][it] = entropy
            corrected_samples[order][it] = miller_madow(entropy, n_nonempty, n_trans)

        # Decrease from order 3 to order 4
        if max_order >= 4:
            h3 = naive_samples[3][it]
            h4 = naive_samples[4][it]
            decrease_3_to_4[it] = h3 - h4

        if (it + 1) % 200 == 0:
            print(f"      ... {it + 1}/{n_iterations}")

    return naive_samples, corrected_samples, decrease_3_to_4


# ─────────────────────────────────────────────────────────────────────
# Full analysis for one cluster count
# ─────────────────────────────────────────────────────────────────────

def run_analysis(calls, n_clusters, label):
    """Run complete Markov order analysis at given k."""
    print(f"\n  ═══ {label} (k = {n_clusters}) ═══")

    labels, X, km = cluster_by_frequency(calls, n_clusters)
    sequences = build_sequences(calls, labels)
    flat = []
    for seq in sequences:
        flat.extend(seq)

    print(f"    Sequences: {len(sequences)}")
    print(f"    Calls in sequences: {len(flat)}")

    # ── Point estimates ──
    results = compute_conditional_entropy(flat, n_clusters, max_order=4)

    print(f"\n    Point estimates:")
    print(f"    {'Order':>7s}  {'H_naive':>10s}  {'H_corrected':>12s}  {'Contexts':>10s}  {'Transitions':>12s}  {'Decrease':>10s}")
    print(f"    {'─' * 7}  {'─' * 10}  {'─' * 12}  {'─' * 10}  {'─' * 12}  {'─' * 10}")

    prev_h = None
    for order, entropy, n_nonempty, n_trans in results:
        h_corrected = miller_madow(entropy, n_nonempty, n_trans)
        decrease = f"{prev_h - entropy:.4f}" if prev_h is not None else "─"
        print(f"    {order:>7d}  {entropy:>10.4f}  {h_corrected:>12.4f}  {n_nonempty:>10d}  {n_trans:>12d}  {decrease:>10s}")
        prev_h = entropy

    # ── BIC model selection ──
    print(f"\n    BIC model selection:")
    print(f"    {'Order':>7s}  {'BIC':>14s}  {'logL':>14s}  {'n_params':>10s}")
    print(f"    {'─' * 7}  {'─' * 14}  {'─' * 14}  {'─' * 10}")

    bic_results = []
    for order in range(1, 6):
        bic, logL, n_params, n_obs = compute_bic(flat, n_clusters, order)
        bic_results.append((order, bic, logL, n_params))
        print(f"    {order:>7d}  {bic:>14.1f}  {logL:>14.1f}  {n_params:>10d}")

    best_order = min(bic_results, key=lambda x: x[1])[0]
    print(f"    BIC-preferred order: {best_order}")

    # ── Bootstrap CIs ──
    print(f"\n    Bootstrap 95% CIs (1000 iterations, session-level resampling):")
    naive_samples, corrected_samples, decrease_3_4 = bootstrap_entropy(
        sequences, n_clusters, n_iterations=1000, max_order=4)

    print(f"\n    {'Order':>7s}  {'H_naive [95% CI]':>30s}  {'H_corrected [95% CI]':>30s}")
    print(f"    {'─' * 7}  {'─' * 30}  {'─' * 30}")

    for order in range(5):
        n_mean = naive_samples[order].mean()
        n_lo = np.percentile(naive_samples[order], 2.5)
        n_hi = np.percentile(naive_samples[order], 97.5)
        c_mean = corrected_samples[order].mean()
        c_lo = np.percentile(corrected_samples[order], 2.5)
        c_hi = np.percentile(corrected_samples[order], 97.5)
        print(f"    {order:>7d}  {n_mean:.4f} [{n_lo:.4f}, {n_hi:.4f}]      {c_mean:.4f} [{c_lo:.4f}, {c_hi:.4f}]")

    # Key test: does decrease from order 3 to 4 include zero?
    d_mean = decrease_3_4.mean()
    d_lo = np.percentile(decrease_3_4, 2.5)
    d_hi = np.percentile(decrease_3_4, 97.5)

    print(f"\n    Entropy decrease from order 3 to 4:")
    print(f"      Point estimate: {d_mean:.4f} bits")
    print(f"      Bootstrap 95% CI: [{d_lo:.4f}, {d_hi:.4f}]")

    if d_lo > 0:
        print(f"      *** CI EXCLUDES zero — decrease from order 3 to 4 is REAL")
        print(f"      *** Markov order > 4 claim SUPPORTED")
    else:
        print(f"      CI INCLUDES zero — decrease from order 3 to 4 may be noise")
        print(f"      Markov order > 4 claim needs softening")

    return {
        'best_bic_order': best_order,
        'decrease_3_4_mean': d_mean,
        'decrease_3_4_ci': (d_lo, d_hi),
        'ci_excludes_zero': d_lo > 0,
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  VULNERABILITY 4: Markov Order Confidence Intervals")
    print("  Is the order > 4 claim robust?")
    print("=" * 70)

    all_rows = load_annotations()
    srkw = [r for r in all_rows if r['KW'] == '1'
            and r['AnnotationLevel'] == 'Call' and r['Ecotype'] == 'SRKW']
    print(f"\n  Total SRKW call-level annotations: {len(srkw)}")

    # Run at k=3 (original)
    result_k3 = run_analysis(srkw, n_clusters=3, label="Original analysis")

    # Run at k=5 (robustness check — 5^4 = 625 contexts)
    result_k5 = run_analysis(srkw, n_clusters=5, label="Robustness check")

    # ── Summary ──
    print(f"\n  {'═' * 60}")
    print(f"  SUMMARY")
    print(f"  {'═' * 60}")

    print(f"\n  {'k':>5s}  {'BIC order':>10s}  {'Decrease 3→4':>14s}  {'95% CI':>20s}  {'Robust?':>10s}")
    print(f"  {'─' * 5}  {'─' * 10}  {'─' * 14}  {'─' * 20}  {'─' * 10}")

    for k, r in [('k=3', result_k3), ('k=5', result_k5)]:
        ci_str = f"[{r['decrease_3_4_ci'][0]:.4f}, {r['decrease_3_4_ci'][1]:.4f}]"
        robust = "YES" if r['ci_excludes_zero'] else "NO"
        print(f"  {k:>5s}  {r['best_bic_order']:>10d}  {r['decrease_3_4_mean']:>14.4f}  {ci_str:>20s}  {robust:>10s}")

    if result_k3['ci_excludes_zero'] and result_k5['ci_excludes_zero']:
        print(f"\n  *** BOTH k=3 and k=5: order 3→4 decrease is real")
        print(f"  *** Markov order > 4 claim is ROBUST")
    elif result_k3['ci_excludes_zero']:
        print(f"\n  k=3: claim supported. k=5: claim weakened by sparsity.")
        print(f"  Recommend: report as 'Markov order ≥ 4' with k=3, note k=5 result.")
    else:
        print(f"\n  WARNING: Decrease 3→4 CI includes zero at k=3")
        print(f"  Recommend: soften to 'Markov order ≥ 3'")

    print()


if __name__ == "__main__":
    main()
