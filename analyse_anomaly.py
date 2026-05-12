#!/usr/bin/env python3
"""
The 2019 anomaly + NRKW comparison.

Finding 11 showed a 7× spike in C2 usage for SRKW in 2019 — the worst
Chinook salmon shortage on record. This script digs into that anomaly
and compares SRKW with Northern Residents (NRKW) as a control group.

If SRKW acoustic behaviour shifted during a food crisis but NRKW didn't,
that links communication directly to ecological stress. Hydrophone data
becomes a welfare indicator.
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


def cluster_calls(calls, n_clusters=3):
    """Cluster calls by frequency metadata."""
    from sklearn.cluster import KMeans

    features = []
    valid = []
    for i, c in enumerate(calls):
        try:
            lo = float(c['LowFreqHz']) if c['LowFreqHz'] != 'NA' else None
            hi = float(c['HighFreqHz']) if c['HighFreqHz'] != 'NA' else None
            dur = float(c['FileEndSec']) - float(c['FileBeginSec'])
            if lo and hi and lo > 0 and hi > 0 and dur > 0.05:
                features.append([(lo+hi)/2/10000, (hi-lo)/10000, min(dur,10)/10])
                valid.append(i)
        except (ValueError, KeyError):
            pass

    if len(features) < n_clusters * 5:
        return {}, np.array([]), None

    X = np.array(features)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return {valid[i]: labels[i] for i in range(len(valid))}, X, km


def get_year(call):
    try:
        return int(call['UTC'][:4])
    except (ValueError, IndexError, KeyError):
        return None


def build_sequences(calls, labels, max_gap=30.0):
    by_file = defaultdict(list)
    for i, c in enumerate(calls):
        if i not in labels:
            continue
        try:
            by_file[c['Soundfile']].append((float(c['FileBeginSec']), i, labels[i]))
        except (ValueError, KeyError):
            pass

    sequences = []
    for fname, entries in by_file.items():
        sorted_e = sorted(entries, key=lambda x: x[0])
        seq = [sorted_e[0]]
        for j in range(1, len(sorted_e)):
            if sorted_e[j][0] - sorted_e[j-1][0] <= max_gap:
                seq.append(sorted_e[j])
            else:
                if len(seq) >= 2:
                    sequences.append([(lab, begin) for begin, idx, lab in seq])
                seq = [sorted_e[j]]
        if len(seq) >= 2:
            sequences.append([(lab, begin) for begin, idx, lab in seq])
    return sequences


# ─────────────────────────────────────────────────────────────────────
# THE 2019 ANOMALY
# ─────────────────────────────────────────────────────────────────────

def analyse_2019_anomaly(all_rows):
    print("=" * 70)
    print("  THE 2019 ANOMALY: Did SRKW change their calls during famine?")
    print("=" * 70)
    print()

    srkw = [r for r in all_rows if r['KW'] == '1'
            and r['AnnotationLevel'] == 'Call' and r['Ecotype'] == 'SRKW']

    labels, X, km = cluster_calls(srkw)
    if not labels:
        print("  Insufficient data")
        return

    n_clusters = km.n_clusters

    # Per-year analysis
    by_year = defaultdict(lambda: {'labels': Counter(), 'calls': [],
                                    'datasets': Counter(), 'durations': []})
    for i, c in enumerate(srkw):
        if i not in labels:
            continue
        year = get_year(c)
        if year and 2005 <= year <= 2025:
            by_year[year]['labels'][labels[i]] += 1
            by_year[year]['datasets'][c['Dataset']] += 1
            try:
                dur = float(c['FileEndSec']) - float(c['FileBeginSec'])
                by_year[year]['durations'].append(dur)
            except:
                pass
            try:
                by_year[year]['calls'].append({
                    'label': labels[i],
                    'lo': float(c['LowFreqHz']) if c['LowFreqHz'] != 'NA' else None,
                    'hi': float(c['HighFreqHz']) if c['HighFreqHz'] != 'NA' else None,
                    'dur': dur,
                })
            except:
                pass

    years = sorted(by_year.keys())

    # ── Cluster distribution by year ──
    print(f"  Year-by-year cluster distribution:")
    print(f"  {'Year':>6s}  {'N':>6s}  " + "  ".join(f"{'C'+str(i)+'%':>6s}" for i in range(n_clusters)) + "  {'Sources':>8s}  {'MeanDur':>8s}")
    print(f"  {'─'*6}  {'─'*6}  " + "  ".join('─'*6 for _ in range(n_clusters)) + f"  {'─'*8}  {'─'*8}")

    year_data = []
    for year in years:
        yd = by_year[year]
        total = sum(yd['labels'].values())
        if total < 10:
            continue
        dist = [yd['labels'].get(c, 0) / total for c in range(n_clusters)]
        n_sources = len(yd['datasets'])
        mean_dur = np.mean(yd['durations']) if yd['durations'] else 0
        year_data.append((year, total, dist, n_sources, mean_dur))

        pcts = "  ".join(f"{d*100:6.1f}" for d in dist)
        marker = "  ◄◄◄ ANOMALY" if year == 2019 else ""
        print(f"  {year:>6d}  {total:>6d}  {pcts}  {n_sources:>8d}  {mean_dur:>8.2f}s{marker}")

    # ── Deep dive on 2019 ──
    print(f"\n  ═══ 2019 DEEP DIVE ═══")
    print()

    y2019 = by_year.get(2019)
    if not y2019 or sum(y2019['labels'].values()) < 10:
        print("  Insufficient 2019 data")
        return

    total_2019 = sum(y2019['labels'].values())
    print(f"  2019 calls: {total_2019}")
    print(f"  2019 recording sources: {dict(y2019['datasets'].most_common())}")

    # Compare 2019 acoustic properties vs other years
    calls_2019 = [c for c in y2019['calls'] if c['lo'] is not None]
    calls_other = []
    for year in years:
        if year == 2019:
            continue
        calls_other.extend([c for c in by_year[year]['calls'] if c['lo'] is not None])

    if calls_2019 and calls_other:
        centers_2019 = np.array([(c['lo'] + c['hi']) / 2 for c in calls_2019])
        bws_2019 = np.array([c['hi'] - c['lo'] for c in calls_2019])
        durs_2019 = np.array([c['dur'] for c in calls_2019])

        centers_other = np.array([(c['lo'] + c['hi']) / 2 for c in calls_other])
        bws_other = np.array([c['hi'] - c['lo'] for c in calls_other])
        durs_other = np.array([c['dur'] for c in calls_other])

        print(f"\n  Acoustic comparison (2019 vs all other years):")
        print(f"  {'Metric':>20s}  {'2019':>15s}  {'Other years':>15s}  {'Diff':>8s}")
        print(f"  {'─'*20}  {'─'*15}  {'─'*15}  {'─'*8}")

        for name, a, b in [
            ("Center freq (Hz)", centers_2019, centers_other),
            ("Bandwidth (Hz)", bws_2019, bws_other),
            ("Duration (s)", durs_2019, durs_other),
        ]:
            diff_pct = (a.mean() - b.mean()) / (b.mean() + 1e-12) * 100
            print(f"  {name:>20s}  {a.mean():>9.0f}±{a.std():>4.0f}  {b.mean():>9.0f}±{b.std():>4.0f}  {diff_pct:>+7.1f}%")

        # Statistical test: are 2019 calls different?
        from scipy.stats import mannwhitneyu

        print(f"\n  Mann-Whitney U tests (2019 vs other years):")
        for name, a, b in [
            ("Center frequency", centers_2019, centers_other),
            ("Bandwidth", bws_2019, bws_other),
            ("Duration", durs_2019, durs_other),
        ]:
            if len(a) >= 5 and len(b) >= 5:
                U, p = mannwhitneyu(a, b, alternative='two-sided')
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                print(f"    {name:>20s}: U={U:.0f}, p={p:.2e} {sig}")

    # ── 2019 transition analysis ──
    print(f"\n  2019 transition analysis:")
    srkw_2019 = [r for r in srkw if get_year(r) == 2019]
    labels_2019 = {i: labels[i] for i in range(len(srkw)) if i in labels and get_year(srkw[i]) == 2019}
    seqs_2019 = build_sequences(srkw_2019, {i - min(labels_2019.keys()): v for i, v in labels_2019.items()
                                             } if labels_2019 else {})

    # Simpler: rebuild from scratch for 2019
    labels_2019_fresh, _, _ = cluster_calls(srkw_2019)
    seqs_2019 = build_sequences(srkw_2019, labels_2019_fresh)

    if seqs_2019:
        bigrams_2019 = np.zeros((n_clusters, n_clusters), dtype=int)
        for seq in seqs_2019:
            for j in range(len(seq) - 1):
                bigrams_2019[seq[j][0], seq[j+1][0]] += 1

        total_bi = bigrams_2019.sum()
        if total_bi > 0:
            print(f"    Sequences: {len(seqs_2019)}, transitions: {total_bi}")
            self_rate = sum(bigrams_2019[i, i] for i in range(n_clusters)) / total_bi
            print(f"    Self-transition rate: {self_rate:.4f}")

            # Compare with overall self-transition rate
            srkw_all_labels, _, _ = cluster_calls(srkw)
            seqs_all = build_sequences(srkw, srkw_all_labels)
            bigrams_all = np.zeros((n_clusters, n_clusters), dtype=int)
            for seq in seqs_all:
                for j in range(len(seq) - 1):
                    bigrams_all[seq[j][0], seq[j+1][0]] += 1
            total_all = bigrams_all.sum()
            if total_all > 0:
                self_rate_all = sum(bigrams_all[i, i] for i in range(n_clusters)) / total_all
                print(f"    Overall self-transition rate: {self_rate_all:.4f}")
                diff = self_rate - self_rate_all
                print(f"    Difference: {diff:+.4f}")
                if abs(diff) > 0.05:
                    direction = "MORE repetitive" if diff > 0 else "LESS repetitive"
                    print(f"    *** 2019 sequences are {direction} than average")

    print()
    return by_year


# ─────────────────────────────────────────────────────────────────────
# NRKW COMPARISON
# ─────────────────────────────────────────────────────────────────────

def analyse_nrkw_comparison(all_rows):
    print("=" * 70)
    print("  NRKW vs SRKW: The control group")
    print("=" * 70)
    print()

    srkw = [r for r in all_rows if r['KW'] == '1'
            and r['AnnotationLevel'] == 'Call' and r['Ecotype'] == 'SRKW']
    nrkw = [r for r in all_rows if r['KW'] == '1'
            and r['AnnotationLevel'] == 'Call' and r['Ecotype'] == 'NRKW']

    print(f"  SRKW calls: {len(srkw)}")
    print(f"  NRKW calls: {len(nrkw)}")

    if len(nrkw) < 50:
        print("  Insufficient NRKW data")
        return

    # ── Basic comparison ──
    for name, calls in [("SRKW", srkw), ("NRKW", nrkw)]:
        valid = []
        for c in calls:
            try:
                lo = float(c['LowFreqHz']) if c['LowFreqHz'] != 'NA' else None
                hi = float(c['HighFreqHz']) if c['HighFreqHz'] != 'NA' else None
                dur = float(c['FileEndSec']) - float(c['FileBeginSec'])
                if lo and hi and lo > 0 and hi > 0:
                    valid.append({'center': (lo+hi)/2, 'bw': hi-lo, 'dur': dur})
            except:
                pass

        if valid:
            centers = np.array([v['center'] for v in valid])
            bws = np.array([v['bw'] for v in valid])
            durs = np.array([v['dur'] for v in valid])
            print(f"\n  {name} ({len(valid)} with freq data):")
            print(f"    Center freq: {centers.mean():.0f}±{centers.std():.0f} Hz")
            print(f"    Bandwidth:   {bws.mean():.0f}±{bws.std():.0f} Hz")
            print(f"    Duration:    {durs.mean():.2f}±{durs.std():.2f}s")

    # ── Clustering comparison ──
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    print(f"\n  ═══ CLUSTERING COMPARISON ═══")
    for name, calls in [("SRKW", srkw), ("NRKW", nrkw)]:
        labels, X, km = cluster_calls(calls)
        if not labels:
            continue

        best_k, best_sil = 2, -1
        for k in range(2, min(12, len(X) // 10)):
            km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
            lab = km_test.fit_predict(X)
            sil = silhouette_score(X, lab, sample_size=min(2000, len(X)))
            if sil > best_sil:
                best_sil = sil
                best_k = k
        print(f"\n  {name}: best k={best_k} (silhouette={best_sil:.4f}, n={len(X)})")

    # ── Information theory comparison ──
    print(f"\n  ═══ INFORMATION THEORY ═══")

    for name, calls in [("SRKW", srkw), ("NRKW", nrkw)]:
        labels, X, km = cluster_calls(calls, n_clusters=3)
        if not labels:
            continue

        sequences = build_sequences(calls, labels)
        if not sequences:
            continue

        # Marginal entropy
        cluster_counts = Counter(labels.values())
        total = sum(cluster_counts.values())
        H_marginal = -sum((c/total) * math.log2(c/total) for c in cluster_counts.values() if c > 0)

        # Bigram transitions
        n_c = km.n_clusters
        bigrams = np.zeros((n_c, n_c), dtype=int)
        for seq in sequences:
            for j in range(len(seq) - 1):
                bigrams[seq[j][0], seq[j+1][0]] += 1

        total_bi = bigrams.sum()
        if total_bi == 0:
            continue

        H_cond = 0
        for i in range(n_c):
            rt = bigrams[i].sum()
            if rt == 0:
                continue
            for j in range(n_c):
                if bigrams[i, j] > 0:
                    p = bigrams[i, j] / rt
                    H_cond -= (rt / total_bi) * p * math.log2(p)

        MI = H_marginal - H_cond
        MI_ratio = MI / H_marginal if H_marginal > 0 else 0

        # Self-transition
        self_rate = sum(bigrams[i, i] for i in range(n_c)) / total_bi

        print(f"\n  {name}:")
        print(f"    H(call):      {H_marginal:.4f} bits")
        print(f"    H(next|prev): {H_cond:.4f} bits")
        print(f"    MI:           {MI:.4f} bits")
        print(f"    MI/H:         {MI_ratio:.1%}")
        print(f"    Self-trans:   {self_rate:.4f}")
        print(f"    Sequences:    {len(sequences)}")
        print(f"    Transitions:  {total_bi}")

    # ── NRKW temporal analysis (does 2019 show anomaly?) ──
    print(f"\n  ═══ NRKW TEMPORAL: Does 2019 show an anomaly? ═══")

    nrkw_labels, _, _ = cluster_calls(nrkw)
    n_clusters = 3

    by_year = defaultdict(Counter)
    for i, c in enumerate(nrkw):
        if i not in nrkw_labels:
            continue
        year = get_year(c)
        if year and 2005 <= year <= 2025:
            by_year[year][nrkw_labels[i]] += 1

    years = sorted(by_year.keys())
    print(f"\n  NRKW cluster distribution by year:")
    print(f"  {'Year':>6s}  {'N':>6s}  " + "  ".join(f"C{i}%" for i in range(n_clusters)))

    for year in years:
        total = sum(by_year[year].values())
        if total < 5:
            continue
        dist = [by_year[year].get(c, 0) / total for c in range(n_clusters)]
        pcts = "  ".join(f"{d*100:5.1f}" for d in dist)
        marker = "  ◄ 2019" if year == 2019 else ""
        print(f"  {year:>6d}  {total:>6d}  {pcts}{marker}")

    # ── Markov order comparison ──
    print(f"\n  ═══ MARKOV ORDER COMPARISON ═══")

    for name, calls in [("SRKW", srkw), ("NRKW", nrkw)]:
        labels, X, km = cluster_calls(calls, n_clusters=3)
        if not labels:
            continue

        sequences = build_sequences(calls, labels)
        flat = []
        for seq in sequences:
            flat.extend([lab for lab, _ in seq])

        if len(flat) < 50:
            continue

        print(f"\n  {name} Markov order analysis ({len(flat)} calls):")
        prev_H = None
        for order in range(0, 5):
            contexts = defaultdict(Counter)
            for i in range(order, len(flat) - 1):
                ctx = tuple(flat[i-order:i+1]) if order > 0 else (flat[i],)
                contexts[ctx][flat[i+1]] += 1

            total = sum(sum(v.values()) for v in contexts.values())
            H = 0
            for ctx, counts in contexts.items():
                ct = sum(counts.values())
                for c in counts.values():
                    if c > 0:
                        p = c / ct
                        H -= (ct / total) * p * math.log2(p)

            if prev_H is not None:
                red = prev_H - H
                pct = red / prev_H * 100 if prev_H > 0 else 0
                print(f"    Order {order}: H={H:.4f} bits  (reduction: {red:.4f}, {pct:.1f}%)")
            else:
                print(f"    Order {order}: H={H:.4f} bits")
            prev_H = H

    # ── Cross-population Procrustes ──
    print(f"\n  ═══ SRKW ↔ NRKW PROCRUSTES ALIGNMENT ═══")

    srkw_labels, srkw_X, srkw_km = cluster_calls(srkw, n_clusters=3)
    nrkw_labels, nrkw_X, nrkw_km = cluster_calls(nrkw, n_clusters=3)

    if srkw_km is not None and nrkw_km is not None:
        from scipy.linalg import orthogonal_procrustes

        c1 = srkw_km.cluster_centers_
        c2 = nrkw_km.cluster_centers_
        n = min(len(c1), len(c2))

        # Greedy matching
        dists = np.array([[np.linalg.norm(c1[a] - c2[b]) for b in range(len(c2))]
                          for a in range(len(c1))])
        matched_1, matched_2 = [], []
        used_1, used_2 = set(), set()
        for d, a, b in sorted([(dists[a,b], a, b) for a in range(len(c1)) for b in range(len(c2))]):
            if a in used_1 or b in used_2:
                continue
            matched_1.append(a)
            matched_2.append(b)
            used_1.add(a)
            used_2.add(b)
            if len(matched_1) >= n:
                break

        S = c1[matched_1]
        T = c2[matched_2]

        try:
            R, scale = orthogonal_procrustes(S, T)
            aligned = S @ R
            disparity = float(np.linalg.norm(aligned - T) / (np.linalg.norm(T) + 1e-12))
        except:
            disparity = float('inf')

        print(f"\n  SRKW ↔ NRKW centroid alignment:")
        print(f"    Disparity: {disparity:.4f}")
        print(f"    Matched centroids: {n}")

        # Compare with SRKW ↔ TKW disparity for context
        tkw = [r for r in all_rows if r['KW'] == '1'
               and r['AnnotationLevel'] == 'Call' and r['Ecotype'] == 'TKW']
        tkw_labels, tkw_X, tkw_km = cluster_calls(tkw, n_clusters=3)
        if tkw_km is not None:
            c3 = tkw_km.cluster_centers_
            n2 = min(len(c1), len(c3))
            dists2 = np.array([[np.linalg.norm(c1[a] - c3[b]) for b in range(len(c3))]
                               for a in range(len(c1))])
            m1, m2 = [], []
            u1, u2 = set(), set()
            for d, a, b in sorted([(dists2[a,b], a, b) for a in range(len(c1)) for b in range(len(c3))]):
                if a in u1 or b in u2:
                    continue
                m1.append(a)
                m2.append(b)
                u1.add(a)
                u2.add(b)
                if len(m1) >= n2:
                    break
            try:
                R2, _ = orthogonal_procrustes(c1[m1], c3[m2])
                disp2 = float(np.linalg.norm(c1[m1] @ R2 - c3[m2]) / (np.linalg.norm(c3[m2]) + 1e-12))
            except:
                disp2 = float('inf')

            print(f"    (For comparison: SRKW ↔ TKW disparity: {disp2:.4f})")
            if disparity < disp2:
                print(f"    SRKW is closer to NRKW than to TKW — as expected for related populations")

    print()


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "▓" * 70)
    print("▓  ORCA-ENGINE: 2019 ANOMALY + NRKW COMPARISON")
    print("▓  Connecting communication to conservation")
    print("▓" * 70)
    print()

    all_rows = load_annotations()

    # The 2019 anomaly
    by_year = analyse_2019_anomaly(all_rows)

    # NRKW comparison
    analyse_nrkw_comparison(all_rows)

    # Summary
    print("▓" * 70)
    print("▓  SUMMARY")
    print("▓" * 70)
    print()
    print("  Two questions answered:")
    print("  1. Did SRKW change their calls during the 2018-2019 prey crisis?")
    print("  2. How does SRKW communication compare to NRKW (healthy control)?")
    print()


if __name__ == "__main__":
    main()
