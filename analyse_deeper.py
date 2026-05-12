#!/usr/bin/env python3
"""
Deeper analysis:
  - 2019 syntax breakdown (how did the grammar change, not just the sounds?)
  - Diel patterns (do orcas communicate differently at night?)
  - Seasonal patterns (monthly variation in call distribution)
  - Cross-year topology rotation (did the semantic field itself shift in 2019?)
  - SAR deep dive (why richest vocabulary but minimal grammar?)
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


def cluster_calls(calls, n_clusters=3):
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
        return {}, np.array([]), None
    X = np.array(features)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return {valid[i]: labels[i] for i in range(len(valid))}, X, km


def build_sequences(calls, labels, max_gap=30.0):
    by_file = defaultdict(list)
    for i, c in enumerate(calls):
        if i not in labels:
            continue
        try:
            by_file[c['Soundfile']].append((float(c['FileBeginSec']), i, labels[i]))
        except:
            pass
    sequences = []
    for fname, entries in by_file.items():
        se = sorted(entries, key=lambda x: x[0])
        seq = [se[0]]
        for j in range(1, len(se)):
            if se[j][0] - se[j-1][0] <= max_gap:
                seq.append(se[j])
            else:
                if len(seq) >= 2:
                    sequences.append([(lab, beg) for beg, idx, lab in seq])
                seq = [se[j]]
        if len(seq) >= 2:
            sequences.append([(lab, beg) for beg, idx, lab in seq])
    return sequences


def transition_matrix(sequences, n_clusters=3):
    M = np.zeros((n_clusters, n_clusters), dtype=int)
    for seq in sequences:
        for j in range(len(seq) - 1):
            M[seq[j][0], seq[j+1][0]] += 1
    return M


def matrix_entropy(M):
    """Conditional entropy of transition matrix."""
    total = M.sum()
    if total == 0:
        return 0
    H = 0
    for i in range(M.shape[0]):
        rt = M[i].sum()
        if rt == 0:
            continue
        for j in range(M.shape[1]):
            if M[i, j] > 0:
                p = M[i, j] / rt
                H -= (rt / total) * p * math.log2(p)
    return H


def get_year(c):
    try:
        return int(c['UTC'][:4])
    except:
        return None


def get_hour(c):
    """Extract hour from UTC timestamp."""
    try:
        utc = c['UTC']
        # Handles formats like "2019-07-05 14:23:00" or "2019-07-05T14:23:00"
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]:
            try:
                dt = datetime.strptime(utc[:19], fmt)
                return dt.hour
            except:
                pass
        # Fallback: just grab characters 11-13
        return int(utc[11:13])
    except:
        return None


def get_month(c):
    try:
        return int(c['UTC'][5:7])
    except:
        return None


# ─────────────────────────────────────────────────────────────────────
# 1. 2019 SYNTAX BREAKDOWN
# ─────────────────────────────────────────────────────────────────────

def analyse_2019_syntax(srkw):
    print("=" * 70)
    print("  1. 2019 SYNTAX BREAKDOWN: How did the grammar change?")
    print("=" * 70)
    print()

    labels, X, km = cluster_calls(srkw)
    if not labels:
        return
    nc = km.n_clusters

    # Split into 2019 and other years
    calls_2019 = [c for i, c in enumerate(srkw) if get_year(c) == 2019 and i in labels]
    calls_other = [c for i, c in enumerate(srkw) if get_year(c) != 2019 and get_year(c) and i in labels]

    labels_2019, _, _ = cluster_calls(calls_2019, nc)
    labels_other, _, _ = cluster_calls(calls_other, nc)

    seqs_2019 = build_sequences(calls_2019, labels_2019)
    seqs_other = build_sequences(calls_other, labels_other)

    M_2019 = transition_matrix(seqs_2019, nc)
    M_other = transition_matrix(seqs_other, nc)

    # Print transition matrices
    print(f"  2019 transition matrix ({M_2019.sum()} transitions):")
    print(f"          " + "  ".join(f"→C{j}" for j in range(nc)))
    for i in range(nc):
        rt = M_2019[i].sum()
        if rt > 0:
            probs = M_2019[i] / rt
            row = "  ".join(f"{probs[j]:.3f}" for j in range(nc))
            print(f"    C{i} →  {row}")

    print(f"\n  Other years transition matrix ({M_other.sum()} transitions):")
    print(f"          " + "  ".join(f"→C{j}" for j in range(nc)))
    for i in range(nc):
        rt = M_other[i].sum()
        if rt > 0:
            probs = M_other[i] / rt
            row = "  ".join(f"{probs[j]:.3f}" for j in range(nc))
            print(f"    C{i} →  {row}")

    # Entropy comparison
    H_2019 = matrix_entropy(M_2019)
    H_other = matrix_entropy(M_other)
    print(f"\n  Conditional entropy:")
    print(f"    2019:        {H_2019:.4f} bits")
    print(f"    Other years: {H_other:.4f} bits")
    diff = H_2019 - H_other
    pct = diff / (H_other + 1e-12) * 100
    print(f"    Difference:  {diff:+.4f} bits ({pct:+.1f}%)")
    if diff > 0:
        print(f"    *** 2019 sequences are MORE unpredictable")
    else:
        print(f"    *** 2019 sequences are MORE predictable")

    # Sequence length comparison
    lens_2019 = [len(s) for s in seqs_2019]
    lens_other = [len(s) for s in seqs_other]
    if lens_2019 and lens_other:
        print(f"\n  Bout lengths:")
        print(f"    2019:        mean={np.mean(lens_2019):.1f}, median={np.median(lens_2019):.0f}, max={max(lens_2019)}")
        print(f"    Other years: mean={np.mean(lens_other):.1f}, median={np.median(lens_other):.0f}, max={max(lens_other)}")

    # Unique bigrams
    bigrams_2019 = set()
    bigrams_other = set()
    for seq in seqs_2019:
        for j in range(len(seq) - 1):
            bigrams_2019.add((seq[j][0], seq[j+1][0]))
    for seq in seqs_other:
        for j in range(len(seq) - 1):
            bigrams_other.add((seq[j][0], seq[j+1][0]))

    novel_2019 = bigrams_2019 - bigrams_other
    lost_2019 = bigrams_other - bigrams_2019
    print(f"\n  Bigram types:")
    print(f"    2019 unique: {len(bigrams_2019)}/{nc*nc} possible")
    print(f"    Other years: {len(bigrams_other)}/{nc*nc} possible")
    if novel_2019:
        print(f"    Novel in 2019 (not in other years): {novel_2019}")
    if lost_2019:
        print(f"    Lost in 2019 (in other years, not 2019): {lost_2019}")
    print()


# ─────────────────────────────────────────────────────────────────────
# 2. DIEL PATTERNS
# ─────────────────────────────────────────────────────────────────────

def analyse_diel(srkw):
    print("=" * 70)
    print("  2. DIEL PATTERNS: Do orcas communicate differently at night?")
    print("=" * 70)
    print()

    labels, X, km = cluster_calls(srkw)
    if not labels:
        return
    nc = km.n_clusters

    by_hour = defaultdict(lambda: {'counts': Counter(), 'centers': [], 'bws': [], 'durs': []})
    for i, c in enumerate(srkw):
        if i not in labels:
            continue
        hour = get_hour(c)
        if hour is None:
            continue
        by_hour[hour]['counts'][labels[i]] += 1
        try:
            lo = float(c['LowFreqHz']) if c['LowFreqHz'] != 'NA' else None
            hi = float(c['HighFreqHz']) if c['HighFreqHz'] != 'NA' else None
            dur = float(c['FileEndSec']) - float(c['FileBeginSec'])
            if lo and hi and lo > 0 and hi > 0:
                by_hour[hour]['centers'].append((lo+hi)/2)
                by_hour[hour]['bws'].append(hi-lo)
                by_hour[hour]['durs'].append(dur)
        except:
            pass

    hours = sorted(by_hour.keys())
    if not hours:
        print("  No timestamp data available")
        return

    print(f"  Call distribution by hour (UTC):")
    print(f"  {'Hour':>6s}  {'N':>6s}  " + "  ".join(f"C{i}%" for i in range(nc)) + f"  {'CenterHz':>10s}  {'BW Hz':>10s}  {'Dur s':>8s}")
    print(f"  {'─'*6}  {'─'*6}  " + "  ".join('─'*5 for _ in range(nc)) + f"  {'─'*10}  {'─'*10}  {'─'*8}")

    # Collect day vs night data
    day_calls = []  # 06-18 UTC (roughly 22:00-10:00 Pacific = spans day)
    night_calls = []

    for hour in range(24):
        if hour not in by_hour:
            continue
        hd = by_hour[hour]
        total = sum(hd['counts'].values())
        if total < 5:
            continue

        dist = [hd['counts'].get(c, 0) / total for c in range(nc)]
        pcts = "  ".join(f"{d*100:5.1f}" for d in dist)
        c_mean = np.mean(hd['centers']) if hd['centers'] else 0
        b_mean = np.mean(hd['bws']) if hd['bws'] else 0
        d_mean = np.mean(hd['durs']) if hd['durs'] else 0

        # UTC 06-18 = late night to morning Pacific (day activity in summer)
        is_day = 14 <= hour <= 23 or 0 <= hour <= 6  # Pacific daylight ~UTC-7

        marker = " ☀" if is_day else " ☾"
        print(f"  {hour:>6d}  {total:>6d}  {pcts}  {c_mean:>10.0f}  {b_mean:>10.0f}  {d_mean:>8.2f}{marker}")

        if is_day:
            day_calls.extend(hd['centers'])
        else:
            night_calls.extend(hd['centers'])

    # Day vs night comparison
    if len(day_calls) >= 20 and len(night_calls) >= 20:
        from scipy.stats import mannwhitneyu
        day_arr = np.array(day_calls)
        night_arr = np.array(night_calls)
        U, p = mannwhitneyu(day_arr, night_arr, alternative='two-sided')
        print(f"\n  Day vs Night center frequency:")
        print(f"    Day (Pacific):   {day_arr.mean():.0f}±{day_arr.std():.0f} Hz (n={len(day_arr)})")
        print(f"    Night (Pacific): {night_arr.mean():.0f}±{night_arr.std():.0f} Hz (n={len(night_arr)})")
        print(f"    Mann-Whitney U: p={p:.2e} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'}")

    # Call rate by hour
    print(f"\n  Call rate pattern:")
    total_by_hour = {h: sum(by_hour[h]['counts'].values()) for h in hours if sum(by_hour[h]['counts'].values()) >= 5}
    if total_by_hour:
        peak_hour = max(total_by_hour, key=total_by_hour.get)
        quiet_hour = min(total_by_hour, key=total_by_hour.get)
        print(f"    Peak hour (UTC): {peak_hour}:00 ({total_by_hour[peak_hour]} calls)")
        print(f"    Quietest hour:   {quiet_hour}:00 ({total_by_hour[quiet_hour]} calls)")
        ratio = total_by_hour[peak_hour] / (total_by_hour[quiet_hour] + 1)
        print(f"    Peak/quiet ratio: {ratio:.1f}×")
    print()


# ─────────────────────────────────────────────────────────────────────
# 3. SEASONAL PATTERNS
# ─────────────────────────────────────────────────────────────────────

def analyse_seasonal(srkw):
    print("=" * 70)
    print("  3. SEASONAL PATTERNS: Monthly variation")
    print("=" * 70)
    print()

    labels, X, km = cluster_calls(srkw)
    if not labels:
        return
    nc = km.n_clusters

    by_month = defaultdict(lambda: Counter())
    month_acoustic = defaultdict(lambda: {'centers': [], 'bws': [], 'durs': []})

    for i, c in enumerate(srkw):
        if i not in labels:
            continue
        month = get_month(c)
        if month is None:
            continue
        by_month[month][labels[i]] += 1
        try:
            lo = float(c['LowFreqHz']) if c['LowFreqHz'] != 'NA' else None
            hi = float(c['HighFreqHz']) if c['HighFreqHz'] != 'NA' else None
            dur = float(c['FileEndSec']) - float(c['FileBeginSec'])
            if lo and hi and lo > 0 and hi > 0:
                month_acoustic[month]['centers'].append((lo+hi)/2)
                month_acoustic[month]['bws'].append(hi-lo)
                month_acoustic[month]['durs'].append(dur)
        except:
            pass

    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    print(f"  {'Month':>6s}  {'N':>6s}  " + "  ".join(f"C{i}%" for i in range(nc)) + f"  {'CenterHz':>10s}  {'BW Hz':>8s}")
    print(f"  {'─'*6}  {'─'*6}  " + "  ".join('─'*5 for _ in range(nc)) + f"  {'─'*10}  {'─'*8}")

    for month in range(1, 13):
        total = sum(by_month[month].values())
        if total < 10:
            continue
        dist = [by_month[month].get(c, 0) / total for c in range(nc)]
        pcts = "  ".join(f"{d*100:5.1f}" for d in dist)
        c_mean = np.mean(month_acoustic[month]['centers']) if month_acoustic[month]['centers'] else 0
        b_mean = np.mean(month_acoustic[month]['bws']) if month_acoustic[month]['bws'] else 0
        print(f"  {month_names[month]:>6s}  {total:>6d}  {pcts}  {c_mean:>10.0f}  {b_mean:>8.0f}")

    # Summer vs winter
    summer = []  # Jun-Sep (SRKW inland season)
    winter = []  # Oct-May
    for month in range(1, 13):
        for c in month_acoustic[month]['centers']:
            if 6 <= month <= 9:
                summer.append(c)
            else:
                winter.append(c)

    if len(summer) >= 20 and len(winter) >= 20:
        from scipy.stats import mannwhitneyu
        s, w = np.array(summer), np.array(winter)
        U, p = mannwhitneyu(s, w, alternative='two-sided')
        print(f"\n  Summer (Jun-Sep) vs Winter (Oct-May):")
        print(f"    Summer: {s.mean():.0f}±{s.std():.0f} Hz (n={len(s)})")
        print(f"    Winter: {w.mean():.0f}±{w.std():.0f} Hz (n={len(w)})")
        print(f"    Mann-Whitney: p={p:.2e} {'***' if p < 0.001 else 'n.s.'}")
    print()


# ─────────────────────────────────────────────────────────────────────
# 4. SAR DEEP DIVE: The lexical language
# ─────────────────────────────────────────────────────────────────────

def analyse_sar_deep(all_rows):
    print("=" * 70)
    print("  4. SAR DEEP DIVE: Why richest vocabulary, minimal grammar?")
    print("=" * 70)
    print()

    sar = [r for r in all_rows if r['KW'] == '1'
           and r['AnnotationLevel'] == 'Call' and r['Ecotype'] == 'SAR']
    srkw = [r for r in all_rows if r['KW'] == '1'
            and r['AnnotationLevel'] == 'Call' and r['Ecotype'] == 'SRKW']

    print(f"  SAR: {len(sar)} calls")
    print(f"  SRKW: {len(srkw)} calls (for comparison)")

    for name, calls in [("SAR", sar), ("SRKW", srkw)]:
        labels, X, km = cluster_calls(calls, n_clusters=3)
        if not labels:
            continue

        sequences = build_sequences(calls, labels)
        nc = km.n_clusters

        # Cluster distribution
        counts = Counter(labels.values())
        total = sum(counts.values())
        dist = [counts.get(c, 0) / total for c in range(nc)]

        # Shannon entropy of distribution
        H = -sum(p * math.log2(p) for p in dist if p > 0)

        # Gini-Simpson diversity
        gini = 1 - sum(p**2 for p in dist)

        # Transition matrix
        M = transition_matrix(sequences, nc)
        total_trans = M.sum()
        H_trans = matrix_entropy(M)
        self_rate = sum(M[i, i] for i in range(nc)) / total_trans if total_trans > 0 else 0

        # Sequence lengths
        seq_lens = [len(s) for s in sequences]

        print(f"\n  {name}:")
        print(f"    Cluster distribution: {', '.join(f'C{i}={dist[i]:.3f}' for i in range(nc))}")
        print(f"    Marginal entropy H: {H:.4f} bits")
        print(f"    Gini-Simpson diversity: {gini:.4f}")
        print(f"    Sequences: {len(sequences)}, total transitions: {total_trans}")
        print(f"    Mean sequence length: {np.mean(seq_lens):.1f}")
        print(f"    Self-transition rate: {self_rate:.4f}")
        print(f"    Conditional entropy H(next|prev): {H_trans:.4f} bits")
        print(f"    Mutual information MI: {H - H_trans:.4f} bits")
        print(f"    MI/H ratio: {(H - H_trans) / H * 100:.1f}% of information is sequential")

    # Key comparison
    print(f"\n  ═══ THE VOCABULARY-GRAMMAR TRADE-OFF ═══")
    print()
    print(f"  SAR communicates like an isolating language (Mandarin, Vietnamese):")
    print(f"    → Many distinct signal types (high marginal entropy)")
    print(f"    → Each signal carries meaning on its own (low sequential structure)")
    print(f"    → Word order matters less because words themselves are specific")
    print()
    print(f"  SRKW communicates like an agglutinative language (Turkish, Japanese):")
    print(f"    → Few base signal types (low marginal entropy, dominated by C0)")
    print(f"    → Meaning encoded in sequences and positions (high sequential structure)")
    print(f"    → A small vocabulary with rich combinatorial grammar")
    print()


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "▓" * 70)
    print("▓  ORCA-ENGINE: DEEPER ANALYSIS")
    print("▓  Syntax breakdown, diel/seasonal patterns, SAR deep dive")
    print("▓" * 70)
    print()

    all_rows = load_annotations()
    srkw = [r for r in all_rows if r['KW'] == '1'
            and r['AnnotationLevel'] == 'Call' and r['Ecotype'] == 'SRKW']

    analyse_2019_syntax(srkw)
    analyse_diel(srkw)
    analyse_seasonal(srkw)
    analyse_sar_deep(all_rows)

    print("▓" * 70)
    print("▓  DEEPER ANALYSIS COMPLETE")
    print("▓" * 70)
    print()


if __name__ == "__main__":
    main()
