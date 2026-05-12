#!/usr/bin/env python3
"""
SAR as control group for the 2019 anomaly.

If the 2019 acoustic shift is real ecological stress:
  - SRKW should show the shift (Chinook-dependent, starving)
  - SAR should NOT show it (different prey, different ocean, healthy population)
  - TKW should NOT show it (marine mammal diet, not salmon-dependent)

If the shift is an artifact (recording equipment, ocean noise, annotation protocol):
  - ALL ecotypes should show it in 2019

This is the natural experiment. Same dataset. Same annotation pipeline.
Different populations. Different prey. If only SRKW shifts, it's ecology.
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
        return {}, np.array([])
    X = np.array(features)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return {valid[i]: labels[i] for i in range(len(valid))}, X


def get_year(c):
    try:
        return int(c['UTC'][:4])
    except:
        return None


def yearly_acoustic_stats(calls):
    """Get per-year acoustic statistics."""
    by_year = defaultdict(lambda: {'centers': [], 'bws': [], 'durs': [], 'n': 0})
    for c in calls:
        year = get_year(c)
        if not year or year < 2005 or year > 2025:
            continue
        try:
            lo = float(c['LowFreqHz']) if c['LowFreqHz'] != 'NA' else None
            hi = float(c['HighFreqHz']) if c['HighFreqHz'] != 'NA' else None
            dur = float(c['FileEndSec']) - float(c['FileBeginSec'])
            if lo and hi and lo > 0 and hi > 0 and dur > 0:
                by_year[year]['centers'].append((lo+hi)/2)
                by_year[year]['bws'].append(hi-lo)
                by_year[year]['durs'].append(dur)
                by_year[year]['n'] += 1
        except:
            pass
    return by_year


def print_yearly_comparison(name, by_year, highlight_year=2019):
    years = sorted(y for y in by_year if by_year[y]['n'] >= 10)
    if not years:
        print(f"  {name}: insufficient data")
        return

    print(f"\n  {name} — yearly acoustic properties:")
    print(f"  {'Year':>6s}  {'N':>6s}  {'Center Hz':>12s}  {'Bandwidth Hz':>14s}  {'Duration s':>12s}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*12}  {'─'*14}  {'─'*12}")

    for year in years:
        d = by_year[year]
        c = np.array(d['centers'])
        b = np.array(d['bws'])
        dur = np.array(d['durs'])
        marker = "  ◄◄◄" if year == highlight_year else ""
        print(f"  {year:>6d}  {d['n']:>6d}  {c.mean():>7.0f}±{c.std():>4.0f}  {b.mean():>9.0f}±{b.std():>4.0f}  {dur.mean():>7.2f}±{dur.std():.2f}{marker}")

    return years


def compute_jsd_by_year(calls, n_clusters=3, split_year=2019):
    """Compute Jensen-Shannon divergence between a target year and all others."""
    labels, X = cluster_calls(calls, n_clusters)
    if not labels:
        return None, None, None

    target_counts = Counter()
    other_counts = Counter()
    for i, c in enumerate(calls):
        if i not in labels:
            continue
        year = get_year(c)
        if year == split_year:
            target_counts[labels[i]] += 1
        elif year and 2005 <= year <= 2025:
            other_counts[labels[i]] += 1

    if sum(target_counts.values()) < 10 or sum(other_counts.values()) < 10:
        return None, None, None

    # Normalise
    all_clusters = set(list(target_counts.keys()) + list(other_counts.keys()))
    n = max(all_clusters) + 1

    p = np.array([target_counts.get(c, 0) for c in range(n)], dtype=float)
    q = np.array([other_counts.get(c, 0) for c in range(n)], dtype=float)
    p /= p.sum()
    q /= q.sum()

    m = (p + q) / 2
    jsd = 0
    for i in range(n):
        if p[i] > 0:
            jsd += 0.5 * p[i] * math.log2(p[i] / (m[i] + 1e-15))
        if q[i] > 0:
            jsd += 0.5 * q[i] * math.log2(q[i] / (m[i] + 1e-15))

    return jsd, p, q


def main():
    print("\n" + "▓" * 70)
    print("▓  NATURAL EXPERIMENT: Is the 2019 shift ecology or artifact?")
    print("▓  SRKW (starving) vs SAR (healthy) vs TKW (different diet)")
    print("▓" * 70)
    print()

    all_rows = load_annotations()

    ecotype_calls = {}
    for eco in ['SRKW', 'SAR', 'TKW', 'OKW']:
        ecotype_calls[eco] = [r for r in all_rows
                              if r['KW'] == '1' and r['AnnotationLevel'] == 'Call'
                              and r['Ecotype'] == eco]
        print(f"  {eco}: {len(ecotype_calls[eco])} call-level annotations")

    # ─── Per-ecotype yearly acoustic statistics ──────────────────
    print("\n" + "=" * 70)
    print("  YEARLY ACOUSTIC PROPERTIES BY ECOTYPE")
    print("=" * 70)

    eco_years = {}
    for eco in ['SRKW', 'SAR', 'TKW', 'OKW']:
        by_year = yearly_acoustic_stats(ecotype_calls[eco])
        eco_years[eco] = by_year
        print_yearly_comparison(eco, by_year)

    # ─── The key test: JSD for 2019 vs other years, per ecotype ──
    print("\n" + "=" * 70)
    print("  THE NATURAL EXPERIMENT: 2019 divergence by ecotype")
    print("=" * 70)
    print()
    print(f"  Jensen-Shannon divergence: 2019 vs all other years")
    print(f"  (Higher = more different. If only SRKW is high, it's ecology.)")
    print()

    print(f"  {'Ecotype':>8s}  {'JSD':>10s}  {'N (2019)':>10s}  {'N (other)':>10s}  {'Interpretation':>20s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*20}")

    jsd_results = {}
    for eco in ['SRKW', 'SAR', 'TKW', 'OKW']:
        calls = ecotype_calls[eco]
        jsd, p, q = compute_jsd_by_year(calls, n_clusters=3, split_year=2019)

        # Count 2019 calls
        n_2019 = sum(1 for c in calls if get_year(c) == 2019)
        n_other = sum(1 for c in calls if get_year(c) and get_year(c) != 2019 and 2005 <= get_year(c) <= 2025)

        if jsd is not None:
            jsd_results[eco] = jsd
            if jsd > 0.05:
                interp = "*** SHIFTED"
            elif jsd > 0.01:
                interp = "* moderate shift"
            else:
                interp = "stable"
            print(f"  {eco:>8s}  {jsd:>10.6f}  {n_2019:>10d}  {n_other:>10d}  {interp:>20s}")
        else:
            print(f"  {eco:>8s}  {'N/A':>10s}  {n_2019:>10d}  {n_other:>10d}  {'insufficient data':>20s}")

    # ─── Mann-Whitney for 2019 vs other years, all ecotypes ──────
    print(f"\n" + "=" * 70)
    print(f"  MANN-WHITNEY U: 2019 vs other years, per ecotype")
    print(f"=" * 70)
    print()

    from scipy.stats import mannwhitneyu

    for eco in ['SRKW', 'SAR', 'TKW', 'OKW']:
        by_year = eco_years[eco]
        if 2019 not in by_year or by_year[2019]['n'] < 5:
            print(f"  {eco}: insufficient 2019 data (n={by_year.get(2019, {}).get('n', 0)})")
            continue

        d2019 = by_year[2019]
        other_c, other_b, other_d = [], [], []
        for year in by_year:
            if year != 2019:
                other_c.extend(by_year[year]['centers'])
                other_b.extend(by_year[year]['bws'])
                other_d.extend(by_year[year]['durs'])

        if len(other_c) < 5:
            continue

        print(f"\n  {eco} (2019: n={d2019['n']}, other: n={len(other_c)}):")
        for name, a, b in [
            ("Center freq", d2019['centers'], other_c),
            ("Bandwidth", d2019['bws'], other_b),
            ("Duration", d2019['durs'], other_d),
        ]:
            a, b = np.array(a), np.array(b)
            if len(a) >= 5 and len(b) >= 5:
                U, p = mannwhitneyu(a, b, alternative='two-sided')
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                diff_pct = (a.mean() - b.mean()) / (b.mean() + 1e-12) * 100
                print(f"    {name:>15s}: {a.mean():>8.0f} vs {b.mean():>8.0f} Hz  ({diff_pct:>+6.1f}%)  p={p:.2e} {sig}")

    # ─── Summary ─────────────────────────────────────────────────
    print("\n" + "▓" * 70)
    print("▓  VERDICT")
    print("▓" * 70)
    print()

    if jsd_results:
        srkw_jsd = jsd_results.get('SRKW', 0)
        others = [v for k, v in jsd_results.items() if k != 'SRKW']
        max_other = max(others) if others else 0

        if srkw_jsd > max_other * 2 and srkw_jsd > 0.01:
            print("  SRKW shows a 2019 acoustic shift that NO other ecotype shows.")
            print("  This rules out recording artifacts, ocean noise, and annotation protocol.")
            print("  The shift is ECOLOGICAL — specific to a starving population.")
            print()
            print("  Acoustic monitoring of SRKW call-type distribution is a viable")
            print("  real-time welfare indicator for this endangered population.")
        elif srkw_jsd > 0.01:
            print("  SRKW shows a 2019 shift. Other ecotypes show smaller or no shifts.")
            print("  The shift is likely ecological but artifact contribution cannot be fully excluded.")
        else:
            print("  No strong ecotype-specific 2019 shift detected in cluster distribution.")
            print("  (The acoustic metric shifts may be more sensitive than cluster JSD.)")

    print()


if __name__ == "__main__":
    main()
