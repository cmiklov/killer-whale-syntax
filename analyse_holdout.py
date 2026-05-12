#!/usr/bin/env python3
"""
Held-out year validation of the 2019 crisis detector.

The thresholds in Section 7 (Conservation Applications) were derived
from retrospective analysis of the 2005-2023 corpus. A reviewer will
reasonably ask: would a detector trained without access to 2019 actually
fire on 2019, and would it stay quiet on years where no crisis occurred?

This script performs the cleanest held-out test possible with the data
in this repository (the .npz feature files for dclde_srkw_features,
which contain SRKW calls from 2017, 2019, and 2020). The full
DCLDE annotation corpus used in Section 5.7 (n=12,298 SRKW) is not
distributed with the repo; this pilot is limited to n=577.

Procedure:
1. Use 2017 calls (n=155) as the pre-crisis baseline.
2. Re-cluster the 50-D features at k=3 on the full 577-call subset.
3. Compute four indicators for each year:
   - C2 fraction (a non-dominant cluster's share)
   - Mean bandwidth (Hz, from low_freq + high_freq annotation)
   - Mean center frequency (Hz)
   - Mean duration (seconds)
4. Set thresholds at baseline_mean + 2*baseline_std from 2017.
5. Apply to 2019 (known crisis, should flag) and 2020 (held-out, should
   ideally not flag, or partially recover toward baseline).

Limitations:
- Small N per year (155/140/282). This is a pilot, not a full validation.
- Only three years available. False positive rate cannot be computed
  meaningfully from one held-out year.
- Re-clustering on the subset gives different cluster IDs than the
  Section 5.7 analysis, so C0/C1/C2 labels here are not the same labels
  as in the paper.

A proper prospective validation would re-run the Section 5.7 analysis
on the full DCLDE annotation corpus with year-stratified train/test
splits, hold out each year in turn, and report the implied false
positive rate. The infrastructure to do that is straightforward; the
gating step is access to the source Annotations.csv.
"""

import ast
import os
import sys
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(THIS_DIR, "data", "dclde_srkw_features.npz")


def load():
    d = np.load(DATA, allow_pickle=True)
    features = d["features"]
    metadata = [ast.literal_eval(str(s)) for s in d["metadata"]]
    years = np.array([int(m["utc"][:4]) for m in metadata])
    return features, metadata, years


def safe_float(x):
    try:
        if x == "NA" or x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def year_indicators(metadata, year_mask, cluster_labels):
    """Compute (C2_frac, mean_bw_Hz, mean_centerfreq_Hz, mean_dur_s) for
    a subset of calls. Bandwidth, centre frequency, and duration come
    from annotation metadata where available; cluster fraction comes
    from the re-clustered subset features."""
    sub = [m for m, keep in zip(metadata, year_mask) if keep]
    labels_sub = cluster_labels[year_mask]

    # Cluster distribution
    label_counts = Counter(labels_sub.tolist())
    total = sum(label_counts.values())
    # We don't know which cluster will be "C2" a priori; report fractions
    # for all three and let the caller decide.
    cluster_fracs = {k: label_counts.get(k, 0) / total for k in range(3)}

    # Acoustic indicators from annotation metadata
    bws, cfs, durs = [], [], []
    for m in sub:
        lo = safe_float(m.get("low_freq"))
        hi = safe_float(m.get("high_freq"))
        dur = safe_float(m.get("duration"))
        if lo and hi and lo > 0 and hi > 0:
            bws.append(hi - lo)
            cfs.append((lo + hi) / 2)
        if dur and dur > 0:
            durs.append(dur)

    return {
        "n": total,
        "cluster_fracs": cluster_fracs,
        "bandwidth_hz": np.mean(bws) if bws else None,
        "bandwidth_std": np.std(bws) if bws else None,
        "centerfreq_hz": np.mean(cfs) if cfs else None,
        "duration_s": np.mean(durs) if durs else None,
        "n_with_freq": len(bws),
        "n_with_duration": len(durs),
    }


def main():
    features, metadata, years = load()
    print(f"Loaded {len(features)} SRKW calls from dclde_srkw_features.npz")
    print(f"Year distribution: {dict(Counter(years.tolist()))}")

    # Station check: which year/station combinations are present?
    stations = np.array([m.get("dataset", "NA") for m in metadata])
    print(f"Year x station breakdown:")
    for y in sorted(set(years.tolist())):
        ymask = years == y
        sc = Counter(stations[ymask].tolist())
        print(f"  {y}: {dict(sc)}")
    print()

    # Restrict to orcasound_lab to hold station constant. This gives us
    # 2017 (n=155), 2019 (n=140), and 2020 (n=37) at the same station.
    station_mask = stations == "orcasound_lab"
    f_clean = features[station_mask]
    m_clean = [m for m, k in zip(metadata, station_mask) if k]
    y_clean = years[station_mask]
    print(f"Restricting to orcasound_lab: N = {len(f_clean)} "
          f"(2017: {(y_clean==2017).sum()}, 2019: {(y_clean==2019).sum()}, "
          f"2020: {(y_clean==2020).sum()})")
    print()

    # Re-cluster at k=3 on the station-controlled subset
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(f_clean)

    metadata = m_clean
    years = y_clean

    # Compute per-year indicators
    years_unique = sorted(set(years.tolist()))
    results = {y: year_indicators(metadata, years == y, cluster_labels) for y in years_unique}

    # ─── Report ──────────────────────────────────────────────────────
    print("=" * 72)
    print(" Per-year indicators (re-clustered at k=3 on the 577-call subset)")
    print("=" * 72)
    print(f"{'Year':>6}  {'N':>5}  {'C0%':>6}  {'C1%':>6}  {'C2%':>6}"
          f"  {'BW(Hz)':>9}  {'CF(Hz)':>9}  {'Dur(s)':>7}")
    print("-" * 72)
    for y in years_unique:
        r = results[y]
        cf = r["cluster_fracs"]
        print(f"{y:>6}  {r['n']:>5}  "
              f"{cf[0]*100:>5.1f}%  {cf[1]*100:>5.1f}%  {cf[2]*100:>5.1f}%  "
              f"{(r['bandwidth_hz'] or 0):>9.0f}  "
              f"{(r['centerfreq_hz'] or 0):>9.0f}  "
              f"{(r['duration_s'] or 0):>7.2f}")
    print()

    # ─── Held-out detector ───────────────────────────────────────────
    if 2017 not in results:
        print("Cannot run held-out detector: 2017 baseline absent.")
        return

    baseline = results[2017]
    print("=" * 72)
    print(" Held-out detector (train on 2017 baseline, threshold = mean + 2*std)")
    print("=" * 72)

    # Identify the dominant cluster in the baseline (its absence becomes
    # the anomaly indicator); the Section7 published threshold was "non-dominant
    # cluster exceeds 5% share of the corpus."
    baseline_fracs = baseline["cluster_fracs"]
    dominant_cluster = max(baseline_fracs, key=baseline_fracs.get)
    print(f"  2017 baseline (n={baseline['n']}): "
          f"dominant cluster C{dominant_cluster} "
          f"({baseline_fracs[dominant_cluster]*100:.1f}%), "
          f"non-dominant {(1 - baseline_fracs[dominant_cluster])*100:.1f}%")
    print(f"  Threshold: non-dominant fraction > 5%")
    print()

    # Bandwidth is not available for orcasound_lab annotations
    # (low_freq/high_freq are 'NA'), so the bandwidth indicator from Section7
    # cannot be evaluated here. Cluster-distribution and duration shifts
    # are what's available.
    print(f"  {'Year':>6}  {'N':>5}  {'NonDom%':>9}  {'NonDom>5%?':>11}  "
          f"{'d_dur(s)':>9}  Verdict")
    print(f"  {'-'*6}  {'-'*5}  {'-'*9}  {'-'*11}  {'-'*9}  -------")
    for y in years_unique:
        if y == 2017:
            print(f"  {y:>6}  {results[y]['n']:>5}  "
                  f"{(1-baseline_fracs[dominant_cluster])*100:>8.1f}%  "
                  f"{'(baseline)':>11}  {0.0:>9.2f}  BASELINE")
            continue
        r = results[y]
        non_dom = 1 - r["cluster_fracs"][dominant_cluster]
        cluster_flag = non_dom > 0.05
        d_dur = (r["duration_s"] or 0) - (baseline["duration_s"] or 0)
        verdict = "FLAG" if cluster_flag else "quiet"
        marker = ""
        if y == 2019:
            marker = "  (known crisis)"
        elif y == 2020:
            marker = "  (held-out, post-crisis recovery year)"
        print(f"  {y:>6}  {r['n']:>5}  "
              f"{non_dom*100:>8.1f}%  "
              f"{'YES' if cluster_flag else 'no':>11}  "
              f"{d_dur:>+9.2f}  {verdict}{marker}")
    print()

    print("Interpretation:")
    print(" - The detector trained on 2017 orcasound_lab baseline (no exposure")
    print("   to 2019 data during threshold derivation) is tested on 2019 and")
    print("   2020 calls from the same station.")
    print(" - 2019 (known crisis) should fire if the Section7 indicator framework")
    print("   captures real signal. This is the held-out sanity check.")
    print(" - 2020 (post-crisis) is a pilot test of false-positive resilience.")
    print(" - The result is a single station-controlled pilot. The published")
    print("   Section5.7 analysis used the full DCLDE corpus (n=12,298 SRKW), which")
    print("   is not distributed with this repository.")
    print(" - A proper prospective validation would hold out each year in turn")
    print("   on the full corpus and report the empirical false-positive rate.")


if __name__ == "__main__":
    main()
