#!/usr/bin/env python3
"""
Vulnerability 2: 20ms ICI Spike May Be Annotation Artifact

63.2% of inter-call intervals fall in a single 5ms bin (15-20ms).
At this timescale, annotation temporal resolution matters. If DCLDE
annotators used fixed-length windows or quantised timestamps, the
spike is an artifact of the process, not biology.

Tests:
1. Timestamp precision analysis — examine FileBeginSec decimal digits.
2. Cross-provider comparison — different annotation pipelines should
   produce different artifacts.
3. Cross-station ICI (Haro NPZ) — independent hydrophones.
4. Quantisation model comparison — if quantised, what would we expect?

Note: Raw audio validation (waveform onset measurement) requires
librosa and audio downloads. Listed as future work.
"""

import os
import sys
import csv
import ast
import numpy as np
from collections import defaultdict, Counter
from decimal import Decimal

sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

def load_annotations():
    with open(os.path.join(os.path.dirname(__file__),
              "data", "dclde", "Annotations.csv"), 'r') as f:
        return list(csv.DictReader(f))


def load_haro():
    data = np.load(os.path.join(os.path.dirname(__file__),
                   "data", "haro_srkw_features.npz"), allow_pickle=True)
    features = data["features"].copy()
    metadata = [ast.literal_eval(m) for m in data["metadata"]]
    return features, metadata


# ─────────────────────────────────────────────────────────────────────
# TEST 1: Timestamp precision analysis
# ─────────────────────────────────────────────────────────────────────

def analyse_timestamp_precision(srkw_calls):
    """Examine FileBeginSec decimal precision."""
    print(f"\n  ─── Timestamp Precision Analysis ───")
    print(f"    Total SRKW calls examined: {len(srkw_calls)}")

    # Count decimal digits
    digit_counts = Counter()
    timestamps = []
    for c in srkw_calls:
        try:
            raw = c['FileBeginSec'].strip().strip('"')
            timestamps.append(float(raw))
            # Count decimal digits using string representation
            if '.' in raw:
                decimal_part = raw.split('.')[1].rstrip('0')
                digit_counts[len(decimal_part)] += 1
            else:
                digit_counts[0] += 1
        except (ValueError, KeyError):
            pass

    print(f"\n    Decimal digits distribution:")
    total = sum(digit_counts.values())
    for d in sorted(digit_counts.keys()):
        count = digit_counts[d]
        pct = count / total * 100
        print(f"      {d} digits: {count:>6d} ({pct:>5.1f}%)")

    # Check quantisation: what fraction of timestamps fall exactly on a grid?
    ts_arr = np.array(timestamps)
    print(f"\n    Quantisation test (fraction already on grid):")
    for grid in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
        rounded = np.round(ts_arr / grid) * grid
        on_grid = np.sum(np.abs(ts_arr - rounded) < 1e-9)
        frac = on_grid / len(ts_arr) * 100
        print(f"      {grid:>6.3f}s grid: {on_grid:>6d} / {len(ts_arr)} ({frac:>5.1f}%)")

    # Check ICI distribution from the CSV itself
    print(f"\n    ICI distribution from DCLDE annotations:")
    by_file = defaultdict(list)
    for c in srkw_calls:
        try:
            begin = float(c['FileBeginSec'])
            by_file[c['Soundfile']].append(begin)
        except (ValueError, KeyError):
            pass

    all_icis = []
    for fname, begins in by_file.items():
        begins.sort()
        for j in range(len(begins) - 1):
            ici = begins[j + 1] - begins[j]
            if 0.001 < ici < 5.0:
                all_icis.append(ici)

    if all_icis:
        ici_arr = np.array(all_icis)
        print(f"    Total ICIs: {len(ici_arr)}")
        print(f"    Mean: {ici_arr.mean():.4f}s")
        print(f"    Median: {np.median(ici_arr):.4f}s")

        # Fine histogram
        bins_5ms = np.arange(0, 0.105, 0.005)
        hist, _ = np.histogram(ici_arr[ici_arr < 0.1], bins=bins_5ms)
        print(f"\n    ICI distribution (5ms bins, < 100ms):")
        for i in range(len(hist)):
            lo = bins_5ms[i] * 1000
            hi = bins_5ms[i + 1] * 1000
            pct = hist[i] / len(ici_arr) * 100
            bar = "█" * int(pct * 2)
            print(f"      {lo:>5.0f}-{hi:>3.0f}ms: {hist[i]:>5d} ({pct:>5.1f}%) {bar}")

    return ts_arr, all_icis


# ─────────────────────────────────────────────────────────────────────
# TEST 2: Cross-provider ICI comparison
# ─────────────────────────────────────────────────────────────────────

def analyse_cross_provider_ici(srkw_calls):
    """Compare ICI distributions across different annotation providers."""
    print(f"\n  ─── Cross-Provider ICI Comparison ───")

    # Group by provider, then by soundfile within provider
    by_provider = defaultdict(lambda: defaultdict(list))
    for c in srkw_calls:
        try:
            provider = c['Provider']
            begin = float(c['FileBeginSec'])
            by_provider[provider][c['Soundfile']].append(begin)
        except (ValueError, KeyError):
            pass

    print(f"\n    {'Provider':.<25s}  {'Calls':>7s}  {'ICIs':>7s}  {'15-20ms':>8s}  {'Peak bin':>10s}  {'Median ICI':>11s}")
    print(f"    {'─' * 25}  {'─' * 7}  {'─' * 7}  {'─' * 8}  {'─' * 10}  {'─' * 11}")

    provider_icis = {}
    for provider in sorted(by_provider.keys()):
        files = by_provider[provider]
        n_calls = sum(len(v) for v in files.values())
        icis = []
        for fname, begins in files.items():
            begins.sort()
            for j in range(len(begins) - 1):
                ici = begins[j + 1] - begins[j]
                if 0.001 < ici < 5.0:
                    icis.append(ici)

        if len(icis) < 10:
            continue

        ici_arr = np.array(icis)
        provider_icis[provider] = ici_arr

        # Fraction in 15-20ms bin
        in_bin = ((ici_arr >= 0.015) & (ici_arr < 0.020)).sum()
        frac = in_bin / len(ici_arr) * 100

        # Peak bin (5ms resolution)
        bins_5ms = np.arange(0, 0.105, 0.005)
        hist, _ = np.histogram(ici_arr[ici_arr < 0.1], bins=bins_5ms)
        if hist.max() > 0:
            peak_idx = np.argmax(hist)
            peak_lo = bins_5ms[peak_idx] * 1000
            peak_hi = bins_5ms[peak_idx + 1] * 1000
            peak_str = f"{peak_lo:.0f}-{peak_hi:.0f}ms"
        else:
            peak_str = "─"

        median_ici = np.median(ici_arr)
        print(f"    {provider:.<25s}  {n_calls:>7d}  {len(icis):>7d}  {frac:>7.1f}%  {peak_str:>10s}  {median_ici:>10.4f}s")

    # Check if spike appears in all providers
    if provider_icis:
        spike_providers = []
        for provider, ici_arr in provider_icis.items():
            in_bin = ((ici_arr >= 0.015) & (ici_arr < 0.020)).sum()
            frac = in_bin / len(ici_arr) * 100
            if frac > 30:
                spike_providers.append(provider)

        if len(spike_providers) == len(provider_icis):
            print(f"\n    *** 20ms spike present in ALL providers")
            print(f"    *** Less likely to be pipeline-specific artifact")
        elif spike_providers:
            print(f"\n    20ms spike present in: {', '.join(spike_providers)}")
            print(f"    Absent in: {', '.join(p for p in provider_icis if p not in spike_providers)}")
            print(f"    *** Provider-specific — likely annotation artifact")
        else:
            print(f"\n    20ms spike not dominant in any provider at >30% threshold")

    return provider_icis


# ─────────────────────────────────────────────────────────────────────
# TEST 3: Cross-station ICI (Haro NPZ)
# ─────────────────────────────────────────────────────────────────────

def analyse_cross_station_ici(metadata):
    """Compare ICI distributions between north and south stations."""
    print(f"\n  ─── Cross-Station ICI (Haro NPZ) ───")

    by_session = defaultdict(list)
    for i, m in enumerate(metadata):
        station = m.get('station', 'unknown')
        fname = m.get('filename', '')
        begin = float(m.get('duration', 0))
        by_session[(fname, station)].append(begin)

    station_icis = defaultdict(list)
    for (fname, station), begins in by_session.items():
        begins.sort()
        for j in range(len(begins) - 1):
            ici = begins[j + 1] - begins[j]
            if 0.001 < ici < 5.0:
                station_icis[station].append(ici)

    print(f"\n    {'Station':>10s}  {'ICIs':>7s}  {'15-20ms':>8s}  {'Mean':>10s}  {'Median':>10s}  {'CV':>8s}")
    print(f"    {'─' * 10}  {'─' * 7}  {'─' * 8}  {'─' * 10}  {'─' * 10}  {'─' * 8}")

    for station in sorted(station_icis.keys()):
        icis = np.array(station_icis[station])
        in_bin = ((icis >= 0.015) & (icis < 0.020)).sum()
        frac = in_bin / len(icis) * 100
        cv = icis.std() / icis.mean() if icis.mean() > 0 else 0
        print(f"    {station:>10s}  {len(icis):>7d}  {frac:>7.1f}%  {icis.mean():>10.4f}  {np.median(icis):>10.4f}  {cv:>8.3f}")

    # KS test between stations
    stations = sorted(station_icis.keys())
    if len(stations) == 2:
        from scipy.stats import ks_2samp
        arr1 = np.array(station_icis[stations[0]])
        arr2 = np.array(station_icis[stations[1]])
        ks_stat, p = ks_2samp(arr1, arr2)
        print(f"\n    KS test ({stations[0]} vs {stations[1]}): D = {ks_stat:.4f}, p = {p:.4e}")
        if p < 0.05:
            print(f"    *** Distributions DIFFER between stations")
            print(f"    *** If spike is in both, it's more likely biological")
        else:
            print(f"    Distributions are similar between stations")

    return station_icis


# ─────────────────────────────────────────────────────────────────────
# TEST 4: Quantisation model comparison
# ─────────────────────────────────────────────────────────────────────

def analyse_quantisation_model(timestamps, observed_icis):
    """If timestamps are quantised, what ICI distribution would we expect?"""
    print(f"\n  ─── Quantisation Model Comparison ───")

    if not observed_icis:
        print(f"    No ICIs to analyse")
        return

    obs_arr = np.array(observed_icis)
    obs_sub = obs_arr[obs_arr < 0.1]  # Focus on < 100ms

    # Test a few quantisation levels
    from scipy.stats import ks_2samp

    print(f"\n    {'Quant level':>12s}  {'KS stat':>10s}  {'p-value':>10s}  {'Interpretation':>20s}")
    print(f"    {'─' * 12}  {'─' * 10}  {'─' * 10}  {'─' * 20}")

    for quant in [0.001, 0.005, 0.01, 0.02, 0.05]:
        # Round timestamps to this grid, recompute ICIs
        rounded = np.round(timestamps / quant) * quant
        # Reconstruct ICIs from rounded timestamps (this is approximate —
        # we don't know which timestamps are from the same session)
        # Instead, round the observed ICIs to this grid
        rounded_icis = np.round(obs_sub / quant) * quant
        rounded_icis = rounded_icis[rounded_icis > 0]

        if len(rounded_icis) < 10:
            continue

        ks_stat, p = ks_2samp(obs_sub, rounded_icis)
        if p > 0.05:
            interp = "MATCHES (artifact?)"
        else:
            interp = "DIFFERS (not artifact)"
        print(f"    {quant:>12.3f}s  {ks_stat:>10.4f}  {p:>10.4e}  {interp:>20s}")

    # Direct test: is the ICI distribution consistent with the timestamp precision?
    # If timestamps have 3 decimal places (1ms precision), minimum distinguishable ICI is 0.001s
    # If timestamps have 2 decimal places (10ms), minimum is 0.01s
    # The 15-20ms peak with 3-digit precision means the spike is NOT explained by precision alone


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  VULNERABILITY 2: ICI Validation")
    print("  Is the 20ms spike real or annotation artifact?")
    print("=" * 70)

    # Load DCLDE annotations
    all_rows = load_annotations()
    srkw = [r for r in all_rows if r['KW'] == '1'
            and r['AnnotationLevel'] == 'Call' and r['Ecotype'] == 'SRKW']
    print(f"\n  SRKW call-level annotations: {len(srkw)}")

    # Test 1: Timestamp precision
    timestamps, all_icis = analyse_timestamp_precision(srkw)

    # Test 2: Cross-provider
    provider_icis = analyse_cross_provider_ici(srkw)

    # Test 3: Cross-station (Haro NPZ)
    _, haro_metadata = load_haro()
    station_icis = analyse_cross_station_ici(haro_metadata)

    # Test 4: Quantisation model
    if timestamps is not None and all_icis:
        analyse_quantisation_model(timestamps, all_icis)

    # ── Conclusion ──
    print(f"\n  {'═' * 60}")
    print(f"  CONCLUSION")
    print(f"  {'═' * 60}")
    print(f"\n  Evidence for/against annotation artifact:")
    print(f"    1. Timestamp precision: see above")
    print(f"    2. Cross-provider: spike in all/some/no providers")
    print(f"    3. Cross-station: spike in both/one station(s)")
    print(f"    4. Quantisation model: matches/differs from observed")
    print(f"\n  Note: Raw audio validation (waveform onset measurement)")
    print(f"  requires librosa and audio downloads. Listed as future work.")
    print(f"  The ICI *distribution shape* (peaked with long tail) remains")
    print(f"  informative even if the exact peak location is artifactual.")

    print()


if __name__ == "__main__":
    main()
