#!/usr/bin/env python3
"""
Vulnerability 3: Acoustic Priming as Alternative to Topological Syntax

The topology-syntax result (d = 1.34-1.57) could be explained by acoustic
priming — animals repeating similar sounds (production inertia) rather than
topological grammar generating sequential structure.

Three tests to rule it out:
1. Exclude self-transitions: recompute d on cross-transition pairs only.
   If d stays large when you remove the 80-97% self-repetitions, priming
   can't explain it.
2. Cross-individual (voice boundary) test: if topology-syntax holds across
   speaker changes within a session, it's coordination, not priming.
3. Cross-station test: if topology predicts syntax between calls from
   different physical positions (likely different individuals), priming
   is ruled out.
"""

import os
import sys
import ast
import re
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

def load_and_normalise(path, group_key='station'):
    """Per-group Z-score + L2 normalisation. From analyse_boltzmann.py."""
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


def build_sessions(metadata):
    """Build sessions keyed by (filename, station/provider)."""
    by_file = defaultdict(list)
    for i, m in enumerate(metadata):
        # Use station if available, otherwise provider, otherwise 'default'
        group = m.get('station', m.get('provider', 'default'))
        by_file[(m.get('filename', str(i)), group)].append(
            (float(m.get('duration', 0)), i))
    for key in by_file:
        by_file[key].sort()
    return by_file


# ─────────────────────────────────────────────────────────────────────
# Cohen's d
# ─────────────────────────────────────────────────────────────────────

def cohens_d(a, b):
    """Cohen's d between two arrays."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    diff = np.mean(a) - np.mean(b)
    pooled_std = np.sqrt((np.std(a) ** 2 + np.std(b) ** 2) / 2)
    return diff / pooled_std if pooled_std > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────────────────────────────

def cluster_calls(normed, n_clusters=3):
    """KMeans on normalised 50D features. Returns labels array."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return km.fit_predict(normed)


# ─────────────────────────────────────────────────────────────────────
# TEST 1: Self-transition exclusion
# ─────────────────────────────────────────────────────────────────────

def test_exclude_self_transitions(name, normed, metadata, group_key, n_clusters=3):
    """Recompute d on cross-transition pairs only."""
    print(f"\n    {name} (k={n_clusters}):")

    labels = cluster_calls(normed, n_clusters)
    by_file = build_sessions(metadata)
    rng = np.random.RandomState(42)

    # Collect adjacent pairs, tagged by transition type
    self_sims = []
    cross_sims = []

    for key, entries in by_file.items():
        for j in range(len(entries) - 1):
            i1 = entries[j][1]
            i2 = entries[j + 1][1]
            sim = float(np.dot(normed[i1], normed[i2]))
            if labels[i1] == labels[i2]:
                self_sims.append(sim)
            else:
                cross_sims.append(sim)

    all_adj = np.array(self_sims + cross_sims)
    self_arr = np.array(self_sims)
    cross_arr = np.array(cross_sims)

    # Random pairs — all pairs
    rand_all = []
    for _ in range(len(all_adj)):
        i1, i2 = rng.choice(len(normed), 2, replace=False)
        rand_all.append(float(np.dot(normed[i1], normed[i2])))
    rand_all = np.array(rand_all)

    # Random cross-cluster pairs only
    rand_cross = []
    attempts = 0
    while len(rand_cross) < len(cross_arr) and attempts < len(cross_arr) * 20:
        i1, i2 = rng.choice(len(normed), 2, replace=False)
        if labels[i1] != labels[i2]:
            rand_cross.append(float(np.dot(normed[i1], normed[i2])))
        attempts += 1
    rand_cross = np.array(rand_cross)

    # Random self-cluster pairs only
    rand_self = []
    attempts = 0
    while len(rand_self) < len(self_arr) and attempts < len(self_arr) * 20:
        i1, i2 = rng.choice(len(normed), 2, replace=False)
        if labels[i1] == labels[i2]:
            rand_self.append(float(np.dot(normed[i1], normed[i2])))
        attempts += 1
    rand_self = np.array(rand_self)

    d_all = cohens_d(all_adj, rand_all)
    d_self = cohens_d(self_arr, rand_self) if len(rand_self) > 0 else 0.0
    d_cross = cohens_d(cross_arr, rand_cross) if len(rand_cross) > 0 else 0.0

    total = len(all_adj)
    self_frac = len(self_sims) / total * 100 if total > 0 else 0
    cross_frac = len(cross_sims) / total * 100 if total > 0 else 0

    print(f"      All pairs:             d = {d_all:.4f}  (n = {total})")
    print(f"      Self-transition only:  d = {d_self:.4f}  (n = {len(self_sims)}, {self_frac:.1f}% of pairs)")
    print(f"      Cross-transition only: d = {d_cross:.4f}  (n = {len(cross_sims)}, {cross_frac:.1f}% of pairs)")

    if len(cross_sims) > 0:
        print(f"      Cross adj mean sim:    {cross_arr.mean():.4f}")
        print(f"      Cross rand mean sim:   {rand_cross.mean():.4f}" if len(rand_cross) > 0 else "")

    if d_cross > 0.2:
        print(f"      *** Cross-transition d = {d_cross:.2f} — priming CANNOT explain coupling")
    elif d_cross > 0:
        print(f"      * Weak cross-transition effect — priming partially explains coupling")
    else:
        print(f"      Cross-transition d ≤ 0 — priming may explain the coupling")

    return {'d_all': d_all, 'd_self': d_self, 'd_cross': d_cross,
            'n_self': len(self_sims), 'n_cross': len(cross_sims)}


# ─────────────────────────────────────────────────────────────────────
# TEST 2: Cross-voice (cross-individual) test
# ─────────────────────────────────────────────────────────────────────

def detect_voices(normed, by_file, min_calls=20):
    """
    Per session: PCA to 10 dims, KMeans k=2..7, select by silhouette.
    Same algorithm as analyse_deepest.py.
    Returns dict: session_key -> (voice_labels, n_voices, indices).
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    voice_info = {}

    for key, entries in by_file.items():
        if len(entries) < min_calls:
            continue

        indices = [idx for _, idx in entries]
        session_feats = normed[indices]

        # PCA
        n_components = min(10, len(indices) - 1, session_feats.shape[1])
        if n_components < 2:
            continue
        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(session_feats)

        # Try k=2..7, pick best silhouette
        best_k = 1
        best_sil = -1
        best_labels = np.zeros(len(indices), dtype=int)

        max_k = min(8, len(indices) // 5)
        for k in range(2, max_k):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lab = km.fit_predict(reduced)
            sil = silhouette_score(reduced, lab)
            if sil > best_sil:
                best_sil = sil
                best_k = k
                best_labels = lab

        voice_info[key] = (best_labels, best_k, indices)

    return voice_info


def test_cross_voice(normed, metadata):
    """Test topology-syntax across voice boundaries within sessions."""
    print(f"\n  ═══ TEST 2: Cross-Voice Pairs (SRKW Haro) ═══")

    by_file = build_sessions(metadata)
    voice_info = detect_voices(normed, by_file, min_calls=20)

    if not voice_info:
        print(f"    No sessions with sufficient calls for voice detection")
        return None

    n_sessions = len(voice_info)
    n_multi = sum(1 for _, (_, nv, _) in voice_info.items() if nv >= 2)
    mean_voices = np.mean([nv for _, (_, nv, _) in voice_info.items()])

    print(f"    Sessions analysed: {n_sessions}")
    print(f"    Sessions with 2+ voices: {n_multi}")
    print(f"    Mean voices per session: {mean_voices:.1f}")

    # Collect adjacent pairs split by same-voice vs cross-voice
    same_voice_sims = []
    cross_voice_sims = []

    for key, (voice_labels, n_voices, indices) in voice_info.items():
        if n_voices < 2:
            continue
        entries = by_file[key]
        for j in range(len(entries) - 1):
            i1 = entries[j][1]
            i2 = entries[j + 1][1]
            # Map to position in session
            pos1 = indices.index(i1)
            pos2 = indices.index(i2)
            sim = float(np.dot(normed[i1], normed[i2]))
            if voice_labels[pos1] == voice_labels[pos2]:
                same_voice_sims.append(sim)
            else:
                cross_voice_sims.append(sim)

    if not cross_voice_sims:
        print(f"    No cross-voice adjacent pairs found")
        return None

    # Random pairs from multi-voice sessions
    rng = np.random.RandomState(42)
    all_multi_indices = []
    for key, (_, nv, indices) in voice_info.items():
        if nv >= 2:
            all_multi_indices.extend(indices)

    rand_sims = []
    for _ in range(len(same_voice_sims) + len(cross_voice_sims)):
        i1, i2 = rng.choice(all_multi_indices, 2, replace=False)
        rand_sims.append(float(np.dot(normed[i1], normed[i2])))
    rand_arr = np.array(rand_sims)

    same_arr = np.array(same_voice_sims)
    cross_arr = np.array(cross_voice_sims)

    d_same = cohens_d(same_arr, rand_arr)
    d_cross = cohens_d(cross_arr, rand_arr)

    print(f"\n    Same-voice adjacent pairs:  n = {len(same_voice_sims)}, mean sim = {same_arr.mean():.4f}")
    print(f"    Cross-voice adjacent pairs: n = {len(cross_voice_sims)}, mean sim = {cross_arr.mean():.4f}")
    print(f"    Random pairs:               n = {len(rand_sims)}, mean sim = {rand_arr.mean():.4f}")
    print(f"\n    d (same-voice vs random):  {d_same:.4f}")
    print(f"    d (cross-voice vs random): {d_cross:.4f}")

    if d_cross > 0.2:
        print(f"    *** Topology predicts syntax even ACROSS speakers (d = {d_cross:.2f})")
        print(f"    *** This is coordination, not acoustic priming")
    elif d_cross > 0:
        print(f"    * Weak cross-voice effect (d = {d_cross:.2f})")
    else:
        print(f"    Cross-voice d ≤ 0 — cannot rule out priming with this test")

    return {'d_same': d_same, 'd_cross': d_cross,
            'n_same': len(same_voice_sims), 'n_cross': len(cross_voice_sims)}


# ─────────────────────────────────────────────────────────────────────
# TEST 3: Cross-station test
# ─────────────────────────────────────────────────────────────────────

def test_cross_station(normed, metadata):
    """Test topology-syntax across hydrophone stations."""
    print(f"\n  ═══ TEST 3: Cross-Station Pairs (SRKW Haro) ═══")

    # Group by timestamp extracted from filename, split by station
    by_timestamp = defaultdict(lambda: {'north': [], 'south': []})
    for i, m in enumerate(metadata):
        fname = m.get('filename', '')
        station = m.get('station', '')
        match = re.search(r'(\d{8}T\d{6}Z)', fname)
        if match and station in ('north', 'south'):
            ts = match.group(1)
            begin = float(m.get('duration', 0))
            by_timestamp[ts][station].append((begin, i))

    # Keep only timestamps with both stations
    paired = {ts: data for ts, data in by_timestamp.items()
              if data['north'] and data['south']}

    print(f"    Paired timestamps (both stations): {len(paired)}")

    if not paired:
        print(f"    No paired timestamps found")
        return None

    # For each paired window, find cross-station response pairs
    cross_station_sims = []
    for ts, data in paired.items():
        north = sorted(data['north'])
        south = sorted(data['south'])

        for n_time, n_idx in north:
            nearest_delta = None
            nearest_s_idx = None
            for s_time, s_idx in south:
                delta = s_time - n_time
                if 0.05 < delta < 5.0:
                    if nearest_delta is None or delta < nearest_delta:
                        nearest_delta = delta
                        nearest_s_idx = s_idx
            if nearest_s_idx is not None:
                sim = float(np.dot(normed[n_idx], normed[nearest_s_idx]))
                cross_station_sims.append(sim)

        # Also south→north
        for s_time, s_idx in south:
            nearest_delta = None
            nearest_n_idx = None
            for n_time, n_idx in north:
                delta = n_time - s_time
                if 0.05 < delta < 5.0:
                    if nearest_delta is None or delta < nearest_delta:
                        nearest_delta = delta
                        nearest_n_idx = n_idx
            if nearest_n_idx is not None:
                sim = float(np.dot(normed[s_idx], normed[nearest_n_idx]))
                cross_station_sims.append(sim)

    if not cross_station_sims:
        print(f"    No cross-station response pairs found")
        return None

    cross_arr = np.array(cross_station_sims)

    # Random cross-station pairs
    rng = np.random.RandomState(42)
    all_north = []
    all_south = []
    for ts, data in paired.items():
        all_north.extend(idx for _, idx in data['north'])
        all_south.extend(idx for _, idx in data['south'])

    rand_cross = []
    for _ in range(len(cross_station_sims)):
        n_idx = rng.choice(all_north)
        s_idx = rng.choice(all_south)
        rand_cross.append(float(np.dot(normed[n_idx], normed[s_idx])))
    rand_arr = np.array(rand_cross)

    d = cohens_d(cross_arr, rand_arr)

    print(f"    Cross-station response pairs: {len(cross_station_sims)}")
    print(f"    Cross-station pair mean sim:  {cross_arr.mean():.4f}")
    print(f"    Random cross-station mean sim: {rand_arr.mean():.4f}")
    print(f"    d (cross-station vs random):  {d:.4f}")

    if d > 0.2:
        print(f"    *** Topology predicts syntax ACROSS stations (d = {d:.2f})")
        print(f"    *** Different positions = likely different individuals = priming ruled out")
    elif d > 0:
        print(f"    * Weak cross-station effect (d = {d:.2f})")
    else:
        print(f"    Cross-station d ≤ 0 — cannot rule out priming with this test")

    return {'d': d, 'n_pairs': len(cross_station_sims),
            'mean_cross': cross_arr.mean(), 'mean_random': rand_arr.mean()}


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  VULNERABILITY 3: Acoustic Priming vs Topological Syntax")
    print("  Is it grammar or acoustic inertia?")
    print("=" * 70)

    # ── Test 1: Self-transition exclusion (all ecotypes) ──
    print(f"\n  ═══ TEST 1: Self-Transition Exclusion ═══")

    test1_results = {}
    for name, path, group_key in [
        ("SRKW", "data/haro_srkw_features.npz", "station"),
        ("TKW", "data/tkw_features.npz", "provider"),
        ("OKW", "data/okw_features.npz", "provider"),
    ]:
        normed, metadata = load_and_normalise(path, group_key)
        test1_results[name] = test_exclude_self_transitions(
            name, normed, metadata, group_key)

    # ── Test 2 & 3: SRKW Haro only ──
    normed_haro, metadata_haro = load_and_normalise(
        "data/haro_srkw_features.npz", "station")

    test2_result = test_cross_voice(normed_haro, metadata_haro)
    test3_result = test_cross_station(normed_haro, metadata_haro)

    # ── Summary ──
    print(f"\n  {'═' * 60}")
    print(f"  SUMMARY: Can acoustic priming explain topology-syntax coupling?")
    print(f"  {'═' * 60}")

    print(f"\n  {'Test':.<40s}  {'d':>8s}  {'n':>8s}  {'Priming ruled out?':>20s}")
    print(f"  {'─' * 40}  {'─' * 8}  {'─' * 8}  {'─' * 20}")

    for name in ['SRKW', 'TKW', 'OKW']:
        r = test1_results[name]
        ruled = "YES" if r['d_cross'] > 0.2 else "WEAK" if r['d_cross'] > 0 else "NO"
        print(f"  {f'{name} all pairs':.<40s}  {r['d_all']:>8.4f}  {r['n_self'] + r['n_cross']:>8d}  {'(baseline)':>20s}")
        print(f"  {f'{name} cross-transition only':.<40s}  {r['d_cross']:>8.4f}  {r['n_cross']:>8d}  {ruled:>20s}")

    if test2_result:
        ruled = "YES" if test2_result['d_cross'] > 0.2 else "WEAK" if test2_result['d_cross'] > 0 else "NO"
        print(f"  {'SRKW cross-voice only':.<40s}  {test2_result['d_cross']:>8.4f}  {test2_result['n_cross']:>8d}  {ruled:>20s}")

    if test3_result:
        ruled = "YES" if test3_result['d'] > 0.2 else "WEAK" if test3_result['d'] > 0 else "NO"
        print(f"  {'SRKW cross-station only':.<40s}  {test3_result['d']:>8.4f}  {test3_result['n_pairs']:>8d}  {ruled:>20s}")

    # Overall verdict
    cross_ds = [test1_results[n]['d_cross'] for n in ['SRKW', 'TKW', 'OKW']]
    all_cross_positive = all(d > 0.2 for d in cross_ds)

    if all_cross_positive:
        print(f"\n  *** VERDICT: Acoustic priming CANNOT explain topology-syntax coupling")
        print(f"  *** Cross-transition d > 0.2 for all ecotypes")
    else:
        weak = [n for n, d in zip(['SRKW', 'TKW', 'OKW'], cross_ds) if d <= 0.2]
        print(f"\n  WARNING: Cross-transition effect weak for: {', '.join(weak)}")

    print()


if __name__ == "__main__":
    main()
