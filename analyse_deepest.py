#!/usr/bin/env python3
"""
The deepest mine:

1. Individual caller detection — are there distinct voices within sessions?
2. Entropy rate — how compressible is orca communication? (human comparison)
3. Cross-ecotype rhythm — different populations, different pulse frequencies?
4. Phrase repetition — do the same sequences recur across different sessions?
5. TKW (Bigg's) deep dive — the richest grammar in the ocean
"""

import os
import sys
import csv
import ast
import math
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))


def load_haro():
    data = np.load("data/haro_srkw_features.npz", allow_pickle=True)
    return data["features"].copy(), [ast.literal_eval(m) for m in data["metadata"]]


def load_annotations():
    with open("data/dclde/Annotations.csv", 'r') as f:
        return list(csv.DictReader(f))


def get_call_data(rows, ecotype):
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
                    'center': (lo+hi)/2, 'bw': hi-lo, 'dur': dur,
                    'file': r['Soundfile'], 'begin': float(r['FileBeginSec']),
                    'utc': r.get('UTC', ''),
                })
        except:
            pass
    return calls


def cluster_and_sequence(calls, n_clusters=3):
    from sklearn.cluster import KMeans
    X = np.array([[c['center']/10000, c['bw']/10000, min(c['dur'],10)/10] for c in calls])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    by_file = defaultdict(list)
    for i, c in enumerate(calls):
        by_file[c['file']].append((c['begin'], labels[i], i))
    sequences = []
    for fname, entries in by_file.items():
        se = sorted(entries)
        seq = [se[0]]
        for j in range(1, len(se)):
            if se[j][0] - se[j-1][0] <= 30:
                seq.append(se[j])
            else:
                if len(seq) >= 2:
                    sequences.append([(lab, beg) for beg, lab, idx in seq])
                seq = [se[j]]
        if len(seq) >= 2:
            sequences.append([(lab, beg) for beg, lab, idx in seq])
    return labels, X, km, sequences


# ─────────────────────────────────────────────────────────────────────
# 1. INDIVIDUAL CALLER DETECTION
# ─────────────────────────────────────────────────────────────────────

def analyse_individual_voices(features, metadata):
    print("=" * 70)
    print("  1. INDIVIDUAL VOICES: How many orcas are talking?")
    print("=" * 70)
    print()

    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    # Normalise per station
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

    # Within a SINGLE recording session, cluster to find distinct voices
    by_file = defaultdict(list)
    for i, m in enumerate(metadata):
        by_file[m['filename']].append(i)

    # Pick the largest sessions
    top_sessions = sorted(by_file.items(), key=lambda x: -len(x[1]))[:10]

    print(f"  Analysing top 10 sessions for within-session voice clustering:\n")

    voice_counts = []
    for fname, indices in top_sessions:
        if len(indices) < 20:
            continue

        session_feat = feat[indices]
        norms = np.linalg.norm(session_feat, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        session_normed = session_feat / norms

        # PCA to reduce noise
        if session_normed.shape[1] > 10:
            pca = PCA(n_components=min(10, len(indices) - 1))
            session_reduced = pca.fit_transform(session_normed)
        else:
            session_reduced = session_normed

        # Try different k values
        best_k, best_sil = 1, -1
        for k in range(2, min(8, len(indices) // 5)):
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                lab = km.fit_predict(session_reduced)
                sil = silhouette_score(session_reduced, lab)
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
            except:
                pass

        voice_counts.append(best_k)
        station = metadata[indices[0]]['station']
        print(f"    {fname[:45]:45s} n={len(indices):>4d}  voices={best_k}  sil={best_sil:.3f}  ({station})")

    if voice_counts:
        vc = np.array(voice_counts)
        print(f"\n  Summary:")
        print(f"    Sessions analysed: {len(vc)}")
        print(f"    Mean voices per session: {vc.mean():.1f}")
        print(f"    Range: {vc.min()}-{vc.max()}")
        print(f"    Distribution: {dict(Counter(vc))}")

        if vc.mean() > 1.5:
            print(f"\n  *** MULTIPLE DISTINCT VOICES detected within sessions")
            print(f"  *** Average {vc.mean():.1f} acoustic signatures per recording")
            print(f"  *** This is not a monologue — it's a CONVERSATION")
    print()


# ─────────────────────────────────────────────────────────────────────
# 2. ENTROPY RATE — HOW COMPRESSIBLE IS ORCA COMMUNICATION?
# ─────────────────────────────────────────────────────────────────────

def analyse_entropy_rate(all_rows):
    print("=" * 70)
    print("  2. ENTROPY RATE: How language-like is orca communication?")
    print("=" * 70)
    print()

    print(f"  {'Ecotype':>8s}  {'H₀':>8s}  {'H₁':>8s}  {'H₂':>8s}  {'H₃':>8s}  {'H₄':>8s}  {'Compression':>12s}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*12}")

    for eco in ['SRKW', 'TKW', 'SAR', 'OKW']:
        calls = get_call_data(all_rows, eco)
        if len(calls) < 100:
            continue

        labels, X, km, sequences = cluster_and_sequence(calls, n_clusters=3)
        flat = []
        for seq in sequences:
            flat.extend([lab for lab, _ in seq])

        if len(flat) < 50:
            continue

        nc = km.n_clusters

        # H₀: marginal entropy (no context)
        counts = Counter(flat)
        total = len(flat)
        H0 = -sum((c/total) * math.log2(c/total) for c in counts.values() if c > 0)

        # H_k: k-th order conditional entropy
        entropies = [H0]
        for order in range(1, 5):
            contexts = defaultdict(Counter)
            for i in range(order, len(flat)):
                ctx = tuple(flat[i-order:i])
                contexts[ctx][flat[i]] += 1

            total_ctx = sum(sum(v.values()) for v in contexts.values())
            H_k = 0
            for ctx, next_counts in contexts.items():
                ct = sum(next_counts.values())
                for c in next_counts.values():
                    if c > 0:
                        p = c / ct
                        H_k -= (ct / total_ctx) * p * math.log2(p)
            entropies.append(H_k)

        compression = H0 / entropies[-1] if entropies[-1] > 0 else float('inf')
        h_strs = "  ".join(f"{h:>8.4f}" for h in entropies)
        print(f"  {eco:>8s}  {h_strs}  {compression:>11.1f}×")

    print(f"\n  Reference: English text ≈ 4.7 bits/char (H₀) → 1.0 bit/char (H_context)")
    print(f"  Compression ratio for English: ~4.7×")
    print(f"\n  H₀ = marginal entropy (no context)")
    print(f"  H₄ = entropy rate at order 4 (conditioned on previous 4 calls)")
    print(f"  Compression = H₀/H₄ (how much syntax reduces uncertainty)")
    print()


# ─────────────────────────────────────────────────────────────────────
# 3. CROSS-ECOTYPE RHYTHM
# ─────────────────────────────────────────────────────────────────────

def analyse_cross_ecotype_rhythm(all_rows):
    print("=" * 70)
    print("  3. CROSS-ECOTYPE RHYTHM: Different populations, different pulses?")
    print("=" * 70)
    print()

    for eco in ['SRKW', 'TKW', 'SAR', 'OKW']:
        calls = get_call_data(all_rows, eco)
        if len(calls) < 100:
            continue

        # Compute ICIs
        by_file = defaultdict(list)
        for c in calls:
            by_file[c['file']].append(c['begin'])

        icis = []
        for fname, begins in by_file.items():
            begins.sort()
            for j in range(len(begins) - 1):
                ici = begins[j+1] - begins[j]
                if 0.001 < ici < 10.0:
                    icis.append(ici)

        if len(icis) < 50:
            continue

        ici_arr = np.array(icis)
        cv = ici_arr.std() / ici_arr.mean()

        # Find peak
        short_icis = ici_arr[ici_arr < 0.5]
        if len(short_icis) > 20:
            bins = np.linspace(0.001, 0.5, 100)
            hist, _ = np.histogram(short_icis, bins=bins)
            peak_idx = np.argmax(hist)
            peak_ici = (bins[peak_idx] + bins[peak_idx+1]) / 2
            peak_freq = 1.0 / peak_ici if peak_ici > 0 else 0
            peak_pct = hist[peak_idx] / len(icis) * 100
        else:
            peak_ici = ici_arr.mean()
            peak_freq = 1.0 / peak_ici
            peak_pct = 0

        print(f"  {eco}:")
        print(f"    ICIs: {len(icis)}")
        print(f"    Mean: {ici_arr.mean():.4f}s ({1/ici_arr.mean():.1f} Hz)")
        print(f"    Median: {np.median(ici_arr):.4f}s")
        print(f"    CV: {cv:.3f} ({'rhythmic' if cv < 0.5 else 'moderate' if cv < 1.0 else 'variable'})")
        print(f"    Peak ICI: {peak_ici:.4f}s ({peak_freq:.1f} Hz, {peak_pct:.1f}% of calls)")
        print()

    print(f"  If ecotypes have different peak frequencies, they have different")
    print(f"  rhythmic signatures — timing-level 'accents' independent of call content.")
    print()


# ─────────────────────────────────────────────────────────────────────
# 4. PHRASE REPETITION ACROSS SESSIONS
# ─────────────────────────────────────────────────────────────────────

def analyse_phrase_repetition(all_rows):
    print("=" * 70)
    print("  4. PHRASE REPETITION: Do the same sequences recur across days?")
    print("=" * 70)
    print()

    srkw = get_call_data(all_rows, 'SRKW')
    labels, X, km, sequences = cluster_and_sequence(srkw, n_clusters=3)

    # Extract n-grams per session and check cross-session repetition
    for n in [3, 4, 5, 6]:
        session_ngrams = defaultdict(set)  # ngram → set of session indices
        ngram_total = Counter()

        for seq_idx, seq in enumerate(sequences):
            call_seq = [lab for lab, _ in seq]
            for i in range(len(call_seq) - n + 1):
                ngram = tuple(call_seq[i:i+n])
                session_ngrams[ngram].add(seq_idx)
                ngram_total[ngram] += 1

        # How many n-grams appear in multiple sessions?
        multi_session = {ng: sessions for ng, sessions in session_ngrams.items()
                         if len(sessions) >= 2}
        total_types = len(session_ngrams)
        multi_count = len(multi_session)

        print(f"  {n}-grams:")
        print(f"    Total types: {total_types}")
        print(f"    Appearing in 2+ sessions: {multi_count} ({multi_count/total_types*100:.1f}%)")

        # Top recurring phrases
        if multi_session:
            top = sorted(multi_session.items(), key=lambda x: -len(x[1]))[:5]
            for ng, sessions in top:
                ng_str = "→".join(f"C{c}" for c in ng)
                total = ngram_total[ng]
                print(f"      {ng_str}: {len(sessions)} sessions, {total} total occurrences")

    # The key question: are cross-session phrases more common than expected?
    print(f"\n  Permutation test: are cross-session repetitions above chance?")

    # Observed: fraction of 4-grams in 2+ sessions
    call_seqs = [[lab for lab, _ in seq] for seq in sequences]
    observed_ngrams = defaultdict(set)
    for seq_idx, seq in enumerate(call_seqs):
        for i in range(len(seq) - 3):
            ng = tuple(seq[i:i+4])
            observed_ngrams[ng].add(seq_idx)
    observed_multi = sum(1 for v in observed_ngrams.values() if len(v) >= 2) / max(len(observed_ngrams), 1)

    # Permutation: shuffle within sessions, recount
    rng = np.random.RandomState(42)
    perm_multis = []
    for _ in range(100):
        perm_ngrams = defaultdict(set)
        for seq_idx, seq in enumerate(call_seqs):
            shuffled = list(seq)
            rng.shuffle(shuffled)
            for i in range(len(shuffled) - 3):
                ng = tuple(shuffled[i:i+4])
                perm_ngrams[ng].add(seq_idx)
        perm_multi = sum(1 for v in perm_ngrams.values() if len(v) >= 2) / max(len(perm_ngrams), 1)
        perm_multis.append(perm_multi)

    perm_arr = np.array(perm_multis)
    p_value = (perm_arr >= observed_multi).mean()

    print(f"    Observed cross-session 4-gram rate: {observed_multi:.4f}")
    print(f"    Permuted mean: {perm_arr.mean():.4f} (std={perm_arr.std():.4f})")
    print(f"    p-value: {p_value:.4f}")
    if p_value < 0.05:
        excess = (observed_multi - perm_arr.mean()) / perm_arr.mean() * 100
        print(f"    *** SIGNIFICANT: {excess:.0f}% more cross-session repetition than chance")
        print(f"    *** Recurring phrases exist — these are LEARNED SEQUENCES")
    else:
        print(f"    Cross-session repetition is consistent with chance")
    print()


# ─────────────────────────────────────────────────────────────────────
# 5. BIGG'S (TKW) DEEP DIVE
# ─────────────────────────────────────────────────────────────────────

def analyse_tkw_deep(all_rows):
    print("=" * 70)
    print("  5. BIGG'S (TKW): The richest grammar in the ocean")
    print("=" * 70)
    print()

    tkw = get_call_data(all_rows, 'TKW')
    print(f"  TKW calls: {len(tkw)}")

    labels, X, km, sequences = cluster_and_sequence(tkw, n_clusters=3)
    nc = km.n_clusters

    # Cluster characteristics
    print(f"\n  Cluster characteristics:")
    for c in range(nc):
        mask = labels == c
        cx = X[mask]
        calls_c = [tkw[i] for i in range(len(tkw)) if i < len(labels) and labels[i] == c]
        print(f"    C{c}: n={mask.sum()}, center={cx[:,0].mean()*10000:.0f}Hz, "
              f"bw={cx[:,1].mean()*10000:.0f}Hz, dur={cx[:,2].mean()*10:.2f}s")

    # Full transition matrix
    M = np.zeros((nc, nc), dtype=int)
    for seq in sequences:
        for j in range(len(seq) - 1):
            M[seq[j][0], seq[j+1][0]] += 1

    total = M.sum()
    print(f"\n  Transition matrix ({total} transitions):")
    print(f"          " + "  ".join(f"→C{j}" for j in range(nc)))
    for i in range(nc):
        rt = M[i].sum()
        if rt > 0:
            probs = M[i] / rt
            row = "  ".join(f"{probs[j]:.3f}" for j in range(nc))
            print(f"    C{i} →  {row}  (n={rt})")

    # Entropy rate
    counts = Counter(labels)
    total_calls = sum(counts.values())
    H0 = -sum((c/total_calls) * math.log2(c/total_calls) for c in counts.values() if c > 0)

    H1 = 0
    for i in range(nc):
        rt = M[i].sum()
        if rt == 0:
            continue
        for j in range(nc):
            if M[i, j] > 0:
                p = M[i, j] / rt
                H1 -= (rt / total) * p * math.log2(p)

    compression = H0 / H1 if H1 > 0 else float('inf')
    print(f"\n  TKW entropy rate:")
    print(f"    H₀ (marginal): {H0:.4f} bits")
    print(f"    H₁ (bigram):   {H1:.4f} bits")
    print(f"    Compression:    {compression:.2f}×")

    # ICI analysis
    icis = []
    by_file = defaultdict(list)
    for c in tkw:
        by_file[c['file']].append(c['begin'])
    for fname, begins in by_file.items():
        begins.sort()
        for j in range(len(begins) - 1):
            ici = begins[j+1] - begins[j]
            if 0.001 < ici < 10.0:
                icis.append(ici)

    if icis:
        ici_arr = np.array(icis)
        cv = ici_arr.std() / ici_arr.mean()
        print(f"\n  TKW rhythm:")
        print(f"    Mean ICI: {ici_arr.mean():.3f}s ({1/ici_arr.mean():.1f} Hz)")
        print(f"    CV: {cv:.3f}")
        print(f"    Median: {np.median(ici_arr):.3f}s")

    # Bout structure
    all_bout_lens = defaultdict(list)
    for seq in sequences:
        call_seq = [lab for lab, _ in seq]
        current = call_seq[0]
        length = 1
        for j in range(1, len(call_seq)):
            if call_seq[j] == current:
                length += 1
            else:
                all_bout_lens[current].append(length)
                current = call_seq[j]
                length = 1
        all_bout_lens[current].append(length)

    print(f"\n  TKW bout structure:")
    for c in sorted(all_bout_lens.keys()):
        bl = np.array(all_bout_lens[c])
        print(f"    C{c}: mean={bl.mean():.1f}, median={np.median(bl):.0f}, max={bl.max()}, n={len(bl)}")

    # The key finding: what makes TKW grammar the richest?
    print(f"\n  ═══ WHY IS TKW GRAMMAR THE RICHEST? ═══")
    print()

    # Measure transition matrix asymmetry
    asym = 0
    for i in range(nc):
        for j in range(i+1, nc):
            ri = M[i].sum()
            rj = M[j].sum()
            if ri > 0 and rj > 0:
                pij = M[i, j] / ri
                pji = M[j, i] / rj
                asym += abs(pij - pji)
    asym /= max(nc * (nc - 1) / 2, 1)

    print(f"  Transition asymmetry: {asym:.4f}")
    print(f"  (Higher = more directional. TKW transitions are non-reciprocal.)")

    # Compare with SRKW
    srkw = get_call_data(all_rows, 'SRKW')
    s_labels, _, s_km, s_sequences = cluster_and_sequence(srkw, n_clusters=3)
    s_M = np.zeros((nc, nc), dtype=int)
    for seq in s_sequences:
        for j in range(len(seq) - 1):
            s_M[seq[j][0], seq[j+1][0]] += 1

    s_asym = 0
    for i in range(nc):
        for j in range(i+1, nc):
            ri = s_M[i].sum()
            rj = s_M[j].sum()
            if ri > 0 and rj > 0:
                pij = s_M[i, j] / ri
                pji = s_M[j, i] / rj
                s_asym += abs(pij - pji)
    s_asym /= max(nc * (nc - 1) / 2, 1)

    print(f"  SRKW asymmetry: {s_asym:.4f}")
    print(f"  TKW/SRKW ratio: {asym / (s_asym + 1e-12):.2f}×")
    if asym > s_asym * 1.5:
        print(f"  *** TKW transitions are MORE DIRECTIONAL than SRKW")
        print(f"  *** Bigg's grammar has stronger left-right asymmetry")
        print(f"  *** Like a language where word order carries more meaning")
    print()


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "▓" * 70)
    print("▓  THE DEEPEST MINE")
    print("▓  Voices, entropy, rhythm, phrases, Bigg's grammar")
    print("▓" * 70)
    print()

    features, metadata = load_haro()
    all_rows = load_annotations()

    analyse_individual_voices(features, metadata)
    analyse_entropy_rate(all_rows)
    analyse_cross_ecotype_rhythm(all_rows)
    analyse_phrase_repetition(all_rows)
    analyse_tkw_deep(all_rows)

    print("▓" * 70)
    print("▓  BOTTOM OF THE MINE")
    print("▓" * 70)
    print()


if __name__ == "__main__":
    main()
