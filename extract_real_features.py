#!/usr/bin/env python3
"""
Extract REAL acoustic features from the Ford-Osborne 2018 audio exemplars.

Maps each audio file to its call type, extracts 50D feature vectors using
librosa, and saves to a numpy file that the orca-engine can load.

This replaces mock_features() with real spectral, temporal, modulation,
and fingerprint features extracted from actual orca vocalisation recordings.
"""

import os
import sys
import re
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import librosa
from orca.features import (
    N_FEATURES, N_SPECTRAL, N_TEMPORAL, N_COMPONENT, N_MODULATION,
    N_STRUCTURAL, N_SPECTRAL_BANDS,
)

# Map audio filenames to canonical call types
# Some files have variants (S06-a, S06-b, S33-RJ-a, etc.)
# We take the first/primary exemplar for each call type.
AUDIO_DIR = os.path.join(
    os.path.dirname(__file__),
    "data", "signals-srkw", "signals", "processed", "calls", "Ford-Osborne-2018", "mp3"
)

# Primary exemplar for each call type (prefer base form over variants)
EXEMPLAR_MAP = {
    "S01": "FO-S01.mp3",
    "S02": "FO-S02.mp3",
    "S03": "FO-S03.mp3",
    "S04": "FO-S04.mp3",
    "S05": "FO-S05.mp3",
    "S06": "FO-S06-a.mp3",
    "S07": "FO-S07.mp3",
    "S09": "FO-S09.mp3",
    "S10": "FO-S10.mp3",
    "S13": "FO-S13.mp3",
    "S14": "FO-S14.mp3",
    "S16": "FO-S16.mp3",
    "S17": "FO-S17.mp3",
    "S18": "FO-S18-R.mp3",
    "S19": "FO-S19.mp3",
    "S22": "FO-S22-a.mp3",
    "S30": "FO-S30.mp3",
    "S31": "FO-S31-a.mp3",
    "S33": "FO-S33-R.mp3",
    "S34": "FO-S34-R.mp3",
    "S36": "FO-S36.mp3",
    "S37": "FO-S37.mp3",
    "S38": "FO-S38-a.mp3",
    "S39": "FO-S39.mp3",
    "S40": "FO-S40-R-a.mp3",
    "S42": "FO-S42.mp3",
    "S44": "FO-S44-a.mp3",
    "S45": "FO-S45.mp3",
    "S46": "FO-S46.mp3",
}


def extract_acoustic_features(audio_path: str) -> np.ndarray:
    """
    Extract 50D acoustic feature vector from an audio file using librosa.

    Returns: (N_FEATURES,) numpy array
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    if len(y) == 0:
        return np.zeros(N_FEATURES)

    # ─── Spectral shape (6D) ─────────────────────────────────────
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    spectral_flux = np.mean(np.diff(np.abs(librosa.stft(y=y)), axis=1))
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    spectral = np.array([
        spectral_centroid / 11025.0,    # normalize by Nyquist
        spectral_bandwidth / 11025.0,
        spectral_rolloff / 22050.0,     # normalize by sample rate
        spectral_flux,
        spectral_contrast / 50.0,       # typical range ~0-50 dB
        spectral_flatness,              # already 0-1
    ])

    # ─── Temporal envelope (5D) ──────────────────────────────────
    # Soft classification: how pulsed, tonal, mixed, burst-like, silent
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]

    rms_std = np.std(rms) / (np.mean(rms) + 1e-12)  # coefficient of variation
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr) / (np.mean(zcr) + 1e-12)

    # Heuristic soft classification
    pulsed = min(rms_std, 1.0)                    # high amplitude variation = pulsed
    tonal = max(0, 1.0 - zcr_std)                 # stable zero-crossing = tonal
    mixed = min(pulsed * tonal, 1.0)              # both pulsed and tonal
    burst = min(max(rms_std - 1.0, 0) / 2.0, 1.0)  # extreme amplitude variation
    silence = max(0, 1.0 - np.mean(rms) * 20)    # low energy = silence

    total = pulsed + tonal + mixed + burst + silence + 1e-12
    temporal = np.array([pulsed, tonal, mixed, burst, silence]) / total

    # ─── Click vs tonal components (2D) ──────────────────────────
    # Use onset strength for click-like content
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    click_energy = np.mean(onset_env) / (np.max(onset_env) + 1e-12)
    tonal_energy = 1.0 - click_energy

    components = np.array([click_energy, tonal_energy])

    # ─── FM contour (3D) ─────────────────────────────────────────
    # Pitch tracking for frequency modulation
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=100, fmax=10000, sr=sr
    )
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([0])
    if len(f0_valid) == 0:
        f0_valid = np.array([0])

    mean_freq = np.mean(f0_valid) / 10000.0       # normalize by fmax
    mod_depth = np.std(f0_valid) / (np.mean(f0_valid) + 1e-12)  # relative modulation
    # Modulation rate: how fast the pitch changes
    if len(f0_valid) > 1:
        mod_rate = np.mean(np.abs(np.diff(f0_valid))) / (np.mean(f0_valid) + 1e-12)
    else:
        mod_rate = 0.0

    modulation = np.array([mean_freq, min(mod_depth, 2.0) / 2.0, min(mod_rate, 2.0) / 2.0])

    # ─── Structural (8D) ─────────────────────────────────────────
    # These come from the catalogue metadata, not audio — filled in later
    structural = np.zeros(N_STRUCTURAL)
    structural[0] = len(y) / sr  # duration in seconds

    # ─── Spectral fingerprint (26D) ──────────────────────────────
    # Log-mel energy in 26 bands
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_SPECTRAL_BANDS)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    fingerprint = np.mean(mel_db, axis=1)  # average over time
    # Normalize to [0, 1] range
    fp_min = fingerprint.min()
    fp_max = fingerprint.max()
    if fp_max - fp_min > 0:
        fingerprint = (fingerprint - fp_min) / (fp_max - fp_min)
    else:
        fingerprint = np.zeros(N_SPECTRAL_BANDS)

    return np.concatenate([spectral, temporal, components, modulation,
                           structural, fingerprint])


def main():
    print("=" * 60)
    print("  Extracting REAL acoustic features from Ford-Osborne audio")
    print("=" * 60)
    print()

    if not os.path.isdir(AUDIO_DIR):
        print(f"  ERROR: Audio directory not found: {AUDIO_DIR}")
        print(f"  Run: git clone https://github.com/orcasound/signals-srkw data/signals-srkw")
        return

    features = {}
    errors = []

    for call_type, filename in sorted(EXEMPLAR_MAP.items()):
        path = os.path.join(AUDIO_DIR, filename)
        if not os.path.exists(path):
            errors.append(f"  MISSING: {call_type} → {filename}")
            continue

        try:
            feat = extract_acoustic_features(path)
            features[call_type] = feat
            # Print summary
            duration = feat[N_SPECTRAL + N_TEMPORAL + N_COMPONENT + N_MODULATION]
            spectral_centroid = feat[0] * 11025
            print(f"  {call_type:5s}  duration={duration:.2f}s  centroid={spectral_centroid:.0f}Hz  "
                  f"fingerprint_energy={np.mean(feat[-N_SPECTRAL_BANDS:]):.3f}")
        except Exception as e:
            errors.append(f"  ERROR: {call_type} → {e}")

    if errors:
        print(f"\n  Issues:")
        for e in errors:
            print(f"    {e}")

    # Save features
    output_path = os.path.join(os.path.dirname(__file__), "data", "srkw_acoustic_features.npz")
    np.savez(output_path,
             call_types=np.array(list(features.keys())),
             features=np.array(list(features.values())))

    print(f"\n  Extracted features for {len(features)} call types")
    print(f"  Saved to: {output_path}")
    print(f"  Feature dimensions: {N_FEATURES}")

    # Quick sanity check: which calls are most/least similar acoustically?
    if len(features) >= 2:
        print(f"\n  Acoustic similarity (top 5 most similar pairs):")
        types = list(features.keys())
        vecs = np.array(list(features.values()))
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        vecs_norm = vecs / norms

        sim_matrix = vecs_norm @ vecs_norm.T
        pairs = []
        for i in range(len(types)):
            for j in range(i + 1, len(types)):
                pairs.append((types[i], types[j], sim_matrix[i, j]))
        pairs.sort(key=lambda x: -x[2])

        for ct1, ct2, sim in pairs[:5]:
            print(f"    {ct1} ↔ {ct2}: {sim:.4f}")

        print(f"\n  Acoustic similarity (top 5 most DIFFERENT pairs):")
        for ct1, ct2, sim in pairs[-5:]:
            print(f"    {ct1} ↔ {ct2}: {sim:.4f}")


if __name__ == "__main__":
    main()
