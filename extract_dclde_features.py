#!/usr/bin/env python3
"""
Download SRKW audio segments from the DCLDE 2027 dataset and extract
per-call acoustic features using librosa.

This is the real deal: 14,240 SRKW call annotations with precise timing,
backed by hydrophone recordings from the Salish Sea. Each call is extracted
as a segment, features are computed, and the result is a feature matrix
where rows represent individual orca calls — not call types, CALLS.

We sample across providers/locations to get diversity, then aggregate
features per call type for topology analysis.
"""

import os
import sys
import csv
import time
import numpy as np
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from extract_real_features import extract_acoustic_features
from orca.features import N_FEATURES

GCS_BASE = "https://storage.googleapis.com/noaa-passive-bioacoustic/dclde/2027/dclde_2027_killer_whales"

# Map provider names to GCS path prefixes
PROVIDER_TO_PATH = {
    "OrcaSound":       "orcasound/audio",
    "JASCO_VFPA":      "vfpa/audio",
    "JASCO_VFPA_ONC":  "vfpa/audio",
    "SMRUConsulting":   "smru/audio",
    "SIMRES":          "simres/audio",
    "DFO_CRP":         "dfo_crp/audio",
    "ONC":             "onc/audio",
    "SIO":             "scripps/audio",
}

# Map dataset names to subdirectories (best guess from file paths)
DATASET_TO_SUBDIR = {
    "orcasound_lab":    "orcasound_lab",
    "bush_point":       "bush_point",
    "port_townsend":    "port_townsend",
    "HaroStraitNorth":  "VFPA-HaroStrait-NB",
    "HaroStraitSouth":  "VFPA-HaroStrait-SB",
    "BoundaryPass":     "BoundaryPass",
    "LimeKiln":         "Lime Kiln",
    "Tekteksen":        "tekteksen",
}


def load_srkw_annotations():
    """Load SRKW call-level annotations from the DCLDE CSV."""
    path = os.path.join(os.path.dirname(__file__), "data", "dclde", "Annotations.csv")
    with open(path, 'r') as f:
        rows = list(csv.DictReader(f))

    srkw = [r for r in rows if r['KW'] == '1'
            and r['AnnotationLevel'] == 'Call'
            and r['Ecotype'] == 'SRKW']
    return srkw


def gcs_audio_url(annotation):
    """Construct GCS URL for an annotation's audio file."""
    provider = annotation['Provider']
    dataset = annotation['Dataset']
    filename = annotation['Soundfile']

    provider_path = PROVIDER_TO_PATH.get(provider)
    if not provider_path:
        return None

    subdir = DATASET_TO_SUBDIR.get(dataset, dataset.lower())
    return f"{GCS_BASE}/{provider_path}/{subdir}/{filename}"


def download_and_extract(url, start_sec, end_sec, max_retries=2):
    """Download audio, extract segment, compute features."""
    import subprocess
    import librosa

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Download with curl
        result = subprocess.run(
            ['curl', '-s', '-f', '-o', tmp_path, url],
            capture_output=True, timeout=60
        )
        if result.returncode != 0:
            return None

        # Load and extract segment
        y, sr = librosa.load(tmp_path, sr=22050, mono=True,
                             offset=start_sec, duration=end_sec - start_sec)
        if len(y) < sr * 0.05:  # less than 50ms = too short
            return None

        # Write segment to temp file for feature extraction
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as seg:
            seg_path = seg.name
        import soundfile as sf
        sf.write(seg_path, y, sr)

        features = extract_acoustic_features(seg_path)
        os.unlink(seg_path)
        return features

    except Exception as e:
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    print("=" * 60)
    print("  DCLDE SRKW: Real per-call acoustic feature extraction")
    print("=" * 60)
    print()

    annotations = load_srkw_annotations()
    print(f"  Total SRKW call annotations: {len(annotations)}")

    # Sample strategy: take calls from multiple locations for diversity
    # Focus on Orcasound (easy path mapping) and a sample from JASCO/SMRU
    target_datasets = ['orcasound_lab', 'bush_point', 'port_townsend']
    target_calls = [a for a in annotations if a['Dataset'] in target_datasets]
    print(f"  Orcasound calls (lab + bush_point + port_townsend): {len(target_calls)}")

    # Group by audio file to minimize downloads
    by_file = {}
    for ann in target_calls:
        fname = ann['Soundfile']
        if fname not in by_file:
            by_file[fname] = []
        by_file[fname].append(ann)

    print(f"  Unique audio files: {len(by_file)}")

    # Sample: take up to 30 files (to keep download reasonable)
    # Prioritize files with many calls (more data per download)
    sorted_files = sorted(by_file.items(), key=lambda x: -len(x[1]))
    sample_files = sorted_files[:30]
    total_calls_in_sample = sum(len(v) for _, v in sample_files)
    print(f"  Sampling {len(sample_files)} files ({total_calls_in_sample} calls)")
    print()

    all_features = []
    all_metadata = []
    success = 0
    fail = 0

    for file_idx, (filename, file_annotations) in enumerate(sample_files):
        ann = file_annotations[0]
        url = gcs_audio_url(ann)
        if not url:
            fail += len(file_annotations)
            continue

        print(f"  [{file_idx+1}/{len(sample_files)}] {filename} ({len(file_annotations)} calls)")

        # Download once, extract segments
        import subprocess, tempfile, librosa, soundfile as sf

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name

        result = subprocess.run(
            ['curl', '-s', '-f', '-o', tmp_path, url],
            capture_output=True, timeout=120
        )

        if result.returncode != 0:
            print(f"    DOWNLOAD FAILED: {url[:80]}")
            fail += len(file_annotations)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            continue

        try:
            file_size = os.path.getsize(tmp_path)
            y_full, sr = librosa.load(tmp_path, sr=22050, mono=True)
            file_dur = len(y_full) / sr
        except Exception as e:
            print(f"    LOAD FAILED: {e}")
            fail += len(file_annotations)
            os.unlink(tmp_path)
            continue

        for ann in file_annotations:
            try:
                start = float(ann['FileBeginSec'])
                end = float(ann['FileEndSec'])
                dur = end - start
                if dur < 0.05 or dur > 30 or start >= file_dur:
                    fail += 1
                    continue

                # Extract segment
                start_sample = int(start * sr)
                end_sample = min(int(end * sr), len(y_full))
                y_seg = y_full[start_sample:end_sample]

                if len(y_seg) < sr * 0.05:
                    fail += 1
                    continue

                # Write segment and extract features
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as seg:
                    seg_path = seg.name
                sf.write(seg_path, y_seg, sr)
                features = extract_acoustic_features(seg_path)
                os.unlink(seg_path)

                all_features.append(features)
                all_metadata.append({
                    'dataset': ann['Dataset'],
                    'provider': ann['Provider'],
                    'utc': ann['UTC'],
                    'low_freq': ann['LowFreqHz'],
                    'high_freq': ann['HighFreqHz'],
                    'duration': dur,
                    'filename': filename,
                })
                success += 1

            except Exception as e:
                fail += 1

        os.unlink(tmp_path)
        print(f"    extracted {success} calls so far ({fail} failed)")

    print(f"\n  Results: {success} calls extracted, {fail} failed")

    if success == 0:
        print("  No calls extracted. Check GCS paths.")
        return

    # Save
    features_array = np.array(all_features)
    output_path = os.path.join(os.path.dirname(__file__), "data", "dclde_srkw_features.npz")
    np.savez(output_path,
             features=features_array,
             metadata=np.array([str(m) for m in all_metadata]))

    print(f"\n  Saved {success} call features to {output_path}")
    print(f"  Feature shape: {features_array.shape}")

    # Quick analysis: spectral diversity
    centroids = features_array[:, 0] * 11025  # de-normalize spectral centroid
    print(f"\n  Spectral centroid across all calls:")
    print(f"    min={centroids.min():.0f} Hz, mean={centroids.mean():.0f} Hz, max={centroids.max():.0f} Hz")
    print(f"    std={centroids.std():.0f} Hz")

    # Pairwise similarity distribution
    norms = np.linalg.norm(features_array, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normed = features_array / norms

    # Sample pairwise similarities (full matrix too big)
    n = min(500, len(normed))
    sample_sims = []
    rng = np.random.RandomState(42)
    indices = rng.choice(len(normed), n, replace=False) if len(normed) > n else range(len(normed))
    sub = normed[indices]
    sim_matrix = sub @ sub.T
    upper = sim_matrix[np.triu_indices(len(sub), k=1)]
    print(f"\n  Pairwise acoustic similarity ({len(upper)} pairs sampled):")
    print(f"    mean={upper.mean():.4f}, std={upper.std():.4f}")
    print(f"    min={upper.min():.4f}, max={upper.max():.4f}")


if __name__ == "__main__":
    main()
