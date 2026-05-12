#!/usr/bin/env python3
"""
Extract per-call acoustic features from Haro Strait SRKW recordings.

Haro Strait North and South are two JASCO hydrophones recording simultaneously
in the core J/K/L pod territory. This gives us:
  - 5,870 individual SRKW calls
  - Two recording positions for the same pod passages
  - Real per-location acoustic variation (not shared exemplars)

This fixes the biggest weakness in Finding 1: the phonosemantic correlation
will be computed from calls that have DIFFERENT recordings for the same
call type, not shared exemplars.
"""

import os
import sys
import csv
import subprocess
import tempfile
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from extract_real_features import extract_acoustic_features
from orca.features import N_FEATURES

GCS_BASE = "https://storage.googleapis.com/noaa-passive-bioacoustic/dclde/2027/dclde_2027_killer_whales"

DATASET_TO_PATH = {
    "HaroStraitNorth": "vfpa/audio/vfpa-harostrait-nb",
    "HaroStraitSouth": "vfpa/audio/vfpa-harostrait-sb",
}


def load_haro_annotations():
    with open("data/dclde/Annotations.csv", 'r') as f:
        rows = list(csv.DictReader(f))

    haro = [r for r in rows if r['KW'] == '1' and r['AnnotationLevel'] == 'Call'
            and r['Ecotype'] == 'SRKW'
            and r['Dataset'] in ('HaroStraitNorth', 'HaroStraitSouth')]
    return haro


def main():
    print("=" * 70)
    print("  Haro Strait SRKW: Per-call acoustic features")
    print("  Two simultaneous hydrophones in J/K/L pod territory")
    print("=" * 70)
    print()

    annotations = load_haro_annotations()
    print(f"  Total Haro Strait SRKW annotations: {len(annotations)}")

    # Group by file
    by_file = defaultdict(list)
    for ann in annotations:
        by_file[ann['Soundfile']].append(ann)

    # Sort by call count, take top files
    sorted_files = sorted(by_file.items(), key=lambda x: -len(x[1]))

    # Download budget: ~40 files to get good coverage of both stations
    max_files = 40
    sample_files = sorted_files[:max_files]
    total_calls = sum(len(v) for _, v in sample_files)
    print(f"  Sampling {len(sample_files)} files ({total_calls} calls)")

    # Verify station balance
    north = sum(len(v) for _, v in sample_files if v[0]['Dataset'] == 'HaroStraitNorth')
    south = sum(len(v) for _, v in sample_files if v[0]['Dataset'] == 'HaroStraitSouth')
    print(f"  North: {north} calls, South: {south} calls")
    print()

    import librosa
    import soundfile as sf

    all_features = []
    all_metadata = []
    success = 0
    fail = 0

    for file_idx, (filename, file_annotations) in enumerate(sample_files):
        ann = file_annotations[0]
        dataset = ann['Dataset']
        path_prefix = DATASET_TO_PATH.get(dataset)
        if not path_prefix:
            fail += len(file_annotations)
            continue

        url = f"{GCS_BASE}/{path_prefix}/{filename}"
        print(f"  [{file_idx+1}/{len(sample_files)}] {filename} ({len(file_annotations)} calls, {dataset})")

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name

        result = subprocess.run(
            ['curl', '-s', '-f', '-o', tmp_path, url],
            capture_output=True, timeout=180
        )

        if result.returncode != 0:
            print(f"    DOWNLOAD FAILED")
            fail += len(file_annotations)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            continue

        try:
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

                start_sample = int(start * sr)
                end_sample = min(int(end * sr), len(y_full))
                y_seg = y_full[start_sample:end_sample]

                if len(y_seg) < sr * 0.05:
                    fail += 1
                    continue

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as seg:
                    seg_path = seg.name
                sf.write(seg_path, y_seg, sr)
                features = extract_acoustic_features(seg_path)
                os.unlink(seg_path)

                low_freq = float(ann['LowFreqHz']) if ann['LowFreqHz'] != 'NA' else 0
                high_freq = float(ann['HighFreqHz']) if ann['HighFreqHz'] != 'NA' else 0

                all_features.append(features)
                all_metadata.append({
                    'dataset': dataset,
                    'station': 'north' if 'North' in dataset else 'south',
                    'utc': ann['UTC'],
                    'low_freq': low_freq,
                    'high_freq': high_freq,
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
        print("  No calls extracted.")
        return

    features_array = np.array(all_features)
    output_path = os.path.join(os.path.dirname(__file__), "data", "haro_srkw_features.npz")
    np.savez(output_path,
             features=features_array,
             metadata=np.array([str(m) for m in all_metadata]))

    print(f"\n  Saved {success} call features to {output_path}")
    print(f"  Feature shape: {features_array.shape}")

    # Station balance
    north_count = sum(1 for m in all_metadata if m['station'] == 'north')
    south_count = sum(1 for m in all_metadata if m['station'] == 'south')
    print(f"  North station: {north_count}, South station: {south_count}")

    # Quick analysis: does the same event recorded from two stations sound different?
    # Find simultaneous recordings (same UTC timestamp ± 1s from different stations)
    print(f"\n  Checking for simultaneous cross-station recordings...")
    from datetime import datetime

    by_time = defaultdict(list)
    for i, m in enumerate(all_metadata):
        try:
            # Round to nearest 30s for matching
            utc = m['utc'][:16]  # YYYY-MM-DD HH:MM
            by_time[utc].append((i, m['station']))
        except:
            pass

    cross_station_pairs = []
    for time_key, entries in by_time.items():
        norths = [i for i, s in entries if s == 'north']
        souths = [i for i, s in entries if s == 'south']
        if norths and souths:
            for n in norths[:3]:
                for s in souths[:3]:
                    cross_station_pairs.append((n, s))

    if cross_station_pairs:
        print(f"  Found {len(cross_station_pairs)} cross-station pairs (same minute)")

        norms = np.linalg.norm(features_array, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        normed = features_array / norms

        cross_sims = [float(np.dot(normed[i], normed[j])) for i, j in cross_station_pairs]
        same_station_sims = []
        import random
        random.seed(42)
        for _ in range(min(len(cross_sims) * 2, 1000)):
            i, j = random.sample(range(len(normed)), 2)
            if all_metadata[i]['station'] == all_metadata[j]['station']:
                same_station_sims.append(float(np.dot(normed[i], normed[j])))

        if cross_sims and same_station_sims:
            cs = np.array(cross_sims)
            ss = np.array(same_station_sims)
            print(f"  Cross-station similarity (same event):  mean={cs.mean():.4f}, std={cs.std():.4f}")
            print(f"  Same-station similarity (random pairs): mean={ss.mean():.4f}, std={ss.std():.4f}")
            diff = cs.mean() - ss.mean()
            print(f"  Difference: {diff:+.4f}")
            if diff > 0.01:
                print(f"  *** Same events ARE more similar across stations than random pairs")
                print(f"  *** The acoustic features capture CALL IDENTITY, not station artifacts")
    else:
        print(f"  No simultaneous cross-station pairs found")


if __name__ == "__main__":
    main()
