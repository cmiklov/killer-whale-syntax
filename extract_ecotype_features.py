#!/usr/bin/env python3
"""
Extract full 50D acoustic features for TKW, SAR, and OKW from DCLDE audio.

This upgrades the cross-ecotype comparison from 3D annotation metadata
to full 50D librosa features — the same quality as Haro Strait SRKW.

Strategy: sample the top files per ecotype (most calls per file = most
data per download). Target ~500-1000 calls per ecotype.
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

# Provider → GCS path mapping
PROVIDER_PATH = {
    "SIO":            "scripps/audio",
    "UAF_NGOS":       "uaf/audio",
    "ONC":            "onc/audio",
    "JASCO_VFPA":     "vfpa/audio",
    "JASCO_VFPA_ONC": "vfpa/audio",
    "SIMRES":         "simres/audio",
}

# Dataset → subdirectory mapping (lowercase for GCS)
DATASET_SUBDIR = {
    "Cpe_Elz":           "ce",
    "Quin_Can":          "qc",
    "MS_671879205":      "ms",
    "KB_67424266":       "kb",
    "HE_67424266":       "he",
    "BarkleyCanyon":     "barkleycanyon",
    "StraitofGeorgia":   "straitofgeorgia_globus-robertsbank",
    "Field_HTI":         "field",
    "Field_SondTrap":    "field",
    "KB_67383303":       "kb",
    "MS_6897":           "ms",
    "HaroStraitSouth":   "vfpa-harostrait-sb",
    "BoundaryPass":      "boundarypass",
    "Tekteksen":         "eastpoint",
}


def load_ecotype_annotations(ecotype):
    with open("data/dclde/Annotations.csv", 'r') as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r['KW'] == '1' and r['AnnotationLevel'] == 'Call'
            and r['Ecotype'] == ecotype]


def gcs_url(ann):
    provider = ann['Provider']
    dataset = ann['Dataset']
    filename = ann['Soundfile']

    provider_path = PROVIDER_PATH.get(provider)
    if not provider_path:
        return None

    subdir = DATASET_SUBDIR.get(dataset, dataset.lower())
    return f"{GCS_BASE}/{provider_path}/{subdir}/{filename}"


def extract_ecotype(ecotype, max_files=25):
    """Extract features for one ecotype."""
    print(f"\n{'='*70}")
    print(f"  {ecotype}: Extracting full 50D features")
    print(f"{'='*70}\n")

    annotations = load_ecotype_annotations(ecotype)
    print(f"  Total {ecotype} call annotations: {len(annotations)}")

    # Group by file, take top files
    by_file = defaultdict(list)
    for ann in annotations:
        by_file[ann['Soundfile']].append(ann)

    sorted_files = sorted(by_file.items(), key=lambda x: -len(x[1]))
    sample_files = sorted_files[:max_files]
    total_calls = sum(len(v) for _, v in sample_files)
    print(f"  Sampling {len(sample_files)} files ({total_calls} calls)")

    import librosa
    import soundfile as sf

    all_features = []
    all_metadata = []
    success = 0
    fail = 0

    for file_idx, (filename, file_anns) in enumerate(sample_files):
        ann = file_anns[0]
        url = gcs_url(ann)
        if not url:
            print(f"  [{file_idx+1}/{len(sample_files)}] {filename[:50]} — NO PATH")
            fail += len(file_anns)
            continue

        print(f"  [{file_idx+1}/{len(sample_files)}] {filename[:50]} ({len(file_anns)} calls)")

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                ['curl', '-s', '-f', '--max-time', '600', '-o', tmp_path, url],
                capture_output=True, timeout=660
            )
        except subprocess.TimeoutExpired:
            result = type('R', (), {'returncode': 1})()

        if result.returncode != 0:
            # Try .flac extension
            flac_url = url.rsplit('.', 1)[0] + '.flac' if '.' in url else url + '.flac'
            try:
                result = subprocess.run(
                    ['curl', '-s', '-f', '--max-time', '600', '-o', tmp_path, flac_url],
                    capture_output=True, timeout=660
                )
            except subprocess.TimeoutExpired:
                result = type('R', (), {'returncode': 1})()
            if result.returncode != 0:
                print(f"    DOWNLOAD FAILED")
                fail += len(file_anns)
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                continue

        try:
            y_full, sr = librosa.load(tmp_path, sr=22050, mono=True)
            file_dur = len(y_full) / sr
        except Exception as e:
            print(f"    LOAD FAILED: {e}")
            fail += len(file_anns)
            os.unlink(tmp_path)
            continue

        for ann in file_anns:
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

                all_features.append(features)
                all_metadata.append({
                    'ecotype': ecotype,
                    'dataset': ann['Dataset'],
                    'provider': ann['Provider'],
                    'utc': ann.get('UTC', ''),
                    'duration': dur,
                    'filename': filename,
                })
                success += 1
            except Exception as e:
                fail += 1

        os.unlink(tmp_path)
        print(f"    {success} extracted ({fail} failed)")

    print(f"\n  {ecotype} results: {success} calls, {fail} failed")

    if success > 0:
        features_array = np.array(all_features)
        output_path = f"data/{ecotype.lower()}_features.npz"
        np.savez(output_path,
                 features=features_array,
                 metadata=np.array([str(m) for m in all_metadata]))
        print(f"  Saved to {output_path} ({features_array.shape})")

    return success, fail


def main():
    print("▓" * 70)
    print("▓  FULL 50D FEATURES: TKW + SAR + OKW")
    print("▓  Upgrading cross-ecotype comparison from 3D metadata to 50D audio")
    print("▓" * 70)

    results = {}
    import sys
    ecotypes = sys.argv[1:] if len(sys.argv) > 1 else ['TKW', 'SAR', 'OKW']
    for eco in ecotypes:
        s, f = extract_ecotype(eco, max_files=20)
        results[eco] = (s, f)

    print(f"\n{'▓'*70}")
    print(f"▓  EXTRACTION COMPLETE")
    print(f"{'▓'*70}\n")
    for eco, (s, f) in results.items():
        print(f"  {eco}: {s} calls extracted, {f} failed")
    print()


if __name__ == "__main__":
    main()
