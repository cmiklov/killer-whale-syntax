#!/usr/bin/env python3
"""
Finding 46: Do sessions trace similar paths through acoustic space?

Track the 50D centroid as it moves over the course of each session.
If sessions follow similar trajectories, there's a shared narrative
template — conversations aren't just structured, they're patterned.
"""

import os
import sys
import ast
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))


def load_and_normalise():
    data = np.load("data/haro_srkw_features.npz", allow_pickle=True)
    features = data["features"].copy()
    metadata = [ast.literal_eval(m) for m in data["metadata"]]

    by_station = defaultdict(list)
    for i, m in enumerate(metadata):
        by_station[m['station']].append(i)
    for station, indices in by_station.items():
        sf = features[indices]
        mean, std = sf.mean(axis=0), sf.std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        for i in indices:
            features[i] = (features[i] - mean) / std

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.where(norms > 0, norms, 1.0), metadata


def main():
    print("=" * 70)
    print("  ACOUSTIC TRAJECTORIES: Do sessions follow similar paths?")
    print("=" * 70)
    print()

    normed, metadata = load_and_normalise()

    # Build sessions (40+ calls)
    by_file = defaultdict(list)
    for i, m in enumerate(metadata):
        by_file[(m['filename'], m['station'])].append(
            (float(m.get('duration', 0)), i))

    sessions = []
    for key, entries in by_file.items():
        entries.sort()
        if len(entries) >= 40:
            sessions.append([idx for _, idx in entries])

    print(f"  Sessions (40+ calls): {len(sessions)}")
    print(f"  Mean length: {np.mean([len(s) for s in sessions]):.0f} calls")

    # For each session, compute trajectory: rolling centroid in windows of 10
    window = 10
    trajectories = []

    for seq in sessions:
        n = len(seq)
        if n < window * 3:
            continue
        traj = []
        for start in range(0, n - window + 1, window // 2):  # 50% overlap
            chunk = normed[seq[start:start + window]]
            centroid = chunk.mean(axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-12)
            traj.append(centroid)
        if len(traj) >= 4:
            trajectories.append(np.array(traj))

    print(f"  Trajectories extracted: {len(trajectories)}")
    print(f"  Mean trajectory length: {np.mean([len(t) for t in trajectories]):.1f} windows")

    # Compute trajectory similarity: DTW-like comparison
    # Simplified: compare normalised position (0-1) trajectories
    # Resample each trajectory to 10 points
    n_points = 10
    resampled = []
    for traj in trajectories:
        indices = np.linspace(0, len(traj) - 1, n_points).astype(int)
        resampled.append(traj[indices])

    resampled = np.array(resampled)  # (n_sessions, n_points, 50D)

    # Pairwise trajectory similarity: average cosine sim at each position
    n_traj = len(resampled)
    position_sims = np.zeros(n_points)
    pair_count = 0

    for i in range(n_traj):
        for j in range(i + 1, n_traj):
            for p in range(n_points):
                sim = float(np.dot(resampled[i, p], resampled[j, p]))
                position_sims[p] += sim
            pair_count += 1

    position_sims /= max(pair_count, 1)

    print(f"\n  Cross-session similarity by normalised position:")
    print(f"  {'Position':>10s}  {'Similarity':>12s}  {'Bar':>30s}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*30}")

    for p in range(n_points):
        pos = p / (n_points - 1)
        bar = "█" * int(position_sims[p] * 60)
        print(f"  {pos:>10.1f}  {position_sims[p]:>12.6f}  {bar}")

    # Is the trajectory shape consistent?
    # Compare: similarity at START vs MIDDLE vs END
    start_sim = position_sims[:3].mean()
    mid_sim = position_sims[3:7].mean()
    end_sim = position_sims[7:].mean()

    print(f"\n  Phase similarities:")
    print(f"    Start (0.0-0.2): {start_sim:.6f}")
    print(f"    Middle (0.3-0.6): {mid_sim:.6f}")
    print(f"    End (0.7-1.0): {end_sim:.6f}")

    # Step-to-step direction consistency
    # For each trajectory, compute direction vectors between consecutive points
    # Then check if directions are consistent across sessions
    print(f"\n  Direction consistency (do sessions move the SAME WAY?):")

    direction_sims = np.zeros(n_points - 1)
    dir_count = 0

    for i in range(n_traj):
        for j in range(i + 1, n_traj):
            for p in range(n_points - 1):
                dir_i = resampled[i, p + 1] - resampled[i, p]
                dir_j = resampled[j, p + 1] - resampled[j, p]
                norm_i = np.linalg.norm(dir_i)
                norm_j = np.linalg.norm(dir_j)
                if norm_i > 1e-8 and norm_j > 1e-8:
                    direction_sims[p] += float(np.dot(dir_i / norm_i, dir_j / norm_j))
            dir_count += 1

    direction_sims /= max(dir_count, 1)

    print(f"  {'Step':>10s}  {'Direction sim':>14s}  {'Interpretation':>20s}")
    print(f"  {'─'*10}  {'─'*14}  {'─'*20}")

    for p in range(n_points - 1):
        pos = (p + 0.5) / (n_points - 1)
        interp = "same direction" if direction_sims[p] > 0.1 else "random" if direction_sims[p] > -0.1 else "opposite"
        print(f"  {pos:>10.2f}  {direction_sims[p]:>14.6f}  {interp:>20s}")

    mean_dir = direction_sims.mean()
    print(f"\n  Mean direction consistency: {mean_dir:.6f}")
    if mean_dir > 0.05:
        print(f"  *** Sessions move through acoustic space in SIMILAR DIRECTIONS")
        print(f"  *** There is a shared trajectory template")
    elif mean_dir > 0:
        print(f"  * Weak directional consistency")
    else:
        print(f"  No shared trajectory — each session takes its own path")

    print()


if __name__ == "__main__":
    main()
