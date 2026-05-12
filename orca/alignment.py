"""
Cross-pod alignment: translating between orca dialects via topology.

No lor-engine equivalent. This is the payoff — the thing that makes
the R/D approach genuinely novel for cetacean communication research.

The insight: you don't translate words. You align topologies. Find the
rotation/reflection that maps one pod's semantic field onto another's.
The mapping IS the translation.

Validation: aligned calls should share behavioural contexts. If they
do, the topology is real. If they don't, we learned something too.
"""

import numpy as np
from dataclasses import dataclass, field
from .kernel import OrcaKernel
from .field import OrcaField


@dataclass
class AlignmentResult:
    """Result of aligning two pod fields."""
    rotation: np.ndarray                    # (dim, dim) orthogonal matrix
    scale: float                            # scaling factor
    disparity: float                        # Procrustes distance (lower = better)
    aligned_vectors: np.ndarray             # source vectors after alignment
    shared_calls: list[tuple[int, int]]     # (source_id, target_id) matched pairs
    context_agreement: float                # fraction of pairs sharing context


def align_pods(
    source_field: OrcaField,
    target_field: OrcaField,
    source_ids: list[int] | None = None,
    target_ids: list[int] | None = None,
    anchor_pairs: list[tuple[int, int]] | None = None,
    reduce_dim: int | None = 10,
) -> AlignmentResult:
    """
    Find the rotation that best aligns source pod's field to target pod's field.

    If anchor_pairs is given, uses only those as correspondence points.
    If anchor_pairs is None, uses Hungarian algorithm on similarity matrix
    to find optimal pairing, then Procrustes to find optimal rotation.

    reduce_dim: reduce to this dimensionality before alignment (PCA).
        With only 12-15 call types per pod in 64D, Procrustes is
        underdetermined. PCA reduction mitigates this.

    Returns AlignmentResult with the rotation, matched pairs, and validation score.
    """
    if source_ids is None:
        source_ids = [r.id for r in source_field.kernel.all_roots()]
    if target_ids is None:
        target_ids = [r.id for r in target_field.kernel.all_roots()]

    # Get vectors
    source_vecs = np.array([source_field.vector(rid) for rid in source_ids])
    target_vecs = np.array([target_field.vector(rid) for rid in target_ids])

    dim = source_vecs.shape[1]

    # Optional PCA reduction
    if reduce_dim is not None and reduce_dim < dim:
        # PCA on combined data
        combined = np.vstack([source_vecs, target_vecs])
        mean = np.mean(combined, axis=0)
        centered = combined - mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        basis = Vt[:reduce_dim]  # top reduce_dim components

        source_reduced = (source_vecs - mean) @ basis.T
        target_reduced = (target_vecs - mean) @ basis.T
        work_dim = reduce_dim
    else:
        source_reduced = source_vecs
        target_reduced = target_vecs
        work_dim = dim
        basis = None
        mean = None

    if anchor_pairs is not None:
        # Use provided correspondences
        src_idx = [source_ids.index(s) for s, _ in anchor_pairs]
        tgt_idx = [target_ids.index(t) for _, t in anchor_pairs]
        S = source_reduced[src_idx]
        T = target_reduced[tgt_idx]
        matched_pairs = anchor_pairs
    else:
        # Find optimal pairing via similarity matrix + greedy matching
        n_src = len(source_ids)
        n_tgt = len(target_ids)

        # Compute cross-similarity matrix
        src_norms = np.linalg.norm(source_reduced, axis=1, keepdims=True)
        src_norms = np.where(src_norms > 0, src_norms, 1.0)
        tgt_norms = np.linalg.norm(target_reduced, axis=1, keepdims=True)
        tgt_norms = np.where(tgt_norms > 0, tgt_norms, 1.0)

        sim_matrix = (source_reduced / src_norms) @ (target_reduced / tgt_norms).T

        # Greedy matching (Hungarian would be better but this suffices)
        matched_pairs = []
        used_tgt = set()
        n_pairs = min(n_src, n_tgt)
        sim_flat = []
        for i in range(n_src):
            for j in range(n_tgt):
                sim_flat.append((sim_matrix[i, j], i, j))
        sim_flat.sort(reverse=True)

        used_src = set()
        for sim_val, i, j in sim_flat:
            if i in used_src or j in used_tgt:
                continue
            matched_pairs.append((source_ids[i], target_ids[j]))
            used_src.add(i)
            used_tgt.add(j)
            if len(matched_pairs) >= n_pairs:
                break

        src_idx = [source_ids.index(s) for s, _ in matched_pairs]
        tgt_idx = [target_ids.index(t) for _, t in matched_pairs]
        S = source_reduced[src_idx]
        T = target_reduced[tgt_idx]

    # Orthogonal Procrustes: find R such that ||S @ R - T||² is minimized
    try:
        from scipy.linalg import orthogonal_procrustes
        R, scale = orthogonal_procrustes(S, T)
    except ImportError:
        # Fallback: SVD-based Procrustes
        M = S.T @ T
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        scale = 1.0

    # Apply rotation
    aligned_source = source_reduced @ R

    # Compute disparity (Procrustes distance)
    aligned_matched = S @ R
    disparity = float(np.linalg.norm(aligned_matched - T) / max(np.linalg.norm(T), 1e-12))

    # Validate: check context agreement
    context_agreement = _compute_context_agreement(
        matched_pairs, source_field.kernel, target_field.kernel
    )

    # Lift rotation back to full dimension if PCA was applied
    if basis is not None:
        R_full = basis.T @ R @ basis
    else:
        R_full = R

    return AlignmentResult(
        rotation=R_full,
        scale=float(scale) if isinstance(scale, (int, float)) else 1.0,
        disparity=disparity,
        aligned_vectors=aligned_source,
        shared_calls=matched_pairs,
        context_agreement=context_agreement,
    )


def cross_pod_similarity(
    result: AlignmentResult,
    source_field: OrcaField,
    target_field: OrcaField,
) -> list[tuple[str, str, float]]:
    """
    After alignment, report which call types in source correspond to which in target.

    Returns: [(source_call_type, target_call_type, similarity), ...]
    """
    similarities = []
    for src_id, tgt_id in result.shared_calls:
        src_root = source_field.kernel.by_id(src_id)
        tgt_root = target_field.kernel.by_id(tgt_id)

        src_vec = source_field.vector(src_id)
        tgt_vec = target_field.vector(tgt_id)
        sim = float(np.dot(src_vec, tgt_vec) / (
            np.linalg.norm(src_vec) * np.linalg.norm(tgt_vec) + 1e-12
        ))

        similarities.append((
            f"{src_root.call_type}({src_root.pod})",
            f"{tgt_root.call_type}({tgt_root.pod})",
            sim,
        ))

    return sorted(similarities, key=lambda x: -x[2])


def validate_alignment(
    result: AlignmentResult,
    source_kernel: OrcaKernel,
    target_kernel: OrcaKernel,
) -> float:
    """
    Validate: aligned calls should share behavioural contexts.

    Returns fraction of aligned pairs with at least one matching context tag.
    """
    return _compute_context_agreement(result.shared_calls, source_kernel, target_kernel)


def _compute_context_agreement(
    pairs: list[tuple[int, int]],
    source_kernel: OrcaKernel,
    target_kernel: OrcaKernel,
) -> float:
    """Compute fraction of pairs sharing at least one context tag."""
    if not pairs:
        return 0.0

    agreements = 0
    for src_id, tgt_id in pairs:
        try:
            src_contexts = set(source_kernel.by_id(src_id).contexts)
            tgt_contexts = set(target_kernel.by_id(tgt_id).contexts)
            if src_contexts & tgt_contexts:
                agreements += 1
        except KeyError:
            pass

    return agreements / len(pairs)
