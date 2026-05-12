"""
Dual semantic field: the bridge between orca acoustics and cognition.

Each call type becomes a vector. Compounds become operations on vectors.
Reaction-diffusion dynamics over the field find stable attractors.
Attractors ARE call meanings. Comprehension IS relaxation to the nearest attractor.

The key innovation over the lor-engine's SemanticField: TWO fields running
in parallel — acoustic topology (what calls sound like) and context topology
(when calls are used). The combined field is the working space.

If acoustic and context distances correlate, orcas have sound symbolism.
If they don't, the dual-field separation is doing real work.
"""

import math
import numpy as np
from .kernel import OrcaKernel
from .features import (
    extract_features, extract_acoustic_only, extract_context_only,
    N_FEATURES, N_ACOUSTIC, N_CONTEXT,
)


class _SubField:
    """
    A single semantic field — projected feature vectors + R/D dynamics.

    This is the internal workhorse, equivalent to lor-engine's SemanticField.
    The OrcaField wraps two of these (acoustic + context) plus a combined view.
    """

    def __init__(self, raw_features: np.ndarray, dim: int, seed: int):
        self.dim = dim
        self.n_roots = raw_features.shape[0]
        self.n_raw = raw_features.shape[1]

        # Johnson-Lindenstrauss random projection (seeded for reproducibility)
        rng = np.random.RandomState(seed=seed)
        self._projection = rng.randn(self.n_raw, dim) / math.sqrt(dim)

        # Project and normalize to unit hypersphere
        self.root_vectors = raw_features @ self._projection
        norms = np.linalg.norm(self.root_vectors, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        self.root_vectors = self.root_vectors / norms


class OrcaField:
    """
    The dual semantic field of orca communication.

    Two sub-fields (acoustic + context) project into separate spaces.
    The combined field merges them for R/D dynamics.
    All lor-engine SemanticField methods are available on the combined field.
    """

    def __init__(self, kernel: OrcaKernel, dim: int = 64, seed: int = None,
                 acoustic_weight: float = 0.6, context_weight: float = 0.4):
        """
        Initialize the dual field from a kernel.

        dim: embedding dimension for each sub-field.
        seed: random seed (defaults to kernel size, matching lor-engine pattern).
        acoustic_weight: weight for acoustic field in combined space.
        context_weight: weight for context field in combined space.
        """
        self.kernel = kernel
        self.dim = dim
        self.n_roots = len(kernel)
        self.acoustic_weight = acoustic_weight
        self.context_weight = context_weight

        if seed is None:
            seed = len(kernel)

        # Extract features for all roots
        all_roots = kernel.all_roots()
        raw_features = np.array([extract_features(r) for r in all_roots])
        acoustic_features = np.array([extract_acoustic_only(r) for r in all_roots])
        context_features = np.array([extract_context_only(r) for r in all_roots])

        # Build three sub-fields with different seeds
        self._acoustic = _SubField(acoustic_features, dim, seed=seed)
        self._context = _SubField(context_features, dim, seed=seed + 1)

        # Combined field: weighted blend of acoustic and context vectors
        combined_raw = (
            acoustic_weight * self._acoustic.root_vectors +
            context_weight * self._context.root_vectors
        )
        norms = np.linalg.norm(combined_raw, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        self.root_vectors = combined_raw / norms

        # Build index: root_id → row index
        self._id_to_idx = {r.id: i for i, r in enumerate(all_roots)}
        self._idx_to_id = {i: r.id for i, r in enumerate(all_roots)}

        # Store raw features for inspection
        self.raw_features = raw_features

    # ─── Vector Access ────────────────────────────────────────────

    def vector(self, root_id: int) -> np.ndarray:
        """Get the normalized combined vector for a call type."""
        return self.root_vectors[self._id_to_idx[root_id]].copy()

    def acoustic_vector(self, root_id: int) -> np.ndarray:
        """Get the acoustic-only vector for a call type."""
        return self._acoustic.root_vectors[self._id_to_idx[root_id]].copy()

    def context_vector(self, root_id: int) -> np.ndarray:
        """Get the context-only vector for a call type."""
        return self._context.root_vectors[self._id_to_idx[root_id]].copy()

    # ─── Compound Operations ──────────────────────────────────────

    def compose(self, modifier_id: int, head_id: int, alpha: float = 0.4) -> np.ndarray:
        """
        Compute the compound vector: modifier modifies head.

        Head-dominant (weight 1-alpha), modifier inflects (weight alpha).
        Nonlinear interaction term captures emergent meaning.
        """
        v_mod = self.vector(modifier_id)
        v_head = self.vector(head_id)

        blend = (1.0 - alpha) * v_head + alpha * v_mod
        interaction = v_mod * v_head
        compound_vec = blend + 0.3 * interaction

        norm = np.linalg.norm(compound_vec)
        if norm > 0:
            compound_vec = compound_vec / norm
        return compound_vec

    def compose_triple(self, r1_id: int, r2_id: int, r3_id: int) -> np.ndarray:
        """Compute a three-call compound vector: r1 modifies (r2 modifies r3)."""
        inner = self.compose(r2_id, r3_id)
        v_r1 = self.vector(r1_id)
        blend = 0.7 * inner + 0.3 * v_r1
        interaction = v_r1 * inner
        result = blend + 0.2 * interaction
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        return result

    # ─── Similarity & Nearest Attractor ───────────────────────────

    def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        d1 = np.linalg.norm(v1)
        d2 = np.linalg.norm(v2)
        if d1 == 0 or d2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (d1 * d2))

    def acoustic_similarity(self, id_a: int, id_b: int) -> float:
        """Cosine similarity in acoustic space only."""
        return self.similarity(self.acoustic_vector(id_a), self.acoustic_vector(id_b))

    def context_similarity(self, id_a: int, id_b: int) -> float:
        """Cosine similarity in context space only."""
        return self.similarity(self.context_vector(id_a), self.context_vector(id_b))

    def nearest_roots(self, v: np.ndarray, k: int = 5) -> list[tuple[int, str, float]]:
        """
        Find the k nearest call-type attractors to a point in the field.

        Returns: [(root_id, call_type, similarity), ...]
        """
        v_norm = v / (np.linalg.norm(v) + 1e-12)
        sims = self.root_vectors @ v_norm
        top_k = np.argsort(sims)[-k:][::-1]

        results = []
        for idx in top_k:
            rid = self._idx_to_id[idx]
            root = self.kernel.by_id(rid)
            results.append((rid, root.call_type, float(sims[idx])))
        return results

    # ─── Reaction-Diffusion Dynamics ──────────────────────────────

    def activate(self, root_ids: list[int], strengths: list[float] | None = None) -> np.ndarray:
        """
        Create an activation pattern from multiple call types.

        Multiple calls active simultaneously create an interference pattern.
        The field will relax to the nearest attractor.
        """
        if strengths is None:
            strengths = [1.0] * len(root_ids)

        activation = np.zeros(self.dim)
        for rid, s in zip(root_ids, strengths):
            activation += s * self.vector(rid)

        norm = np.linalg.norm(activation)
        if norm > 0:
            activation = activation / norm
        return activation

    def diffuse(
        self,
        field_state: np.ndarray,
        dt: float = 0.01,
        diffusion_rate: float = 0.1,
        feed_rate: float = 0.04,
        kill_rate: float = 0.06,
    ) -> np.ndarray:
        """
        One step of Gray-Scott-inspired R/D dynamics.

        Same parameters as lor-engine. The activator is the field state.
        The inhibitor is the distance from the nearest attractor.
        """
        nearest = self.nearest_roots(field_state, k=3)
        if not nearest:
            return field_state

        # Reaction: pull toward nearest attractor
        attractor = self.vector(nearest[0][0])
        reaction = attractor - field_state

        # Inhibition from competing attractors
        inhibition = np.zeros(self.dim)
        for rid, call_type, sim in nearest[1:]:
            competitor = self.vector(rid)
            inhibition += sim * (competitor - field_state)

        # Gray-Scott dynamics
        attractor_pull = diffusion_rate * reaction
        competition = 0.05 * inhibition
        feed = feed_rate * (1.0 - np.linalg.norm(field_state)) * field_state
        kill = -kill_rate * field_state

        update = field_state + dt * (attractor_pull + competition + feed + kill)

        norm = np.linalg.norm(update)
        if norm > 0:
            update = update / norm
        return update

    def relax(
        self,
        initial: np.ndarray,
        steps: int = 100,
        dt: float = 0.01,
        convergence_threshold: float = 1e-4,
        **kwargs,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Relax the field from an initial state to the nearest attractor.

        Returns: (final_state, trajectory)
        """
        state = initial.copy()
        trajectory = [state.copy()]

        for step in range(steps):
            new_state = self.diffuse(state, dt=dt, **kwargs)
            delta = np.linalg.norm(new_state - state)
            state = new_state
            trajectory.append(state.copy())

            if delta < convergence_threshold:
                break

        return state, trajectory

    def think(self, root_ids: list[int], steps: int = 200) -> list[tuple[int, str, float]]:
        """
        Activate multiple call types and let the field find what emerges.

        Returns: nearest call types to the converged state.
        """
        activation = self.activate(root_ids)
        converged, trajectory = self.relax(activation, steps=steps)
        return self.nearest_roots(converged, k=10)

    # ─── Dual-Field Analysis ─────────────────────────────────────

    def phonosemantic_correlation(self) -> float:
        """
        Compute correlation between acoustic and context distance matrices.

        If high: calls that sound similar ARE used similarly (sound symbolism).
        If low: acoustic form and function are independent (arbitrary signifiers).
        This is the key diagnostic for orca language structure.
        """
        n = self.n_roots
        if n < 3:
            return 0.0

        # Compute pairwise distances in each sub-field
        acoustic_dists = []
        context_dists = []
        for i in range(n):
            for j in range(i + 1, n):
                id_i = self._idx_to_id[i]
                id_j = self._idx_to_id[j]
                acoustic_dists.append(1.0 - self.acoustic_similarity(id_i, id_j))
                context_dists.append(1.0 - self.context_similarity(id_i, id_j))

        if len(acoustic_dists) < 2:
            return 0.0

        # Pearson correlation
        a = np.array(acoustic_dists)
        c = np.array(context_dists)
        a_mean = a - np.mean(a)
        c_mean = c - np.mean(c)
        denom = np.linalg.norm(a_mean) * np.linalg.norm(c_mean)
        if denom < 1e-12:
            return 0.0
        return float(np.dot(a_mean, c_mean) / denom)

    def distance_matrix(self) -> np.ndarray:
        """Compute pairwise cosine distance matrix between all call types."""
        sim = self.root_vectors @ self.root_vectors.T
        return 1.0 - sim

    def pod_cluster(self, pod: str, k: int = 10) -> list[tuple[str, float]]:
        """
        Find call types that cluster near a pod's centroid.

        Analogous to lor-engine's constellation() method.
        """
        matching = []
        for root in self.kernel.by_pod(pod):
            matching.append(root.id)

        if not matching:
            return []

        centroid = np.mean([self.vector(rid) for rid in matching], axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        results = self.nearest_roots(centroid, k=k)
        return [(call_type, sim) for _, call_type, sim in results]
