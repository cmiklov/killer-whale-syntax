"""
Opcode discovery: finding systematic modification axes in orca vocalisations.

The lor-engine defines 7 opcodes (I/L/LI/IL/LL/II/E). The orca-engine
DISCOVERS its opcodes from data by clustering the delta vectors between
discrete calls and their variable-call variants.

A variable call is a recognisable variant of a discrete call type —
same basic structure but with acoustic modifications (pitch shift,
duration change, added pulsing, etc.). If these modifications are
systematic (the same kind of change applied to different base calls),
they are opcodes.

Until real data is available, the 7 Genesis opcodes serve as a
structural scaffold.
"""

import numpy as np
from dataclasses import dataclass, field
from .kernel import OrcaKernel
from .phonology import (
    _L, _I, _LI, _IL, _LL, _II, _E, _BANG, N_OPS,
    COMPOUND_TABLE, COMPLEMENT_TABLE, MODIFIER_TABLE,
    degeminate_opcodes, opcode_name, opcode_energy,
)


# ─── Genesis Fallback Opcodes ────────────────────────────────────────

class Opcode:
    """Fallback opcode using the 7 Genesis opcodes."""
    L = _L
    I = _I
    LI = _LI
    IL = _IL
    LL = _LL
    II = _II
    E = _E
    BANG = _BANG

    ALL = [_L, _I, _LI, _IL, _LL, _II, _E]
    NAMES = {_L: 'L', _I: 'I', _LI: 'LI', _IL: 'IL', _LL: 'LL', _II: 'II', _E: 'E', _BANG: 'BANG'}


# ─── Discovered Opcodes ──────────────────────────────────────────────

@dataclass
class DiscoveredOpcode:
    """An opcode discovered from clustering variable-call delta vectors."""
    id: int
    name: str                          # auto-generated or human-assigned
    delta_centroid: np.ndarray         # average delta vector for this transformation
    examples: list[tuple[int, int]] = field(default_factory=list)  # (base_id, variant_id)
    productivity: float = 0.0          # fraction of roots this applies to
    consistency: float = 0.0           # angular variance (lower = more consistent)

    def __repr__(self):
        return (f"DiscoveredOpcode({self.id}, '{self.name}', "
                f"productivity={self.productivity:.2f}, consistency={self.consistency:.3f})")


class OpcodeDiscovery:
    """
    Discover opcodes from variable calls by clustering delta vectors.

    Algorithm:
    1. For each variable call, find nearest discrete call (its "base")
    2. Compute delta = variable_vector - base_vector
    3. Cluster deltas (DBSCAN — no k required, handles noise)
    4. Each cluster = candidate opcode
    5. Validate: productive (multiple bases), consistent (low angular variance)
    """

    def __init__(self, field):
        self.field = field
        self.discovered: list[DiscoveredOpcode] = []

    def discover(
        self,
        variable_calls: list,
        discrete_ids: list[int] | None = None,
        min_cluster_size: int = 3,
        eps: float = 0.5,
    ) -> list[DiscoveredOpcode]:
        """
        Discover opcodes from variable call data.

        variable_calls: list of OrcaRoot objects representing variable calls.
        discrete_ids: IDs of discrete (base) call types. If None, uses all kernel roots.
        min_cluster_size: minimum examples for a valid opcode.
        eps: DBSCAN neighborhood radius.

        Returns list of DiscoveredOpcode.
        """
        if discrete_ids is None:
            discrete_ids = [r.id for r in self.field.kernel.all_roots()]

        # Get discrete call vectors
        discrete_vectors = np.array([self.field.vector(rid) for rid in discrete_ids])

        # Compute deltas: for each variable call, find nearest discrete and compute delta
        deltas = []
        pairs = []  # (base_id, variable_call)
        for vc in variable_calls:
            if vc.features is None or len(vc.features) == 0:
                continue

            # Find nearest discrete call by cosine similarity
            vc_vec = self.field.vector(vc.id) if vc.id in self.field._id_to_idx else None
            if vc_vec is None:
                continue

            sims = discrete_vectors @ vc_vec
            best_idx = np.argmax(sims)
            base_id = discrete_ids[best_idx]

            delta = vc_vec - self.field.vector(base_id)
            deltas.append(delta)
            pairs.append((base_id, vc.id))

        if len(deltas) < min_cluster_size:
            self.discovered = []
            return []

        deltas = np.array(deltas)

        # Cluster deltas using DBSCAN
        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=eps, min_samples=min_cluster_size, metric='cosine')
            labels = clustering.fit_predict(deltas)
        except ImportError:
            # Fallback: simple k-means-like clustering
            labels = self._simple_cluster(deltas, min_cluster_size)

        # Build opcodes from clusters
        opcodes = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # noise

        for label_id in sorted(unique_labels):
            mask = labels == label_id
            cluster_deltas = deltas[mask]
            cluster_pairs = [pairs[i] for i in range(len(pairs)) if mask[i]]

            centroid = np.mean(cluster_deltas, axis=0)

            # Consistency: angular variance of deltas around centroid
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
            angles = []
            for d in cluster_deltas:
                d_norm = d / (np.linalg.norm(d) + 1e-12)
                cos_angle = np.clip(np.dot(d_norm, centroid_norm), -1, 1)
                angles.append(np.arccos(cos_angle))
            consistency = float(np.std(angles)) if angles else 1.0

            # Productivity: unique base roots involved
            unique_bases = len(set(p[0] for p in cluster_pairs))
            productivity = unique_bases / max(len(discrete_ids), 1)

            opcodes.append(DiscoveredOpcode(
                id=int(label_id),
                name=f"OP{label_id}",
                delta_centroid=centroid,
                examples=cluster_pairs,
                productivity=productivity,
                consistency=consistency,
            ))

        self.discovered = opcodes
        return opcodes

    def apply_opcode(self, root_id: int, opcode: DiscoveredOpcode) -> np.ndarray:
        """Apply a discovered opcode to a call type vector."""
        v = self.field.vector(root_id)
        result = v + opcode.delta_centroid
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        return result

    def detect_opcode(self, base_id: int, variant_id: int) -> DiscoveredOpcode | None:
        """Detect which opcode transforms base into variant."""
        if not self.discovered:
            return None

        delta = self.field.vector(variant_id) - self.field.vector(base_id)
        delta_norm = delta / (np.linalg.norm(delta) + 1e-12)

        best_opcode = None
        best_sim = -1.0
        for op in self.discovered:
            centroid_norm = op.delta_centroid / (np.linalg.norm(op.delta_centroid) + 1e-12)
            sim = float(np.dot(delta_norm, centroid_norm))
            if sim > best_sim:
                best_sim = sim
                best_opcode = op

        return best_opcode if best_sim > 0.5 else None

    def _simple_cluster(self, deltas: np.ndarray, min_size: int) -> np.ndarray:
        """Fallback clustering when sklearn is not available."""
        # Simple greedy clustering by cosine similarity
        n = len(deltas)
        labels = np.full(n, -1)
        norms = np.linalg.norm(deltas, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        normalized = deltas / norms

        cluster_id = 0
        for i in range(n):
            if labels[i] != -1:
                continue
            # Find all points similar to this one
            sims = normalized @ normalized[i]
            similar = np.where(sims > 0.5)[0]
            if len(similar) >= min_size:
                for j in similar:
                    if labels[j] == -1:
                        labels[j] = cluster_id
                cluster_id += 1

        return labels
