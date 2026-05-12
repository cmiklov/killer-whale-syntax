"""
Query interface: parse path expressions and dispatch to the engine.

Adapted from lor-engine's navigate.py. Path syntax for orca calls:

  S1                  → look up call type S1
  S1:S7               → compound: S1 modifies S7
  S1:S7:S3            → triple compound
  ?S1                 → reverse lookup (all pods with S1)
  @3                  → look up by root ID
  #VIL                → look up by palindromic address
  pod:J               → all call types for J-pod
  S1.*                → all compounds involving S1 as modifier
  *.S1                → all compounds involving S1 as head
  think:S1,S7         → co-activate S1 and S7, run R/D, see what emerges
  align:J:K           → align J-pod and K-pod fields
  field               → field diagnostics (phonosemantic correlation)
"""

from .kernel import OrcaKernel
from .field import OrcaField
from .trie import Trie, build_trie
from .lookup import decompose_sequence
from .alignment import align_pods, cross_pod_similarity


class OrcaEngine:
    """The orca-engine query interface."""

    def __init__(self, kernel: OrcaKernel = None, catalogue_path: str = None):
        if kernel is not None:
            self.kernel = kernel
        elif catalogue_path is not None:
            self.kernel = OrcaKernel(catalogue_path=catalogue_path)
        else:
            raise ValueError("Must provide kernel or catalogue_path")

        self.field = OrcaField(self.kernel)
        self.trie = build_trie(self.kernel)

        # Cache per-pod fields for alignment
        self._pod_fields: dict[str, OrcaField] = {}

    def _pod_field(self, pod: str) -> OrcaField:
        """Get or create a field for a specific pod."""
        if pod not in self._pod_fields:
            roots = self.kernel.by_pod(pod)
            if not roots:
                return None
            from .kernel import OrcaKernel
            pod_kernel = OrcaKernel(roots=roots)
            self._pod_fields[pod] = OrcaField(pod_kernel)
        return self._pod_fields[pod]

    def query(self, expr: str) -> list[dict]:
        """
        Parse and execute a query expression.

        Returns list of result dicts.
        """
        expr = expr.strip()
        if not expr:
            return [{"error": "Empty query"}]

        # Reverse lookup: ?label
        if expr.startswith("?"):
            return self._reverse_lookup(expr[1:])

        # By ID: @3
        if expr.startswith("@"):
            return self._by_id(expr[1:])

        # By address: #VIL
        if expr.startswith("#"):
            return self._by_address(expr[1:])

        # Pod query: pod:J
        if expr.startswith("pod:"):
            return self._pod_query(expr[4:])

        # Think: think:S1,S7
        if expr.startswith("think:"):
            return self._think(expr[6:])

        # Align: align:J:K
        if expr.startswith("align:"):
            parts = expr[6:].split(":")
            if len(parts) == 2:
                return self._align(parts[0], parts[1])
            return [{"error": f"Align needs two pods: align:J:K"}]

        # Field diagnostics
        if expr == "field":
            return self._field_diagnostics()

        # Wildcard: S1.* or *.S1
        if ".*" in expr:
            root_part = expr.replace(".*", "")
            return self._all_compounds(root_part, position="modifier")
        if "*." in expr:
            root_part = expr.replace("*.", "")
            return self._all_compounds(root_part, position="head")

        # Compound: S1:S7 or S1:S7:S3
        if ":" in expr:
            return self._compound(expr)

        # Single call type lookup
        return self._single_lookup(expr)

    def _single_lookup(self, call_type: str) -> list[dict]:
        matches = self.kernel.by_call_type(call_type)
        if not matches:
            return [{"error": f"Unknown call type: {call_type}"}]

        results = []
        for root in matches:
            nearest = self.field.nearest_roots(self.field.vector(root.id), k=5)
            results.append({
                "call_type": root.call_type,
                "id": root.id,
                "pod": root.pod,
                "address": root.address,
                "description": root.description,
                "contexts": root.contexts,
                "frequency": root.frequency,
                "nearest": [(ct, f"{sim:.3f}") for _, ct, sim in nearest[1:]],
            })
        return results

    def _compound(self, expr: str) -> list[dict]:
        parts = expr.split(":")
        root_ids = []
        for part in parts:
            matches = self.kernel.by_call_type(part.strip())
            if not matches:
                return [{"error": f"Unknown call type: {part}"}]
            root_ids.append(matches[0].id)

        if len(root_ids) == 2:
            cv = self.field.compose(root_ids[0], root_ids[1])
        elif len(root_ids) == 3:
            cv = self.field.compose_triple(root_ids[0], root_ids[1], root_ids[2])
        else:
            return [{"error": "Compounds support 2-3 call types"}]

        nearest = self.field.nearest_roots(cv, k=5)
        return [{
            "compound": expr,
            "root_ids": root_ids,
            "nearest": [(ct, f"{sim:.3f}") for _, ct, sim in nearest],
        }]

    def _reverse_lookup(self, label: str) -> list[dict]:
        return decompose_sequence(label, self.kernel, self.trie)

    def _by_id(self, id_str: str) -> list[dict]:
        try:
            rid = int(id_str)
            root = self.kernel.by_id(rid)
            return self._single_lookup(root.call_type)
        except (ValueError, KeyError):
            return [{"error": f"Unknown ID: {id_str}"}]

    def _by_address(self, addr: str) -> list[dict]:
        root = self.kernel.by_address(addr)
        if root is None:
            return [{"error": f"Unknown address: {addr}"}]
        return self._single_lookup(root.call_type)

    def _pod_query(self, pod: str) -> list[dict]:
        roots = self.kernel.by_pod(pod.strip())
        if not roots:
            return [{"error": f"Unknown pod: {pod}"}]
        return [{
            "pod": pod,
            "count": len(roots),
            "call_types": [
                {"call_type": r.call_type, "id": r.id, "contexts": r.contexts}
                for r in roots
            ],
        }]

    def _think(self, expr: str) -> list[dict]:
        call_types = [ct.strip() for ct in expr.split(",")]
        root_ids = []
        for ct in call_types:
            matches = self.kernel.by_call_type(ct)
            if not matches:
                return [{"error": f"Unknown call type: {ct}"}]
            root_ids.append(matches[0].id)

        results = self.field.think(root_ids)
        return [{
            "think": call_types,
            "emerged": [(ct, f"{sim:.3f}") for _, ct, sim in results],
        }]

    def _align(self, pod_a: str, pod_b: str) -> list[dict]:
        field_a = self._pod_field(pod_a.strip())
        field_b = self._pod_field(pod_b.strip())
        if field_a is None:
            return [{"error": f"Unknown pod: {pod_a}"}]
        if field_b is None:
            return [{"error": f"Unknown pod: {pod_b}"}]

        result = align_pods(field_a, field_b, reduce_dim=5)
        sims = cross_pod_similarity(result, field_a, field_b)

        return [{
            "alignment": f"{pod_a}↔{pod_b}",
            "disparity": f"{result.disparity:.4f}",
            "context_agreement": f"{result.context_agreement:.2f}",
            "matched_pairs": len(result.shared_calls),
            "correspondences": [
                {"source": src, "target": tgt, "similarity": f"{sim:.3f}"}
                for src, tgt, sim in sims
            ],
        }]

    def _field_diagnostics(self) -> list[dict]:
        corr = self.field.phonosemantic_correlation()
        return [{
            "phonosemantic_correlation": f"{corr:.4f}",
            "interpretation": (
                "Strong sound symbolism" if corr > 0.5 else
                "Moderate correlation" if corr > 0.2 else
                "Weak/no correlation — acoustic form and function are independent"
            ),
            "n_roots": self.field.n_roots,
            "dim": self.field.dim,
            "pods": self.kernel.pods(),
        }]

    def _all_compounds(self, call_type: str, position: str, limit: int = 20) -> list[dict]:
        matches = self.kernel.by_call_type(call_type)
        if not matches:
            return [{"error": f"Unknown call type: {call_type}"}]

        root = matches[0]
        results = []
        for other in self.kernel.all_roots():
            if other.id == root.id:
                continue
            if position == "modifier":
                cv = self.field.compose(root.id, other.id)
                label = f"{root.call_type}:{other.call_type}"
            else:
                cv = self.field.compose(other.id, root.id)
                label = f"{other.call_type}:{root.call_type}"

            nearest = self.field.nearest_roots(cv, k=1)
            results.append({
                "compound": label,
                "nearest": nearest[0][1],
                "similarity": f"{nearest[0][2]:.3f}",
            })
            if len(results) >= limit:
                break

        return results
