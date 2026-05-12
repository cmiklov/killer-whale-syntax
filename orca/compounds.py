"""
Compound detection: finding stereotyped call sequences.

The lor-engine COMPUTES compounds from roots + degemination.
The orca-engine DETECTS compounds from observed call sequences.
This is the fundamental difference: discovery vs definition.

A compound is a pair (or triple) of call types that co-occur
significantly more often than chance, with consistent timing,
and possibly with acoustic boundary smoothing.
"""

import numpy as np
from dataclasses import dataclass, field
from collections import Counter
from .kernel import OrcaKernel
from .phonology import boundary_smoothing_score


@dataclass
class CompoundCandidate:
    """A detected stereotyped call sequence — candidate compound."""
    modifier_id: int
    head_id: int
    count: int                          # how many times observed
    contexts: list[str] = field(default_factory=list)
    boundary_smoothing: float = 0.0     # acoustic smoothing at junction
    mean_gap: float = 0.0              # average time gap between calls (seconds)

    @property
    def label(self) -> str:
        return f"{self.modifier_id}:{self.head_id}"


def detect_compounds(
    sequences: list[list[int]],
    kernel: OrcaKernel,
    min_count: int = 3,
    max_gap: float = 2.0,
    sequence_contexts: list[list[str]] | None = None,
) -> list[CompoundCandidate]:
    """
    Detect stereotyped call-type pairs in observed sequences.

    sequences: list of call sequences, each a list of root IDs in temporal order.
    kernel: the orca kernel for feature lookup.
    min_count: minimum number of observations to be a candidate.
    max_gap: maximum time between calls to count as adjacent (seconds).
    sequence_contexts: optional context tags per sequence.

    Returns list of CompoundCandidate, sorted by count descending.
    """
    pair_counts: Counter = Counter()
    pair_contexts: dict[tuple[int, int], list[str]] = {}

    for seq_idx, seq in enumerate(sequences):
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counts[pair] += 1

            if sequence_contexts and seq_idx < len(sequence_contexts):
                key = pair
                if key not in pair_contexts:
                    pair_contexts[key] = []
                pair_contexts[key].extend(sequence_contexts[seq_idx])

    candidates = []
    for (mod_id, head_id), count in pair_counts.most_common():
        if count < min_count:
            break

        # Compute boundary smoothing if both roots have features
        smoothing = 0.0
        try:
            mod_root = kernel.by_id(mod_id)
            head_root = kernel.by_id(head_id)
            if len(mod_root.features) > 0 and len(head_root.features) > 0:
                smoothing = boundary_smoothing_score(mod_root.features, head_root.features)
        except KeyError:
            pass

        contexts = list(set(pair_contexts.get((mod_id, head_id), [])))

        candidates.append(CompoundCandidate(
            modifier_id=mod_id,
            head_id=head_id,
            count=count,
            contexts=contexts,
            boundary_smoothing=smoothing,
        ))

    return candidates


def compound_vector(field, modifier_id: int, head_id: int) -> np.ndarray:
    """Compute compound vector via field composition."""
    return field.compose(modifier_id, head_id)


def compound_label(kernel: OrcaKernel, mod_id: int, head_id: int) -> str:
    """Generate compound label: 'S1:S7'."""
    mod = kernel.by_id(mod_id)
    head = kernel.by_id(head_id)
    return f"{mod.call_type}:{head.call_type}"


def detect_triples(
    sequences: list[list[int]],
    kernel: OrcaKernel,
    min_count: int = 2,
) -> list[tuple[int, int, int, int]]:
    """
    Detect stereotyped call-type triples in observed sequences.

    Returns list of (r1_id, r2_id, r3_id, count) sorted by count.
    """
    triple_counts: Counter = Counter()

    for seq in sequences:
        for i in range(len(seq) - 2):
            triple = (seq[i], seq[i + 1], seq[i + 2])
            triple_counts[triple] += 1

    results = [
        (r1, r2, r3, count)
        for (r1, r2, r3), count in triple_counts.most_common()
        if count >= min_count
    ]
    return results
