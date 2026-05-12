"""
Reverse lookup: decompose a call sequence into constituent call types.

Adapted from lor-engine's word decomposition. Instead of suffix stripping
and consonant degemination reversal, uses call-type label matching.
"""

from .kernel import OrcaKernel
from .trie import Trie


def decompose_sequence(
    sequence_label: str,
    kernel: OrcaKernel,
    trie: Trie,
) -> list[dict]:
    """
    Decompose a call sequence label into constituent call types.

    Input: a compound label like "S1:S7" or a single call type "S1".
    Returns list of decomposition dicts.
    """
    results = []

    # Handle explicit compound notation (colon-separated)
    if ":" in sequence_label:
        parts = sequence_label.split(":")
        root_ids = []
        root_types = []
        valid = True
        for part in parts:
            matches = kernel.by_call_type(part.strip())
            if matches:
                root_ids.append(matches[0].id)
                root_types.append(matches[0].call_type)
            else:
                valid = False
                break
        if valid:
            results.append({
                "roots": root_ids,
                "root_types": root_types,
                "compound_label": ":".join(root_types),
            })
        return results

    # Single call type lookup
    matches = kernel.by_call_type(sequence_label)
    if matches:
        for m in matches:
            results.append({
                "roots": [m.id],
                "root_types": [m.call_type],
                "compound_label": m.call_type,
                "pod": m.pod,
                "description": m.description,
            })
        return results

    # Try trie prefix matching for unknown labels
    prefixes = trie.find_prefixes(sequence_label)
    for rid, remainder in prefixes:
        root = kernel.by_id(rid)
        if remainder:
            # Check if remainder matches another call type
            sub_matches = kernel.by_call_type(remainder)
            for sub in sub_matches:
                results.append({
                    "roots": [rid, sub.id],
                    "root_types": [root.call_type, sub.call_type],
                    "compound_label": f"{root.call_type}:{sub.call_type}",
                })
        else:
            results.append({
                "roots": [rid],
                "root_types": [root.call_type],
                "compound_label": root.call_type,
            })

    return results
