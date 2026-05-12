"""
Orca kernel: the discovered call types that are the atoms of orca communication.

Unlike the lor-engine's 644 defined roots, the orca kernel is discovered from
recorded call-type catalogues. Each root is a discrete call type — a stereotyped
vocalisation identified by researchers through decades of field observation.

The kernel supports loading from CSV catalogues or from programmatically
constructed OrcaRoot lists (for testing with mock data).
"""

import csv
import os
import numpy as np
from dataclasses import dataclass, field

from .address import int_to_palindrome, palindrome_to_int


@dataclass
class OrcaRoot:
    """A discrete call type — the atomic unit of orca communication."""
    id: int
    call_type: str              # e.g. "S1", "S2i", "S10"
    features: np.ndarray        # acoustic feature vector (N_FEATURES,)
    contexts: list[str]         # behavioural contexts: ["foraging", "socializing", ...]
    frequency: float            # occurrence rate (0-1 normalized)
    pod: str                    # pod identifier: "J", "K", "L", etc.
    description: str            # human description if available
    address: str                # palindromic positional address

    def __repr__(self):
        return f"OrcaRoot({self.id}, '{self.call_type}', pod='{self.pod}', addr='{self.address}')"


CATALOGUE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "catalogues"
)


def parse_catalogue(filepath: str) -> list[OrcaRoot]:
    """
    Parse a call-type catalogue CSV and return list of OrcaRoot objects.

    Expected CSV columns: id, call_type, pod, description, contexts, frequency
    Contexts are pipe-separated: "foraging|socializing"
    """
    roots = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = int(row["id"])
            call_type = row["call_type"].strip()
            pod = row["pod"].strip()
            description = row.get("description", "").strip().strip('"')
            contexts = [c.strip() for c in row.get("contexts", "").split("|") if c.strip()]
            frequency = float(row.get("frequency", 0.0))
            address = int_to_palindrome(rid)

            # Features are not in the CSV — they come from audio analysis
            # or mock generation. Initialize as empty.
            features = np.zeros(0)

            roots.append(OrcaRoot(
                id=rid,
                call_type=call_type,
                features=features,
                contexts=contexts,
                frequency=frequency,
                pod=pod,
                description=description,
                address=address,
            ))

    return roots


class OrcaKernel:
    """
    The orca kernel — discrete call types with palindromic addresses.

    Analogous to lor-engine's Kernel but discovered from data, not defined.
    """

    def __init__(self, roots: list[OrcaRoot] = None, catalogue_path: str = None):
        if roots is None:
            if catalogue_path is None:
                raise ValueError("Must provide either roots or catalogue_path")
            roots = parse_catalogue(catalogue_path)

        self.roots = roots
        self._by_id = {r.id: r for r in roots}
        self._by_call_type: dict[str, list[OrcaRoot]] = {}
        for r in roots:
            self._by_call_type.setdefault(r.call_type, []).append(r)
        self._by_pod: dict[str, list[OrcaRoot]] = {}
        for r in roots:
            self._by_pod.setdefault(r.pod, []).append(r)
        self._by_address = {r.address: r for r in roots}

    def __len__(self):
        return len(self.roots)

    def __getitem__(self, root_id: int) -> OrcaRoot:
        return self._by_id[root_id]

    def by_call_type(self, call_type: str) -> list[OrcaRoot]:
        """Look up roots by call type string. May return multiple (cross-pod)."""
        return self._by_call_type.get(call_type, [])

    def by_pod(self, pod: str) -> list[OrcaRoot]:
        """Get all call types for a specific pod."""
        return self._by_pod.get(pod, [])

    def by_address(self, address: str) -> OrcaRoot | None:
        """Look up a root by its palindromic address."""
        return self._by_address.get(address)

    def by_id(self, root_id: int) -> OrcaRoot:
        """Look up a root by its registry ID."""
        return self._by_id[root_id]

    def all_roots(self) -> list[OrcaRoot]:
        """Return all roots."""
        return self.roots

    def pods(self) -> list[str]:
        """Return all pod identifiers."""
        return sorted(self._by_pod.keys())

    def validate(self) -> list[str]:
        """Validate that all addresses round-trip correctly."""
        errors = []
        for r in self.roots:
            decoded = palindrome_to_int(r.address)
            if decoded != r.id:
                errors.append(
                    f"Root {r.id} ({r.call_type}): address '{r.address}' "
                    f"decodes to {decoded}, expected {r.id}"
                )
            re_encoded = int_to_palindrome(r.id)
            if re_encoded != r.address:
                errors.append(
                    f"Root {r.id} ({r.call_type}): re-encoded as '{re_encoded}', "
                    f"expected '{r.address}'"
                )
        return errors
