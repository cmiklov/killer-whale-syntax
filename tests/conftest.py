"""Shared fixtures for orca-engine tests."""

import pytest
import numpy as np
from orca.kernel import OrcaRoot, OrcaKernel
from orca.features import mock_features, N_FEATURES
from orca.address import int_to_palindrome


# ─── Mock Catalogue ──────────────────────────────────────────────────
# Based on real Southern Resident Killer Whale call-type catalogues.
# Call types S1-S19 are documented for J/K/L pods.
# These mock roots use realistic names but synthetic features.

MOCK_ROOTS = [
    # J-pod discrete calls
    (1, "S1", "J", "Discrete pulsed call, two-component", ["foraging", "socializing"], 0.15),
    (2, "S2i", "J", "Rising upcall variant", ["travel", "socializing"], 0.08),
    (3, "S3", "J", "Rapid pulsed call", ["foraging"], 0.12),
    (4, "S4", "J", "Multi-component wavering", ["socializing"], 0.10),
    (5, "S7", "J", "Flat tonal call", ["travel"], 0.06),
    (6, "S10", "J", "High-frequency whistle", ["socializing", "foraging"], 0.09),
    (7, "S16", "J", "Low-frequency pulsed", ["rest"], 0.04),
    # K-pod discrete calls (some shared with J, some unique)
    (8, "S1", "K", "Discrete pulsed call, two-component", ["foraging", "socializing"], 0.14),
    (9, "S3", "K", "Rapid pulsed call", ["foraging"], 0.11),
    (10, "S12", "K", "Ascending tonal whistle", ["travel", "socializing"], 0.07),
    (11, "S13", "K", "Burst-pulse sequence", ["foraging"], 0.08),
    (12, "S19", "K", "Complex frequency-modulated", ["socializing"], 0.05),
    # L-pod discrete calls
    (13, "S1", "L", "Discrete pulsed call, two-component", ["foraging", "socializing"], 0.13),
    (14, "S2i", "L", "Rising upcall variant", ["travel"], 0.06),
    (15, "S5", "L", "Descending tonal", ["rest", "socializing"], 0.07),
]


def _build_mock_roots() -> list[OrcaRoot]:
    roots = []
    for rid, call_type, pod, desc, contexts, freq in MOCK_ROOTS:
        # Generate unique features using call_type + pod for uniqueness
        features = mock_features(f"{call_type}_{pod}")
        address = int_to_palindrome(rid)
        roots.append(OrcaRoot(
            id=rid,
            call_type=call_type,
            features=features,
            contexts=contexts,
            frequency=freq,
            pod=pod,
            description=desc,
            address=address,
        ))
    return roots


@pytest.fixture
def mock_roots():
    """List of mock OrcaRoot objects."""
    return _build_mock_roots()


@pytest.fixture
def kernel(mock_roots):
    """An OrcaKernel loaded with mock data."""
    return OrcaKernel(roots=mock_roots)


@pytest.fixture
def j_pod_kernel(mock_roots):
    """Kernel with only J-pod roots."""
    j_roots = [r for r in mock_roots if r.pod == "J"]
    return OrcaKernel(roots=j_roots)


@pytest.fixture
def k_pod_kernel(mock_roots):
    """Kernel with only K-pod roots."""
    k_roots = [r for r in mock_roots if r.pod == "K"]
    return OrcaKernel(roots=k_roots)
