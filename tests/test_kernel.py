"""Tests for the orca kernel — call type loading, lookups, address round-trips."""

import numpy as np
from orca.kernel import OrcaRoot, OrcaKernel
from orca.address import int_to_palindrome, palindrome_to_int
from orca.features import N_FEATURES


class TestKernelBasics:
    def test_kernel_loads(self, kernel):
        assert len(kernel) == 15
        print(f"  Kernel loaded: {len(kernel)} call types")

    def test_all_roots_have_features(self, kernel):
        for root in kernel.all_roots():
            assert root.features is not None
            assert len(root.features) == N_FEATURES
            assert not np.all(root.features == 0), f"Root {root.call_type} has zero features"

    def test_all_roots_have_contexts(self, kernel):
        for root in kernel.all_roots():
            assert isinstance(root.contexts, list)
            assert len(root.contexts) > 0, f"Root {root.call_type} has no contexts"


class TestKernelLookups:
    def test_by_id(self, kernel):
        root = kernel.by_id(1)
        assert root.call_type == "S1"
        assert root.pod == "J"

    def test_by_call_type(self, kernel):
        # S1 appears in J, K, and L pods
        roots = kernel.by_call_type("S1")
        assert len(roots) == 3
        pods = {r.pod for r in roots}
        assert pods == {"J", "K", "L"}
        print(f"  S1 found in pods: {sorted(pods)}")

    def test_by_pod(self, kernel):
        j_roots = kernel.by_pod("J")
        assert len(j_roots) == 7
        k_roots = kernel.by_pod("K")
        assert len(k_roots) == 5
        l_roots = kernel.by_pod("L")
        assert len(l_roots) == 3
        print(f"  J:{len(j_roots)}, K:{len(k_roots)}, L:{len(l_roots)}")

    def test_by_address(self, kernel):
        root = kernel.by_id(1)
        found = kernel.by_address(root.address)
        assert found is not None
        assert found.id == 1

    def test_pods_list(self, kernel):
        pods = kernel.pods()
        assert pods == ["J", "K", "L"]

    def test_missing_call_type(self, kernel):
        roots = kernel.by_call_type("NONEXISTENT")
        assert roots == []

    def test_missing_pod(self, kernel):
        roots = kernel.by_pod("Z")
        assert roots == []


class TestAddressRoundTrip:
    def test_all_addresses_round_trip(self, kernel):
        errors = kernel.validate()
        assert errors == [], f"Address validation failed:\n" + "\n".join(errors)
        print(f"  All {len(kernel)} addresses round-trip correctly")

    def test_address_encoding(self):
        for i in range(20):
            addr = int_to_palindrome(i)
            decoded = palindrome_to_int(addr)
            assert decoded == i, f"{i} → '{addr}' → {decoded}"

    def test_unique_addresses(self, kernel):
        addresses = [r.address for r in kernel.all_roots()]
        assert len(addresses) == len(set(addresses)), "Duplicate addresses found"
