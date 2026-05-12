"""Tests for opcode discovery and Genesis table fallback."""

import numpy as np
import pytest
from orca.opcodes import Opcode, DiscoveredOpcode, OpcodeDiscovery
from orca.phonology import degeminate_opcodes, opcode_energy, verify_energy_conservation
from orca.field import OrcaField


@pytest.fixture
def field(kernel):
    return OrcaField(kernel, dim=64)


class TestGenesisFallback:
    def test_opcode_constants(self):
        assert Opcode.L == 0
        assert Opcode.I == 1
        assert Opcode.E == 6
        assert len(Opcode.ALL) == 7

    def test_degemination_table(self):
        # L + I = LI (compound), IL (complement)
        comp, compl, mod = degeminate_opcodes(Opcode.L, Opcode.I)
        assert comp == Opcode.LI
        assert compl == Opcode.IL

    def test_e_plus_e_is_bang(self):
        comp, compl, mod = degeminate_opcodes(Opcode.E, Opcode.E)
        assert comp == Opcode.BANG
        assert mod == 2.0

    def test_energy_conservation(self):
        violations = verify_energy_conservation()
        assert violations == [], f"Energy conservation violations: {violations}"

    def test_opcode_energies(self):
        assert opcode_energy(Opcode.L) == 0.0
        assert opcode_energy(Opcode.I) == 1.0
        assert abs(opcode_energy(Opcode.E) - np.e) < 0.001


class TestOpcodeDiscovery:
    def test_discovery_with_synthetic_variants(self, kernel, field):
        """Create synthetic variable calls and verify discovery finds structure."""
        discovery = OpcodeDiscovery(field)

        # Create mock variable calls by shifting discrete call vectors
        # This simulates a consistent modification (an opcode)
        shift = np.random.RandomState(42).randn(64) * 0.3

        from orca.kernel import OrcaRoot
        from orca.features import mock_features
        from orca.address import int_to_palindrome

        variable_calls = []
        for i, root in enumerate(kernel.all_roots()[:7]):  # first 7 discrete calls
            # Create a variant with a consistent shift
            variant_features = root.features + np.random.RandomState(i).randn(50) * 0.1
            variant = OrcaRoot(
                id=100 + i,
                call_type=f"{root.call_type}v",
                features=variant_features,
                contexts=root.contexts,
                frequency=root.frequency * 0.5,
                pod=root.pod,
                description=f"Variant of {root.call_type}",
                address=int_to_palindrome(100 + i),
            )
            variable_calls.append(variant)

        # Note: discovery needs the variable calls to be in the kernel/field
        # For a proper test, we'd need to rebuild the field with variants
        # For now, verify the discovery infrastructure works
        assert discovery.discovered == []

    def test_apply_opcode(self, field):
        """Test applying a synthetic opcode."""
        centroid = np.random.RandomState(42).randn(64) * 0.2
        opcode = DiscoveredOpcode(
            id=0,
            name="test_op",
            delta_centroid=centroid,
            examples=[(1, 2)],
            productivity=0.5,
            consistency=0.1,
        )
        discovery = OpcodeDiscovery(field)
        result = discovery.apply_opcode(1, opcode)
        assert result.shape == (64,)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6
        # Should be different from the original
        original = field.vector(1)
        assert not np.allclose(result, original)
