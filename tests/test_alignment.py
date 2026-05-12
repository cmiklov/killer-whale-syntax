"""Tests for cross-pod alignment — topology translation between dialects."""

import numpy as np
import pytest
from orca.field import OrcaField
from orca.alignment import align_pods, cross_pod_similarity, validate_alignment


@pytest.fixture
def j_field(j_pod_kernel):
    return OrcaField(j_pod_kernel, dim=64)


@pytest.fixture
def k_field(k_pod_kernel):
    return OrcaField(k_pod_kernel, dim=64)


@pytest.fixture
def full_field(kernel):
    return OrcaField(kernel, dim=64)


class TestSelfAlignment:
    def test_self_alignment_low_disparity(self, j_field):
        """Aligning a field to itself should give near-zero disparity."""
        result = align_pods(j_field, j_field, reduce_dim=5)
        assert result.disparity < 0.1, f"Self-alignment disparity too high: {result.disparity}"
        print(f"  Self-alignment disparity: {result.disparity:.4f}")

    def test_self_alignment_perfect_context(self, j_field):
        """Self-alignment should have perfect context agreement."""
        result = align_pods(j_field, j_field, reduce_dim=5)
        assert result.context_agreement == 1.0, (
            f"Self-alignment context agreement should be 1.0, got {result.context_agreement}"
        )


class TestCrossPodAlignment:
    def test_alignment_produces_result(self, j_field, k_field):
        """Cross-pod alignment should produce a valid result."""
        result = align_pods(j_field, k_field, reduce_dim=5)
        assert result.rotation is not None
        assert result.disparity >= 0.0
        assert 0.0 <= result.context_agreement <= 1.0
        assert len(result.shared_calls) > 0
        print(f"  J↔K disparity: {result.disparity:.4f}")
        print(f"  J↔K context agreement: {result.context_agreement:.2f}")
        print(f"  J↔K matched pairs: {len(result.shared_calls)}")

    def test_alignment_symmetry(self, j_field, k_field):
        """Aligning J→K and K→J should give similar disparity."""
        result_jk = align_pods(j_field, k_field, reduce_dim=5)
        result_kj = align_pods(k_field, j_field, reduce_dim=5)
        # Disparities should be in same ballpark (not necessarily identical)
        ratio = max(result_jk.disparity, result_kj.disparity) / (
            min(result_jk.disparity, result_kj.disparity) + 1e-12
        )
        assert ratio < 5.0, f"Asymmetric alignment: J→K={result_jk.disparity}, K→J={result_kj.disparity}"

    def test_cross_pod_similarity_report(self, j_field, k_field):
        """cross_pod_similarity should return readable results."""
        result = align_pods(j_field, k_field, reduce_dim=5)
        sims = cross_pod_similarity(result, j_field, k_field)
        assert len(sims) > 0
        for src_label, tgt_label, sim in sims:
            assert isinstance(src_label, str)
            assert isinstance(tgt_label, str)
            assert -1.0 <= sim <= 1.0
        print(f"  Top alignments: {sims[:3]}")


class TestAnchorPairs:
    def test_anchored_alignment(self, j_field, k_field):
        """Alignment with known correspondences should work."""
        # S1 appears in both J and K pods
        j_ids = [r.id for r in j_field.kernel.all_roots()]
        k_ids = [r.id for r in k_field.kernel.all_roots()]
        # Use first root from each as anchor
        anchor = [(j_ids[0], k_ids[0])]
        result = align_pods(j_field, k_field, anchor_pairs=anchor, reduce_dim=None)
        assert result.disparity >= 0.0
        assert len(result.shared_calls) == 1


class TestValidation:
    def test_context_validation(self, j_field, k_field):
        """validate_alignment should return a float between 0 and 1."""
        result = align_pods(j_field, k_field, reduce_dim=5)
        score = validate_alignment(result, j_field.kernel, k_field.kernel)
        assert 0.0 <= score <= 1.0
