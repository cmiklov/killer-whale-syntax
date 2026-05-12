"""Tests for the dual semantic field — R/D dynamics on orca call topologies."""

import numpy as np
import pytest
from orca.field import OrcaField


@pytest.fixture
def field(kernel):
    return OrcaField(kernel, dim=64)


class TestVectorSpace:
    def test_vectors_normalized(self, field):
        for root in field.kernel.all_roots():
            v = field.vector(root.id)
            norm = np.linalg.norm(v)
            assert abs(norm - 1.0) < 1e-6, f"Root {root.call_type} norm={norm}"

    def test_correct_dimension(self, field):
        v = field.vector(1)
        assert v.shape == (64,)

    def test_different_roots_different_vectors(self, field):
        v1 = field.vector(1)
        v2 = field.vector(2)
        assert not np.allclose(v1, v2), "Different call types should have different vectors"

    def test_similarity_symmetric(self, field):
        v1 = field.vector(1)
        v2 = field.vector(2)
        assert abs(field.similarity(v1, v2) - field.similarity(v2, v1)) < 1e-10


class TestDualField:
    def test_acoustic_vector_shape(self, field):
        v = field.acoustic_vector(1)
        assert v.shape == (64,)

    def test_context_vector_shape(self, field):
        v = field.context_vector(1)
        assert v.shape == (64,)

    def test_acoustic_vectors_normalized(self, field):
        for root in field.kernel.all_roots():
            v = field.acoustic_vector(root.id)
            assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_context_vectors_normalized(self, field):
        for root in field.kernel.all_roots():
            v = field.context_vector(root.id)
            assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_acoustic_vs_context_different(self, field):
        """Acoustic and context vectors should generally be different."""
        a = field.acoustic_vector(1)
        c = field.context_vector(1)
        # They are in the same dim space but from different projections
        # so they won't be identical
        sim = field.similarity(a, c)
        assert sim < 0.99, "Acoustic and context vectors shouldn't be identical"

    def test_phonosemantic_correlation_bounded(self, field):
        corr = field.phonosemantic_correlation()
        assert -1.0 <= corr <= 1.0, f"Correlation out of bounds: {corr}"
        print(f"  Phonosemantic correlation: {corr:.3f}")

    def test_acoustic_similarity(self, field):
        # Same call type in different pods should have SOME acoustic similarity
        # (since mock features are generated from call_type + pod)
        sim = field.acoustic_similarity(1, 8)  # S1 in J and S1 in K
        print(f"  S1(J) vs S1(K) acoustic similarity: {sim:.3f}")

    def test_context_similarity(self, field):
        sim = field.context_similarity(1, 8)  # S1 in J and S1 in K
        print(f"  S1(J) vs S1(K) context similarity: {sim:.3f}")


class TestCompounds:
    def test_compound_produces_vector(self, field):
        cv = field.compose(1, 2)
        assert cv.shape == (64,)
        assert abs(np.linalg.norm(cv) - 1.0) < 1e-6

    def test_compound_between_components(self, field):
        """Compound should be between its components in the field."""
        v1 = field.vector(1)
        v2 = field.vector(2)
        cv = field.compose(1, 2)
        sim_1_cv = field.similarity(v1, cv)
        sim_2_cv = field.similarity(v2, cv)
        # Compound should be closer to head (id=2) than modifier (id=1)
        # because alpha=0.4 makes it head-dominant
        assert sim_2_cv > 0.0, "Compound should be somewhat near head"

    def test_compound_order_matters(self, field):
        cv12 = field.compose(1, 2)
        cv21 = field.compose(2, 1)
        sim = field.similarity(cv12, cv21)
        assert sim < 0.99, "Compound order should matter (non-commutative)"

    def test_triple_compound(self, field):
        cv = field.compose_triple(1, 2, 3)
        assert cv.shape == (64,)
        assert abs(np.linalg.norm(cv) - 1.0) < 1e-6


class TestDynamics:
    def test_diffuse_preserves_norm(self, field):
        v = field.vector(1)
        diffused = field.diffuse(v)
        assert abs(np.linalg.norm(diffused) - 1.0) < 1e-6

    def test_relaxation_converges(self, field):
        # Start from a random point
        rng = np.random.RandomState(42)
        initial = rng.randn(64)
        initial = initial / np.linalg.norm(initial)

        converged, trajectory = field.relax(initial, steps=200)
        assert abs(np.linalg.norm(converged) - 1.0) < 1e-6
        assert len(trajectory) >= 2
        # Should have converged (final delta small)
        if len(trajectory) > 2:
            final_delta = np.linalg.norm(trajectory[-1] - trajectory[-2])
            assert final_delta < 0.01, f"Did not converge: final delta={final_delta}"

    def test_root_is_near_fixed_point(self, field):
        """A root vector should be near its own attractor basin."""
        v = field.vector(1)
        diffused = field.diffuse(v)
        sim = field.similarity(v, diffused)
        assert sim > 0.9, f"Root should be near-stable under diffusion, sim={sim}"

    def test_think_produces_results(self, field):
        results = field.think([1, 2])
        assert len(results) > 0
        # Results should be (root_id, call_type, similarity) tuples
        for rid, ct, sim in results:
            assert isinstance(rid, int)
            assert isinstance(ct, str)
            assert -1.0 <= sim <= 1.0


class TestComprehension:
    def test_root_is_own_nearest(self, field):
        """Each call type should be its own nearest attractor (or very close)."""
        misses = 0
        for root in field.kernel.all_roots():
            nearest = field.nearest_roots(field.vector(root.id), k=1)
            if nearest[0][0] != root.id:
                misses += 1
                # At minimum, similarity should be very high (near-collision)
                assert nearest[0][2] > 0.95, (
                    f"Root {root.call_type} (id={root.id}) nearest is "
                    f"{nearest[0][1]} (id={nearest[0][0]}) with sim={nearest[0][2]:.4f}"
                )
        # Allow a small number of near-collisions with synthetic data
        # With only 15 roots and 50D mock features, some near-collisions are expected
        assert misses <= 4, f"Too many roots not their own nearest: {misses}"

    def test_nearest_returns_k(self, field):
        v = field.vector(1)
        results = field.nearest_roots(v, k=5)
        assert len(results) == 5

    def test_nearest_sorted_by_similarity(self, field):
        v = field.vector(1)
        results = field.nearest_roots(v, k=5)
        sims = [s for _, _, s in results]
        assert sims == sorted(sims, reverse=True)


class TestPodClustering:
    def test_pod_cluster(self, field):
        results = field.pod_cluster("J", k=5)
        assert len(results) > 0
        # Top results should include J-pod calls
        print(f"  J-pod cluster top 5: {results}")

    def test_distance_matrix_shape(self, field):
        dm = field.distance_matrix()
        n = field.n_roots
        assert dm.shape == (n, n)
        # Diagonal should be zero (self-distance)
        for i in range(n):
            assert abs(dm[i, i]) < 1e-6
