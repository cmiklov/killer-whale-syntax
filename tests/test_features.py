"""Tests for acoustic feature extraction."""

import numpy as np
from orca.features import (
    N_FEATURES, N_SPECTRAL, N_TEMPORAL, N_COMPONENT, N_MODULATION,
    N_STRUCTURAL, N_SPECTRAL_BANDS, N_ACOUSTIC, N_CONTEXT,
    mock_features, extract_features, extract_acoustic_only, extract_context_only,
)


class TestFeatureDimensions:
    def test_n_features_sum(self):
        expected = N_SPECTRAL + N_TEMPORAL + N_COMPONENT + N_MODULATION + N_STRUCTURAL + N_SPECTRAL_BANDS
        assert N_FEATURES == expected == 50

    def test_n_acoustic(self):
        expected = N_SPECTRAL + N_TEMPORAL + N_COMPONENT + N_MODULATION + N_SPECTRAL_BANDS
        assert N_ACOUSTIC == expected == 42

    def test_n_context(self):
        assert N_CONTEXT == N_STRUCTURAL == 8


class TestMockFeatures:
    def test_shape(self):
        features = mock_features("S1")
        assert features.shape == (N_FEATURES,)

    def test_deterministic(self):
        f1 = mock_features("S1")
        f2 = mock_features("S1")
        assert np.array_equal(f1, f2), "Mock features should be deterministic"

    def test_different_calls_different_features(self):
        f1 = mock_features("S1")
        f2 = mock_features("S2i")
        assert not np.array_equal(f1, f2), "Different calls should have different features"

    def test_no_all_zero(self):
        for call in ["S1", "S2i", "S3", "S7", "S10", "S12", "S19"]:
            f = mock_features(call)
            assert not np.all(f == 0), f"Mock features for {call} should not be all zeros"

    def test_no_nan_or_inf(self):
        for call in ["S1", "S2i", "S3", "S7"]:
            f = mock_features(call)
            assert not np.any(np.isnan(f)), f"NaN in features for {call}"
            assert not np.any(np.isinf(f)), f"Inf in features for {call}"

    def test_spectral_positive(self):
        f = mock_features("S1")
        spectral = f[:N_SPECTRAL]
        assert np.all(spectral >= 0), "Spectral features should be non-negative"

    def test_temporal_sums_to_one(self):
        f = mock_features("S1")
        temporal = f[N_SPECTRAL:N_SPECTRAL + N_TEMPORAL]
        assert abs(np.sum(temporal) - 1.0) < 0.01, "Temporal envelope should sum to ~1"


class TestFeatureExtraction:
    def test_from_root_features(self, kernel):
        root = kernel.by_id(1)
        features = extract_features(root)
        assert features.shape == (N_FEATURES,)
        # Should use the stored features
        assert np.array_equal(features, root.features)

    def test_acoustic_only_shape(self, kernel):
        root = kernel.by_id(1)
        acoustic = extract_acoustic_only(root)
        assert acoustic.shape == (N_ACOUSTIC,)

    def test_context_only_shape(self, kernel):
        root = kernel.by_id(1)
        context = extract_context_only(root)
        assert context.shape == (N_CONTEXT,)

    def test_acoustic_plus_context_covers_all(self, kernel):
        """Acoustic and context features together should cover all features."""
        root = kernel.by_id(1)
        full = extract_features(root)
        acoustic = extract_acoustic_only(root)
        context = extract_context_only(root)
        assert len(acoustic) + len(context) == N_FEATURES

    def test_all_roots_extractable(self, kernel):
        for root in kernel.all_roots():
            features = extract_features(root)
            assert features.shape == (N_FEATURES,), f"Bad shape for {root.call_type}"
