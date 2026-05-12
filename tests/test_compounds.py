"""Tests for compound detection and sequence analysis."""

import numpy as np
import pytest
from orca.compounds import detect_compounds, compound_vector, compound_label, detect_triples
from orca.field import OrcaField


@pytest.fixture
def field(kernel):
    return OrcaField(kernel, dim=64)


class TestCompoundDetection:
    def test_detect_pairs(self, kernel):
        # Create mock sequences with repeated pairs
        sequences = [
            [1, 2, 3, 1, 2],
            [1, 2, 5, 6],
            [4, 1, 2, 7],
        ]
        candidates = detect_compounds(sequences, kernel, min_count=2)
        assert len(candidates) > 0
        # (1, 2) appears 4 times: seq1[0:1], seq1[3:4], seq2[0:1], seq3[1:2]
        top = candidates[0]
        assert top.modifier_id == 1
        assert top.head_id == 2
        assert top.count == 4
        print(f"  Top compound: {top.modifier_id}:{top.head_id} (count={top.count})")

    def test_min_count_filters(self, kernel):
        sequences = [[1, 2], [3, 4], [5, 6]]
        candidates = detect_compounds(sequences, kernel, min_count=2)
        assert len(candidates) == 0, "No pair appears twice, should find nothing"

    def test_boundary_smoothing_computed(self, kernel):
        sequences = [[1, 2, 1, 2, 1, 2]]
        candidates = detect_compounds(sequences, kernel, min_count=2)
        assert len(candidates) > 0
        # Smoothing should be computed (non-zero for roots with features)
        assert candidates[0].boundary_smoothing >= 0.0

    def test_contexts_collected(self, kernel):
        sequences = [[1, 2], [1, 2], [1, 2]]
        contexts = [["foraging"], ["socializing"], ["foraging"]]
        candidates = detect_compounds(sequences, kernel, min_count=2, sequence_contexts=contexts)
        assert len(candidates) > 0
        assert len(candidates[0].contexts) > 0


class TestCompoundVectors:
    def test_compound_vector_shape(self, field):
        cv = compound_vector(field, 1, 2)
        assert cv.shape == (64,)
        assert abs(np.linalg.norm(cv) - 1.0) < 1e-6

    def test_compound_label(self, kernel):
        label = compound_label(kernel, 1, 2)
        assert label == "S1:S2i"


class TestTripleDetection:
    def test_detect_triples(self, kernel):
        sequences = [
            [1, 2, 3, 1, 2, 3],
            [1, 2, 3, 4, 5],
        ]
        triples = detect_triples(sequences, kernel, min_count=2)
        assert len(triples) > 0
        # (1,2,3) appears 3 times: seq1[0:2], seq1[3:5], seq2[0:2]
        assert triples[0] == (1, 2, 3, 3)
