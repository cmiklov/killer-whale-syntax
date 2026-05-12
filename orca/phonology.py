"""
Phonological rules: the 7x7 Genesis table + acoustic boundary smoothing.

The Genesis table is copied verbatim from lor-engine/lor/phonology.py.
It is the same table that runs the fluid-engine, palindrome-engine,
chem-engine, and genesis visualization. Same two operations. Same seven opcodes.

The acoustic boundary smoothing replaces consonant degemination — when two
call types are adjacent in a sequence, the spectral transition at the
boundary can be smooth (analogous to degemination) or abrupt.
"""

import math
import numpy as np

# ─── The 7×7 Genesis Table ──────────────────────────────────────────
#
# Copied verbatim from lor-engine. The physics is universal.

_L, _I, _LI, _IL, _LL, _II, _E = 0, 1, 2, 3, 4, 5, 6
_BANG = -1  # E + E — genesis event

OPCODE_NAMES = ['L', 'I', 'LI', 'IL', 'LL', 'II', 'E']
N_OPS = 7

COMPOUND_TABLE = [
    _L,   _LI,  _L,   _L,   _L,   _I,   _IL,
    _IL,  _II,  _I,   _I,   _IL,  _II,  _II,
    _L,   _I,   _I,   _LL,  _LI,  _II,  _I,
    _L,   _I,   _LL,  _I,   _IL,  _II,  _L,
    _L,   _LI,  _IL,  _LI,  _LL,  _I,   _LI,
    _I,   _II,  _II,  _II,  _I,   _II,  _E,
    _I,   _LI,  _IL,  _LL,  _II,  _E,   _BANG,
]

COMPLEMENT_TABLE = [
    _L,   _IL,  _LI,  _IL,  _L,   _I,   _E,
    _LI,  _L,   _LI,  _IL,  _LI,  _I,   _E,
    _LI,  _LI,  _L,   _I,   _L,   _LI,  _E,
    _IL,  _IL,  _I,   _L,   _L,   _IL,  _E,
    _L,   _IL,  _L,   _L,   _LL,  _I,   _E,
    _I,   _I,   _LI,  _IL,  _I,   _II,  _L,
    _E,   _E,   _E,   _E,   _E,   _L,   _E,
]

MODIFIER_TABLE = [
     0.0,  0.1,  0.1,  0.1,  0.1, -1.0,  0.8,
     0.1, -0.5, -0.1, -0.1,  0.3, -1.0,  0.6,
     0.1, -0.1, -0.3,  1.0,  0.1, -1.0,  0.4,
     0.1, -0.1,  1.0, -0.3,  0.1, -1.0,  0.4,
     0.1,  0.3,  0.1,  0.1,  0.2, -1.0,  0.7,
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  1.0,
     0.8,  0.6,  0.4,  0.4,  0.7,  1.0,  2.0,
]

_ENERGY = [0.0, 1.0, 0.5, 0.5, 0.0, 2.0, math.e]


def degeminate_opcodes(a: int, b: int) -> tuple[int, int, float]:
    """
    Degeminate two center opcodes through the 7x7 Genesis table.

    Returns (compound, complement, modifier_strength).
    """
    key = a * N_OPS + b
    return COMPOUND_TABLE[key], COMPLEMENT_TABLE[key], MODIFIER_TABLE[key]


def opcode_name(idx: int) -> str:
    if idx == _BANG:
        return "BANG"
    return OPCODE_NAMES[idx]


def opcode_index(name: str) -> int:
    if name == "BANG":
        return _BANG
    return OPCODE_NAMES.index(name)


def opcode_energy(idx: int) -> float:
    if idx == _BANG:
        return 2 * math.e
    return _ENERGY[idx]


def is_e_contaminated(idx: int) -> bool:
    return idx == _E or idx == _BANG


def e_contamination_chain(a: int, b: int) -> bool:
    return a == _E or b == _E


def genesis_sequence(opcode: int) -> int:
    """E advances each state one day forward: L → I → LI → IL → LL → II → E → BANG."""
    seq = [_L, _I, _LI, _IL, _LL, _II, _E]
    if opcode == _E:
        return _BANG
    idx = seq.index(opcode)
    return seq[(idx + 1) % len(seq)]


def verify_energy_conservation():
    """Verify the 7x7 table conserves energy for all rational pairs."""
    violations = []
    for a in range(N_OPS):
        for b in range(N_OPS):
            comp, compl, _ = degeminate_opcodes(a, b)
            if comp == _BANG:
                continue
            if compl == _E or comp == _E:
                continue
            e_in = _ENERGY[a] + _ENERGY[b]
            e_out = _ENERGY[comp] + _ENERGY[compl]
            deficit = abs(e_in - e_out)
            if deficit > 0.001:
                violations.append((OPCODE_NAMES[a], OPCODE_NAMES[b], e_in, e_out, deficit))
    return violations


# ─── Acoustic Boundary Smoothing ─────────────────────────────────────
#
# The orca analogue of consonant degemination.
# When two calls are adjacent in a sequence, the spectral transition
# can be smooth (bonding, compound tendency) or abrupt (independent calls).

def boundary_smoothing_score(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """
    Measure acoustic smoothing at the junction of two calls.

    Analogous to degemination: when similar acoustic features meet at a boundary,
    the transition is smoother. Higher score = more smoothing = stronger compound bond.

    Score range: 0.0 (maximally abrupt) to 1.0 (maximally smooth).
    """
    if len(features_a) == 0 or len(features_b) == 0:
        return 0.0

    # Use cosine similarity of the feature vectors
    norm_a = np.linalg.norm(features_a)
    norm_b = np.linalg.norm(features_b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    cosine_sim = np.dot(features_a, features_b) / (norm_a * norm_b)
    # Map from [-1, 1] to [0, 1]
    return float((cosine_sim + 1.0) / 2.0)


def acoustic_degeminate(features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
    """
    Produce the 'degeminated' acoustic boundary — the smoothed transition.

    When two adjacent calls have similar spectral endpoints, the boundary
    features are averaged (smoothed). When they're dissimilar, the boundary
    is the concatenation midpoint.

    Returns a feature vector representing the boundary region.
    """
    smoothing = boundary_smoothing_score(features_a, features_b)

    # Weighted average: more smoothing = more blending
    boundary = smoothing * (features_a + features_b) / 2.0 + (1.0 - smoothing) * features_b

    norm = np.linalg.norm(boundary)
    if norm > 0:
        boundary = boundary / norm
    return boundary
