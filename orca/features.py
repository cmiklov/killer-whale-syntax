"""
Acoustic feature extraction for orca vocalisations.

Replaces the lor-engine's 82D phonological feature extraction with acoustic
features appropriate for orca call analysis. The feature space is designed
to capture the same structural dimensions:

  Lor-engine (human phonology)        Orca-engine (acoustic features)
  ─────────────────────────────       ─────────────────────────────────
  Place of articulation (6D)    →     Spectral shape (6D)
  Manner of articulation (5D)   →     Temporal envelope (5D)
  Voicing (2D)                  →     Click vs tonal components (2D)
  Vowel space (3D)              →     FM contour (3D)
  Structural (8D)               →     Call structure + context (8D)
  Character bag (26D)           →     Spectral fingerprint (26D)

Total: 50 dimensions (vs lor's 82, but same JL projection to 64D).

When librosa is available, features can be extracted from real audio.
When it's not, mock_features() generates deterministic synthetic vectors
for testing. All tests run on synthetic data.
"""

import numpy as np

# ─── Feature Dimensions ─────────────────────────────────────────────

# Spectral shape (replaces Place of articulation)
# Captures WHERE in frequency space the energy sits
N_SPECTRAL = 6  # centroid, bandwidth, rolloff, flux, contrast, flatness

# Temporal envelope (replaces Manner of articulation)
# Captures HOW the sound unfolds in time
N_TEMPORAL = 5  # pulsed, tonal, mixed, burst, silence (soft classification)

# Click vs tonal components (replaces Voicing)
# Captures the fundamental acoustic mode
N_COMPONENT = 2  # click_energy_ratio, tonal_energy_ratio

# Frequency modulation contour (replaces Vowel space)
# Captures the melodic shape of the call
N_MODULATION = 3  # mean_freq, modulation_depth, modulation_rate

# Structural features (replaces Structural)
# Call-level and context-level information
N_STRUCTURAL = 8  # duration, repetition_count, inter_click_interval, frequency,
                   # pod_flag, social_context, feeding_context, travel_context

# Spectral fingerprint (replaces Character bag)
# Energy distribution across frequency bands — the call's acoustic "signature"
N_SPECTRAL_BANDS = 26  # log-mel energy in 26 frequency bands

# Total raw features
N_FEATURES = N_SPECTRAL + N_TEMPORAL + N_COMPONENT + N_MODULATION + N_STRUCTURAL + N_SPECTRAL_BANDS
# = 6 + 5 + 2 + 3 + 8 + 26 = 50

# Acoustic features only (excluding structural/context)
N_ACOUSTIC = N_SPECTRAL + N_TEMPORAL + N_COMPONENT + N_MODULATION + N_SPECTRAL_BANDS
# = 6 + 5 + 2 + 3 + 26 = 42

# Context features only
N_CONTEXT = N_STRUCTURAL  # = 8


def extract_spectral(audio_features: dict) -> np.ndarray:
    """Extract spectral shape features (6D)."""
    return np.array([
        audio_features.get("centroid", 0.0),
        audio_features.get("bandwidth", 0.0),
        audio_features.get("rolloff", 0.0),
        audio_features.get("flux", 0.0),
        audio_features.get("contrast", 0.0),
        audio_features.get("flatness", 0.0),
    ])


def extract_temporal(audio_features: dict) -> np.ndarray:
    """Extract temporal envelope features (5D soft classification)."""
    return np.array([
        audio_features.get("pulsed", 0.0),
        audio_features.get("tonal", 0.0),
        audio_features.get("mixed", 0.0),
        audio_features.get("burst", 0.0),
        audio_features.get("silence", 0.0),
    ])


def extract_components(audio_features: dict) -> np.ndarray:
    """Extract click vs tonal component ratios (2D)."""
    return np.array([
        audio_features.get("click_energy", 0.0),
        audio_features.get("tonal_energy", 0.0),
    ])


def extract_modulation(audio_features: dict) -> np.ndarray:
    """Extract frequency modulation contour features (3D)."""
    return np.array([
        audio_features.get("mean_freq", 0.0),
        audio_features.get("mod_depth", 0.0),
        audio_features.get("mod_rate", 0.0),
    ])


def extract_structural(root) -> np.ndarray:
    """
    Extract structural and context features (8D).

    Uses the OrcaRoot's metadata rather than audio analysis.
    """
    vec = np.zeros(N_STRUCTURAL)
    vec[0] = getattr(root, "duration", 0.0)
    vec[1] = getattr(root, "repetition_count", 0.0)
    vec[2] = getattr(root, "ici", 0.0)  # inter-click interval
    vec[3] = root.frequency

    # Context flags (binary)
    contexts = root.contexts if hasattr(root, "contexts") else []
    vec[4] = 1.0 if root.pod in ("J", "K", "L") else 0.5  # known resident pod
    vec[5] = 1.0 if "socializing" in contexts else 0.0
    vec[6] = 1.0 if "foraging" in contexts else 0.0
    vec[7] = 1.0 if "travel" in contexts else 0.0

    return vec


def extract_spectral_fingerprint(audio_features: dict) -> np.ndarray:
    """Extract spectral fingerprint — energy in 26 frequency bands (26D)."""
    bands = audio_features.get("mel_bands", None)
    if bands is not None:
        bands = np.array(bands[:N_SPECTRAL_BANDS])
        if len(bands) < N_SPECTRAL_BANDS:
            bands = np.pad(bands, (0, N_SPECTRAL_BANDS - len(bands)))
        return bands
    return np.zeros(N_SPECTRAL_BANDS)


def extract_features(root, audio_features: dict = None) -> np.ndarray:
    """
    Extract full feature vector for an orca call type.

    If audio_features dict is provided, extracts from it.
    If root already has features stored, uses those.
    Otherwise returns zeros.

    Returns: (N_FEATURES,) array
    """
    if audio_features is not None:
        spectral = extract_spectral(audio_features)
        temporal = extract_temporal(audio_features)
        components = extract_components(audio_features)
        modulation = extract_modulation(audio_features)
        structural = extract_structural(root)
        fingerprint = extract_spectral_fingerprint(audio_features)
        return np.concatenate([spectral, temporal, components, modulation,
                               structural, fingerprint])

    # If root has precomputed features with correct size, use them
    if hasattr(root, "features") and root.features is not None and len(root.features) == N_FEATURES:
        return root.features.copy()

    # Fallback: structural features only (from metadata)
    vec = np.zeros(N_FEATURES)
    structural = extract_structural(root)
    vec[N_SPECTRAL + N_TEMPORAL + N_COMPONENT + N_MODULATION:
        N_SPECTRAL + N_TEMPORAL + N_COMPONENT + N_MODULATION + N_STRUCTURAL] = structural
    return vec


def extract_acoustic_only(root, audio_features: dict = None) -> np.ndarray:
    """Extract only acoustic features (no context/structural). Returns (N_ACOUSTIC,) array."""
    full = extract_features(root, audio_features)
    # Acoustic = spectral + temporal + component + modulation + fingerprint
    # = indices [0:16] and [24:50] (skip structural at [16:24])
    acoustic_start = N_SPECTRAL + N_TEMPORAL + N_COMPONENT + N_MODULATION
    return np.concatenate([
        full[:acoustic_start],           # spectral + temporal + component + modulation
        full[acoustic_start + N_STRUCTURAL:]  # spectral fingerprint
    ])


def extract_context_only(root, audio_features: dict = None) -> np.ndarray:
    """Extract only context/structural features. Returns (N_CONTEXT,) array."""
    full = extract_features(root, audio_features)
    start = N_SPECTRAL + N_TEMPORAL + N_COMPONENT + N_MODULATION
    return full[start:start + N_STRUCTURAL]


def mock_features(call_type: str, seed: int = 7) -> np.ndarray:
    """
    Generate deterministic synthetic features for testing.

    Each call_type string produces a unique but reproducible feature vector.
    The vector has realistic structure: spectral features are correlated,
    context flags are binary, etc.

    This allows all tests to pass without real orca data.
    """
    rng = np.random.RandomState(seed=sum(ord(c) for c in call_type) * seed)

    # Spectral shape (positive, normalized)
    spectral = np.abs(rng.randn(N_SPECTRAL))
    spectral = spectral / (np.max(spectral) + 1e-8)

    # Temporal envelope (soft classification, sums to ~1)
    temporal = np.abs(rng.randn(N_TEMPORAL))
    temporal = temporal / (np.sum(temporal) + 1e-8)

    # Component ratios (0-1, sum to 1)
    click_ratio = rng.uniform(0, 1)
    components = np.array([click_ratio, 1.0 - click_ratio])

    # Modulation (normalized)
    modulation = np.abs(rng.randn(N_MODULATION))
    modulation = modulation / (np.max(modulation) + 1e-8)

    # Structural (mix of continuous and binary)
    structural = np.zeros(N_STRUCTURAL)
    structural[0] = rng.uniform(0.1, 2.0)    # duration (seconds)
    structural[1] = rng.randint(1, 10)        # repetition count
    structural[2] = rng.uniform(0.01, 0.5)    # ICI
    structural[3] = rng.uniform(0.01, 0.3)    # frequency of occurrence
    structural[4] = 1.0                        # known pod flag
    structural[5] = float(rng.random() > 0.5)  # social context
    structural[6] = float(rng.random() > 0.5)  # feeding context
    structural[7] = float(rng.random() > 0.5)  # travel context

    # Spectral fingerprint (positive, log-mel-like)
    fingerprint = np.abs(rng.randn(N_SPECTRAL_BANDS))
    fingerprint = fingerprint / (np.max(fingerprint) + 1e-8)

    return np.concatenate([spectral, temporal, components, modulation,
                           structural, fingerprint])
