"""
Per-User Calibration for Emotion Detection

Provides personalised emotion detection by comparing live embeddings against
a user's calibration baselines via cosine similarity. This corrects for
individual differences (e.g., elderly resting face misclassified as sad).

Calibration flow:
    1. Onboarding: capture neutral face (5s) + happy face (5s)
    2. Average ~25 embeddings per state → store as baseline
    3. Compute adaptive thresholds from inter-baseline similarity
    4. At runtime: compare live embedding to baselines → override raw model
       when similarity is high enough, fall back to raw model otherwise

Components:
    - GenericBaseline: stores per-state embeddings for a user
    - GenericCalibratedDetector: 5-rule decision logic for calibrated predictions
    - cosine_similarity, average_embeddings: utility functions

Usage:
    baseline = GenericBaseline(user_id='alice', modality='deepface_emb')
    baseline.REQUIRED_STATES = ['neutral', 'happy']
    baseline.add_state('neutral', neutral_embedding)
    baseline.add_state('happy', happy_embedding)

    detector = GenericCalibratedDetector(calibrated_emotions={'Happiness', 'Neutral'})
    detector.set_baseline(baseline)
    detector.set_adaptive_thresholds(thresholds)

    raw = detector.get_raw_prediction(extraction_result)
    cal = detector.get_calibrated_prediction(extraction_result)
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


# ============================================================================
# Utility Functions
# ============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector is zero-length.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def average_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    """Average a list of embeddings into a single centroid.

    Used during calibration to produce one stable baseline embedding
    from ~25 captured frames.

    Raises:
        ValueError: If the list is empty.
    """
    if not embeddings:
        raise ValueError("Cannot average empty list of embeddings")
    return np.mean(np.stack(embeddings), axis=0)


def average_values(values: List[float]) -> float:
    """Average a list of float values. Returns 0.0 if empty."""
    if not values:
        return 0.0
    return float(np.mean(values))


# ============================================================================
# Baseline Storage
# ============================================================================

@dataclass
class GenericBaseline:
    """Model-agnostic per-user baseline storage.

    Stores one embedding per calibration state (e.g., neutral, happy).
    Works with any embedding dimension — the same class is used for
    both face (1024-dim) and audio (768/1024-dim) baselines.
    """
    user_id: str
    modality: str  # e.g., 'deepface_emb', 'audio'
    created_at: datetime = field(default_factory=datetime.now)

    # Embeddings for each calibration state (state_name → embedding)
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)

    # Optional metadata
    metadata: Dict = field(default_factory=dict)

    # States required for calibration to be considered complete.
    # Override per instance: baseline.REQUIRED_STATES = ['neutral', 'happy']
    REQUIRED_STATES = ['neutral', 'happy']

    def add_state(self, state: str, embedding: np.ndarray):
        """Add a calibration state with its embedding."""
        self.embeddings[state] = embedding

    def get_embedding(self, state: str) -> Optional[np.ndarray]:
        """Get embedding for a state, or None if not captured."""
        return self.embeddings.get(state)

    def is_complete(self) -> bool:
        """Check if all required states have been captured."""
        return all(state in self.embeddings for state in self.REQUIRED_STATES)

    def get_states(self) -> List[str]:
        """Get list of captured state names."""
        return list(self.embeddings.keys())

    def embedding_dim(self) -> Optional[int]:
        """Get embedding dimensionality (from first stored embedding)."""
        if self.embeddings:
            first = next(iter(self.embeddings.values()))
            return first.shape[0]
        return None


# ============================================================================
# Calibrated Detector
# ============================================================================

class GenericCalibratedDetector:
    """Calibrated emotion detector using cosine similarity to baselines.

    Compares live embeddings against stored baselines and applies a 5-rule
    decision logic to produce calibrated predictions. The rules balance
    calibration accuracy with raw model fallback for non-calibrated emotions.

    Decision rules (evaluated in order):
        1. Calibration match: closest baseline passes threshold → use calibration
           (if raw model strongly detects non-calibrated emotion, threshold is boosted)
        2. Raw override: raw model confidently detects non-calibrated emotion → trust raw
        3. Deviation floor: far from all baselines → use best non-calibrated emotion
        4. Rejected calibrated: raw says calibrated emotion but rejected → fallback
        5. Default: raw model output
    """

    def __init__(
        self,
        calibrated_emotions: Optional[set] = None,
        similarity_threshold: float = 0.80,
        neutral_threshold: float = 0.85,
        raw_override_confidence: float = 0.60,
        deviation_floor: float = 0.60,
    ):
        """
        Args:
            calibrated_emotions: Emotion labels we have baselines for.
                Defaults to {'Happy', 'Neutral'}.
            similarity_threshold: Cosine similarity threshold for non-neutral states.
            neutral_threshold: Stricter threshold for neutral state.
            raw_override_confidence: Trust raw model for non-calibrated emotions
                above this confidence.
            deviation_floor: If max similarity to ALL baselines is below this,
                user has deviated into uncalibrated territory.
        """
        self.baseline: Optional[GenericBaseline] = None
        self.calibrated_emotions = calibrated_emotions or {'Happy', 'Neutral'}
        self.similarity_threshold = similarity_threshold
        self.neutral_threshold = neutral_threshold
        self.raw_override_confidence = raw_override_confidence
        self.deviation_floor = deviation_floor

    def set_baseline(self, baseline: GenericBaseline):
        """Set the user's calibration baseline."""
        self.baseline = baseline

    def set_adaptive_thresholds(self, thresholds: Dict):
        """Apply adaptive thresholds computed from calibration data.

        Typically called after calibration with thresholds derived from
        the inter-baseline cosine similarity.
        """
        self.similarity_threshold = thresholds['similarity_threshold']
        self.neutral_threshold = thresholds['neutral_threshold']
        self.deviation_floor = thresholds['deviation_floor']
        self.raw_override_confidence = thresholds['raw_override_confidence']

    def get_raw_prediction(self, extraction_result: Dict) -> Dict:
        """Format raw model output for display and logging.

        Args:
            extraction_result: Output from face/audio extractor's extract().

        Returns:
            Dict with: emotion, confidence, emotion_probs, valence, arousal.
        """
        return {
            'emotion': extraction_result['top_emotion'],
            'confidence': extraction_result['confidence'],
            'emotion_probs': extraction_result.get('emotion_probs', {}),
        }

    def get_calibrated_prediction(self, extraction_result: Dict) -> Dict:
        """Get calibrated prediction by comparing live embedding to baselines.

        Applies the 5-rule decision logic. If no baseline is set or incomplete,
        returns raw prediction with calibrated=False.

        Args:
            extraction_result: Output from extractor's extract(). Must include
                'embedding', 'top_emotion', 'confidence', 'emotion_probs'.

        Returns:
            Dict with: calibrated, emotion, confidence, emotion_source,
                similarities, closest_baseline.
        """
        if self.baseline is None or not self.baseline.is_complete():
            raw = self.get_raw_prediction(extraction_result)
            raw['calibrated'] = False
            raw['warning'] = 'No calibration available'
            return raw

        current_embedding = extraction_result['embedding']

        # Compute cosine similarities to all baseline states
        similarities = {}
        for state, baseline_emb in self.baseline.embeddings.items():
            similarities[state] = cosine_similarity(current_embedding, baseline_emb)

        closest_state = max(similarities, key=similarities.get)
        closest_similarity = similarities[closest_state]

        raw_emotion = extraction_result['top_emotion']
        raw_confidence = extraction_result['confidence']
        below_deviation_floor = closest_similarity < self.deviation_floor
        emotion_probs = extraction_result.get('emotion_probs', {})

        # Helper: find best non-calibrated emotion from raw probabilities
        def _best_non_cal():
            non_cal = {k: v for k, v in emotion_probs.items()
                       if k not in self.calibrated_emotions}
            if non_cal and max(non_cal.values()) > 0.05:
                return max(non_cal, key=non_cal.get)
            return None

        # If raw model strongly detects a non-calibrated emotion (>90%),
        # boost the threshold so only a very strong calibration match can override.
        raw_is_strong_noncat = (
            raw_emotion not in self.calibrated_emotions and raw_confidence > 0.90
        )
        effective_sim_threshold = self.similarity_threshold + (0.05 if raw_is_strong_noncat else 0)
        effective_neu_threshold = self.neutral_threshold + (0.05 if raw_is_strong_noncat else 0)

        # Rule 1: Closest baseline passes threshold → use calibration
        if closest_state == 'neutral' and closest_similarity > effective_neu_threshold:
            calibrated_emotion = 'Neutral'
            emotion_source = 'calibration'
        elif closest_similarity > effective_sim_threshold:
            state_to_emotion = {'neutral': 'Neutral', 'happy': 'Happy'}
            calibrated_emotion = state_to_emotion.get(closest_state, closest_state.title())
            emotion_source = 'calibration'

        # Rule 2: Raw model confidently detects non-calibrated emotion
        elif raw_emotion not in self.calibrated_emotions and raw_confidence > self.raw_override_confidence:
            calibrated_emotion = raw_emotion
            emotion_source = 'raw_model'

        # Rule 3: Below deviation floor → user in uncalibrated territory
        elif below_deviation_floor:
            best = _best_non_cal()
            if best:
                calibrated_emotion = best
                emotion_source = 'deviation_fallback'
            else:
                calibrated_emotion = raw_emotion
                emotion_source = 'raw_model'

        # Rule 4: Raw says calibrated emotion but rejected → use best non-cal
        elif raw_emotion in self.calibrated_emotions:
            best = _best_non_cal()
            if best:
                calibrated_emotion = best
                emotion_source = 'fallback'
            else:
                calibrated_emotion = raw_emotion
                emotion_source = 'raw_model'

        # Rule 5: Otherwise → raw model
        else:
            calibrated_emotion = raw_emotion
            emotion_source = 'raw_model'

        # Confidence: use cosine similarity for calibration, raw conf otherwise
        if emotion_source == 'calibration':
            calibrated_confidence = closest_similarity
        else:
            calibrated_confidence = raw_confidence
        calibrated_confidence = max(0.0, min(1.0, calibrated_confidence))

        return {
            'calibrated': True,
            'emotion': calibrated_emotion,
            'confidence': calibrated_confidence,
            'emotion_source': emotion_source,
            'similarities': similarities,
            'closest_baseline': closest_state,
        }
