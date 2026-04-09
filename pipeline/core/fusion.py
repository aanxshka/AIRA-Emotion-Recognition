"""
Late Fusion for Multimodal Emotion Recognition (V3 Rule-Based)

Combines face (DeepFace) and audio (Emotion2Vec) emotion probabilities
into a single fused prediction mapped to Russell's Circumplex quadrants.

The two models use different label sets:
    DeepFace:    Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral (7)
    Emotion2Vec: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise, Other, Unknown (9)

Both are aligned to 7 shared categories before fusion:
    Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

Modality weighting (V3 — Happy-protected asymmetric):
    Audio Neutral/None        → 85% face / 15% audio
    Agreement                 → 50/50
    Audio negative + face≠Happy → 35% face / 65% audio (rescue negative emotions)
    Audio negative + face=Happy → 55% face / 45% audio (protect Happy)
    General disagreement      → 55% face / 45% audio

Usage:
    fusion = ProbabilityFusion()
    result = fusion.fuse(face_result, audio_result)
    # result.emotion → 'Sad'
    # result.quadrant → 'Q3'
    # result.face_weight → 0.35
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple


# ============================================================================
# Constants
# ============================================================================

# 7 shared emotion categories between DeepFace and Emotion2Vec
SHARED_EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# DeepFace label → shared label.
# Tolerant mapping: accepts both raw labels (Happiness, Sadness) and
# calibrated/aligned labels (Happy, Sad) without breaking.
# Uses += accumulation so aliased keys don't overwrite each other.
FACE_TO_SHARED = {
    'Anger': 'Angry',
    'Angry': 'Angry',
    'Disgust': 'Disgust',
    'Fear': 'Fear',
    'Happiness': 'Happy',
    'Happy': 'Happy',
    'Neutral': 'Neutral',
    'Sadness': 'Sad',
    'Sad': 'Sad',
    'Surprise': 'Surprise',
}

# Emotion2Vec label → shared label (drops Other, Unknown)
AUDIO_TO_SHARED = {
    'Angry': 'Angry',
    'Disgust': 'Disgust',
    'Fear': 'Fear',
    'Happy': 'Happy',
    'Neutral': 'Neutral',
    'Sad': 'Sad',
    'Surprise': 'Surprise',
}

# Standard V-A coordinates for each emotion (Russell's Circumplex).
# Used to map discrete emotion predictions to quadrants.
EMOTION_VA_LOOKUP = {
    'Happy':    ( 0.8,  0.5),   # Q1: positive valence, high arousal
    'Angry':    (-0.6,  0.7),   # Q2: negative valence, high arousal
    'Fear':     (-0.6,  0.6),   # Q2: negative valence, high arousal
    'Disgust':  (-0.6,  0.3),   # Q2: negative valence, mild arousal
    'Surprise': ( 0.2,  0.7),   # Q1: mild positive, high arousal
    'Sad':      (-0.7, -0.3),   # Q3: negative valence, low arousal
    'Neutral':  ( 0.0,  0.0),   # Center
}

QUADRANT_LABELS = {
    'Q1': 'Q1 (Happy/Excited)',
    'Q2': 'Q2 (Angry/Anxious)',
    'Q3': 'Q3 (Sad/Depressed)',
    'Q4': 'Q4 (Calm/Content)',
    'Neutral': 'Neutral',
}


# ============================================================================
# Utility Functions
# ============================================================================

def va_to_quadrant(valence: float, arousal: float, threshold: float = 0.1) -> str:
    """Map V-A coordinates to Russell's Circumplex quadrant.

    Args:
        valence: -1 (negative) to +1 (positive)
        arousal: -1 (low energy) to +1 (high energy)
        threshold: values within this range of zero are classified as Neutral

    Returns:
        One of: 'Q1', 'Q2', 'Q3', 'Q4', 'Neutral'
    """
    if abs(valence) < threshold and abs(arousal) < threshold:
        return 'Neutral'
    elif valence >= 0 and arousal >= 0:
        return 'Q1'
    elif valence < 0 and arousal >= 0:
        return 'Q2'
    elif valence < 0 and arousal < 0:
        return 'Q3'
    else:
        return 'Q4'


def align_face_probs(face_probs: Dict[str, float]) -> Dict[str, float]:
    """Align face model probabilities to 7 shared categories.

    Handles both DeepFace labels (Happiness, Sadness) and already-aligned
    labels (Happy, Sad) by summing into shared buckets. Drops any labels
    not in FACE_TO_SHARED (e.g., HSEmotion's Contempt). Renormalizes to sum to 1.
    """
    aligned = {em: 0.0 for em in SHARED_EMOTIONS}
    for face_name, shared_name in FACE_TO_SHARED.items():
        aligned[shared_name] += face_probs.get(face_name, 0.0)

    total = sum(aligned.values())
    if total > 0:
        aligned = {k: v / total for k, v in aligned.items()}
    return aligned


def align_audio_probs(audio_probs: Dict[str, float]) -> Dict[str, float]:
    """Align Emotion2Vec probabilities to 7 shared categories.

    Drops Other and Unknown classes, renormalizes to sum to 1.
    """
    aligned = {}
    for audio_name, shared_name in AUDIO_TO_SHARED.items():
        aligned[shared_name] = audio_probs.get(audio_name, 0.0)

    total = sum(aligned.values())
    if total > 0:
        aligned = {k: v / total for k, v in aligned.items()}
    return aligned


def compute_modality_weights(
    face_emotion: Optional[str], face_conf: float,
    audio_emotion: Optional[str], audio_conf: float,
) -> Tuple[float, float]:
    """Compute face/audio weights based on what each model detected.

    V3 rule set (Happy-protected asymmetric audio boost):
        - Audio Neutral/None → 85/15 (audio has no signal)
        - Agreement → 50/50
        - Audio negative + face NOT Happy → 35/65 (rescue negative emotions)
        - Audio negative + face IS Happy → 55/45 (protect DeepFace's strongest class)
        - General disagreement → 55/45

    Returns:
        (face_weight, audio_weight) tuple summing to 1.0
    """
    if audio_emotion is None or audio_emotion == 'Neutral':
        return 0.85, 0.15

    face_shared = FACE_TO_SHARED.get(face_emotion, face_emotion) if face_emotion else None

    if face_shared is not None and audio_emotion == face_shared:
        return 0.50, 0.50

    # Audio detects negative emotion → trust audio more, but protect Happy.
    # DeepFace struggles with negative emotions (0-18% on Disgust/Fear/Sad)
    # and often misclassifies them (e.g., Disgust→Angry).
    # DeepFace is strong on Happy (80%) — don't let false audio override it.
    if (audio_emotion in ('Sad', 'Angry', 'Fear', 'Disgust')
            and audio_conf >= 0.50
            and face_shared != 'Happy'):
        return 0.35, 0.65

    return 0.55, 0.45


# ============================================================================
# FusionResult
# ============================================================================

@dataclass
class FusionResult:
    """Container for fusion output. Includes per-modality info for debugging."""
    emotion: str
    confidence: float
    quadrant: str # 'Q1', 'Q2', 'Q3', 'Q4', or 'Neutral'
    quadrant_label: str # e.g., 'Q3 (Sad/Depressed)'

    fused_valence: Optional[float]
    fused_arousal: Optional[float]

    face_emotion: Optional[str]
    face_confidence: Optional[float]
    audio_emotion: Optional[str]
    audio_confidence: Optional[float]

    fusion_version: int # 1 = ProbabilityFusion (V3), 3 = MLPFusion
    face_weight: float
    audio_weight: float
    modalities_present: str # 'both', 'face_only', 'audio_only', 'none'
    fused_probs: Optional[Dict[str, float]] = None  # Full 7-class distribution


# ============================================================================
# Probability Fusion (V3)
# ============================================================================

def _default_result(fusion_version: int) -> FusionResult:
    """Return a default Neutral result when no modalities are present."""
    return FusionResult(
        emotion='Neutral', confidence=0.0,
        quadrant='Neutral', quadrant_label=QUADRANT_LABELS['Neutral'],
        fused_valence=None, fused_arousal=None,
        face_emotion=None, face_confidence=None,
        audio_emotion=None, audio_confidence=None,
        fusion_version=fusion_version,
        face_weight=0.0, audio_weight=0.0,
        modalities_present='none',
    )


class ProbabilityFusion:
    """V3 rule-based late fusion via weighted probability averaging.

    Aligns face and audio probabilities to 7 shared categories,
    applies signal-based modality weighting, and returns the
    top emotion from the weighted average distribution.

    Handles missing modalities: face-only, audio-only, both, or none.
    """

    def fuse(self, face_result: Optional[Dict], audio_result: Optional[Dict]) -> FusionResult:
        """Fuse face and audio predictions into a single emotion output.

        Args:
            face_result: Dict with top_emotion, confidence, emotion_probs. Or None.
            audio_result: Dict with top_emotion, confidence, emotion_probs. Or None.

        Returns:
            FusionResult with fused emotion, quadrant, weights, per-modality info.
        """
        if face_result is None and audio_result is None:
            return _default_result(fusion_version=1)

        # Face only
        if face_result is not None and audio_result is None:
            aligned = align_face_probs(face_result['emotion_probs'])
            top = max(aligned, key=aligned.get)
            va = EMOTION_VA_LOOKUP.get(top, (0.0, 0.0))
            q = va_to_quadrant(*va)
            return FusionResult(
                emotion=top, confidence=face_result['confidence'],
                quadrant=q, quadrant_label=QUADRANT_LABELS[q],
                fused_valence=None, fused_arousal=None,
                face_emotion=face_result['top_emotion'], face_confidence=face_result['confidence'],
                audio_emotion=None, audio_confidence=None,
                fusion_version=1, face_weight=1.0, audio_weight=0.0,
                modalities_present='face_only',
                fused_probs=aligned,
            )

        # Audio only
        if face_result is None and audio_result is not None:
            aligned = align_audio_probs(audio_result['emotion_probs'])
            top = max(aligned, key=aligned.get)
            va = EMOTION_VA_LOOKUP.get(top, (0.0, 0.0))
            q = va_to_quadrant(*va)
            return FusionResult(
                emotion=top, confidence=audio_result['confidence'],
                quadrant=q, quadrant_label=QUADRANT_LABELS[q],
                fused_valence=None, fused_arousal=None,
                face_emotion=None, face_confidence=None,
                audio_emotion=audio_result['top_emotion'], audio_confidence=audio_result['confidence'],
                fusion_version=1, face_weight=0.0, audio_weight=1.0,
                modalities_present='audio_only',
                fused_probs=aligned,
            )

        # Both present: weighted fusion
        face_conf = face_result['confidence']
        audio_conf = audio_result['confidence']

        fw, aw = compute_modality_weights(
            face_result['top_emotion'], face_conf,
            audio_result['top_emotion'], audio_conf,
        )

        face_aligned = align_face_probs(face_result['emotion_probs'])
        audio_aligned = align_audio_probs(audio_result['emotion_probs'])

        fused_probs = {}
        for em in SHARED_EMOTIONS:
            fused_probs[em] = fw * face_aligned.get(em, 0.0) + aw * audio_aligned.get(em, 0.0)

        top = max(fused_probs, key=fused_probs.get)
        fused_conf = fused_probs[top]
        va = EMOTION_VA_LOOKUP.get(top, (0.0, 0.0))
        q = va_to_quadrant(*va)

        return FusionResult(
            emotion=top, confidence=fused_conf,
            quadrant=q, quadrant_label=QUADRANT_LABELS[q],
            fused_valence=None, fused_arousal=None,
            face_emotion=face_result['top_emotion'], face_confidence=face_conf,
            audio_emotion=audio_result['top_emotion'], audio_confidence=audio_conf,
            fusion_version=1, face_weight=fw, audio_weight=aw,
            modalities_present='both',
            fused_probs=fused_probs,
        )
