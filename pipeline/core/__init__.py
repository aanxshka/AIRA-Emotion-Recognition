"""
AIRA Emotion Detection Pipeline — Core Module

Re-exports all public classes and functions for convenient imports:
    from pipeline.core import DeepFaceEmotionEmbeddingExtractor, MLPFusion, ...
"""

# Face model
from pipeline.core.face_extractor import DeepFaceEmotionEmbeddingExtractor

# Audio model
from pipeline.core.audio_extractor import Emotion2VecExtractor

# Calibration
from pipeline.core.calibration import (
    GenericBaseline,
    GenericCalibratedDetector,
    cosine_similarity,
    average_embeddings,
    average_values,
)

# Fusion (V3 rule-based)
from pipeline.core.fusion import (
    ProbabilityFusion,
    FusionResult,
    SHARED_EMOTIONS,
    QUADRANT_LABELS,
    EMOTION_VA_LOOKUP,
    align_face_probs,
    align_audio_probs,
    compute_modality_weights,
    va_to_quadrant,
)

# Fusion (MLP learned)
from pipeline.core.mlp_fusion import MLPFusion

# Fusion adapter (calibration → fusion bridge)
from pipeline.core.fusion_adapter import build_face_result
