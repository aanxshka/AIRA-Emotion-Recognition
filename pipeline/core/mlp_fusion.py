"""
MLP Learned Fusion for Multimodal Emotion Recognition

A small neural network trained on RAVDESS to learn optimal combination
weights for face + audio emotion probabilities. Drop-in replacement for
ProbabilityFusion — same FusionResult interface.

Architecture: 14-dim input [7 face + 7 audio aligned probs]
              → 32 (ReLU, Dropout 0.3)
              → 16 (ReLU, Dropout 0.3)
              → 7-class softmax output

Trained on 1140 RAVDESS clips (19 actors), 5-fold CV: 91.8% ± 3.7%.
Weight analysis shows audio is 1.3x more influential than face overall,
but face contributes meaningfully for Happy (0.389 vs 0.266 audio).

Usage:
    fusion = MLPFusion(model_path='pipeline/models/mlp_fusion.pt')
    result = fusion.fuse(face_result, audio_result)
    # result.emotion → 'Sad'
    # result.quadrant → 'Q3'
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional

from pipeline.core.fusion import (
    FusionResult,
    SHARED_EMOTIONS,
    QUADRANT_LABELS,
    align_face_probs,
    align_audio_probs,
    va_to_quadrant,
    EMOTION_VA_LOOKUP,
)


# ============================================================================
# MLP Model
# ============================================================================

class FusionMLP(nn.Module):
    """Small MLP for emotion fusion.

    Takes 14-dim input (7 aligned face probs + 7 aligned audio probs),
    outputs 7-class logits (one per shared emotion).
    """

    def __init__(self, input_dim=14, hidden1=32, hidden2=16, num_classes=7, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# MLPFusion (same interface as ProbabilityFusion)
# ============================================================================

class MLPFusion:
    """Learned fusion using a trained MLP on face + audio probability vectors.

    Drop-in replacement for ProbabilityFusion — produces the same FusionResult.
    When the model is not loaded, falls back to a simple 50/50 average.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = FusionMLP()
        self.model.eval()
        self.loaded = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def load(self, model_path: str):
        """Load trained weights from a .pt file."""
        state = torch.load(model_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.loaded = True
        print(f"[MLPFusion] Loaded from {model_path}")

    def save(self, model_path: str):
        """Save trained weights to a .pt file."""
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        print(f"[MLPFusion] Saved to {model_path}")

    def _build_input(self, face_result: Optional[Dict],
                     audio_result: Optional[Dict]) -> np.ndarray:
        """Build 14-dim input vector from face + audio results.

        Aligns both probability distributions to 7 shared categories,
        then concatenates: [face_angry, ..., face_surprise, audio_angry, ..., audio_surprise].
        Missing modality gets uniform 1/7 probabilities.
        """
        if face_result and 'emotion_probs' in face_result:
            face_aligned = align_face_probs(face_result['emotion_probs'])
        else:
            face_aligned = {em: 1.0 / 7 for em in SHARED_EMOTIONS}

        if audio_result and 'emotion_probs' in audio_result:
            audio_aligned = align_audio_probs(audio_result['emotion_probs'])
        else:
            audio_aligned = {em: 1.0 / 7 for em in SHARED_EMOTIONS}

        face_vec = [face_aligned.get(em, 0.0) for em in SHARED_EMOTIONS]
        audio_vec = [audio_aligned.get(em, 0.0) for em in SHARED_EMOTIONS]

        return np.array(face_vec + audio_vec, dtype=np.float32)

    def fuse(self, face_result: Optional[Dict],
             audio_result: Optional[Dict]) -> FusionResult:
        """Fuse face + audio results using the trained MLP.

        Same interface as ProbabilityFusion.fuse(). When model is not loaded,
        falls back to 50/50 average of aligned probabilities.

        Args:
            face_result: Dict with top_emotion, confidence, emotion_probs. Or None.
            audio_result: Dict with top_emotion, confidence, emotion_probs. Or None.

        Returns:
            FusionResult with fused emotion, quadrant, per-modality info.
        """
        has_face = face_result is not None and 'emotion_probs' in (face_result or {})
        has_audio = audio_result is not None and 'emotion_probs' in (audio_result or {})

        if not has_face and not has_audio:
            return FusionResult(
                emotion='Neutral', confidence=0.0,
                quadrant='Neutral', quadrant_label=QUADRANT_LABELS['Neutral'],
                fused_valence=None, fused_arousal=None,
                face_emotion=None, face_confidence=None,
                audio_emotion=None, audio_confidence=None,
                fusion_version=3, face_weight=0.0, audio_weight=0.0,
                modalities_present='none',
            )

        modalities = 'both' if (has_face and has_audio) else (
            'face_only' if has_face else 'audio_only')

        # Only use MLP when BOTH modalities have real data.
        # MLP was trained on real face+audio pairs; feeding it uniform 1/7
        # for a missing modality produces unreliable outputs (e.g., Fear on
        # a neutral face with no audio). For single-modality, use aligned
        # probabilities directly.
        if has_face and has_audio and self.loaded:
            input_vec = self._build_input(face_result, audio_result)
            with torch.no_grad():
                logits = self.model(torch.tensor(input_vec).unsqueeze(0))
                probs = torch.softmax(logits, dim=1).numpy()[0]
        elif has_face:
            probs_dict = align_face_probs(face_result['emotion_probs'])
            probs = np.array([probs_dict.get(em, 0.0) for em in SHARED_EMOTIONS])
        elif has_audio:
            probs_dict = align_audio_probs(audio_result['emotion_probs'])
            probs = np.array([probs_dict.get(em, 0.0) for em in SHARED_EMOTIONS])
        else:
            probs = np.ones(7) / 7

        top_idx = int(np.argmax(probs))
        emotion = SHARED_EMOTIONS[top_idx]
        confidence = float(probs[top_idx])

        va = EMOTION_VA_LOOKUP.get(emotion, (0, 0))
        quadrant = va_to_quadrant(va[0], va[1])

        face_emotion = face_result.get('top_emotion') if face_result else None
        face_conf = face_result.get('confidence', 0.0) if face_result else None
        audio_emotion = audio_result.get('top_emotion') if audio_result else None
        audio_conf = audio_result.get('confidence', 0.0) if audio_result else None

        return FusionResult(
            emotion=emotion,
            confidence=confidence,
            quadrant=quadrant,
            quadrant_label=QUADRANT_LABELS.get(quadrant, quadrant),
            fused_valence=None,
            fused_arousal=None,
            face_emotion=face_emotion,
            face_confidence=face_conf,
            audio_emotion=audio_emotion,
            audio_confidence=audio_conf,
            fusion_version=3,
            face_weight=0.5,   # MLP learns weights internally — 0.5 is nominal
            audio_weight=0.5,
            modalities_present=modalities,
        )
