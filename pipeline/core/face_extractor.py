"""
DeepFace Emotion Embedding Extractor

Extracts 1024-dim embeddings from DeepFace's EMOTION model (not the identity model).
Also returns 7-class emotion probabilities via the full model pipeline.

DeepFace's emotion classifier architecture:
    Conv layers → Flatten → Dense(1024) → Dropout → Dense(1024) → Dropout → Dense(7, softmax)

We extract the penultimate Dense(1024) output as the embedding for calibration,
and use DeepFace.analyze() for probabilities (which goes through the full
preprocessing pipeline including resize + normalize).

Usage:
    extractor = DeepFaceEmotionEmbeddingExtractor()
    extractor.load()
    result = extractor.extract(face_rgb_image)
    # result['embedding'] → 1024-dim np.ndarray
    # result['emotion_probs'] → {'Anger': 0.01, ..., 'Neutral': 0.85}
    # result['top_emotion'] → 'Neutral'
    # result['confidence'] → 0.85
"""

import cv2
import numpy as np
from typing import Dict

# Note: DeepFace, TF, and Keras are imported lazily in load() — see comment there.


# DeepFace emotion labels in the exact order the model outputs them.
# Getting this wrong causes emotion swaps (e.g., Surprise ↔ Sadness).
EMOTION_LABELS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']


class DeepFaceEmotionEmbeddingExtractor:
    """
    Extracts 1024-dim emotion embeddings + 7-class probabilities from DeepFace.

    The embedding comes from the penultimate Dense(1024) layer of the emotion CNN.
    Probabilities come from DeepFace.analyze() which runs the full preprocessing
    pipeline (resize to 224x224, pad, normalize to 0-1).

    For calibration: the embedding is used for cosine similarity comparison.
    For fusion: the emotion_probs are aligned and fused with audio probabilities.
    """

    def __init__(self):
        self._deepface = None
        self._emotion_model = None
        self._embedding_model = None

    def load(self, status_callback=None):
        """Load DeepFace emotion model and create embedding sub-model.

        Args:
            status_callback: Optional function to report loading progress.
        """
        if status_callback:
            status_callback("Loading DeepFace emotion embedding model...")

        # Lazy imports: DeepFace must initialize its Keras model before we import
        # tensorflow directly. Top-level TF import causes Keras state mismatch,
        # breaking model.input access on the Sequential emotion model.
        from deepface import DeepFace
        from deepface.models.demography.Emotion import EmotionClient
        import tensorflow as tf

        self._deepface = DeepFace

        # Load the emotion model
        client = EmotionClient()
        self._emotion_model = client.model

        # Create sub-model that outputs penultimate Dense(1024) layer.
        # Architecture: ... → Dense(1024) → Dropout → Dense(1024) → Dropout → Dense(7)
        # layers[-3] is the second Dense(1024) before final Dropout and Dense(7).
        self._embedding_model = tf.keras.Model(
            inputs=self._emotion_model.input,
            outputs=self._emotion_model.layers[-3].output
        )

        if status_callback:
            status_callback("DeepFace emotion embedding model loaded!")

    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for direct model call: RGB → grayscale → 48x48 → [0,1].

        DeepFace normalizes pixels to 0-1 range in its internal pipeline
        (preprocessing.resize_image). Without this normalization, the model
        receives 0-255 values and produces degenerate outputs (100% one class).

        Args:
            face_image: RGB or grayscale face image as numpy array.

        Returns:
            Preprocessed image as (1, 48, 48, 1) float32 array in [0, 1] range.
        """
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_image
        gray = cv2.resize(gray, (48, 48))
        img = np.expand_dims(np.expand_dims(gray, axis=-1), axis=0).astype(np.float32)
        img /= 255.0
        return img

    def extract(self, face_image: np.ndarray) -> Dict:
        """Extract emotion embedding and probabilities from a face image.

        Args:
            face_image: RGB face image as numpy array (H, W, 3).

        Returns:
            Dict with keys:
                embedding: np.ndarray (1024-dim) from penultimate Dense layer
                emotion_probs: Dict[str, float] — 7 emotion probabilities
                top_emotion: str — highest-scoring emotion
                confidence: float — probability of top emotion (0-1)
        """
        if self._embedding_model is None:
            self.load()

        preprocessed = self._preprocess(face_image)

        # Get 1024-dim emotion embedding from sub-model
        embedding = self._embedding_model.predict(preprocessed, verbose=0)[0]

        # Get emotion probabilities from full model
        probs = self._emotion_model.predict(preprocessed, verbose=0)[0]

        emotion_probs = {label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)}
        top_emotion = max(emotion_probs, key=emotion_probs.get)

        return {
            'embedding': embedding,
            'emotion_probs': emotion_probs,
            'top_emotion': top_emotion,
            'confidence': emotion_probs[top_emotion],
        }
