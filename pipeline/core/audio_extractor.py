"""
Emotion2Vec Speech Emotion Extractor

Extracts speech embeddings and 9-class emotion probabilities from audio
using Emotion2Vec (via FunASR). Supports base (768-dim) and large (1024-dim)
model variants.

The large model is recommended — dramatically better on Sad (80% vs 20%)
and Fear (100% vs 75%) compared to base, with the same overall accuracy.

FunASR requires audio input as a file path, so this extractor writes
a temporary WAV file for each inference call.

Usage:
    extractor = Emotion2VecExtractor(model_size='large')
    extractor.load()
    result = extractor.extract(audio_array, sample_rate=16000)
    # result['embedding'] → np.ndarray (768-dim for base, 1024-dim for large)
    # result['emotion_probs'] → {'Angry': 0.05, ..., 'Neutral': 0.70}
    # result['top_emotion'] → 'Neutral'
    # result['confidence'] → 0.70
"""

import os
import tempfile
import numpy as np
import soundfile as sf
from typing import Dict

# Note: FunASR is imported lazily in load() because it downloads model
# weights on first import and has heavy initialization.


# Emotion2Vec outputs bilingual labels — map to clean English names.
EMOTION_MAP = {
    '生气/angry': 'Angry',
    '厌恶/disgusted': 'Disgust',
    '恐惧/fearful': 'Fear',
    '开心/happy': 'Happy',
    '中立/neutral': 'Neutral',
    '其他/other': 'Other',
    '难过/sad': 'Sad',
    '吃惊/surprised': 'Surprise',
    '<unk>': 'Unknown',
}


class Emotion2VecExtractor:
    """Extract embeddings and emotions from audio using Emotion2Vec.

    Supports two model sizes:
        - 'base': Faster, lower accuracy
        - 'large': Slower, significantly better on Sad and Fear

    The model outputs 9 emotion classes (including Other and Unknown),
    which are later aligned to 7 shared classes for fusion.
    """

    def __init__(self, model_size: str = 'large'):
        """
        Args:
            model_size: 'base' or 'large'. Default 'large' for better accuracy.
        """
        self.model_size = model_size
        self.model = None

    def load(self, status_callback=None):
        """Load the Emotion2Vec model from HuggingFace.

        Args:
            status_callback: Optional function to report loading progress.
        """
        if status_callback:
            status_callback(f"Loading Emotion2Vec ({self.model_size})...")

        # Lazy import: FunASR downloads model weights and has heavy init
        from funasr import AutoModel

        model_name = f"iic/emotion2vec_plus_{self.model_size}"
        self.model = AutoModel(model=model_name, hub='hf', disable_update=True)

        if status_callback:
            status_callback("Emotion2Vec loaded!")

    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict:
        """Extract emotion embedding and probabilities from audio.

        Args:
            audio: Audio waveform as numpy array (1D, float32).
            sample_rate: Audio sample rate in Hz (default 16000).

        Returns:
            Dict with keys:
                embedding: np.ndarray speech representation (768-dim base, 1024-dim large)
                emotion_probs: Dict[str, float] — 9 emotion probabilities
                top_emotion: str — highest-scoring emotion
                confidence: float — probability of top emotion (0-1)
        """
        if self.model is None:
            self.load()

        # Ensure audio is float32 and normalized to [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        # FunASR requires a file path — write temp WAV
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio, sample_rate)

        try:
            result = self.model.generate(
                temp_path,
                granularity='utterance',
                extract_embedding=True,
            )

            output = result[0]
            embedding = output['feats'] # Shape varies by model (768 base, 1024 large)
            labels = output['labels']
            scores = np.array(output['scores'])

            # Normalize scores to probabilities
            if scores.sum() > 0:
                scores = scores / scores.sum()

            # Map bilingual labels to clean English names
            emotion_probs = {}
            for label, score in zip(labels, scores):
                clean_name = EMOTION_MAP.get(label, label)
                emotion_probs[clean_name] = float(score)

            top_emotion = max(emotion_probs, key=emotion_probs.get)

            return {
                'embedding': embedding,
                'emotion_probs': emotion_probs,
                'top_emotion': top_emotion,
                'confidence': emotion_probs[top_emotion],
            }

        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
