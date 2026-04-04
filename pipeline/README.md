# AIRA Emotion Detection Pipeline

Real-time multimodal emotion detection combining facial and speech analysis with per-user calibration and late fusion.

## Architecture

```
Webcam → Face Detection (Haar) → DeepFace Emotion CNN → Calibration → Fusion Adapter ─┐
                                                                                        ├→ MLP Fusion → Emotion + Quadrant
Microphone → VAD (FunASR FSMN) → Emotion2Vec Large ───────────────────────────────────┘
                                                                                        ↓
                                                                               CSV → Dashboard
```

## Running the Demo

```bash
cd BT4103-Team-5-AIRA-Emotion-Recognition
pip install -r requirements.txt
PYTHONPATH=. python pipeline/demo.py --camera 1
```

Use `--camera 0` or `--camera 1` depending on your system (check which index is your webcam).

## Modules

| File | Responsibility |
|------|---------------|
| `core/face_extractor.py` | DeepFace emotion model — 1024-dim embeddings + 7-class probabilities |
| `core/audio_extractor.py` | Emotion2Vec large — speech embeddings + 9-class probabilities |
| `core/calibration.py` | Per-user calibration — baseline storage, cosine similarity detection, adaptive thresholds |
| `core/fusion.py` | V3 rule-based fusion — label alignment, Happy-protected asymmetric audio weighting, quadrant mapping |
| `core/mlp_fusion.py` | MLP learned fusion — 14→32→16→7 neural network trained on RAVDESS |
| `core/fusion_adapter.py` | Bridges calibration output into a coherent probability vector for fusion |
| `demo.py` | Full Tkinter GUI with camera, mic, calibration, fusion, and CSV logging |
| `models/mlp_fusion.pt` | Trained MLP weights (7.4KB) |

## How Calibration Works

1. User holds a **neutral face** for 5 seconds — system captures ~25 frames, averages embeddings into a centroid
2. User holds a **happy face** for 5 seconds — same process
3. System computes **adaptive thresholds** from how similar the two baselines are to each other
4. At runtime: each live frame's embedding is compared to baselines via cosine similarity
5. If similarity to a baseline exceeds the threshold → use calibrated label. Otherwise → raw model output

This corrects for individual differences (e.g., elderly resting face misclassified as sad by the raw model).

## How Fusion Works

Two approaches are available (MLP is the default):

**MLP Fusion (learned):**
- Takes 7 aligned face probabilities + 7 aligned audio probabilities = 14-dim input
- Trained on 1140 RAVDESS clips with 5-fold cross-validation (91.8% ± 3.7%)
- Only runs when both modalities are present; falls back to single-modality output otherwise

**V3 Rule-Based Fusion:**
- Weighted average of aligned face + audio probabilities
- Audio Neutral → 85% face / 15% audio (audio has no signal)
- Both agree → 50/50
- Audio negative + face not Happy → 35% face / 65% audio (rescue negative emotions)
- Audio negative + face Happy → 55% face / 45% audio (protect Happy)

## CSV Output

The demo writes to `dashboard/data/emotion_live_data.csv` with columns:

| Column | Type | Range |
|--------|------|-------|
| `timestamp` | datetime (ms) | e.g., "2026-04-04 14:30:00.123" |
| `primary_emotion` | string | Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise |
| `confidence` | float | 0-100 |
| `happy_score` ... `surprise_score` | float | 0-100 (fused probability × 100) |
| `video_signal_quality` | float | 0-100 (rolling face detection rate) |
| `audio_signal_quality` | float | 0-100 (rolling VAD speech rate) |
| `video_feed_active` | bool | True/False |
| `audio_feed_active` | bool | True/False |

## Evaluation Results

Tested on RAVDESS (300 clips, 5 held-out actors, speaker-independent) and AI-generated elderly clips (14 clips, out-of-domain):

| Dataset | Face-only | Audio-only | V3 Rule-Based | MLP Learned |
|---------|-----------|------------|---------------|-------------|
| RAVDESS (300) | 30% | 91% | 87% | **91%** |
| Elderly (14) | 57% | 64% | 71% | **79%** |

Key findings:
- Sadness rescue: face gets 8% → fused gets 68% (audio compensates)
- Happy preserved: 100% (both fusion approaches protect DeepFace's strongest class)
- MLP handles Surprise better than V3 (90% vs 68%)
