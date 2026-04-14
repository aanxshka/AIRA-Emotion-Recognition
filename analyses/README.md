# Analyses

Model evaluation, training, and benchmarking scripts.

## Datasets

The datasets used for training, evaluation, and testing are stored on Google Drive (too large for Git):

**[Google Drive — Shared Datasets](https://drive.google.com/drive/folders/1Yrdzc3_JBe_KiQgWzIdeWBqJQhy1pkH-?usp=sharing)**

Contents:
- RAVDESS (1,440 clips, 24 actors, 8 emotions)
- AI-generated elderly video clips (14 clips)
- YouTube neurological condition clips (8 clips — Parkinson's, Bell's palsy)
- Visual model testing videos (36 clips, team-curated)

## Subfolders

| Folder | Description |
|--------|-------------|
| `evaluation/` | Unified evaluation script (`evaluate_all.py`) that runs Face-only, Audio-only, Probability Rule-Based, and MLP fusion on RAVDESS test set (300 clips), AI-generated elderly clips (14), and neurological YouTube clips (8). Outputs per-dataset CSVs and summary metrics. |
| `training/` | Archival scripts for MLP fusion training: data splitting (`data_split.py`), feature extraction (`extract_features.py`), and model training with 5-fold CV (`train_mlp_fusion.py`). References the calibration_test experiment repo paths. |
| `visual_models_testing/` | Benchmarking scripts and results for 6 visual models: DeepFace, RetinaFace+DeepFace, FER, HSEmotion (3 variants), MediaPipe, Py-feat. Includes a Streamlit comparison app (`hs_fer_mediapipe_evaluation/`) for interactive model comparison. Per-model results include confusion matrices, per-class metrics, and age-group breakdowns. |
| `audio_models_testing/` | Evaluation notebooks for audio models: Emotion2Vec and Wav2Vec, tested on RAVDESS (2,880 samples, 24 speakers). |

## Related Repository

The calibration experimentation repo contains additional work on calibration approaches (embedding similarity, action units, landmarks), DeepFace calibration experiments, and the comparison demo GUIs:

**[andersooi/emotional-detection-calibration-experiment](https://github.com/andersooi/emotional-detection-calibration-experiment)**
