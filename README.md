# AIRA Emotion Detection

**BT4103 Business Analytics Capstone Project — Team 05**

A personalised multimodal emotion detection system for [Prescience PresbyRobotics'](https://www.presciencerobotics.com/) AIRA companion robot. Combines facial analysis (DeepFace) and speech recognition (Emotion2Vec Large) through late fusion, with per-user calibration to adapt to individual expressions.

## Key Results

| Metric | Value |
|--------|-------|
| MLP fusion accuracy (RAVDESS, speaker-independent) | 91% |
| Sadness rescue (face-only → fused) | 8% → 68% |
| Out-of-domain elderly accuracy | 79% |
| Calibration onboarding time | 10 seconds |

## Repository Structure

```
BT4103-Team-5-AIRA-Emotion-Recognition/
├── pipeline/                        # Emotion detection pipeline
│   ├── core/                        # Core modules (fusion, calibration, extractors)
│   ├── models/                      # Trained MLP fusion weights
│   └── demo.py                      # Pipeline GUI (Tkinter) — camera + mic input
├── dashboard/                       # Streamlit monitoring dashboard
│   ├── app.py                       # Main dashboard application
│   ├── model_performance.py         # Evaluation metrics module
│   └── data/                        # Demo data + live data + event logs
├── analyses/
│   ├── evaluation/                  # Unified evaluation script + results
│   ├── training/                    # MLP training scripts (archival)
│   ├── visual_models_testing/        # Visual model benchmarking scripts + results
│   └── audio_models_testing/        # Audio model benchmarking notebooks
├── requirements.txt                 # Python dependencies
└── README.md
```

## Prerequisites

- **Python 3.11+**
- **macOS** (tested on Apple M1 Pro) or Linux

### System Dependencies

These must be installed before pip packages:

**macOS (Homebrew):**
```bash
brew install ffmpeg       # Required by FunASR for audio processing
brew install portaudio    # Required by sounddevice for microphone access
# tkinter comes with Python from Homebrew or python.org installer
# If missing: brew install python-tk@3.11
```

**Windows:**
```bash
# ffmpeg: download from https://ffmpeg.org/download.html and add to PATH
# portaudio: installed automatically by pip sounddevice on Windows
# tkinter: included with standard Python installer from python.org
```

**Linux (apt):**
```bash
sudo apt install ffmpeg portaudio19-dev python3-tk
```

## Installation

```bash
# Clone the repository
git clone https://github.com/andersooi/BT4103-Team-5-AIRA-Emotion-Recognition.git
cd BT4103-Team-5-AIRA-Emotion-Recognition

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

## How to Run

### Pipeline GUI (Emotion Detection)

Launches a Tkinter GUI with live camera and microphone input for real-time emotion detection.

```bash
PYTHONPATH=. venv/bin/python pipeline/demo.py --camera 0
```

- `--camera 0` — default camera (adjust index for your setup)
- First run downloads model weights (~600MB total) and takes ~30 seconds to initialise
- Click **Calibrate Face** to personalise detection (10-second onboarding: neutral 5s + happy 5s)

### Dashboard (Monitoring)

Launches a Streamlit web dashboard for real-time emotion monitoring.

```bash
cd dashboard
../venv/bin/streamlit run app.py
```

- Toggle between **Demo** mode (pre-recorded scenarios) and **Live** mode (reads from pipeline output)
- For live mode, start the pipeline first, then switch the dashboard to Live

### Evaluation

Runs all 4 fusion approaches (face-only, audio-only, rule-based, MLP) on the test datasets.

```bash
PYTHONPATH=. venv/bin/python analyses/evaluation/evaluate_all.py
```

Results are written to `analyses/evaluation/results/`.

## Architecture

```
Camera  → DeepFace (facial emotion)    ─┐
                                         ├→ Calibration → Late Fusion → Russell's Circumplex
Microphone → Emotion2Vec (speech emotion) ─┘       ↑              ↓
                                          VAD gate        Dashboard (Streamlit)
```

- **Calibration**: 10-second onboarding captures neutral + happy baselines. Adaptive thresholds computed from inter-baseline cosine similarity.
- **Late Fusion**: Two approaches — probability rule-based (signal-aware weights) and MLP learned (14-dim → 32 → 16 → 7, trained on RAVDESS).
- **VAD**: FunASR FSMN-VAD filters silence before audio reaches fusion.

## Models Used

| Component | Model | Source | License |
|-----------|-------|--------|---------|
| Face emotion | DeepFace | [serengil/deepface](https://github.com/serengil/deepface) | MIT |
| Speech emotion | Emotion2Vec Large | [FunASR](https://github.com/modelscope/FunASR) | Apache 2.0 |
| Voice activity | FSMN-VAD | [FunASR](https://github.com/modelscope/FunASR) | Apache 2.0 |
| Fusion | MLP (custom) | Trained on RAVDESS | — |

## Team

| Name | GitHub |
|------|--------|
| Anushka Ashirgade | [@aanxshka](https://github.com/aanxshka) |
| Ooi Jia Yu, Anders | [@andersooi](https://github.com/andersooi) |
| Wang Yuqing | [@wyqng123](https://github.com/wyqng123) |
| Sidney Madeline Lawther | [@sidneyML](https://github.com/sidneyML) |
| Sourick Paul | [@sourick23](https://github.com/sourick23) |
| Hu Zhixuan, Shirley | [@shirleyhzx](https://github.com/shirleyhzx) |
