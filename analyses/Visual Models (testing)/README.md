# Visual Models — Emotion Recognition Experiments

## Overview

This folder contains scripts and outputs for evaluating multiple visual emotion recognition models.

The goal of this module was to:
- Compare different visual emotion recognition approaches
- Evaluate performance across emotion classes and age groups
- Identify a suitable model for the final system
- Understand limitations, especially for elderly emotion detection

---

## Models Evaluated

### Feature-Based Models
- MediaPipe (Face Mesh + geometry)
- Py-Feat (Action Units)

### Neural Network Models
- DeepFace
- RetinaFace + DeepFace
- FER (Facial Expression Recognition)
- HSEmotion(enet_b2, enet_b0_vgaf, enet_b0_va_mtl)
- dlib (geometric-based variants)

---

## Folder Structure

### Individual model scripts (Anushka)
Individual evaluator scripts per model, each self-contained and independently runnable.

### hs_fer_mediapipe_evaluation (Sidney)
Extended evaluation of HSEmotion, FERPlus and MediaPipe variants via a full Streamlit app, including:

- Multi-model comparison dashboard
- AI-generated vs real elderly data suitability analysis
- Baseline vs improved model comparison (confidence-weighted frame aggregation)
- Statistical analysis: McNemar's test, Cohen's Kappa, AUC per class
- Tested HSEmotion (3 variants) + FERPlus + MediaPipe with baseline vs improved comparisons

### results
Outputs from different model runs including evaluation metrics, confusion matrices, and model-specific outputs.

---

## How to Run

There is no shared environment or `requirements.txt`.

Each model was tested independently due to dependency conflicts.

### Setup (per model)

1. Create a virtual environment:
    python -m venv venv_name
    source venv_name/bin/activate # Mac/Linux
    venv_name\Scripts\activate # Windows
2. Install required libraries manually depending on the script:
    pip install deepface opencv-python
3. Run the script:
    python deepface_evaluator.py


---

## Important Notes for Handover

- Each script is self-contained and independent
- There is no standardised pipeline across models
- Scripts may require:
  - Manual file path updates
  - Input video adjustments
  - Additional dependencies

Recommended approach:
- Run one model at a time
- Use a separate virtual environment per model
- Inspect scripts before execution

---

## Results

The `results/` folder contains outputs from different model runs:
- Evaluation metrics (Accuracy, Macro F1, AUC)
- Confusion matrices (if generated)
- Model-specific outputs

Each subfolder corresponds to:
- A specific model
- A specific test run

---

## Key Findings

- DeepFace showed the strongest overall performance
- Py-Feat performed reasonably but slightly lower
- FER had strong confidence scores (AUC)
- HSEmotion and MediaPipe underperformed

### Elderly Insight

- All models performed worse on elderly faces
- Subtle expressions were difficult to detect
- Stronger models remained stronger across age groups

This motivated the use of a personalised calibration layer in the final system.

---

## Limitations

- No shared environment setup
- Dataset not included
- Small evaluation sample size
- Scripts are experimental and not standardised

---

## Role in Final System

This module informed:
- Selection of DeepFace as the visual model
- Identification of elderly detection limitations
- Design of the calibration approach

---

## Notes

This folder represents an experimental exploration phase.

It is not a production-ready pipeline, but a record of model evaluation and experimentation.
