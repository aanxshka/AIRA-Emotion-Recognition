"""
Model Performance Module
Contains ML model metrics from real RAVDESS evaluation results.

Confusion matrix, accuracy, and F1 are computed from the 300-clip
RAVDESS test set evaluation (5 held-out actors, speaker-independent).

CPU and latency generators are kept for fallback when live data is unavailable.
"""

import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# 7 shared emotion categories (aligned across DeepFace + Emotion2Vec)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
N = len(EMOTIONS)

# Path to evaluation results (relative to dashboard/ working directory)
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'analyses', 'evaluation', 'results', 'ravdess_results.csv')


def fig_to_b64(fig):
    """Convert matplotlib figure to base64 PNG for embedding in HTML"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def load_evaluation_results(approach='mlp_pred'):
    """Load RAVDESS evaluation results and compute confusion matrix.

    Args:
        approach: Column name for predictions. One of:
            'face_pred', 'audio_pred', 'v3_pred', 'mlp_pred'

    Returns:
        Confusion matrix as numpy array (N x N), or None if file not found.
    """
    if not os.path.exists(RESULTS_PATH):
        return None

    df = pd.read_csv(RESULTS_PATH)
    if 'ground_truth' not in df.columns or approach not in df.columns:
        return None

    mat = np.zeros((N, N), dtype=int)
    for _, row in df.iterrows():
        gt = row['ground_truth']
        pred = row[approach]
        if gt in EMOTIONS and pred in EMOTIONS:
            i = EMOTIONS.index(gt)
            j = EMOTIONS.index(pred)
            mat[i, j] += 1

    return mat


def generate_confusion_matrix(approach='mlp_pred'):
    """Return real confusion matrix from evaluation, or simulated fallback."""
    mat = load_evaluation_results(approach)
    if mat is not None:
        return mat

    # Fallback: simulated matrix (only if evaluation results not available)
    mat = np.zeros((N, N), dtype=int)
    for i in range(N):
        total = random.randint(80, 120)
        correct = int(total * random.uniform(0.55, 0.80))
        mat[i, i] = correct
        remaining = total - correct
        others = [j for j in range(N) if j != i]
        for j in others[:-1]:
            v = random.randint(0, remaining)
            mat[i, j] = v
            remaining -= v
            if remaining <= 0:
                break
        mat[i, others[-1]] = max(0, remaining)
    return mat


def accuracy_from_matrix(mat):
    """Calculate accuracy from confusion matrix"""
    total = mat.sum()
    return mat.diagonal().sum() / total if total > 0 else 0.0


def f1_from_matrix(mat):
    """Calculate macro F1 score from confusion matrix"""
    scores = []
    for i in range(mat.shape[0]):
        tp = mat[i, i]
        fp = mat[:, i].sum() - tp
        fn = mat[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return float(np.mean(scores))


def get_available_approaches():
    """Return list of available prediction approaches from results CSV."""
    if not os.path.exists(RESULTS_PATH):
        return ['mlp_pred']
    df = pd.read_csv(RESULTS_PATH, nrows=0)  # Just read headers
    approaches = [c for c in df.columns if c.endswith('_pred')]
    return approaches if approaches else ['mlp_pred']


def approach_display_name(approach):
    """Convert column name to display label."""
    names = {
        'face_pred': 'Face Only (DeepFace)',
        'audio_pred': 'Audio Only (Emotion2Vec)',
        'v3_pred': 'Probability Rule-Based Fusion',
        'mlp_pred': 'MLP Learned Fusion',
    }
    return names.get(approach, approach)


def generate_cpu_history(n=60):
    """Generate CPU usage history (last n seconds) — fallback when live data unavailable"""
    base = random.uniform(30, 50)
    hist = []
    for _ in range(n):
        base += random.uniform(-4, 4)
        base = max(5, min(90, base))
        hist.append(round(base, 1))
    return hist


def generate_latency():
    """Generate simulated inference latency in ms — fallback when live data unavailable"""
    return random.randint(30, 90)
