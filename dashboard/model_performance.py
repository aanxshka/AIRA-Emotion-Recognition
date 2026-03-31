"""
Model Performance Module
Integrated from frame1/2 branch (commits 370f9a8, 69eba1a)
Contains ML model metrics generation and visualization helpers
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Emotion classes
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
N = len(EMOTIONS)


def fig_to_b64(fig):
    """Convert matplotlib figure to base64 PNG for embedding in HTML"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_confusion_matrix():
    """Generate realistic confusion matrix for emotion classification"""
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
    return mat.diagonal().sum() / mat.sum()


def f1_from_matrix(mat):
    """Calculate macro F1 score from confusion matrix"""
    scores = []
    for i in range(N):
        tp = mat[i, i]
        fp = mat[:, i].sum() - tp
        fn = mat[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return float(np.mean(scores))


def generate_cpu_history(n=60):
    """Generate CPU usage history (last n seconds)"""
    base = random.uniform(30, 50)
    hist = []
    for _ in range(n):
        base += random.uniform(-4, 4)
        base = max(5, min(90, base))
        hist.append(round(base, 1))
    return hist


def generate_latency():
    """Generate simulated inference latency in ms"""
    return random.randint(30, 90)
