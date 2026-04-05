"""
Unified Evaluation: Face-only vs Audio-only vs V3 Rule-Based vs MLP Learned Fusion

Runs all 4 approaches on 3 datasets:
    1. RAVDESS test set (300 clips, 5 held-out actors) — from cached .npz
    2. Elderly AI-generated (14 clips) — from cached .npz
    3. Neurological condition clips (YouTube) — extracted fresh from video files

Outputs:
    - Per-dataset CSV with per-video predictions
    - Summary CSV with accuracy + F1 per approach per dataset
    - Confusion matrices printed to console

Usage:
    cd BT4103-Team-5-AIRA-Emotion-Recognition
    PYTHONPATH=. python analyses/evaluation/evaluate_all.py
"""

import os
import sys
import csv
import time
import numpy as np
import cv2
import soundfile as sf
import tempfile
import subprocess
from collections import defaultdict
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pipeline.core import (
    DeepFaceEmotionEmbeddingExtractor,
    Emotion2VecExtractor,
    ProbabilityFusion,
    MLPFusion,
    FusionResult,
    SHARED_EMOTIONS,
    align_face_probs,
    align_audio_probs,
)


# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
MLP_MODEL_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'pipeline', 'models', 'mlp_fusion.pt')

RAVDESS_FEATURES = os.path.join(DATA_DIR, 'ravdess_test_features.npz')
ELDERLY_FEATURES = os.path.join(DATA_DIR, 'elderly_features.npz')
NEURO_CLIPS_DIR = os.path.join(DATA_DIR, 'neuro_clips')

SAMPLE_RATE = 16000
NUM_FACE_FRAMES = 5

# Neurological clip filename format: {condition}_{person}_{gender}_{emotion}.mp4
NEURO_LABEL_MAP = {
    'happy': 'Happy',
    'sad': 'Sad',
    'neutral': 'Neutral',
    'angry': 'Angry',
    'fear': 'Fear',
    'disgust': 'Disgust',
    'surprise': 'Surprise',
}

DEEPFACE_TO_SHARED = {
    'Anger': 'Angry', 'Disgust': 'Disgust', 'Fear': 'Fear',
    'Happiness': 'Happy', 'Sadness': 'Sad', 'Surprise': 'Surprise',
    'Neutral': 'Neutral',
}


# ============================================================================
# Metrics
# ============================================================================

def compute_f1(gt_list, pred_list, emotion):
    """Compute precision, recall, F1 for one emotion class."""
    tp = sum(1 for g, p in zip(gt_list, pred_list) if g == emotion and p == emotion)
    fp = sum(1 for g, p in zip(gt_list, pred_list) if g != emotion and p == emotion)
    fn = sum(1 for g, p in zip(gt_list, pred_list) if g == emotion and p != emotion)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1


def print_confusion_matrix(gt_list, pred_list, emotions, method_name):
    """Print a confusion matrix to console."""
    print(f"\n  {method_name} Confusion Matrix (rows=true, cols=predicted):")
    print(f"  {'':>12}", end='')
    for em in emotions:
        print(f" {em[:6]:>6}", end='')
    print()
    for true_em in emotions:
        print(f"  {true_em:<12}", end='')
        for pred_em in emotions:
            c = sum(1 for g, p in zip(gt_list, pred_list)
                    if g == true_em and p == pred_em)
            print(f" {c:>6}", end='')
        print()


# ============================================================================
# Video Feature Extraction (for neurological clips)
# ============================================================================

class VideoFeatureExtractor:
    """Extract face + audio features from video files."""

    def __init__(self):
        self.face_ext = None
        self.audio_ext = None
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def load_models(self):
        print("  Loading DeepFace...")
        self.face_ext = DeepFaceEmotionEmbeddingExtractor()
        self.face_ext.load()
        print("  Loading Emotion2Vec large...")
        self.audio_ext = Emotion2VecExtractor(model_size='large')
        self.audio_ext.load()

    def extract_from_video(self, video_path):
        """Extract aligned face + audio probability vectors from one video."""
        # Audio
        audio_probs = None
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        try:
            subprocess.run(
                ['ffmpeg', '-i', video_path, '-ac', '1', '-ar', str(SAMPLE_RATE),
                 '-vn', '-y', temp_path],
                capture_output=True, timeout=30)
            audio, _ = sf.read(temp_path)
            audio = audio.astype(np.float32)
            if len(audio) > SAMPLE_RATE:
                result = self.audio_ext.extract(audio, SAMPLE_RATE)
                audio_probs = result.get('emotion_probs')
        except Exception as e:
            print(f"    Audio failed: {e}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        # Face (sample N frames)
        face_probs = None
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            start = int(total_frames * 0.1)
            end = int(total_frames * 0.9)
            if end <= start:
                start, end = 0, total_frames
            indices = np.linspace(start, end - 1, NUM_FACE_FRAMES, dtype=int)

            all_probs = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                if len(faces) == 0:
                    continue
                x, y, w, h = faces[0]
                m = int(0.1 * w)
                crop = frame[max(0, y-m):min(frame.shape[0], y+h+m),
                             max(0, x-m):min(frame.shape[1], x+w+m)]
                try:
                    analysis = self.face_ext._deepface.analyze(
                        crop, actions=['emotion'],
                        detector_backend='skip',
                        enforce_detection=False, silent=True)
                    raw = analysis[0]['emotion']
                    probs = {
                        'Anger': raw.get('angry', 0) / 100,
                        'Disgust': raw.get('disgust', 0) / 100,
                        'Fear': raw.get('fear', 0) / 100,
                        'Happiness': raw.get('happy', 0) / 100,
                        'Sadness': raw.get('sad', 0) / 100,
                        'Surprise': raw.get('surprise', 0) / 100,
                        'Neutral': raw.get('neutral', 0) / 100,
                    }
                    all_probs.append(probs)
                except Exception:
                    continue
        cap.release()

        if all_probs:
            face_probs = {em: np.mean([p[em] for p in all_probs])
                          for em in all_probs[0]}

        # Align to shared 7-class space
        face_aligned = align_face_probs(face_probs) if face_probs else {em: 1/7 for em in SHARED_EMOTIONS}
        audio_aligned = align_audio_probs(audio_probs) if audio_probs else {em: 1/7 for em in SHARED_EMOTIONS}

        face_vec = np.array([face_aligned[em] for em in SHARED_EMOTIONS], dtype=np.float32)
        audio_vec = np.array([audio_aligned[em] for em in SHARED_EMOTIONS], dtype=np.float32)

        return face_vec, audio_vec


# ============================================================================
# Evaluation Engine
# ============================================================================

def run_predictions(face_vecs, audio_vecs, shared, v3, mlp):
    """Run all 4 approaches on feature vectors."""
    n = len(face_vecs)
    preds = {'face': [], 'audio': [], 'v3': [], 'mlp': []}

    for i in range(n):
        fv, av = face_vecs[i], audio_vecs[i]
        fi, ai = int(np.argmax(fv)), int(np.argmax(av))
        preds['face'].append(shared[fi])
        preds['audio'].append(shared[ai])

        fp = {em: float(fv[j]) for j, em in enumerate(shared)}
        ap = {em: float(av[j]) for j, em in enumerate(shared)}
        fr = {'top_emotion': shared[fi], 'confidence': float(fv[fi]), 'emotion_probs': fp}
        ar = {'top_emotion': shared[ai], 'confidence': float(av[ai]), 'emotion_probs': ap}

        preds['v3'].append(v3.fuse(fr, ar).emotion)
        preds['mlp'].append(mlp.fuse(fr, ar).emotion)

    return preds


def evaluate_dataset(name, gt_names, preds, shared, results_path=None):
    """Evaluate and print results for one dataset."""
    n = len(gt_names)
    methods = ['face', 'audio', 'v3', 'mlp']
    emotions = sorted(set(gt_names))

    print(f"\n{'=' * 75}")
    print(f"  {name} ({n} samples)")
    print(f"{'=' * 75}")

    # Accuracy
    print(f"\n  ACCURACY")
    print(f"  {'Emotion':<12} {'Count':>6} {'Face':>8} {'Audio':>8} {'V3':>8} {'MLP':>8}")
    print(f"  {'-' * 53}")

    totals = {k: 0 for k in methods}
    total_n = 0
    for em in emotions:
        idx = [i for i in range(n) if gt_names[i] == em]
        cnt = len(idx)
        total_n += cnt
        row = f"  {em:<12} {cnt:>6}"
        for method in methods:
            c = sum(1 for i in idx if preds[method][i] == em)
            totals[method] += c
            row += f" {c / cnt:>7.0%}"
        print(row)

    print(f"  {'-' * 53}")
    row = f"  {'Overall':<12} {total_n:>6}"
    for method in methods:
        row += f" {totals[method] / total_n:>7.0%}"
    print(row)

    # F1 Scores
    print(f"\n  F1 SCORES")
    print(f"  {'Emotion':<12} {'Face':>8} {'Audio':>8} {'V3':>8} {'MLP':>8}")
    print(f"  {'-' * 45}")

    macro_f1 = {m: [] for m in methods}
    for em in emotions:
        row = f"  {em:<12}"
        for method in methods:
            _, _, f1 = compute_f1(gt_names, preds[method], em)
            macro_f1[method].append(f1)
            row += f" {f1:>7.2f}"
        print(row)

    print(f"  {'-' * 45}")
    row = f"  {'Macro Avg':<12}"
    for method in methods:
        row += f" {np.mean(macro_f1[method]):>7.2f}"
    print(row)

    # Confusion matrices for V3 and MLP
    for method, label in [('v3', 'V3 Rule-Based'), ('mlp', 'MLP Learned')]:
        print_confusion_matrix(gt_names, preds[method], shared, label)

    # Key metrics
    print(f"\n  KEY METRICS")
    sad_idx = [i for i in range(n) if gt_names[i] == 'Sad']
    if sad_idx:
        sad_face_wrong = [i for i in sad_idx if preds['face'][i] != 'Sad']
        if sad_face_wrong:
            for method in ['v3', 'mlp']:
                rescued = sum(1 for i in sad_face_wrong if preds[method][i] == 'Sad')
                print(f"    Sad rescue ({method.upper()}): {rescued}/{len(sad_face_wrong)}")

    happy_idx = [i for i in range(n) if gt_names[i] == 'Happy']
    if happy_idx:
        happy_face_right = [i for i in happy_idx if preds['face'][i] == 'Happy']
        if happy_face_right:
            for method in ['v3', 'mlp']:
                broken = sum(1 for i in happy_face_right if preds[method][i] != 'Happy')
                print(f"    Happy preserved ({method.upper()}): {len(happy_face_right) - broken}/{len(happy_face_right)}")

    # Save per-video CSV
    if results_path:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ground_truth', 'face_pred', 'audio_pred', 'v3_pred', 'mlp_pred'])
            for i in range(n):
                writer.writerow([gt_names[i], preds['face'][i], preds['audio'][i],
                                 preds['v3'][i], preds['mlp'][i]])
        print(f"\n  Results saved to: {results_path}")

    # Return summary for aggregation
    return {
        'dataset': name,
        'n_samples': n,
        'face_acc': totals['face'] / total_n,
        'audio_acc': totals['audio'] / total_n,
        'v3_acc': totals['v3'] / total_n,
        'mlp_acc': totals['mlp'] / total_n,
        'face_f1': np.mean(macro_f1['face']),
        'audio_f1': np.mean(macro_f1['audio']),
        'v3_f1': np.mean(macro_f1['v3']),
        'mlp_f1': np.mean(macro_f1['mlp']),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 75)
    print("  UNIFIED EVALUATION REPORT")
    print("  Face-only | Audio-only | V3 Rule-Based | MLP Learned")
    print("=" * 75)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load fusion engines
    mlp = MLPFusion(model_path=MLP_MODEL_PATH)
    if not mlp.loaded:
        print(f"WARNING: MLP model not found at {MLP_MODEL_PATH}")
    v3 = ProbabilityFusion()
    shared = list(SHARED_EMOTIONS)

    summaries = []

    # --- Dataset 1: RAVDESS ---
    if os.path.exists(RAVDESS_FEATURES):
        print("\n  Loading RAVDESS test features...")
        data = dict(np.load(RAVDESS_FEATURES, allow_pickle=True))
        gt_names = [str(ln) for ln in data['label_names']]
        preds = run_predictions(data['face_vecs'], data['audio_vecs'], shared, v3, mlp)
        summary = evaluate_dataset(
            "RAVDESS Test Set (300 clips, 5 held-out actors)",
            gt_names, preds, shared,
            results_path=os.path.join(RESULTS_DIR, 'ravdess_results.csv'))
        summaries.append(summary)
    else:
        print(f"\n  SKIPPED: RAVDESS — {RAVDESS_FEATURES} not found")

    # --- Dataset 2: Elderly ---
    if os.path.exists(ELDERLY_FEATURES):
        print("\n  Loading elderly features...")
        data = dict(np.load(ELDERLY_FEATURES, allow_pickle=True))
        gt_names = [str(ln) for ln in data['label_names']]
        preds = run_predictions(data['face_vecs'], data['audio_vecs'], shared, v3, mlp)
        summary = evaluate_dataset(
            "Elderly AI-Generated (14 clips, out-of-domain)",
            gt_names, preds, shared,
            results_path=os.path.join(RESULTS_DIR, 'elderly_results.csv'))
        summaries.append(summary)
    else:
        print(f"\n  SKIPPED: Elderly — {ELDERLY_FEATURES} not found")

    # --- Dataset 3: Neurological ---
    if os.path.exists(NEURO_CLIPS_DIR) and os.listdir(NEURO_CLIPS_DIR):
        print("\n  Processing neurological condition clips...")
        extractor = VideoFeatureExtractor()
        extractor.load_models()

        clips = []
        for fname in sorted(os.listdir(NEURO_CLIPS_DIR)):
            if not (fname.endswith('.mp4') or fname.endswith('.mov')):
                continue
            parts = fname.rsplit('.', 1)[0].split('_')
            if len(parts) < 4:
                print(f"    SKIP: {fname} (can't parse filename)")
                continue
            emotion_raw = parts[-1]
            gt = NEURO_LABEL_MAP.get(emotion_raw)
            if gt is None:
                print(f"    SKIP: {fname} (unknown emotion: {emotion_raw})")
                continue
            clips.append({
                'filepath': os.path.join(NEURO_CLIPS_DIR, fname),
                'filename': fname,
                'condition': parts[0],
                'person': parts[1],
                'emotion': gt,
            })

        print(f"  Found {len(clips)} valid clips")

        face_vecs, audio_vecs, gt_names, filenames = [], [], [], []
        for i, clip in enumerate(clips):
            print(f"    [{i+1}/{len(clips)}] {clip['filename']}")
            fv, av = extractor.extract_from_video(clip['filepath'])
            face_vecs.append(fv)
            audio_vecs.append(av)
            gt_names.append(clip['emotion'])
            filenames.append(clip['filename'])

        face_vecs = np.array(face_vecs)
        audio_vecs = np.array(audio_vecs)

        preds = run_predictions(face_vecs, audio_vecs, shared, v3, mlp)

        # Print per-clip detail for neurological (small dataset)
        print(f"\n  PER-CLIP DETAIL:")
        print(f"  {'Filename':<45} {'GT':<10} {'Face':<10} {'Audio':<10} {'V3':<10} {'MLP':<10}")
        print(f"  {'-' * 95}")
        for i in range(len(gt_names)):
            fc = 'Y' if preds['face'][i] == gt_names[i] else ' '
            ac = 'Y' if preds['audio'][i] == gt_names[i] else ' '
            vc = 'Y' if preds['v3'][i] == gt_names[i] else ' '
            mc = 'Y' if preds['mlp'][i] == gt_names[i] else ' '
            print(f"  {filenames[i]:<45} {gt_names[i]:<10} {preds['face'][i]:<8}{fc} "
                  f"{preds['audio'][i]:<8}{ac} {preds['v3'][i]:<8}{vc} {preds['mlp'][i]:<8}{mc}")

        summary = evaluate_dataset(
            "Neurological Conditions (YouTube clips)",
            gt_names, preds, shared,
            results_path=os.path.join(RESULTS_DIR, 'neuro_results.csv'))
        summaries.append(summary)
    else:
        print(f"\n  SKIPPED: Neurological — {NEURO_CLIPS_DIR} not found or empty")

    # --- Summary CSV ---
    if summaries:
        summary_path = os.path.join(RESULTS_DIR, 'summary.csv')
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
            writer.writeheader()
            writer.writerows(summaries)

        print(f"\n{'=' * 75}")
        print("  SUMMARY")
        print(f"{'=' * 75}")
        print(f"\n  {'Dataset':<45} {'Face':>8} {'Audio':>8} {'V3':>8} {'MLP':>8}")
        print(f"  {'-' * 80}")
        for s in summaries:
            print(f"  {s['dataset']:<45} {s['face_acc']:>7.0%} {s['audio_acc']:>7.0%} "
                  f"{s['v3_acc']:>7.0%} {s['mlp_acc']:>7.0%}")
        print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
