"""
RetinaFace + DeepFace Emotion Evaluator (No-Tkinter CLI Version)

WHY TEST THIS MODEL:
- RetinaFace is a stronger face detector than DeepFace's default (opencv).
- It may handle pose changes, partial occlusion, and older/wrinkled faces better.
- This version avoids tkinter completely and uses filename-based labels.

FILENAME CONVENTION:
  - Young person:  name_emotion.MOV        e.g. anushka_negative.MOV
  - Old person:    name_old_emotion.mp4    e.g. grandma_old_positive.mp4

Supported valence labels in filename:
  - positive
  - negative
  - neutral

OUTPUT:
  results/retinaface{N}/
    - results.csv
    - confusion_matrix.csv
    - per_class_metrics.csv
    - per_class_by_age.csv
    - summary.csv
    - config.json
"""

''' 
Video file
   ↓
OpenCV (cv2)
   • open video
   • read frames
   • show frames
   ↓
RetinaFace
   • detect face location
   • return bounding box
   ↓
Crop face
   ↓
DeepFace
   • run CNN emotion classifier
   ↓
Aggregate frame scores
   ↓
Final valence prediction
'''

import os
import json
import time
import math
from datetime import datetime

import cv2
import pandas as pd
from retinaface import RetinaFace
from deepface import DeepFace


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
ENFORCE_DETECTION = False

POSITIVE_EMOTIONS = {"happy", "surprise"}
NEGATIVE_EMOTIONS = {"angry", "disgust", "fear", "sad"}
NEUTRAL_EMOTIONS = {"neutral"}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".MOV"}


def map_to_valence(label: str) -> str:
    label = label.lower()
    if label in POSITIVE_EMOTIONS:
        return "positive"
    if label in NEGATIVE_EMOTIONS:
        return "negative"
    if label in NEUTRAL_EMOTIONS:
        return "neutral"
    return "unknown"


# ─────────────────────────────────────────────
# Filename parser
# ─────────────────────────────────────────────
def parse_filename(filepath: str):
    stem = os.path.splitext(os.path.basename(filepath))[0].lower()
    parts = stem.split("_")
    valence = None
    for part in reversed(parts):
        if part in ("positive", "negative", "neutral"):
            valence = part
            break
    age_group = "old" if "old" in parts else "young"
    return valence, age_group


# ─────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────
def get_video_paths_from_folder(folder: str):
    paths = []
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if os.path.isfile(full) and os.path.splitext(name)[1] in VIDEO_EXTS:
            paths.append(full)
    return paths


# ─────────────────────────────────────────────
# Core: analyse one video → one prediction
# RetinaFace detects face → crops it → DeepFace runs emotion on crop
# 30 evenly-spaced frames, scores summed across frames
# ─────────────────────────────────────────────
def analyse_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open: {video_path}"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    sample_count = min(30, total_frames)
    sample_indices = set(
        int(i * total_frames / sample_count) for i in range(sample_count)
    )

    valence_sums = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    frames_analysed = 0
    frames_failed = 0
    t0 = time.perf_counter()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in sample_indices:
            try:
                detections = RetinaFace.detect_faces(frame)

                if isinstance(detections, dict) and len(detections) > 0:
                    first_key = list(detections.keys())[0]
                    face = detections[first_key]
                    x1, y1, x2, y2 = face["facial_area"]

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1] - 1, x2)
                    y2 = min(frame.shape[0] - 1, y2)

                    crop = frame[y1:y2, x1:x2]

                    if crop.size == 0:
                        frames_failed += 1
                        frame_idx += 1
                        continue

                    result = DeepFace.analyze(
                        crop,
                        actions=["emotion"],
                        enforce_detection=ENFORCE_DETECTION,
                        silent=True,
                    )
                    if isinstance(result, list):
                        result = result[0]

                    raw_scores = result.get("emotion", {}) or {}

                    frame_valence = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
                    for emotion, score in raw_scores.items():
                        v = map_to_valence(emotion)
                        if v in frame_valence:
                            frame_valence[v] += float(score)

                    frame_total = sum(frame_valence.values()) or 1.0
                    for v in frame_valence:
                        frame_valence[v] = frame_valence[v] / frame_total * 100.0

                    for v in valence_sums:
                        valence_sums[v] += frame_valence[v]

                    frames_analysed += 1
                else:
                    frames_failed += 1

            except Exception:
                frames_failed += 1

            cv2.imshow("Processing...", frame)
            cv2.waitKey(1)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    if frames_analysed == 0:
        return {
            "predicted_valence": "unknown",
            "score_positive": 0.0,
            "score_neutral": 0.0,
            "score_negative": 0.0,
            "predicted_score": 0.0,
            "frames_total": total_frames,
            "frames_analysed": 0,
            "frames_failed": frames_failed,
            "latency_ms": round(latency_ms, 2),
        }

    dominant = max(valence_sums, key=valence_sums.get)

    score_pos = round(valence_sums["positive"] / frames_analysed, 4)
    score_neu = round(valence_sums["neutral"] / frames_analysed, 4)
    score_neg = round(valence_sums["negative"] / frames_analysed, 4)

    return {
        "predicted_valence": dominant,
        "score_positive": score_pos,
        "score_neutral": score_neu,
        "score_negative": score_neg,
        "predicted_score": round(valence_sums[dominant] / frames_analysed, 4),
        "frames_total": total_frames,
        "frames_analysed": frames_analysed,
        "frames_failed": frames_failed,
        "latency_ms": round(latency_ms, 2),
    }


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def compute_per_class_metrics(df, classes=("positive", "neutral", "negative")):
    rows = []
    for cls in classes:
        tp = ((df["ground_truth_valence"] == cls) & (df["predicted_valence"] == cls)).sum()
        fp = ((df["ground_truth_valence"] != cls) & (df["predicted_valence"] == cls)).sum()
        fn = ((df["ground_truth_valence"] == cls) & (df["predicted_valence"] != cls)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if not (math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0)
            else float("nan")
        )
        rows.append({
            "class": cls,
            "precision": round(precision, 4) if not math.isnan(precision) else float("nan"),
            "recall": round(recall, 4) if not math.isnan(recall) else float("nan"),
            "f1": round(f1, 4) if not math.isnan(f1) else float("nan"),
            "support": int((df["ground_truth_valence"] == cls).sum()),
        })

    p_vals = [r["precision"] for r in rows if not math.isnan(r["precision"])]
    r_vals = [r["recall"] for r in rows if not math.isnan(r["recall"])]
    f_vals = [r["f1"] for r in rows if not math.isnan(r["f1"])]

    rows.append({
        "class": "macro_avg",
        "precision": round(sum(p_vals) / len(p_vals), 4) if p_vals else float("nan"),
        "recall": round(sum(r_vals) / len(r_vals), 4) if r_vals else float("nan"),
        "f1": round(sum(f_vals) / len(f_vals), 4) if f_vals else float("nan"),
        "support": len(df),
    })
    return pd.DataFrame(rows)


def compute_auc_onevsrest(df, cls):
    score_col = f"score_{cls}"
    pos = df[df["ground_truth_valence"] == cls][score_col].values
    neg = df[df["ground_truth_valence"] != cls][score_col].values
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    correct = sum((neg < p).sum() for p in pos)
    ties = sum((neg == p).sum() for p in pos)
    return round((correct + 0.5 * ties) / (len(pos) * len(neg)), 4)


def compute_confusion_matrix(df, classes=("positive", "neutral", "negative")):
    rows = []
    for actual in classes:
        row = {"actual \\ predicted": actual}
        for pred in classes:
            row[pred] = int(((df["ground_truth_valence"] == actual) &
                             (df["predicted_valence"] == pred)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def compute_cohen_kappa(df):
    classes = ["positive", "neutral", "negative"]
    n = len(df)
    if n == 0:
        return float("nan")
    observed = sum(
        ((df["ground_truth_valence"] == c) & (df["predicted_valence"] == c)).sum()
        for c in classes
    ) / n
    expected = sum(
        (df["ground_truth_valence"] == c).mean() * (df["predicted_valence"] == c).mean()
        for c in classes
    )
    kappa = (observed - expected) / (1 - expected) if (1 - expected) != 0 else float("nan")
    return round(float(kappa), 4)


# ─────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────
def make_run_dir() -> str:
    os.makedirs("results", exist_ok=True)
    n = 1
    while True:
        run_dir = os.path.join("results", f"retinaface{n}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            return run_dir
        n += 1


def save_results(rows: list, started_at: str) -> str:
    run_dir = make_run_dir()
    df = pd.DataFrame(rows)

    df.to_csv(os.path.join(run_dir, "results.csv"), index=False)

    classes = ("positive", "neutral", "negative")
    valid = df[df["predicted_valence"] != "unknown"].copy()

    compute_confusion_matrix(valid, classes).to_csv(
        os.path.join(run_dir, "confusion_matrix.csv"), index=False
    )

    pcm = compute_per_class_metrics(valid, classes)
    pcm.to_csv(os.path.join(run_dir, "per_class_metrics.csv"), index=False)

    age_rows = []
    for age in ("young", "old"):
        sub = valid[valid["age_group"] == age]
        if len(sub) == 0:
            continue
        for cls in classes:
            tp = ((sub["ground_truth_valence"] == cls) & (sub["predicted_valence"] == cls)).sum()
            fp = ((sub["ground_truth_valence"] != cls) & (sub["predicted_valence"] == cls)).sum()
            fn = ((sub["ground_truth_valence"] == cls) & (sub["predicted_valence"] != cls)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            f1 = (
                2 * prec * rec / (prec + rec)
                if not (math.isnan(prec) or math.isnan(rec) or (prec + rec) == 0)
                else float("nan")
            )
            age_rows.append({
                "class": cls,
                "age_group": age,
                "precision": round(prec, 4) if not math.isnan(prec) else float("nan"),
                "recall": round(rec, 4) if not math.isnan(rec) else float("nan"),
                "f1": round(f1, 4) if not math.isnan(f1) else float("nan"),
                "support": int((sub["ground_truth_valence"] == cls).sum()),
            })
    pd.DataFrame(age_rows).to_csv(
        os.path.join(run_dir, "per_class_by_age.csv"), index=False
    )

    total = len(valid)
    correct = int((valid["ground_truth_valence"] == valid["predicted_valence"]).sum())
    accuracy = round(correct / total * 100, 2) if total else float("nan")
    macro_row = pcm[pcm["class"] == "macro_avg"].iloc[0]
    kappa = compute_cohen_kappa(valid)
    auc_pos = compute_auc_onevsrest(valid, "positive")
    auc_neu = compute_auc_onevsrest(valid, "neutral")
    auc_neg = compute_auc_onevsrest(valid, "negative")
    auc_vals = [x for x in [auc_pos, auc_neu, auc_neg] if not math.isnan(x)]
    macro_auc = round(sum(auc_vals) / len(auc_vals), 4) if auc_vals else float("nan")
    f1_by_cls = {r["class"]: r["f1"] for _, r in pcm[pcm["class"] != "macro_avg"].iterrows()}
    avg_lat = round(df["latency_ms"].mean(), 2)
    p50_lat = round(float(df["latency_ms"].quantile(0.50)), 2)
    p95_lat = round(float(df["latency_ms"].quantile(0.95)), 2)

    summary_rows = [
        {"metric": "model", "value": "RetinaFace+DeepFace",
         "what_it_means": "RetinaFace for detection/cropping, DeepFace CNN for emotion classification"},
        {"metric": "videos_total", "value": total,
         "what_it_means": "Total number of videos processed in this run"},
        {"metric": "accuracy_pct", "value": accuracy,
         "what_it_means": "% of videos where predicted valence matched ground truth. Baseline (always neutral) = 33.3%."},
        {"metric": "macro_f1", "value": macro_row["f1"],
         "what_it_means": "Average F1 across all 3 classes equally. Balances precision and recall."},
        {"metric": "cohen_kappa", "value": kappa,
         "what_it_means": "Agreement beyond chance. 0=random, 1=perfect. Accounts for prediction imbalance."},
        {"metric": "macro_precision", "value": macro_row["precision"],
         "what_it_means": "Average precision across classes. Of all videos predicted as class X, how many actually were X?"},
        {"metric": "macro_recall", "value": macro_row["recall"],
         "what_it_means": "Average recall across classes. Of all actual class X videos, how many did the model catch?"},
        {"metric": "f1_positive", "value": f1_by_cls.get("positive", float("nan")),
         "what_it_means": "F1 for POSITIVE class. Low = model struggles to detect happy faces."},
        {"metric": "f1_neutral", "value": f1_by_cls.get("neutral", float("nan")),
         "what_it_means": "F1 for NEUTRAL class. Low = model confuses resting faces with emotions."},
        {"metric": "f1_negative", "value": f1_by_cls.get("negative", float("nan")),
         "what_it_means": "F1 for NEGATIVE class. Low = model misses or over-predicts upset faces."},
        {"metric": "auc_positive", "value": auc_pos,
         "what_it_means": "AUC for positive vs rest. How well model ranks actual positive videos higher using score_positive. 0.5=random, 1.0=perfect."},
        {"metric": "auc_neutral", "value": auc_neu,
         "what_it_means": "AUC for neutral vs rest using score_neutral."},
        {"metric": "auc_negative", "value": auc_neg,
         "what_it_means": "AUC for negative vs rest using score_negative."},
        {"metric": "macro_auc", "value": macro_auc,
         "what_it_means": "Average AUC across all 3 classes. Single headline confidence-ranking score. Above 0.7=acceptable, above 0.8=good."},
        {"metric": "latency_avg_ms", "value": avg_lat,
         "what_it_means": "Average processing time per video in ms. RetinaFace adds overhead vs plain DeepFace — compare this directly."},
        {"metric": "latency_p50_ms", "value": p50_lat,
         "what_it_means": "Median latency — half of videos processed faster than this."},
        {"metric": "latency_p95_ms", "value": p95_lat,
         "what_it_means": "95th percentile latency — worst-case speed for 95% of videos."},
        {"metric": "started_at", "value": started_at,
         "what_it_means": "Session start time"},
        {"metric": "ended_at", "value": datetime.now().isoformat(timespec="seconds"),
         "what_it_means": "Session end time"},
    ]
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(run_dir, "summary.csv"), index=False
    )

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": "RetinaFace+DeepFace",
            "enforce_detection": ENFORCE_DETECTION,
            "frames_sampled": "min(30, total_frames) evenly spaced",
            "prediction_method": "RetinaFace crop → DeepFace emotion → sum valence scores across frames",
            "auc_scores": "normalised sum / frames_analysed per class",
            "started_at": started_at,
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "videos_processed": [r["video_path"] for r in rows],
        }, f, indent=2)

    return run_dir


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("\n=== RetinaFace + DeepFace Video Evaluator (CLI) ===")

    folder = input("Enter the full path to the folder containing your videos: ").strip()
    if not folder:
        print("No folder entered — exiting.")
        return

    if not os.path.isdir(folder):
        print("That path is not a valid folder.")
        return

    started_at = datetime.now().isoformat(timespec="seconds")

    video_paths = get_video_paths_from_folder(folder)
    if not video_paths:
        print("No video files found in that folder.")
        return

    parsed = []
    skipped = []
    for path in video_paths:
        valence, age_group = parse_filename(path)
        if valence is None:
            skipped.append(os.path.basename(path))
        else:
            parsed.append((path, valence, age_group))

    if skipped:
        print("\nSkipping files with unparseable filenames:")
        for s in skipped:
            print(" -", s)

    if not parsed:
        print("No valid video filenames found.")
        return

    print(f"\nFound {len(parsed)} valid video(s).")
    print("First 10:")
    for p, v, a in parsed[:10]:
        print(f" - {os.path.basename(p)}  →  {v} | {a}")
    if len(parsed) > 10:
        print(f"... and {len(parsed)-10} more")

    confirm = input("\nStart processing? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    rows = []
    for i, (video_path, valence, age_group) in enumerate(parsed, 1):
        print(f"\n[{i}/{len(parsed)}] {os.path.basename(video_path)} | GT: {valence} | {age_group}")
        pred = analyse_video(video_path)
        is_correct = int(pred["predicted_valence"] == valence)

        rows.append({
            "video_num": i,
            "video_path": video_path,
            "age_group": age_group,
            "ground_truth_valence": valence,
            "predicted_valence": pred["predicted_valence"],
            "predicted_score": pred["predicted_score"],
            "score_positive": pred["score_positive"],
            "score_neutral": pred["score_neutral"],
            "score_negative": pred["score_negative"],
            "correct": is_correct,
            "frames_total": pred["frames_total"],
            "frames_analysed": pred["frames_analysed"],
            "frames_failed": pred["frames_failed"],
            "latency_ms": pred["latency_ms"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

        status = "✅" if is_correct else "❌"
        print(
            f"   {status} Predicted: {pred['predicted_valence']} "
            f"(+:{pred['score_positive']:.1f} n:{pred['score_neutral']:.1f} -:{pred['score_negative']:.1f})"
        )

    run_dir = save_results(rows, started_at)
    final_tally = sum(r["correct"] for r in rows)
    accuracy = round(final_tally / len(rows) * 100)

    print(f"\n✅ Saved {len(rows)} video(s) to: {run_dir}")
    print("   results.csv           — one row per video")
    print("   confusion_matrix.csv  — 3x3 actual vs predicted")
    print("   per_class_metrics.csv — precision/recall/f1 per emotion")
    print("   per_class_by_age.csv  — same split by age group")
    print("   summary.csv           — all stats with explanations")
    print("   config.json           — session metadata")
    print(f"\nOverall accuracy: {final_tally}/{len(rows)} ({accuracy}%)")


if __name__ == "__main__":
    main()