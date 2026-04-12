"""
WHY TEST THIS MODEL (MediaPipe FaceLandmarker / Landmarks + Geometry Heuristic)?

- Robot dog camera angle robustness:
  Landmark tracking tends to remain stable even when the camera is below face level (upward pitch),
  which is exactly your robot-dog viewing angle stress case.

- Elderly robustness:
  CNN emotion classifiers can mistake permanent facial wrinkles for expressions (e.g., neutral -> sad).
  Landmark/geometry features rely less on texture and more on shape change (mouth corners, lip opening),
  which can reduce false sadness on neutral faces.

- What this gives you:
  A lightweight, explainable baseline (happy/sad/neutral) + pose proxy metrics, evaluated using the
  same key-based "choose target expression" UI and saved into clear CSV outputs.



The research backs the rationale but not the ceiling. Studies on elderly FER specifically flag that tools like OpenFace 2.0 — 
which use landmark-based facial action unit detection rather than CNN texture features — 
reduce false classifications caused by wrinkles. arXiv However the geometry heuristic you have only 
outputs happy/sad/neutral, which caps its negative-class recall. It's a strong explainability story 
for your presentation, not necessarily a raw accuracy win.
"""

import os
import json
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp


# ─────────────────────────────────────────────
# Landmark indices (MediaPipe FaceMesh 468-point)
# ─────────────────────────────────────────────
L_MOUTH   = 61;  R_MOUTH   = 291
UPPER_LIP = 13;  LOWER_LIP = 14
NOSE_TIP  = 1;   CHIN      = 152
L_EYE_OUT = 33;  R_EYE_OUT = 263
# Mouth corner lift points (more precise than centre)
L_CORNER  = 61;  R_CORNER  = 291
L_UPPER_CORNER = 39; R_UPPER_CORNER = 269   # points above corners


# ─────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────
def to_xy(landmarks, idx, w, h):
    p = landmarks.landmark[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float32)


def predict_from_landmarks(landmarks, w, h):
    """
    Returns (label, score) where label ∈ {happy, sad, neutral}.
    Uses mouth-corner angle + mouth openness + droop — majority-voted
    across frames so single-frame noise doesn't dominate.
    """
    left_m  = to_xy(landmarks, L_MOUTH,   w, h)
    right_m = to_xy(landmarks, R_MOUTH,   w, h)
    upper   = to_xy(landmarks, UPPER_LIP, w, h)
    lower   = to_xy(landmarks, LOWER_LIP, w, h)
    nose    = to_xy(landmarks, NOSE_TIP,  w, h)
    leye    = to_xy(landmarks, L_EYE_OUT, w, h)
    reye    = to_xy(landmarks, R_EYE_OUT, w, h)

    eye_dist    = float(np.linalg.norm(reye - leye)) + 1e-6
    mouth_width = float(np.linalg.norm(right_m - left_m)) / eye_dist
    mouth_open  = float(np.linalg.norm(lower - upper))    / eye_dist

    # Corner angle: are corners higher or lower than mouth centre?
    mouth_mid_y    = (left_m[1] + right_m[1]) / 2.0
    corner_avg_y   = (left_m[1] + right_m[1]) / 2.0
    # Relative to nose: lower y value in image = higher on face = smile
    left_dy  = (nose[1] - left_m[1])  / eye_dist   # positive = corner above nose (smile)
    right_dy = (nose[1] - right_m[1]) / eye_dist
    corner_lift = (left_dy + right_dy) / 2.0        # >0 smile, <0 frown/neutral

    # Droop: mouth centre far below nose = droopy / neutral-to-sad
    droop = (((left_m[1] + right_m[1]) / 2.0) - nose[1]) / eye_dist

    # Scores
    happy_score = mouth_width * 0.40 + mouth_open * 0.20 + max(0.0, corner_lift) * 0.40
    sad_score   = (1.2 - mouth_width) * 0.30 + max(0.0, -corner_lift) * 0.40 + droop * 0.30

    if happy_score > 0.50:
        return "happy",   round(min(happy_score * 100, 100), 2)
    if sad_score > 0.42:
        return "sad",     round(min(sad_score   * 100, 100), 2)
    return "neutral",     round(max(0.0, (1.0 - happy_score - sad_score) * 100), 2)


def map_to_valence(label: str) -> str:
    if label in ("happy",):
        return "positive"
    if label in ("sad", "angry", "disgust", "fear"):
        return "negative"
    if label == "neutral":
        return "neutral"
    return "unknown"


# ─────────────────────────────────────────────
# GUI helpers  (identical UX to DeepFace script)
# ─────────────────────────────────────────────
def pick_video(root):
    path = filedialog.askopenfilename(
        parent=root,
        title="Select a video file — close to quit",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")],
    )
    return path or None


def pick_valence(root):
    win = tk.Toplevel(root); win.title("Ground-truth emotion")
    win.resizable(False, False); win.grab_set()
    tk.Label(win, text="What emotion is shown in this video?",
             font=("Arial", 12), pady=10).pack()
    choice = tk.StringVar(value="")
    for val in ("positive", "neutral", "negative"):
        tk.Radiobutton(win, text=val.capitalize(), variable=choice,
                       value=val, font=("Arial", 11)).pack(anchor="w", padx=30)
    result = {"value": None}
    def confirm():
        if choice.get():
            result["value"] = choice.get(); win.destroy()
        else:
            messagebox.showwarning("Select one", "Please select a valence.", parent=win)
    tk.Button(win, text="OK", command=confirm, width=10, pady=5).pack(pady=12)
    root.wait_window(win)
    return result["value"]


def pick_age_group(root):
    win = tk.Toplevel(root); win.title("Age group")
    win.resizable(False, False); win.grab_set()
    tk.Label(win, text="Age group of the person in the video?",
             font=("Arial", 12), pady=10).pack()
    choice = tk.StringVar(value="")
    for val in ("young", "old"):
        tk.Radiobutton(win, text=val.capitalize(), variable=choice,
                       value=val, font=("Arial", 11)).pack(anchor="w", padx=30)
    result = {"value": None}
    def confirm():
        if choice.get():
            result["value"] = choice.get(); win.destroy()
        else:
            messagebox.showwarning("Select one", "Please select an age group.", parent=win)
    tk.Button(win, text="OK", command=confirm, width=10, pady=5).pack(pady=12)
    root.wait_window(win)
    return result["value"]


# ─────────────────────────────────────────────
# Core: analyse one video → one prediction
# ─────────────────────────────────────────────
def analyse_video(video_path: str, face_mesh) -> dict:
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open: {video_path}"

    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    sample_count   = min(30, total_frames)
    sample_indices = set(
        int(i * total_frames / sample_count) for i in range(sample_count)
    )

    label_votes: list[str] = []
    score_sum       = 0.0
    frames_analysed = 0
    t0              = time.perf_counter()
    frame_idx       = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in sample_indices:
            try:
                h, w = frame.shape[:2]
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)

                if result.multi_face_landmarks:
                    label, score = predict_from_landmarks(
                        result.multi_face_landmarks[0], w, h
                    )
                    label_votes.append(label)
                    score_sum += score
                    frames_analysed += 1

            except Exception:
                pass

            cv2.imshow("Processing…", frame)
            cv2.waitKey(1)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    if not label_votes:
        return {
            "predicted_raw_label": "no_face",
            "predicted_score":     0.0,
            "predicted_valence":   "unknown",
            "frames_sampled":      frames_analysed,
            "latency_ms":          round(latency_ms, 2),
        }

    dominant  = Counter(label_votes).most_common(1)[0][0]
    avg_score = round(score_sum / len(label_votes), 2)

    return {
        "predicted_raw_label": dominant,
        "predicted_score":     avg_score,
        "predicted_valence":   map_to_valence(dominant),
        "frames_sampled":      frames_analysed,
        "latency_ms":          round(latency_ms, 2),
    }


# ─────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────
def make_run_dir() -> str:
    os.makedirs("results", exist_ok=True)
    n = 1
    while True:
        run_dir = os.path.join("results", f"mediapipe{n}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            return run_dir
        n += 1


def save_results(rows: list, started_at: str) -> str:
    run_dir = make_run_dir()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, "results.csv"), index=False)

    total    = len(df)
    correct  = int(df["correct"].sum())
    accuracy = round(correct / total * 100, 2) if total else float("nan")

    pd.DataFrame([{
        "model":        "MediaPipe-FaceMesh-Geometry",
        "videos_total": total,
        "correct":      correct,
        "accuracy_pct": accuracy,
        "started_at":   started_at,
        "ended_at":     datetime.now().isoformat(timespec="seconds"),
    }]).to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model":            "MediaPipe-FaceMesh-Geometry",
            "mediapipe_version": mp.__version__,
            "started_at":       started_at,
            "ended_at":         datetime.now().isoformat(timespec="seconds"),
            "videos_processed": [r["video_path"] for r in rows],
        }, f, indent=2)

    return run_dir


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
def main():
    # Use classic mp.solutions.face_mesh — no .task file needed,
    # no protobuf version conflicts
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    root = tk.Tk()
    root.withdraw()

    rows       = []
    video_num  = 0
    started_at = datetime.now().isoformat(timespec="seconds")

    print("\n=== MediaPipe FaceMesh Video Evaluator ===")
    print("Pick a video to start. Close the file dialog (or click No after a result) to quit.\n")

    while True:
        video_path = pick_video(root)
        if not video_path:
            break

        valence = pick_valence(root)
        if valence is None:
            print("Skipping — no valence selected.")
            continue

        age_group = pick_age_group(root)
        if age_group is None:
            print("Skipping — no age group selected.")
            continue

        video_num += 1
        print(f"[{video_num}] {os.path.basename(video_path)} | GT: {valence} | {age_group}")

        pred       = analyse_video(video_path, face_mesh)
        is_correct = int(pred["predicted_valence"] == valence)

        rows.append({
            "video_num":            video_num,
            "video_path":           video_path,
            "age_group":            age_group,
            "ground_truth_valence": valence,
            "predicted_raw_label":  pred["predicted_raw_label"],
            "predicted_score":      pred["predicted_score"],
            "predicted_valence":    pred["predicted_valence"],
            "correct":              is_correct,
            "frames_sampled":       pred["frames_sampled"],
            "latency_ms":           pred["latency_ms"],
            "timestamp":            datetime.now().isoformat(timespec="seconds"),
        })

        tally = sum(r["correct"] for r in rows)
        msg = (
            f"Predicted:    {pred['predicted_raw_label']} → {pred['predicted_valence']}  "
            f"(score: {pred['predicted_score']:.2f})\n"
            f"Ground truth: {valence}\n\n"
            f"{'✅ CORRECT' if is_correct else '❌ WRONG'}\n\n"
            f"Latency: {pred['latency_ms']:.0f} ms  |  Frames sampled: {pred['frames_sampled']}\n"
            f"Running tally: {tally}/{len(rows)} correct\n\n"
            f"Add another video?"
        )
        if not messagebox.askyesno("Result", msg):
            break

    face_mesh.close()

    if rows:
        run_dir     = save_results(rows, started_at)
        final_tally = sum(r["correct"] for r in rows)
        print(f"\n✅  Saved {len(rows)} video(s) to: {run_dir}")
        print(f"   results.csv  — one row per video")
        print(f"   summary.csv  — overall accuracy")
        print(f"   config.json  — session metadata")
        messagebox.showinfo(
            "Session saved",
            f"Saved {len(rows)} video(s) to:\n{run_dir}\n\n"
            f"Overall accuracy: {final_tally}/{len(rows)} "
            f"({round(final_tally/len(rows)*100)}%)"
        )
    else:
        print("No videos processed — nothing saved.")

    root.destroy()


if __name__ == "__main__":
    main()
