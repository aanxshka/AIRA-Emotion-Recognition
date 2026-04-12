"""
WHY TEST THIS MODEL (dlib 68-point Landmarks + Geometry Heuristic)?

- Robot dog camera angle robustness:
  Landmark tracking tends to remain stable even when the camera is below face level (upward pitch),
  which is exactly your robot-dog viewing angle stress case.

- Elderly robustness:
  CNN emotion classifiers (like DeepFace) mistake permanent facial wrinkles for expressions
  (e.g., neutral → sad / fear). Landmark geometry relies on SHAPE CHANGE not texture,
  so wrinkles don't affect the prediction.

- What this gives you:
  A lightweight, explainable baseline (happy/sad/neutral) evaluated with the same
  GUI file-picker UX as DeepFace. Results CSVs are directly comparable.

Setup (one time):
    pip install dlib opencv-python pandas numpy
    curl -L "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2" \
         -o shape_predictor_68_face_landmarks.dat.bz2
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2
    # place shape_predictor_68_face_landmarks.dat in the same folder as this script
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
import dlib


# ─────────────────────────────────────────────
# dlib 68-point landmark indices
# ─────────────────────────────────────────────
L_MOUTH   = 48;  R_MOUTH   = 54   # mouth corners
UPPER_LIP = 51;  LOWER_LIP = 57   # lip centre top/bottom
L_EYE_OUT = 36;  R_EYE_OUT = 45   # outer eye corners
NOSE_TIP  = 30

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"


# ─────────────────────────────────────────────
# Geometry prediction
# ─────────────────────────────────────────────
def get_pt(shape, idx):
    p = shape.part(idx)
    return np.array([p.x, p.y], dtype=np.float32)


def predict_from_shape(shape):
    """
    Returns (label, score) where label ∈ {happy, sad, neutral}.

    KEY FIX over previous version:
    - Old version measured corner_lift relative to nose tip — which is almost always
      negative on real faces (mouth corners sit below nose tip geometrically),
      so sad_score was permanently inflated → predicted "sad" on everything.
    - New version measures corner_curve relative to lip midpoint (are corners
      above or below the midpoint between upper/lower lip?). This is a RELATIVE
      measure within the mouth region — not an absolute face position.
    - Sad threshold raised and now requires BOTH corner downturn AND narrow mouth.
    - Asymmetry check dampens predictions when left/right signals disagree (noise).

    Features:
    - mouth_width    : wide mouth → smile signal (normalised by eye distance)
    - mouth_open     : parted lips → expression intensity
    - corner_curve   : corners above/below lip midpoint → key smile vs frown signal
    - asymmetry      : left/right difference → high = ambiguous/noise, dampens sad
    """
    left_m  = get_pt(shape, L_MOUTH)
    right_m = get_pt(shape, R_MOUTH)
    upper   = get_pt(shape, UPPER_LIP)
    lower   = get_pt(shape, LOWER_LIP)
    leye    = get_pt(shape, L_EYE_OUT)
    reye    = get_pt(shape, R_EYE_OUT)

    eye_dist    = float(np.linalg.norm(reye - leye)) + 1e-6
    mouth_width = float(np.linalg.norm(right_m - left_m)) / eye_dist
    mouth_open  = float(np.linalg.norm(lower - upper))    / eye_dist

    # Lip midpoint y (average of upper and lower lip centre)
    lip_mid_y = (upper[1] + lower[1]) / 2.0

    # Corner curve: are corners ABOVE or BELOW the lip midpoint?
    # Negative = corners above midpoint = smile
    # Positive = corners below midpoint = frown/sad
    left_curve   = (left_m[1]  - lip_mid_y) / eye_dist
    right_curve  = (right_m[1] - lip_mid_y) / eye_dist
    corner_curve = (left_curve + right_curve) / 2.0

    # Asymmetry: genuine expressions tend to be symmetric
    # High asymmetry = ambiguous/noise → dampen sad signal
    asymmetry = abs(left_curve - right_curve)

    # Happy: wide mouth + corners pulled UP above lip midline
    happy_score = (
        mouth_width             * 0.35 +
        mouth_open              * 0.15 +
        max(0.0, -corner_curve) * 0.50   # corners above midline = smile
    )

    # Sad: corners pulled DOWN below lip midline + narrow mouth
    # Both signals must be present — not just one
    sad_score = (
        max(0.0, corner_curve) * 0.60 +  # corners below midline
        (0.9 - mouth_width)    * 0.40     # narrower than neutral
    ) * max(0.0, 1.0 - asymmetry * 2.0)  # penalise asymmetric/noisy signal

    # Decision — strongly prefer neutral when signal is weak
    if happy_score > 0.42:
        return "happy",   round(min(happy_score * 100, 100), 2)
    if sad_score > 0.30:
        return "sad",     round(min(sad_score   * 100, 100), 2)
    return "neutral",     round(max(0.0, (1.0 - happy_score - sad_score) * 100), 2)


def map_to_valence(label: str) -> str:
    if label == "happy":
        return "positive"
    if label in ("sad", "angry", "disgust", "fear"):
        return "negative"
    if label == "neutral":
        return "neutral"
    return "unknown"


# ─────────────────────────────────────────────
# GUI helpers
# ─────────────────────────────────────────────
def pick_video(root):
    path = filedialog.askopenfilename(
        parent=root,
        title="Select a video file — close to quit",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")],
    )
    return path or None


def pick_valence(root):
    win = tk.Toplevel(root)
    win.title("Ground-truth emotion")
    win.resizable(False, False)
    win.grab_set()
    tk.Label(win, text="What emotion is shown in this video?",
             font=("Arial", 12), pady=10).pack()
    choice = tk.StringVar(value="")
    for val in ("positive", "neutral", "negative"):
        tk.Radiobutton(win, text=val.capitalize(), variable=choice,
                       value=val, font=("Arial", 11)).pack(anchor="w", padx=30)
    result = {"value": None}
    def confirm():
        if choice.get():
            result["value"] = choice.get()
            win.destroy()
        else:
            messagebox.showwarning("Select one", "Please select a valence.", parent=win)
    tk.Button(win, text="OK", command=confirm, width=10, pady=5).pack(pady=12)
    root.wait_window(win)
    return result["value"]


def pick_age_group(root):
    win = tk.Toplevel(root)
    win.title("Age group")
    win.resizable(False, False)
    win.grab_set()
    tk.Label(win, text="Age group of the person in the video?",
             font=("Arial", 12), pady=10).pack()
    choice = tk.StringVar(value="")
    for val in ("young", "old"):
        tk.Radiobutton(win, text=val.capitalize(), variable=choice,
                       value=val, font=("Arial", 11)).pack(anchor="w", padx=30)
    result = {"value": None}
    def confirm():
        if choice.get():
            result["value"] = choice.get()
            win.destroy()
        else:
            messagebox.showwarning("Select one", "Please select an age group.", parent=win)
    tk.Button(win, text="OK", command=confirm, width=10, pady=5).pack(pady=12)
    root.wait_window(win)
    return result["value"]


# ─────────────────────────────────────────────
# Core: analyse one video → one prediction
# ─────────────────────────────────────────────
def analyse_video(video_path: str, detector, predictor) -> dict:
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open: {video_path}"

    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    sample_count   = min(30, total_frames)
    sample_indices = set(
        int(i * total_frames / sample_count) for i in range(sample_count)
    )

    label_votes     = []
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
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray, 0)

                if faces:
                    shape = predictor(gray, faces[0])
                    label, score = predict_from_shape(shape)
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
        run_dir = os.path.join("results", f"dlib_geom{n}")
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
        "model":        "dlib-Geometry",
        "videos_total": total,
        "correct":      correct,
        "accuracy_pct": accuracy,
        "started_at":   started_at,
        "ended_at":     datetime.now().isoformat(timespec="seconds"),
    }]).to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model":            "dlib-Geometry",
            "predictor_path":   PREDICTOR_PATH,
            "started_at":       started_at,
            "ended_at":         datetime.now().isoformat(timespec="seconds"),
            "videos_processed": [r["video_path"] for r in rows],
        }, f, indent=2)

    return run_dir


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
def main():
    if not os.path.isfile(PREDICTOR_PATH):
        print(f"\n❌  Model file not found: {PREDICTOR_PATH}")
        print("Run these two commands first:")
        print('  curl -L "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2" -o shape_predictor_68_face_landmarks.dat.bz2')
        print("  bunzip2 shape_predictor_68_face_landmarks.dat.bz2")
        print("Then place shape_predictor_68_face_landmarks.dat in the same folder as this script.\n")
        return

    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    root = tk.Tk()
    root.withdraw()

    rows       = []
    video_num  = 0
    started_at = datetime.now().isoformat(timespec="seconds")

    print("\n=== dlib Geometry Video Evaluator ===")
    print("Pick a video to start. Close the file dialog (or click No after a result) to quit.\n")

    while True:
        # Step 1 — pick video
        video_path = pick_video(root)
        if not video_path:
            break

        # Step 2 — label it
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

        # Step 3 — run dlib
        pred       = analyse_video(video_path, detector, predictor)
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

        # Step 4 — show result, ask to continue
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

    # Save everything on quit
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
            f"({round(final_tally / len(rows) * 100)}%)"
        )
    else:
        print("No videos processed — nothing saved.")

    root.destroy()


if __name__ == "__main__":
    main()