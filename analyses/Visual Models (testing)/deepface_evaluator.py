import os
import json
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime

import cv2
import pandas as pd
from deepface import DeepFace


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
ENFORCE_DETECTION = False

POSITIVE_EMOTIONS = {"happy", "surprise"}
NEGATIVE_EMOTIONS = {"angry", "disgust", "fear", "sad"}
NEUTRAL_EMOTIONS  = {"neutral"}


def map_to_valence(deepface_label: str) -> str:
    label = deepface_label.lower()
    if label in POSITIVE_EMOTIONS:
        return "positive"
    if label in NEGATIVE_EMOTIONS:
        return "negative"
    if label in NEUTRAL_EMOTIONS:
        return "neutral"
    return "unknown"


# ─────────────────────────────────────────────
# GUI helpers
# ─────────────────────────────────────────────
def pick_video(root: tk.Tk) -> str | None:
    path = filedialog.askopenfilename(
        parent=root,
        title="Select a video file — close to quit",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")],
    )
    return path or None


def pick_valence(root: tk.Tk) -> str | None:
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


def pick_age_group(root: tk.Tk) -> str | None:
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
def analyse_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open: {video_path}"

    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    sample_count  = min(30, total_frames)
    sample_indices = set(
        int(i * total_frames / sample_count) for i in range(sample_count)
    )

    emotion_totals: dict = {}
    frames_analysed = 0
    t0 = time.perf_counter()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in sample_indices:
            try:
                result = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=ENFORCE_DETECTION,
                    silent=True,
                )
                if isinstance(result, list):
                    result = result[0]
                for emotion, score in (result.get("emotion", {}) or {}).items():
                    emotion_totals[emotion] = emotion_totals.get(emotion, 0.0) + float(score)
                frames_analysed += 1
            except Exception:
                pass

            cv2.imshow("Processing video…", frame)
            cv2.waitKey(1)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    if not emotion_totals or frames_analysed == 0:
        return {
            "predicted_raw_label": "no_face",
            "predicted_score":     0.0,
            "predicted_valence":   "unknown",
            "frames_sampled":      frames_analysed,
            "latency_ms":          round(latency_ms, 2),
        }

    averaged = {e: s / frames_analysed for e, s in emotion_totals.items()}
    dominant = max(averaged, key=averaged.get)

    return {
        "predicted_raw_label": dominant,
        "predicted_score":     round(averaged[dominant], 4),
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
        run_dir = os.path.join("results", f"deepface{n}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            return run_dir
        n += 1


def save_results(rows: list, started_at: str) -> str:
    run_dir = make_run_dir()

    # results.csv — one row per video
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, "results.csv"), index=False)

    # summary.csv — overall stats
    total    = len(df)
    correct  = int(df["correct"].sum())
    accuracy = round(correct / total * 100, 2) if total else float("nan")
    pd.DataFrame([{
        "model":        "DeepFace",
        "videos_total": total,
        "correct":      correct,
        "accuracy_pct": accuracy,
        "started_at":   started_at,
        "ended_at":     datetime.now().isoformat(timespec="seconds"),
    }]).to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    # config.json — session metadata
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model":             "DeepFace",
            "enforce_detection": ENFORCE_DETECTION,
            "started_at":        started_at,
            "ended_at":          datetime.now().isoformat(timespec="seconds"),
            "videos_processed":  [r["video_path"] for r in rows],
        }, f, indent=2)

    return run_dir


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
def main():
    root = tk.Tk()
    root.withdraw()

    rows       = []
    video_num  = 0
    started_at = datetime.now().isoformat(timespec="seconds")

    print("\n=== DeepFace Video Evaluator ===")
    print("Pick a video to start. Close the file dialog (or click No after a result) to quit.\n")

    while True:
        # Step 1 — pick video
        video_path = pick_video(root)
        if not video_path:
            break   # user closed dialog → quit

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

        # Step 3 — run DeepFace
        pred       = analyse_video(video_path)
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

    # Save all rows at the very end
    if rows:
        run_dir = save_results(rows, started_at)
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