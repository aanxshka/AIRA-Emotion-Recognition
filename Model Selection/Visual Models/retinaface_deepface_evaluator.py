"""
WHY TEST THIS MODEL (RetinaFace + optional DeepFace emotion)?

- Robot dog angle:
  RetinaFace is a strong face detector that remains reliable under pose changes.
  It also returns key landmarks, allowing pose proxies (left/right asymmetry) to quantify “dog angle” stress.

- Elderly relevance:
  A lot of misclassification comes from bad face crops (wrinkles + partial faces + occlusion).
  Better detection/landmark stability can improve downstream emotion predictions.

- What this does:
  Uses RetinaFace for detection + 5 landmarks (eye/eye/nose/mouth corners)
  Optionally runs DeepFace emotion on the detected face crop.
  Same key-based target UI + auto CSV outputs (summary/confusion/accuracy).
"""

import os
import json
import time
from datetime import datetime
from collections import Counter, deque

import cv2
import numpy as np
import pandas as pd

from retinaface import RetinaFace
from deepface import DeepFace


TARGET_TO_EXPECTED = {
    "neutral": "neutral",
    "slight_smile": "happy",
    "smile": "happy",
    "sad": "sad",
    "tilt_up": "neutral",
    "tilt_down": "neutral",
}

KEY_TO_TARGET = {
    ord("1"): "neutral",
    ord("2"): "slight_smile",
    ord("3"): "smile",
    ord("4"): "sad",
    ord("5"): "tilt_up",
    ord("6"): "tilt_down",
    ord("0"): None,
}

VALID_EMOTIONS = {"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"}


def make_run_dir(model_name="RetinaFace_DeepFace"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", f"{model_name}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def safe_quantile(series: pd.Series, q: float) -> float:
    return float(series.quantile(q)) if len(series) else float("nan")


def majority_vote(window_labels):
    if not window_labels:
        return None
    c = Counter(window_labels)
    return c.most_common(1)[0][0]


def compute_flicker_rate(labels):
    filtered = [x for x in labels if x not in ("no_face", "error", None)]
    if len(filtered) < 2:
        return 0.0
    changes = sum(1 for i in range(1, len(filtered)) if filtered[i] != filtered[i - 1])
    return changes / (len(filtered) - 1) * 100.0


def confusion_table(df):
    if df.empty:
        return pd.DataFrame()
    d = df[(df["target_label"].notna()) & (~df["pred_label"].isin(["no_face", "error"]))].copy()
    if d.empty:
        return pd.DataFrame()
    return pd.crosstab(d["target_label"], d["pred_label"], rownames=["target"], colnames=["pred"])


def pose_proxies_from_5pts(landmarks: dict):
    """
    Landmarks keys: left_eye, right_eye, nose, mouth_left, mouth_right (each (x,y))
    Pose proxies:
      yaw_proxy: nose closer to one eye than the other (left/right asymmetry)
      pitch_proxy: nose-to-mouth vs nose-to-eye vertical relation (rough)
    """
    le = np.array(landmarks["left_eye"], dtype=np.float32)
    re = np.array(landmarks["right_eye"], dtype=np.float32)
    no = np.array(landmarks["nose"], dtype=np.float32)
    ml = np.array(landmarks["mouth_left"], dtype=np.float32)
    mr = np.array(landmarks["mouth_right"], dtype=np.float32)

    eye_dist = np.linalg.norm(re - le) + 1e-6
    left_dx = abs(no[0] - le[0])
    right_dx = abs(re[0] - no[0])
    yaw_proxy = float((left_dx - right_dx) / (left_dx + right_dx + 1e-6))  # [-1,1] ish

    mouth_mid = (ml + mr) / 2.0
    eye_mid = (le + re) / 2.0
    pitch_proxy = float((mouth_mid[1] - no[1]) / (no[1] - eye_mid[1] + 1e-6))  # rough ratio

    return yaw_proxy, pitch_proxy


def main():
    CAMERA_INDEX = 0
    SMOOTH_WINDOW = 10

    # If True: run DeepFace emotion on the detected face crop
    RUN_EMOTION = True

    run_dir = make_run_dir()
    config = {
        "camera_index": CAMERA_INDEX,
        "smooth_window_frames": SMOOTH_WINDOW,
        "run_emotion": RUN_EMOTION,
        "target_to_expected_mapping": TARGET_TO_EXPECTED,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    assert cap.isOpened(), "Webcam not accessible"

    target_label = None
    pred_window = deque(maxlen=SMOOTH_WINDOW)
    rows = []

    print("[RetinaFace+DeepFace Evaluator] Press 1-6 target, 0 pause, q quit.")
    print("Saving to:", run_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        key = cv2.waitKey(1) & 0xFF
        if key in KEY_TO_TARGET:
            target_label = KEY_TO_TARGET[key]
        if key == ord("q"):
            break

        t0 = time.perf_counter()
        status = "ok"
        pred_label_raw = "no_face"
        pred_label = "no_face"
        pred_score = 0.0
        yaw_proxy = 0.0
        pitch_proxy = 0.0
        box = None

        try:
            detections = RetinaFace.detect_faces(frame)

            if isinstance(detections, dict) and len(detections) > 0:
                # take the first detected face
                first_key = list(detections.keys())[0]
                face = detections[first_key]
                box = face["facial_area"]  # [x1,y1,x2,y2]
                landmarks = face["landmarks"]  # dict of 5 points

                yaw_proxy, pitch_proxy = pose_proxies_from_5pts(landmarks)

                if RUN_EMOTION:
                    x1, y1, x2, y2 = box
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
                    crop = frame[y1:y2, x1:x2]

                    result = DeepFace.analyze(crop, actions=["emotion"], enforce_detection=False)
                    if isinstance(result, list):
                        result = result[0]
                    pred_label_raw = result.get("dominant_emotion", "no_face") or "no_face"
                    scores = result.get("emotion", {}) or {}
                    pred_score = float(scores.get(pred_label_raw, 0.0)) if pred_label_raw in scores else 0.0

                    pred_label = pred_label_raw if pred_label_raw in VALID_EMOTIONS else pred_label_raw
                else:
                    # If not running emotion, just output "face_detected"
                    pred_label_raw = "face_detected"
                    pred_label = "face_detected"
            else:
                pred_label_raw = "no_face"
                pred_label = "no_face"

        except Exception:
            status = "error"
            pred_label_raw = "error"
            pred_label = "error"

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        pred_window.append(pred_label)
        pred_label_smooth = majority_vote(list(pred_window)) or pred_label

        expected = TARGET_TO_EXPECTED.get(target_label) if target_label else None
        is_scored = target_label is not None and expected is not None and RUN_EMOTION
        is_valid_pred = pred_label_smooth not in ("no_face", "error") and pred_label_smooth in VALID_EMOTIONS
        is_correct = (pred_label_smooth == expected) if (is_scored and is_valid_pred) else None

        rows.append({
            "unix_ts": time.time(),
            "target_label": target_label,
            "expected_emotion": expected,
            "pred_label_raw": pred_label_raw,
            "pred_label": pred_label_smooth,
            "pred_score": float(pred_score),
            "yaw_proxy": float(yaw_proxy),
            "pitch_proxy": float(pitch_proxy),
            "latency_ms": float(latency_ms),
            "status": status,
            "is_scored": is_scored,
            "is_valid_pred": is_valid_pred,
            "is_correct": is_correct,
        })

        # UI overlay
        overlay = frame.copy()
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)

        df_tmp = pd.DataFrame(rows)
        scored = df_tmp[df_tmp["is_scored"] == True]
        valid_scored = scored[scored["is_valid_pred"] == True]
        acc = float(valid_scored["is_correct"].mean() * 100.0) if len(valid_scored) else 0.0

        cv2.putText(overlay, f"TARGET: {target_label if target_label else 'None'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay, f"PRED: {pred_label_smooth} ({pred_score:.2f}) [{status}]", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(overlay, f"Pose proxy yaw:{yaw_proxy:.2f} pitch:{pitch_proxy:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(overlay, f"Latency: {latency_ms:.1f} ms | Acc: {acc:.1f}%", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(overlay, "Keys: 1 neutral | 2 slight_smile | 3 smile | 4 sad | 5 tilt_up | 6 tilt_down | 0 pause | q quit",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("RetinaFace + DeepFace Evaluator", overlay)

    cap.release()
    cv2.destroyAllWindows()

    # Save outputs
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, "detailed_log.csv"), index=False)

    total = len(df)
    no_face_rate = float(df["pred_label"].isin(["no_face", "error"]).mean() * 100.0) if total else 0.0
    flicker_rate = compute_flicker_rate(df["pred_label"].tolist())

    scored = df[df["is_scored"] == True].copy()
    valid_scored = scored[scored["is_valid_pred"] == True].copy()

    # Accuracy by target
    by_target = []
    for t in sorted(scored["target_label"].dropna().unique()):
        g = scored[(scored["target_label"] == t) & (scored["is_valid_pred"] == True)]
        by_target.append({
            "target_label": t,
            "expected_emotion": TARGET_TO_EXPECTED.get(t),
            "samples": int(len(g)),
            "accuracy_pct": float(g["is_correct"].mean() * 100.0) if len(g) else float("nan"),
        })
    pd.DataFrame(by_target).to_csv(os.path.join(run_dir, "accuracy_by_target.csv"), index=False)

    confusion_table(df).to_csv(os.path.join(run_dir, "confusion_matrix.csv"))

    neutral_rows = valid_scored[valid_scored["target_label"] == "neutral"]
    neutral_to_sad = float((neutral_rows["pred_label"] == "sad").mean() * 100.0) if len(neutral_rows) else float("nan")

    slight_rows = valid_scored[valid_scored["target_label"] == "slight_smile"]
    slight_smile_detect = float((slight_rows["pred_label"] == "happy").mean() * 100.0) if len(slight_rows) else float("nan")

    tilt_rows = df[df["target_label"].isin(["tilt_up", "tilt_down"])].copy()
    tilt_fail_rate = float(tilt_rows["pred_label"].isin(["no_face", "error"]).mean() * 100.0) if len(tilt_rows) else float("nan")

    summary = {
        "run_dir": run_dir,
        "samples_total": int(total),
        "samples_valid_scored": int(len(valid_scored)),
        "overall_scored_accuracy_pct": float(valid_scored["is_correct"].mean() * 100.0) if len(valid_scored) else float("nan"),
        "neutral_misclassified_as_sad_pct": neutral_to_sad,
        "slight_smile_detected_as_happy_pct": slight_smile_detect,
        "tilt_fail_rate_pct_(no_face_or_error)": tilt_fail_rate,
        "no_face_or_error_rate_pct": no_face_rate,
        "label_flicker_rate_pct": float(flicker_rate),
        "latency_avg_ms": float(df["latency_ms"].mean()) if total else float("nan"),
        "latency_p50_ms": safe_quantile(df["latency_ms"], 0.50),
        "latency_p95_ms": safe_quantile(df["latency_ms"], 0.95),
        "latency_max_ms": float(df["latency_ms"].max()) if total else float("nan"),
        "ended_at": datetime.now().isoformat(timespec="seconds"),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    print("\n✅ Saved run to:", run_dir)


if __name__ == "__main__":
    main()