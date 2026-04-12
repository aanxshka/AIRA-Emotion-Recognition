"""
WHY TEST THIS MODEL (HSEmotion)?

- Elderly expressions are often subtle (low-intensity smiles, low-arousal sadness).
  Some models that do "hard" 7-class classification can over-predict sadness on neutral elderly faces.
  HSEmotion is often used as a lightweight alternative and is commonly aligned with affect-style thinking
  (useful for Circumplex-style mapping later).

- Robot dog angle relevance:
  We explicitly test tilt_up / tilt_down (camera below face level) and track no-face failures + stability.

WHAT THIS SCRIPT DOES:
- You choose the "target expression" you are about to perform using number keys.
- The model predicts emotion live.
- The script automatically scores misclassifications + saves clean CSV outputs when you press 'q'.
"""

import os
import json
import time
from datetime import datetime
from collections import Counter, deque

import cv2
import pandas as pd
from hsemotion.facial_emotions import HSEmotionRecognizer

# ---- PyTorch 2.6+ safe loading fix for HSEmotion checkpoints ----
    # Allowlist the exact timm classes referenced by the checkpoint.
try:
    import torch
    import timm
    from timm.models.efficientnet import EfficientNet
    from timm.layers.conv2d_same import Conv2dSame
    from timm.layers.norm_act import BatchNormAct2d

    torch.serialization.add_safe_globals([EfficientNet, Conv2dSame, BatchNormAct2d])
except Exception:
    pass


# ---------------------------
# Target labels (what YOU intend to show) -> expected model label (for scoring)
# We simplify everything to happy/sad/neutral for your elderly-focused evaluation.
# ---------------------------
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
    ord("0"): None,  # pause scoring
}

def make_run_dir(model_name="HSEmotion"):
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

def normalize_to_happy_sad_neutral(raw_label: str) -> str:
    """
    HSEmotion can output various label sets depending on version/model.
    For your evaluation, we map them into: happy / sad / neutral.
    """
    if not raw_label:
        return "no_face"
    l = raw_label.lower()

    # Happy-ish
    if "happy" in l or "joy" in l:
        return "happy"

    # Sad-ish
    if "sad" in l or "depress" in l:
        return "sad"

    # Neutral-ish
    if "neutral" in l:
        return "neutral"

    # Anything else -> treat as neutral for this particular evaluation focus
    return "neutral"


def main():
    CAMERA_INDEX = 0
    SMOOTH_WINDOW = 10  # majority vote smoothing to reduce flicker in metrics

    run_dir = make_run_dir("HSEmotion")
    config = {
        "camera_index": CAMERA_INDEX,
        "smooth_window_frames": SMOOTH_WINDOW,
        "target_to_expected_mapping": TARGET_TO_EXPECTED,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


    # Init model
    model = HSEmotionRecognizer()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    assert cap.isOpened(), "Webcam not accessible"

    target_label = None
    pred_window = deque(maxlen=SMOOTH_WINDOW)
    rows = []

    print("[HSEmotion Evaluator] Press 1-6 to set target, 0 pause, q quit.")
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

        try:
            # HSEmotion expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Different versions expose different predict functions.
            out = None
            if hasattr(model, "predict_emotions"):
                out = model.predict_emotions(rgb)
            elif hasattr(model, "predict"):
                out = model.predict(rgb)

            # Handle common output patterns
            # - tuple (label, score)
            # - dict {label: score}
            # - string label
            if isinstance(out, tuple) and len(out) >= 1:
                pred_label_raw = str(out[0])
                if len(out) >= 2:
                    try:
                        pred_score = float(out[1])
                    except Exception:
                        pred_score = 0.0
            elif isinstance(out, dict) and len(out) > 0:
                best_k = max(out, key=lambda k: out[k])
                pred_label_raw = str(best_k)
                pred_score = float(out[best_k])
            elif isinstance(out, str):
                pred_label_raw = out
            else:
                pred_label_raw = "no_face"

            pred_label = normalize_to_happy_sad_neutral(pred_label_raw)

        except Exception:
            status = "error"
            pred_label_raw = "error"
            pred_label = "error"
            pred_score = 0.0

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        # Smoothing for stability metrics
        pred_window.append(pred_label)
        pred_label_smooth = majority_vote(list(pred_window)) or pred_label

        expected = TARGET_TO_EXPECTED.get(target_label) if target_label else None
        is_scored = target_label is not None and expected is not None
        is_valid_pred = pred_label_smooth not in ("no_face", "error")
        is_correct = (pred_label_smooth == expected) if (is_scored and is_valid_pred) else None

        rows.append({
            "unix_ts": time.time(),
            "target_label": target_label,
            "expected_label": expected,
            "pred_label_raw": pred_label_raw,
            "pred_label": pred_label_smooth,
            "pred_score": float(pred_score),
            "latency_ms": float(latency_ms),
            "status": status,
            "is_scored": is_scored,
            "is_valid_pred": is_valid_pred,
            "is_correct": is_correct,
        })

        # Live overlay stats
        df_tmp = pd.DataFrame(rows)
        scored = df_tmp[df_tmp["is_scored"] == True]
        valid_scored = scored[scored["is_valid_pred"] == True]
        acc = float(valid_scored["is_correct"].mean() * 100.0) if len(valid_scored) else 0.0

        overlay = frame.copy()
        cv2.putText(overlay, f"TARGET: {target_label if target_label else 'None'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay, f"PRED: {pred_label_smooth} ({pred_score:.2f}) [{status}]", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(overlay, f"Latency: {latency_ms:.1f} ms | Acc: {acc:.1f}%", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(overlay, "Keys: 1 neutral | 2 slight_smile | 3 smile | 4 sad | 5 tilt_up | 6 tilt_down | 0 pause | q quit",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("HSEmotion Live Evaluator", overlay)

    cap.release()
    cv2.destroyAllWindows()

    # ---------------------------
    # Save outputs
    # ---------------------------
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
            "expected_label": TARGET_TO_EXPECTED.get(t),
            "samples": int(len(g)),
            "accuracy_pct": float(g["is_correct"].mean() * 100.0) if len(g) else float("nan"),
        })
    pd.DataFrame(by_target).to_csv(os.path.join(run_dir, "accuracy_by_target.csv"), index=False)

    confusion_table(df).to_csv(os.path.join(run_dir, "confusion_matrix.csv"))

    # Elderly-focused metrics
    neutral_rows = valid_scored[valid_scored["target_label"] == "neutral"]
    neutral_to_sad = float((neutral_rows["pred_label"] == "sad").mean() * 100.0) if len(neutral_rows) else float("nan")

    slight_rows = valid_scored[valid_scored["target_label"] == "slight_smile"]
    slight_smile_detect = float((slight_rows["pred_label"] == "happy").mean() * 100.0) if len(slight_rows) else float("nan")

    summary = {
        "run_dir": run_dir,
        "samples_total": int(total),
        "samples_valid_scored": int(len(valid_scored)),
        "overall_scored_accuracy_pct": float(valid_scored["is_correct"].mean() * 100.0) if len(valid_scored) else float("nan"),
        "neutral_misclassified_as_sad_pct": neutral_to_sad,
        "slight_smile_detected_as_happy_pct": slight_smile_detect,
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
    print(" - detailed_log.csv")
    print(" - accuracy_by_target.csv")
    print(" - confusion_matrix.csv")
    print(" - summary.csv")


if __name__ == "__main__":
    main()