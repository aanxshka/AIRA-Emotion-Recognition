import os
import json
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from fer import FER


def make_run_dir(model_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", f"{model_name}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def pct(series: pd.Series, value) -> float:
    if len(series) == 0:
        return 0.0
    return float((series == value).mean() * 100.0)


def safe_quantile(series: pd.Series, q: float) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.quantile(q))


def main():
    # --- Config you can tweak ---
    MODEL_NAME = "FER"
    USE_MTCNN = True  # better detection, slower
    CAMERA_INDEX = 0

    run_dir = make_run_dir(MODEL_NAME)

    # Save config so you can reproduce the run later
    config = {
        "model": MODEL_NAME,
        "use_mtcnn": USE_MTCNN,
        "camera_index": CAMERA_INDEX,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    detector = FER(mtcnn=USE_MTCNN)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    assert cap.isOpened(), "Webcam not accessible"

    rows = []
    print(f"[{MODEL_NAME}] Running. Press 'q' to quit and auto-save results.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        top = detector.top_emotion(rgb)  # (label, score) or None
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        label, score = (top if top else ("no_face", 0.0))

        rows.append({
            "unix_ts": time.time(),
            "label": label,
            "score": float(score),
            "latency_ms": float(latency_ms),
        })

        # Overlay
        cv2.putText(frame, f"{label} ({score:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Latency: {latency_ms:.1f} ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("FER Live Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(rows)

    # --- Save detailed log ---
    detailed_path = os.path.join(run_dir, "fer_detailed.csv")
    df.to_csv(detailed_path, index=False)

    # --- Build summary ---
    total = len(df)
    no_face_rate = pct(df["label"], "no_face") if total else 0.0

    # Exclude no_face for “top emotion distribution”
    df_faces = df[df["label"] != "no_face"].copy()

    top_counts = df_faces["label"].value_counts().head(5).to_dict() if len(df_faces) else {}

    summary = {
        "run_dir": run_dir,
        "model": MODEL_NAME,
        "use_mtcnn": USE_MTCNN,
        "samples_total": int(total),
        "samples_face": int(len(df_faces)),
        "no_face_rate_pct": float(no_face_rate),

        "latency_avg_ms": float(df["latency_ms"].mean()) if total else float("nan"),
        "latency_p50_ms": safe_quantile(df["latency_ms"], 0.50),
        "latency_p95_ms": safe_quantile(df["latency_ms"], 0.95),
        "latency_max_ms": float(df["latency_ms"].max()) if total else float("nan"),

        "top5_emotions_counts": json.dumps(top_counts),
        "ended_at": datetime.now().isoformat(timespec="seconds"),
    }

    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(run_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n✅ Saved results to:", run_dir)
    print(" - Detailed:", detailed_path)
    print(" - Summary :", summary_path)


if __name__ == "__main__":
    main()
