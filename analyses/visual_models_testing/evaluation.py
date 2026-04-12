import streamlit as st
import tempfile
import zipfile
import os
import pandas as pd
from datetime import datetime

from models import load_models
from video_processing import process_video
from evaluation import evaluate_results
from utils import parse_filename


st.set_page_config(page_title="Emotion Model Evaluation", layout="wide")

st.title("Emotion Recognition Model Evaluation")

st.write(
"""
Upload a ZIP file containing video clips.

Naming convention:
- **Young**: `<Name>_<EmotionState>.mp4/.mov`
- **Old**: `<Name>_Old_<EmotionState>.mp4/.mov`

Emotion states: `happy` / `neutral` / `upset`

Example structure:

    dataset.zip
        ├── Sidney_Happy.mp4
        ├── Sidney_Neutral.mp4
        ├── Sidney_Upset.mp4
        ├── Aai_Old_Happy.mov
        ├── Aai_Old_Neutral.mov
        └── Aai_Old_Upset.mov
"""
)

# -----------------------------
# ZIP FILE UPLOADER
# -----------------------------

uploaded_zip = st.file_uploader(
    "Upload Dataset ZIP",
    type=["zip"]
)

# -----------------------------
# FUNCTION: FIND ALL VIDEOS
# -----------------------------

def get_all_videos(root_folder):

    videos = []

    for root, dirs, files in os.walk(root_folder):

        for file in files:

            if file.lower().endswith((".mp4", ".mov", ".avi")):

                videos.append(os.path.join(root, file))

    return videos


# -----------------------------
# MAIN PIPELINE
# -----------------------------

if uploaded_zip:

    st.info("Extracting dataset...")

    temp_dir = tempfile.mkdtemp()

    zip_path = os.path.join(temp_dir, "dataset.zip")

    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    st.success("Dataset extracted")

    # -----------------------------
    # FIND ALL VIDEOS
    # -----------------------------

    video_files = get_all_videos(temp_dir)

    if len(video_files) == 0:
        st.error("No videos found in ZIP file")
        st.stop()

    st.write(f"Found **{len(video_files)} videos**")

    # -----------------------------
    # LOAD MODELS
    # -----------------------------

    st.info("Loading emotion models...")

    models = load_models()

    st.success(f"{len(models)} models loaded")

    # -----------------------------
    # PROCESS VIDEOS
    # -----------------------------

    results = []
    video_num = 1

    for model_name, model in models.items():

        st.write(f"### Evaluating Model: {model_name}")

        for video in video_files:

            filename = os.path.basename(video)

            if filename.startswith("._"):
                continue

            try:
                person, age_group, ground_truth_valence = parse_filename(filename)
            except Exception as e:
                st.warning(f"Skipping improperly named file: {filename} ({e})")
                continue

            pred_raw, pred_score, pred_valence, latency_ms, frames_used = process_video(video, model)

            if pred_raw is None:
                st.warning(f"No face detected in {filename}")
                continue

            correct = 1 if pred_valence == ground_truth_valence else 0

            results.append({
                "video_num": video_num,
                "video_path": video,
                "model": model_name,
                "age_group": age_group,
                "ground_truth_valence": ground_truth_valence,
                "predicted_raw_label": pred_raw,
                "predicted_score": pred_score,
                "predicted_valence": pred_valence,
                "correct": correct,
                "frames_sampled": frames_used,
                "latency_ms": latency_ms,
                "timestamp": datetime.now().isoformat(timespec="seconds")
            })

            video_num += 1

    df = pd.DataFrame(results)

    # -----------------------------
    # MODEL SPEED
    # -----------------------------

    speed_metrics = df.groupby("model")["latency_ms"].mean().reset_index()
    speed_metrics.columns = ["Model", "Avg Latency (ms)"]

    st.write("### Model Speed")
    st.dataframe(speed_metrics)

    # -----------------------------
    # SHOW RAW RESULTS
    # -----------------------------

    st.subheader("Raw Clip Results")

    display_cols = [
        "video_num", "video_path", "model", "age_group",
        "ground_truth_valence", "predicted_raw_label", "predicted_score",
        "predicted_valence", "correct", "frames_sampled", "latency_ms", "timestamp"
    ]

    st.dataframe(df[display_cols])

    # Add person column derived from filename
    df["person"] = df["video_path"].apply(
        lambda p: parse_filename(os.path.basename(p))[0]
        if not os.path.basename(p).startswith("._") else None
    )

    # -----------------------------
    # HELPER: aggregate metrics
    # -----------------------------

    def aggregate_metrics(group_df):
        """Returns accuracy, precision, recall, F1 for a slice of df."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        y_true = group_df["ground_truth_valence"]
        y_pred = group_df["predicted_valence"]
        return pd.Series({
            "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "Recall":    round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "F1 Score":  round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "Clips":     len(group_df),
        })

    # -----------------------------
    # 1. OVERALL RESULTS PER MODEL
    # -----------------------------

    st.subheader("Overall Results per Model")

    overall = df.groupby("model").apply(aggregate_metrics).reset_index()
    overall.rename(columns={"model": "Model"}, inplace=True)
    st.dataframe(overall, use_container_width=True)

    # -----------------------------
    # 2. RESULTS PER AGE GROUP PER MODEL
    # -----------------------------

    st.subheader("Results per Age Group per Model")

    age_metrics = df.groupby(["model", "age_group"]).apply(aggregate_metrics).reset_index()
    age_metrics.rename(columns={"model": "Model", "age_group": "Age Group"}, inplace=True)
    st.dataframe(age_metrics, use_container_width=True)

    # -----------------------------
    # 3. RESULTS PER PERSON PER MODEL
    # -----------------------------

    st.subheader("Results per Person per Model")

    person_metrics = df.groupby(["person", "model"]).apply(aggregate_metrics).reset_index()
    person_metrics.rename(columns={"person": "Person", "model": "Model"}, inplace=True)
    st.dataframe(person_metrics, use_container_width=True)

    # -----------------------------
    # SAVE RESULTS
    # -----------------------------

    st.subheader("Download Results")

    csv_results = df[display_cols + ["person"]].to_csv(index=False).encode("utf-8")
    csv_overall = overall.to_csv(index=False).encode("utf-8")
    csv_age     = age_metrics.to_csv(index=False).encode("utf-8")
    csv_person  = person_metrics.to_csv(index=False).encode("utf-8")

    col1, col2, col3, col4 = st.columns(4)

    col1.download_button("Download Raw Results",      csv_results, "emotion_results.csv",       "text/csv")
    col2.download_button("Download Overall Metrics",  csv_overall, "metrics_overall.csv",        "text/csv")
    col3.download_button("Download Age Group Metrics",csv_age,     "metrics_age_group.csv",      "text/csv")
    col4.download_button("Download Person Metrics",   csv_person,  "metrics_per_person.csv",     "text/csv")

    st.success("Evaluation complete!")