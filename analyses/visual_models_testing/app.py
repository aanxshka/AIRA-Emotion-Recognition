import streamlit as st
import tempfile
import zipfile
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix, roc_auc_score,
    classification_report
)

from models import load_models
from video_processing import process_video
from utils import parse_filename


st.set_page_config(page_title="Emotion Model Evaluation", layout="wide")

# ─────────────────────────────────────────────
# Auto-save directory — Desktop/emotion_results
# Files are overwritten on each new run
# ─────────────────────────────────────────────

SAVE_DIR = Path.home() / "Desktop" / "emotion_results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def autosave(df, filename):
    """Write a DataFrame to the Desktop results folder, overwriting if exists."""
    path = SAVE_DIR / filename
    df.to_csv(path, index=False)


st.title("Emotion Recognition Model Evaluation")

st.write(
"""
Upload a ZIP file containing video clips in a **flat folder** (all files in one folder).

Naming convention:
- **Young**: `<n>_<EmotionState>.mp4/.mov`
- **Old**: `<n>_Old_<EmotionState>.mp4/.mov`

Emotion states: `Positive` / `Neutral` / `Negative`

Example:
```
dataset.zip
    ├── Sidney_Positive.mov
    ├── Sidney_Neutral.mov
    ├── Sidney_Negative.mov
    ├── Aai_Old_Positive.mov
    ├── Aai_Old_Neutral.mov
    └── Aai_Old_Negative.mov
```

> **Auto-save:** Results are saved automatically to `~/Desktop/emotion_results/` after each model completes.
"""
)

# ─────────────────────────────────────────────
# Load models (cached)
# ─────────────────────────────────────────────

@st.cache_resource
def get_all_models():
    return load_models()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

VALENCE_LABELS = ["positive", "neutral", "negative"]
RAW_COLS = [
    "video_num", "filename", "model", "library", "person", "age_group",
    "ground_truth_valence", "predicted_raw_label", "predicted_score",
    "predicted_valence", "score_positive", "score_neutral", "score_negative",
    "correct", "frames_sampled", "latency_ms", "timestamp"
]


def get_all_videos(root_folder):
    videos = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith((".mp4", ".mov", ".avi")):
                videos.append(os.path.join(root, file))
    return videos


def compute_auc_for_group(group):
    """Compute per-class AUC for a DataFrame slice. Returns dict."""
    result = {}
    for cls in VALENCE_LABELS:
        y_true_bin = (group["ground_truth_valence"] == cls).astype(int)
        y_score    = group[f"score_{cls}"]
        if y_true_bin.sum() == 0 or y_true_bin.sum() == len(y_true_bin):
            result[f"AUC_{cls}"] = None
        else:
            try:
                result[f"AUC_{cls}"] = round(roc_auc_score(y_true_bin, y_score), 4)
            except Exception:
                result[f"AUC_{cls}"] = None
    valid = [v for v in result.values() if v is not None]
    result["AUC_mean"] = round(float(np.mean(valid)), 4) if valid else None
    return result


def aggregate_metrics(group_df):
    y_true = group_df["ground_truth_valence"]
    y_pred = group_df["predicted_valence"]
    return pd.Series({
        "Accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "Precision":   round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "Recall":      round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "F1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "F1_macro":    round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Kappa":       round(cohen_kappa_score(y_true, y_pred), 4),
        "Clips":       len(group_df),
    })


def per_class_metrics(group_df):
    y_true = group_df["ground_truth_valence"]
    y_pred = group_df["predicted_valence"]
    report = classification_report(
        y_true, y_pred, labels=VALENCE_LABELS,
        zero_division=0, output_dict=True
    )
    rows = []
    for cls in VALENCE_LABELS:
        r = report.get(cls, {})
        rows.append({
            "Class":     cls,
            "Precision": round(r.get("precision", 0), 4),
            "Recall":    round(r.get("recall", 0), 4),
            "F1":        round(r.get("f1-score", 0), 4),
            "Support":   int(r.get("support", 0)),
        })
    return pd.DataFrame(rows)


def build_summary_csvs(df):
    """Build all summary DataFrames from completed results."""

    # 1. Overall per model
    overall_rows = []
    for model_name, group in df.groupby("model"):
        m   = aggregate_metrics(group)
        row = {"Model": model_name}
        row.update(m.to_dict())
        row.update(compute_auc_for_group(group))
        row["Avg_Latency_ms"] = round(group["latency_ms"].mean(), 2)
        overall_rows.append(row)
    df_overall = pd.DataFrame(overall_rows)

    # 2. Per-class per model
    per_class_rows = []
    for model_name, group in df.groupby("model"):
        pc = per_class_metrics(group)
        pc.insert(0, "Model", model_name)
        per_class_rows.append(pc)
    df_per_class = pd.concat(per_class_rows, ignore_index=True)

    # 3. Per-class by age group per model
    per_class_age_rows = []
    for (model_name, age_group), group in df.groupby(["model", "age_group"]):
        pc = per_class_metrics(group)
        pc.insert(0, "Age_Group", age_group)
        pc.insert(0, "Model", model_name)
        per_class_age_rows.append(pc)
    df_per_class_age = pd.concat(per_class_age_rows, ignore_index=True)

    # 4. Confusion matrices — one row per model, flattened
    cm_rows = []
    for model_name, group in df.groupby("model"):
        cm = confusion_matrix(
            group["ground_truth_valence"],
            group["predicted_valence"],
            labels=VALENCE_LABELS
        )
        df_cm = pd.DataFrame(
            cm,
            index   = [f"actual_{l}" for l in VALENCE_LABELS],
            columns = [f"pred_{l}"   for l in VALENCE_LABELS]
        )
        df_cm.insert(0, "Model", model_name)
        df_cm.insert(1, "actual_class", [f"actual_{l}" for l in VALENCE_LABELS])
        cm_rows.append(df_cm)
    df_confusion = pd.concat(cm_rows, ignore_index=True)

    # 5. Overall by age group per model (with AUC)
    age_rows = []
    for (model_name, age_group), group in df.groupby(["model", "age_group"]):
        m   = aggregate_metrics(group)
        row = {"Model": model_name, "Age_Group": age_group}
        row.update(m.to_dict())
        row.update(compute_auc_for_group(group))
        age_rows.append(row)
    df_age = pd.DataFrame(age_rows)

    # 6. Latency
    df_latency = df.groupby("model")["latency_ms"].agg(
        Avg_ms="mean", Min_ms="min", Max_ms="max"
    ).round(2).reset_index()

    return df_overall, df_per_class, df_per_class_age, df_confusion, df_age, df_latency


# ─────────────────────────────────────────────
# ZIP uploader
# ─────────────────────────────────────────────

uploaded_zip = st.file_uploader("Upload Dataset ZIP", type=["zip"])

if uploaded_zip:

    st.info("Extracting dataset...")
    _base_tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp_uploads")
    os.makedirs(_base_tmp, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=_base_tmp)
    zip_path = os.path.join(temp_dir, "dataset.zip")

    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    st.success("Dataset extracted")

    video_files = get_all_videos(temp_dir)
    video_files = [v for v in video_files if not os.path.basename(v).startswith("._")]

    if len(video_files) == 0:
        st.error("No videos found in ZIP file")
        st.stop()

    st.write(f"Found **{len(video_files)} videos**")

    # ── Model selection ───────────────────────
    st.subheader("Select Models to Run")
    run_hse          = st.checkbox("HSEmotion baseline (enet_b2, enet_b0_vgaf, enet_b0_va_mtl)", value=True)
    run_hse_improved = st.checkbox("HSEmotion improved (confidence-weighted aggregation)", value=False)
    run_ferplus      = st.checkbox("FERPlus OpenCV DNN", value=False)
    run_ferplus_imp  = st.checkbox("FERPlus improved (confidence-weighted aggregation)", value=False)
    run_mediapipe    = st.checkbox("MediaPipe Geometry (landmark-based)", value=False)
    run_mp_improved  = st.checkbox("MediaPipe improved (velocity + eye aperture + temporal consistency)", value=False)

    if not any([run_hse, run_hse_improved, run_ferplus, run_ferplus_imp, run_mediapipe, run_mp_improved]):
        st.warning("Please select at least one model group.")
        st.stop()

    if not st.button("Start Evaluation"):
        st.stop()

    # ── Load selected models ──────────────────
    st.info("Loading models...")
    all_models = get_all_models()

    selected_models = {}
    if run_hse:
        selected_models["HSE_enet_b2"]        = all_models["HSE_enet_b2"]
        selected_models["HSE_enet_b0_vgaf"]   = all_models["HSE_enet_b0_vgaf"]
        selected_models["HSE_enet_b0_va_mtl"] = all_models["HSE_enet_b0_va_mtl"]
    if run_hse_improved:
        selected_models["HSE_enet_b2_improved"]        = all_models["HSE_enet_b2_improved"]
        selected_models["HSE_enet_b0_vgaf_improved"]   = all_models["HSE_enet_b0_vgaf_improved"]
        selected_models["HSE_enet_b0_va_mtl_improved"] = all_models["HSE_enet_b0_va_mtl_improved"]
    if run_ferplus:
        selected_models["FERPlus_OpenCV"]          = all_models["FERPlus_OpenCV"]
    if run_ferplus_imp:
        selected_models["FERPlus_OpenCV_improved"] = all_models["FERPlus_OpenCV_improved"]
    if run_mediapipe:
        selected_models["MediaPipe_Geometry"]          = all_models["MediaPipe_Geometry"]
    if run_mp_improved:
        selected_models["MediaPipe_Geometry_improved"] = all_models["MediaPipe_Geometry_improved"]

    st.success(f"{len(selected_models)} models loaded")
    st.info(f"Auto-saving results to: `{SAVE_DIR}`")

    # ─────────────────────────────────────────
    # PROCESS VIDEOS — auto-save after each model
    # ─────────────────────────────────────────

    all_results = []  # accumulates across all models
    video_num   = 1

    for model_name, (library_type, model) in selected_models.items():

        st.write(f"### Running: {model_name}")
        progress     = st.progress(0)
        model_results = []

        for i, video in enumerate(video_files):

            filename = os.path.basename(video)

            try:
                person, age_group, ground_truth_valence = parse_filename(filename)
            except Exception as e:
                st.warning(f"Skipping: {filename} ({e})")
                progress.progress((i + 1) / len(video_files))
                continue

            result = process_video(video, library_type, model)
            (pred_raw, pred_score, pred_valence,
             score_pos, score_neu, score_neg,
             latency_ms, frames_used) = result

            if pred_raw is None:
                st.warning(f"No face detected in {filename}")
                progress.progress((i + 1) / len(video_files))
                continue

            correct = 1 if pred_valence == ground_truth_valence else 0

            row = {
                "video_num":            video_num,
                "filename":             filename,
                "model":                model_name,
                "library":              library_type,
                "person":               person,
                "age_group":            age_group,
                "ground_truth_valence": ground_truth_valence,
                "predicted_raw_label":  pred_raw,
                "predicted_score":      pred_score,
                "predicted_valence":    pred_valence,
                "score_positive":       score_pos,
                "score_neutral":        score_neu,
                "score_negative":       score_neg,
                "correct":              correct,
                "frames_sampled":       frames_used,
                "latency_ms":           latency_ms,
                "timestamp":            datetime.now().isoformat(timespec="seconds"),
            }
            model_results.append(row)
            all_results.append(row)
            video_num += 1
            progress.progress((i + 1) / len(video_files))

        # ── Auto-save after this model completes ──
        if model_results:
            # 1. Append this model's rows to the running raw CSV
            df_so_far = pd.DataFrame(all_results)
            autosave(df_so_far[RAW_COLS], "raw_results.csv")

            # 2. Rebuild and save all summary CSVs with data collected so far
            try:
                df_ov, df_pc, df_pca, df_cm, df_ag, df_lat = build_summary_csvs(df_so_far)
                autosave(df_ov,  "metrics_overall.csv")
                autosave(df_pc,  "metrics_per_class.csv")
                autosave(df_pca, "metrics_per_class_age.csv")
                autosave(df_cm,  "confusion_matrices.csv")
                autosave(df_ag,  "metrics_age_group.csv")
                autosave(df_lat, "latency_summary.csv")
            except Exception as e:
                st.warning(f"Summary CSVs could not be saved after {model_name}: {e}")

            st.success(f"✅ {model_name} done — results saved to Desktop/emotion_results/")

    # ─────────────────────────────────────────
    # FINAL DISPLAY
    # ─────────────────────────────────────────

    df = pd.DataFrame(all_results)

    if df.empty:
        st.error("No results — check your video filenames match the naming convention.")
        st.stop()

    df_overall, df_per_class, df_per_class_age, df_confusion, df_age, df_latency = build_summary_csvs(df)

    st.subheader("Raw Clip Results")
    st.dataframe(df[RAW_COLS], use_container_width=True)

    st.subheader("1. Overall Metrics per Model")
    st.dataframe(df_overall, use_container_width=True)

    st.subheader("2. Per-Class Metrics per Model")
    st.dataframe(df_per_class, use_container_width=True)

    st.subheader("3. Per-Class Metrics by Age Group per Model")
    st.dataframe(df_per_class_age, use_container_width=True)

    st.subheader("4. Confusion Matrices")
    for model_name, group in df.groupby("model"):
        st.write(f"**{model_name}**")
        cm    = confusion_matrix(group["ground_truth_valence"], group["predicted_valence"], labels=VALENCE_LABELS)
        df_cm = pd.DataFrame(cm, index=[f"actual_{l}" for l in VALENCE_LABELS], columns=[f"pred_{l}" for l in VALENCE_LABELS])
        st.dataframe(df_cm, use_container_width=True)

    st.subheader("5. Overall Results by Age Group per Model")
    st.dataframe(df_age, use_container_width=True)

    st.subheader("6. Latency Summary")
    st.dataframe(df_latency, use_container_width=True)

    # ─────────────────────────────────────────
    # DOWNLOAD BUTTONS (manual backup)
    # ─────────────────────────────────────────

    st.subheader("Download Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.download_button("Raw Results",       df[RAW_COLS].to_csv(index=False).encode(),     "raw_results.csv",           "text/csv")
    col2.download_button("Overall Metrics",   df_overall.to_csv(index=False).encode(),       "metrics_overall.csv",       "text/csv")
    col3.download_button("Per-Class Metrics", df_per_class.to_csv(index=False).encode(),     "metrics_per_class.csv",     "text/csv")
    col4.download_button("Per-Class by Age",  df_per_class_age.to_csv(index=False).encode(), "metrics_per_class_age.csv", "text/csv")

    col5, col6 = st.columns(2)
    col5.download_button("Age Group Metrics", df_age.to_csv(index=False).encode(),           "metrics_age_group.csv",     "text/csv")
    col6.download_button("Latency Summary",   df_latency.to_csv(index=False).encode(),       "latency_summary.csv",       "text/csv")

    st.success(f"🎉 Evaluation complete! All files saved to: {SAVE_DIR}")