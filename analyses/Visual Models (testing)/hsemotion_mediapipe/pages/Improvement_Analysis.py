import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

st.set_page_config(page_title="Improvement Analysis", layout="wide")

st.title("Baseline vs Improved — Does the Improvement Help?")

# ─────────────────────────────────────────────
# What are the improvements?
# ─────────────────────────────────────────────

st.header("What Are the Improvements?")

st.subheader("HSEmotion & FERPlus — Confidence-Weighted Aggregation")
st.write("""
**Baseline behaviour:** Every frame that contains a detected face contributes equally to the final
prediction. The model processes each frame independently, and the final label is the simple majority
vote (or average probability score) across all sampled frames.

**What the improvement does:** Each frame's contribution is weighted by InsightFace's face detection
confidence score (`det_score`). A frame where InsightFace is very confident it found a clean,
well-aligned face gets more weight. A frame where the face is partially occluded, at a bad angle,
or poorly lit gets less weight.

**Why it should theoretically help:** Not all frames are equal quality. For elderly participants
especially, head pose can vary significantly mid-clip — a frame where the person looks away or
is in profile should count for less than a frame with a clear frontal face. By down-weighting
low-quality detections, the aggregated prediction should be more robust and less susceptible
to a single bad frame dominating the result.

**Why it might not always help:** If InsightFace's det_score is not strongly correlated with
emotion-relevant image quality (e.g. it scores a frontal but blurry frame highly), the weighting
adds noise rather than signal. With FRAME_SKIP=1, there are many more frames, so the signal-to-noise
ratio of the weighting matters more.
""")

st.subheader("MediaPipe — Velocity + Eye Aperture + Temporal Consistency")
st.write("""
**Baseline behaviour:** Rule-based geometry thresholds on mouth curve and brow furrow angle.
A single frame's landmark positions are compared to fixed thresholds to classify positive/neutral/negative.

**What the improvement adds:**

1. **Mouth velocity cue** — tracks how much the mouth is moving between consecutive frames.
   Genuine emotional expressions tend to involve dynamic mouth movement; a static face is more
   likely neutral. High velocity toward an open/curved position reinforces a positive classification.

2. **Eye aperture cue** — measures the ratio of eye opening height to width. Fear and surprise
   involve wide-open eyes; sadness involves narrowed eyes. This adds a second geometric channel
   beyond just mouth and brow.

3. **Temporal consistency filter (3-frame window)** — a predicted label must appear in at least
   2 out of 3 consecutive frames before it is accepted. This prevents a single outlier frame
   (e.g. a blink, a momentary expression artifact) from flipping the prediction.

**Why it should theoretically help:** The baseline MediaPipe almost exclusively predicts neutral
because the brow and mouth thresholds are too conservative for subtle elderly expressions. The added
cues give the model more information to distinguish genuine expressions from neutral, and the
temporal filter reduces noise.

**Why it might not always help:** Elderly faces have reduced muscle mobility — expressions are
genuinely more subtle. The velocity and eye aperture thresholds were set heuristically and may
still be too conservative for this population. If the baseline is stuck at neutral, the improvements
may shift some predictions but not fix the fundamental limitation of a geometry-only approach.
""")

st.divider()

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────

st.header("Did the Improvements Actually Help?")

default_path = Path.home() / "Desktop" / "emotion_results" / "raw_results.csv"

if default_path.exists():
    st.success(f"Auto-loaded: `{default_path}`")
    df = pd.read_csv(default_path)
else:
    uploaded = st.file_uploader("Upload raw_results.csv", type=["csv"])
    if not uploaded:
        st.info("No results found at ~/Desktop/emotion_results/raw_results.csv — please upload manually.")
        st.stop()
    df = pd.read_csv(uploaded)

VALENCE_LABELS = ["positive", "neutral", "negative"]

# Pair definitions: (baseline, improved, label)
PAIRS = [
    ("HSE_enet_b2",        "HSE_enet_b2_improved",        "HSEmotion enet_b2"),
    ("HSE_enet_b0_vgaf",   "HSE_enet_b0_vgaf_improved",   "HSEmotion enet_b0_vgaf"),
    ("HSE_enet_b0_va_mtl", "HSE_enet_b0_va_mtl_improved", "HSEmotion enet_b0_va_mtl"),
    ("FERPlus_OpenCV",     "FERPlus_OpenCV_improved",      "FERPlus OpenCV"),
    ("MediaPipe_Geometry", "MediaPipe_Geometry_improved",  "MediaPipe"),
]

available_models = df["model"].unique()
PAIRS = [(b, i, l) for b, i, l in PAIRS if b in available_models and i in available_models]

if not PAIRS:
    st.error("No baseline/improved pairs found in the data.")
    st.stop()

# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def metrics(g):
    y_true, y_pred = g["ground_truth_valence"], g["predicted_valence"]
    try:
        kappa = round(cohen_kappa_score(y_true, y_pred), 4)
    except Exception:
        kappa = None
    return {
        "Accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "F1_macro":    round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "F1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "Kappa":       kappa,
    }

def delta(improved_val, baseline_val):
    if improved_val is None or baseline_val is None:
        return None
    return round(improved_val - baseline_val, 4)

def verdict(acc_delta):
    if acc_delta is None:
        return "—"
    if acc_delta > 0.02:
        return "✅ Improved"
    elif acc_delta < -0.02:
        return "❌ Degraded"
    else:
        return "➡️ No change"

# ─────────────────────────────────────────────
# SECTION 1 — Overall comparison table
# ─────────────────────────────────────────────

st.subheader("1. Overall Metrics — Baseline vs Improved")
st.caption("Δ = Improved minus Baseline. Positive Δ = improvement.")

comp_rows = []
for base_name, imp_name, label in PAIRS:
    base_g = df[df["model"] == base_name]
    imp_g  = df[df["model"] == imp_name]
    bm = metrics(base_g)
    im = metrics(imp_g)

    base_lpf = round((base_g["latency_ms"] / base_g["frames_sampled"].replace(0, np.nan)).mean(), 2)
    imp_lpf  = round((imp_g["latency_ms"]  / imp_g["frames_sampled"].replace(0, np.nan)).mean(), 2)

    comp_rows.append({
        "Model Pair":              label,
        "Baseline Acc":            bm["Accuracy"],
        "Improved Acc":            im["Accuracy"],
        "Δ Accuracy":              delta(im["Accuracy"], bm["Accuracy"]),
        "Baseline F1_macro":       bm["F1_macro"],
        "Improved F1_macro":       im["F1_macro"],
        "Δ F1_macro":              delta(im["F1_macro"], bm["F1_macro"]),
        "Baseline Kappa":          bm["Kappa"],
        "Improved Kappa":          im["Kappa"],
        "Δ Kappa":                 delta(im["Kappa"], bm["Kappa"]),
        "Baseline ms/frame":       base_lpf,
        "Improved ms/frame":       imp_lpf,
        "Δ ms/frame":              round(imp_lpf - base_lpf, 2),
        "Verdict":                 verdict(delta(im["Accuracy"], bm["Accuracy"])),
    })

df_comp = pd.DataFrame(comp_rows)
st.dataframe(df_comp, use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 2 — Per-age-group comparison
# ─────────────────────────────────────────────

st.subheader("2. Accuracy by Age Group — Baseline vs Improved")
st.caption("Does the improvement help more for old or young participants?")

age_rows = []
for base_name, imp_name, label in PAIRS:
    for age in ["young", "old"]:
        base_g = df[(df["model"] == base_name) & (df["age_group"] == age)]
        imp_g  = df[(df["model"] == imp_name)  & (df["age_group"] == age)]
        if len(base_g) == 0 or len(imp_g) == 0:
            continue
        base_acc = round(accuracy_score(base_g["ground_truth_valence"], base_g["predicted_valence"]), 4)
        imp_acc  = round(accuracy_score(imp_g["ground_truth_valence"],  imp_g["predicted_valence"]), 4)
        age_rows.append({
            "Model Pair":   label,
            "Age Group":    age,
            "Baseline Acc": base_acc,
            "Improved Acc": imp_acc,
            "Δ Accuracy":   round(imp_acc - base_acc, 4),
            "Verdict":      verdict(imp_acc - base_acc),
        })

df_age = pd.DataFrame(age_rows)
st.dataframe(df_age, use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 3 — Per-class recall comparison
# ─────────────────────────────────────────────

st.subheader("3. Per-Class Recall — Baseline vs Improved")
st.caption("Did the improvement help detect specific emotion classes better?")

recall_rows = []
for base_name, imp_name, label in PAIRS:
    base_g = df[df["model"] == base_name]
    imp_g  = df[df["model"] == imp_name]
    for cls in VALENCE_LABELS:
        base_recall = round(
            (base_g[base_g["ground_truth_valence"] == cls]["predicted_valence"] == cls).mean(), 4
        ) if len(base_g[base_g["ground_truth_valence"] == cls]) > 0 else 0
        imp_recall = round(
            (imp_g[imp_g["ground_truth_valence"] == cls]["predicted_valence"] == cls).mean(), 4
        ) if len(imp_g[imp_g["ground_truth_valence"] == cls]) > 0 else 0
        recall_rows.append({
            "Model Pair":      label,
            "Class":           cls,
            "Baseline Recall": base_recall,
            "Improved Recall": imp_recall,
            "Δ Recall":        round(imp_recall - base_recall, 4),
            "Verdict":         verdict(imp_recall - base_recall),
        })

df_recall = pd.DataFrame(recall_rows)
st.dataframe(df_recall, use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 4 — Latency cost of improvement
# ─────────────────────────────────────────────

st.subheader("4. Latency Cost of Improvement (per Frame)")
st.caption("Inference time normalised by frames sampled — the only fair comparison since clips vary in duration.")

df["ms_per_frame"] = df["latency_ms"] / df["frames_sampled"].replace(0, np.nan)

lat_rows = []
for base_name, imp_name, label in PAIRS:
    base_g = df[df["model"] == base_name]
    imp_g  = df[df["model"] == imp_name]
    base_lpf = round(base_g["ms_per_frame"].mean(), 2)
    imp_lpf  = round(imp_g["ms_per_frame"].mean(), 2)
    lat_rows.append({
        "Model Pair":        label,
        "Baseline ms/frame": base_lpf,
        "Improved ms/frame": imp_lpf,
        "Δ ms/frame":        round(imp_lpf - base_lpf, 2),
    })

df_lat = pd.DataFrame(lat_rows)
st.dataframe(df_lat, use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 5 — Summary verdict
# ─────────────────────────────────────────────

st.subheader("5. Summary")

for _, row in df_comp.iterrows():
    acc_delta = row["Δ Accuracy"]
    lat_delta = row["Δ ms/frame"]
    verdict_str = row["Verdict"]
    cost = "with no latency cost" if abs(lat_delta) < 5 else f"but adds {lat_delta:+.2f} ms/frame"
    st.write(f"**{row['Model Pair']}:** {verdict_str} (Δ acc = {acc_delta:+.4f}, Δ F1 = {row['Δ F1_macro']:+.4f}) — {cost}")

st.info("""
**Interpretation guide:**
- If improvements show Δ ≈ 0 across all pairs, the confidence-weighting is not adding signal —
  likely because InsightFace det_scores are uniformly high across frames (good lighting, frontal faces).
- If improvements help for old but not young, the weighting is compensating for pose variation
  that is more common in elderly participants.
- A positive Δ accuracy with added ms/frame = worthwhile trade-off only if deployed in non-real-time.
- A zero Δ accuracy with added ms/frame = the improvement is not worth the cost for deployment.
""")

# ─────────────────────────────────────────────
# Downloads
# ─────────────────────────────────────────────

st.header("Download")
col1, col2, col3 = st.columns(3)
col1.download_button("Overall Comparison",    df_comp.to_csv(index=False).encode(),   "improvement_overall.csv",    "text/csv")
col2.download_button("Age Group Comparison",  df_age.to_csv(index=False).encode(),    "improvement_age.csv",        "text/csv")
col3.download_button("Per-Class Recall",      df_recall.to_csv(index=False).encode(), "improvement_per_class.csv",  "text/csv")