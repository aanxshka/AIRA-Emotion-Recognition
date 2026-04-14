import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, classification_report
)

st.set_page_config(page_title="AI vs Real Analysis", layout="wide")

st.title("AI-Generated vs Real Elderly — Data Suitability Analysis")

st.write("""
This page analyses whether AI-generated elderly videos are a suitable supplement
for real elderly data by comparing model performance across three groups:

- **Real Young** — real young participants
- **Real Old** — real elderly participants (Amah, Popo, AnushkaGrandma)
- **AI Old** — AI-generated elderly participants (Japanese, Malay, Chinese, Indian, Korean, Asian)

**Research question:** Is the model significantly better or worse at decoding emotion
on AI-generated elderly faces compared to real elderly faces?
""")

# ─────────────────────────────────────────────
# AI person list — hardcoded
# ─────────────────────────────────────────────

AI_OLD_PERSONS = {"japanese", "malay", "chinese", "indian", "korean", "asian"}

VALENCE_LABELS = ["positive", "neutral", "negative"]


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────

st.subheader("Load Results")

# Try auto-loading from Desktop first
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

st.write(f"Loaded **{len(df)} rows** across **{df['model'].nunique()} models**")

# ─────────────────────────────────────────────
# Assign group labels
# ─────────────────────────────────────────────

def assign_group(row):
    if row["age_group"] == "young":
        return "Real Young"
    elif row["person"] in AI_OLD_PERSONS:
        return "AI Old"
    else:
        return "Real Old"

df["group"] = df.apply(assign_group, axis=1)

group_counts = df[df["model"] == df["model"].iloc[0]]["group"].value_counts()
st.write(f"**Clips per group (per model):** Real Young = {group_counts.get('Real Young',0)} | Real Old = {group_counts.get('Real Old',0)} | AI Old = {group_counts.get('AI Old',0)}")


# ─────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────

GROUPS = ["Real Young", "Real Old", "AI Old"]

def safe_kappa(y_true, y_pred):
    try:
        return round(cohen_kappa_score(y_true, y_pred), 4)
    except Exception:
        return None

def group_metrics(group_df):
    y_true = group_df["ground_truth_valence"]
    y_pred = group_df["predicted_valence"]
    return {
        "Accuracy":   round(accuracy_score(y_true, y_pred), 4),
        "F1_macro":   round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Kappa":      safe_kappa(y_true, y_pred),
        "Clips":      len(group_df),
    }


# ─────────────────────────────────────────────
# 1. ACCURACY BY GROUP PER MODEL
# ─────────────────────────────────────────────

st.subheader("1. Accuracy by Group per Model")
st.caption("Shows whether models perform differently on Real Young, Real Old, and AI Old clips.")

acc_rows = []
for model_name, mdf in df.groupby("model"):
    row = {"Model": model_name}
    for grp in GROUPS:
        gdf = mdf[mdf["group"] == grp]
        if len(gdf) > 0:
            row[f"Acc_{grp}"] = round(accuracy_score(gdf["ground_truth_valence"], gdf["predicted_valence"]), 4)
        else:
            row[f"Acc_{grp}"] = None
    # Gap: AI Old vs Real Old
    if row.get("Acc_AI Old") is not None and row.get("Acc_Real Old") is not None:
        row["Gap (AI - Real Old)"] = round(row["Acc_AI Old"] - row["Acc_Real Old"], 4)
    else:
        row["Gap (AI - Real Old)"] = None
    acc_rows.append(row)

df_acc = pd.DataFrame(acc_rows)
st.dataframe(df_acc, use_container_width=True)

# Highlight the gap direction
avg_gap = df_acc["Gap (AI - Real Old)"].dropna().mean()
if avg_gap > 0.05:
    st.info(f"📈 On average, models are **{round(avg_gap*100,1)}% more accurate** on AI Old than Real Old.")
elif avg_gap < -0.05:
    st.warning(f"📉 On average, models are **{round(abs(avg_gap)*100,1)}% less accurate** on AI Old than Real Old.")
else:
    st.success(f"✅ Accuracy gap between AI Old and Real Old is small ({round(avg_gap*100,1)}%) — AI videos appear representative.")


# ─────────────────────────────────────────────
# 2. F1 MACRO + KAPPA BY GROUP PER MODEL
# ─────────────────────────────────────────────

st.subheader("2. F1 Macro & Kappa by Group per Model")
st.caption("F1 macro treats all three classes equally. Kappa measures agreement beyond chance.")

fk_rows = []
for model_name, mdf in df.groupby("model"):
    row = {"Model": model_name}
    for grp in GROUPS:
        gdf = mdf[mdf["group"] == grp]
        if len(gdf) > 0:
            row[f"F1_{grp}"]    = round(f1_score(gdf["ground_truth_valence"], gdf["predicted_valence"], average="macro", zero_division=0), 4)
            row[f"Kappa_{grp}"] = safe_kappa(gdf["ground_truth_valence"], gdf["predicted_valence"])
        else:
            row[f"F1_{grp}"]    = None
            row[f"Kappa_{grp}"] = None
    fk_rows.append(row)

df_fk = pd.DataFrame(fk_rows)
st.dataframe(df_fk, use_container_width=True)


# ─────────────────────────────────────────────
# 3. PER-CLASS RECALL BY GROUP PER MODEL
# ─────────────────────────────────────────────

st.subheader("3. Per-Class Recall by Group per Model")
st.caption("Recall per emotion class — did the model catch each emotion across each group?")

recall_rows = []
for model_name, mdf in df.groupby("model"):
    for grp in GROUPS:
        gdf = mdf[mdf["group"] == grp]
        if len(gdf) == 0:
            continue
        report = classification_report(
            gdf["ground_truth_valence"], gdf["predicted_valence"],
            labels=VALENCE_LABELS, zero_division=0, output_dict=True
        )
        for cls in VALENCE_LABELS:
            r = report.get(cls, {})
            recall_rows.append({
                "Model":   model_name,
                "Group":   grp,
                "Class":   cls,
                "Recall":  round(r.get("recall", 0), 4),
                "F1":      round(r.get("f1-score", 0), 4),
                "Support": int(r.get("support", 0)),
            })

df_recall = pd.DataFrame(recall_rows)
st.dataframe(df_recall, use_container_width=True)


# ─────────────────────────────────────────────
# 4. SUMMARY: AI SUITABILITY VERDICT PER MODEL
# ─────────────────────────────────────────────

st.subheader("4. AI Suitability Verdict per Model")
st.caption("""
A model's performance gap between Real Old and AI Old tells us whether AI videos 
behave like real elderly data. A small gap (< 10%) suggests AI videos are 
a reasonable supplement. A large gap suggests the AI faces differ in ways 
that confuse the model.
""")

verdict_rows = []
for model_name, mdf in df.groupby("model"):
    real_old_df = mdf[mdf["group"] == "Real Old"]
    ai_old_df   = mdf[mdf["group"] == "AI Old"]

    if len(real_old_df) == 0 or len(ai_old_df) == 0:
        continue

    real_acc = accuracy_score(real_old_df["ground_truth_valence"], real_old_df["predicted_valence"])
    ai_acc   = accuracy_score(ai_old_df["ground_truth_valence"],   ai_old_df["predicted_valence"])
    gap      = ai_acc - real_acc

    real_f1  = f1_score(real_old_df["ground_truth_valence"], real_old_df["predicted_valence"], average="macro", zero_division=0)
    ai_f1    = f1_score(ai_old_df["ground_truth_valence"],   ai_old_df["predicted_valence"],   average="macro", zero_division=0)
    f1_gap   = ai_f1 - real_f1

    if abs(gap) <= 0.10:
        verdict = "✅ Representative"
    elif gap > 0.10:
        verdict = "⚠️ AI easier than real"
    else:
        verdict = "❌ AI harder than real"

    verdict_rows.append({
        "Model":           model_name,
        "Real Old Acc":    round(real_acc, 4),
        "AI Old Acc":      round(ai_acc, 4),
        "Acc Gap":         round(gap, 4),
        "Real Old F1":     round(real_f1, 4),
        "AI Old F1":       round(ai_f1, 4),
        "F1 Gap":          round(f1_gap, 4),
        "Verdict":         verdict,
    })

df_verdict = pd.DataFrame(verdict_rows)
st.dataframe(df_verdict, use_container_width=True)

# Overall summary
rep   = (df_verdict["Verdict"] == "✅ Representative").sum()
easy  = (df_verdict["Verdict"] == "⚠️ AI easier than real").sum()
hard  = (df_verdict["Verdict"] == "❌ AI harder than real").sum()
total = len(df_verdict)

st.write(f"""
**Across all {total} models:**
- ✅ Representative (gap ≤ 10%): **{rep}** models
- ⚠️ AI easier than real (gap > 10%): **{easy}** models  
- ❌ AI harder than real (gap < -10%): **{hard}** models
""")


# ─────────────────────────────────────────────
# 5. DOWNLOADS
# ─────────────────────────────────────────────

st.subheader("Download Analysis")

col1, col2, col3, col4 = st.columns(4)
col1.download_button("Accuracy by Group",    df_acc.to_csv(index=False).encode(),    "ai_accuracy_by_group.csv",    "text/csv")
col2.download_button("F1 & Kappa by Group",  df_fk.to_csv(index=False).encode(),     "ai_f1_kappa_by_group.csv",    "text/csv")
col3.download_button("Per-Class Recall",     df_recall.to_csv(index=False).encode(), "ai_per_class_recall.csv",     "text/csv")
col4.download_button("Suitability Verdict",  df_verdict.to_csv(index=False).encode(),"ai_suitability_verdict.csv",  "text/csv")