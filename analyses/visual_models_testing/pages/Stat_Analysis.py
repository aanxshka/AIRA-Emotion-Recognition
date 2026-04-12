import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import binom as _binom
from scipy.stats import chi2 as _chi2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix, roc_auc_score,
    classification_report
)

st.set_page_config(page_title="Statistical Analysis", layout="wide")

st.title("Full Statistical Analysis")
st.write("""
Complete breakdown of all agreed metrics:
accuracy, precision, recall, F1 (weighted + macro), Cohen's Kappa, AUC per class,
per-class metrics, per-class by age group, confusion matrices, latency, and McNemar's test.
""")

# ─────────────────────────────────────────────
# McNemar helper (no statsmodels needed)
# ─────────────────────────────────────────────

def mcnemar_test(b, c):
    """
    b = model A correct, model B wrong
    c = model A wrong, model B correct
    Exact binomial when b+c < 25, else chi-squared with continuity correction.
    """
    n = b + c
    if n == 0:
        return 1.0, "identical predictions"
    if n < 25:
        p = 2 * min(float(_binom.cdf(b, n, 0.5)), float(_binom.cdf(c, n, 0.5)))
        return round(p, 4), "exact binomial"
    else:
        chi2_stat = (abs(b - c) - 1) ** 2 / n
        p = 1 - float(_chi2.cdf(chi2_stat, df=1))
        return round(p, 4), "chi-squared"

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────

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

st.write(f"Loaded **{len(df)} rows** — **{df['model'].nunique()} models** — **{df['filename'].nunique()} unique clips**")

VALENCE_LABELS = ["positive", "neutral", "negative"]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def safe_kappa(y_true, y_pred):
    try:
        return round(cohen_kappa_score(y_true, y_pred), 4)
    except Exception:
        return None

def compute_auc(group):
    result = {}
    for cls in VALENCE_LABELS:
        y_bin   = (group["ground_truth_valence"] == cls).astype(int)
        y_score = group[f"score_{cls}"]
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            result[f"AUC_{cls}"] = None
        else:
            try:
                result[f"AUC_{cls}"] = round(roc_auc_score(y_bin, y_score), 4)
            except Exception:
                result[f"AUC_{cls}"] = None
    valid = [v for v in result.values() if v is not None]
    result["AUC_mean"] = round(float(np.mean(valid)), 4) if valid else None
    return result

def overall_metrics(g):
    y_true, y_pred = g["ground_truth_valence"], g["predicted_valence"]
    row = {
        "Accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "Precision":   round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "Recall":      round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "F1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "F1_macro":    round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Kappa":       safe_kappa(y_true, y_pred),
        "n":           len(g),
    }
    row.update(compute_auc(g))
    return row

def per_class(g):
    y_true, y_pred = g["ground_truth_valence"], g["predicted_valence"]
    report = classification_report(y_true, y_pred, labels=VALENCE_LABELS,
                                   zero_division=0, output_dict=True)
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

# ─────────────────────────────────────────────
# SECTION 1 — Overall metrics per model
# ─────────────────────────────────────────────

st.header("1. Overall Metrics per Model")
st.caption("Accuracy · Precision · Recall · F1 weighted · F1 macro · Cohen's Kappa · AUC per class · Avg Latency")

rows = []
for model, g in df.groupby("model"):
    row = {"Model": model}
    row.update(overall_metrics(g))
    g_lpf = g["latency_ms"] / g["frames_sampled"].replace(0, np.nan)
    row["Avg_ms_per_frame"] = round(g_lpf.mean(), 2)
    rows.append(row)

df_overall = pd.DataFrame(rows)
st.dataframe(df_overall, use_container_width=True)

best     = df_overall.loc[df_overall["Accuracy"].idxmax(), "Model"]
best_acc = df_overall["Accuracy"].max()
st.info(f"🏆 Best overall accuracy: **{best}** ({round(best_acc*100,1)}%)")

# ─────────────────────────────────────────────
# SECTION 2 — Per-class metrics per model
# ─────────────────────────────────────────────

st.header("2. Per-Class Metrics per Model")
st.caption("Precision · Recall · F1 · Support broken down by positive / neutral / negative")

pc_rows = []
for model, g in df.groupby("model"):
    pc = per_class(g)
    pc.insert(0, "Model", model)
    pc_rows.append(pc)
df_pc = pd.concat(pc_rows, ignore_index=True)
st.dataframe(df_pc, use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 3 — Per-class metrics by age group
# ─────────────────────────────────────────────

st.header("3. Per-Class Metrics by Age Group per Model")
st.caption("Same breakdown split by young vs old — key for elderly robot context")

pca_rows = []
for (model, age), g in df.groupby(["model", "age_group"]):
    pc = per_class(g)
    pc.insert(0, "Age_Group", age)
    pc.insert(0, "Model", model)
    pca_rows.append(pc)
df_pca = pd.concat(pca_rows, ignore_index=True)
st.dataframe(df_pca, use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 4 — Confusion matrices
# ─────────────────────────────────────────────

st.header("4. Confusion Matrices")
st.caption("Rows = actual label · Columns = predicted label")

for model, g in df.groupby("model"):
    cm = confusion_matrix(g["ground_truth_valence"], g["predicted_valence"],
                          labels=VALENCE_LABELS)
    df_cm = pd.DataFrame(
        cm,
        index   = [f"actual_{l}" for l in VALENCE_LABELS],
        columns = [f"pred_{l}"   for l in VALENCE_LABELS]
    )
    st.write(f"**{model}**")
    st.dataframe(df_cm, use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 5 — Age group breakdown per model
# ─────────────────────────────────────────────

st.header("5. Overall Metrics by Age Group per Model")
st.caption("Full metrics split by young vs old including AUC")

age_rows = []
for (model, age), g in df.groupby(["model", "age_group"]):
    row = {"Model": model, "Age_Group": age}
    row.update(overall_metrics(g))
    age_rows.append(row)
df_age = pd.DataFrame(age_rows)
st.dataframe(df_age, use_container_width=True)

# ─────────────────────────────────────────────
# SECTION 6 — Latency summary
# ─────────────────────────────────────────────

st.header("6. Latency per Frame")
st.caption("""
Inference time normalised by frames sampled — the only fair comparison since clips vary in duration.
All values in **milliseconds per frame**.
""")

df["ms_per_frame"] = df["latency_ms"] / df["frames_sampled"].replace(0, np.nan)

df_lat = (
    df.groupby("model")["ms_per_frame"]
    .agg(Avg_ms_per_frame="mean", Min_ms_per_frame="min",
         Max_ms_per_frame="max", Median_ms_per_frame="median")
    .round(2)
    .reset_index()
    .rename(columns={"model": "Model"})
)
st.dataframe(df_lat, use_container_width=True)
fastest = df_lat.loc[df_lat["Avg_ms_per_frame"].idxmin(), "Model"]
st.caption(f"Fastest model: **{fastest}** ({df_lat['Avg_ms_per_frame'].min()} ms/frame avg)")

# ─────────────────────────────────────────────
# SECTION 7 — McNemar's test
# ─────────────────────────────────────────────

st.header("7. McNemar's Test — Pairwise Model Comparison")
st.caption("""
McNemar's test checks whether two models make **different kinds of errors** on the same clips.
It compares disagreements: clips where model A was right but B was wrong, and vice versa.
p < 0.05 suggests statistically significant differences in error patterns.

⚠️ **Sample size caveat (n=36):** The test is severely underpowered at this sample size.
A non-significant result does NOT mean the models perform equally — it likely means
there is insufficient data to detect a real difference. Treat p-values as directional only.
Exact binomial is used when disagreements < 25; chi-squared with continuity correction otherwise.
""")

pivot = df.pivot_table(
    index="filename", columns="model", values="correct", aggfunc="first"
).reset_index()
model_list = [c for c in pivot.columns if c != "filename"]

mc_rows = []
for i in range(len(model_list)):
    for j in range(i + 1, len(model_list)):
        m1, m2 = model_list[i], model_list[j]
        sub = pivot[["filename", m1, m2]].dropna()
        if len(sub) == 0:
            continue
        a = sub[m1].astype(int).values
        b = sub[m2].astype(int).values
        b_val = int(((a == 1) & (b == 0)).sum())  # A right, B wrong
        c_val = int(((a == 0) & (b == 1)).sum())  # A wrong, B right
        p_val, method = mcnemar_test(b_val, c_val)

        if p_val is not None and p_val < 0.05:
            verdict = "✅ p < 0.05"
        elif p_val == 1.0:
            verdict = "— identical"
        else:
            verdict = "❌ p ≥ 0.05"

        mc_rows.append({
            "Model A":        m1,
            "Model B":        m2,
            "A✓ B✗":          b_val,
            "A✗ B✓":          c_val,
            "Disagreements":  b_val + c_val,
            "p-value":        p_val,
            "Method":         method,
            "Verdict":        verdict,
            "n":              len(sub),
        })

df_mc = pd.DataFrame(mc_rows)
st.dataframe(df_mc, use_container_width=True)

n_sig = (df_mc["Verdict"] == "✅ p < 0.05").sum()
if n_sig > 0:
    st.warning(f"⚠️ {n_sig} pair(s) reach p < 0.05 — but note small sample caveat above.")
else:
    st.info("No pairs reach p < 0.05. Expected with n=36 — underpowered to detect significance even if real differences exist.")

# ─────────────────────────────────────────────
# SECTION 8 — Downloads
# ─────────────────────────────────────────────

st.header("Download Analysis")
col1, col2, col3, col4 = st.columns(4)
col1.download_button("Overall Metrics",     df_overall.to_csv(index=False).encode(), "stat_overall.csv",       "text/csv")
col2.download_button("Per-Class Metrics",   df_pc.to_csv(index=False).encode(),      "stat_per_class.csv",     "text/csv")
col3.download_button("Per-Class by Age",    df_pca.to_csv(index=False).encode(),     "stat_per_class_age.csv", "text/csv")
col4.download_button("Age Group Breakdown", df_age.to_csv(index=False).encode(),     "stat_age_group.csv",     "text/csv")
col5, col6 = st.columns(2)
col5.download_button("Latency per Frame",   df_lat.to_csv(index=False).encode(),     "stat_latency_per_frame.csv", "text/csv")
col6.download_button("McNemar's Test",      df_mc.to_csv(index=False).encode(),      "stat_mcnemar.csv",       "text/csv")