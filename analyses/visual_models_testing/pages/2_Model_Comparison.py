import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

st.set_page_config(page_title="Model Comparison", layout="wide")

st.title("Emotion Model Comparison Dashboard")

st.write(
"""
Upload one or more result CSV files generated from model evaluation runs.
The dashboard will automatically merge and compare the models.
"""
)

# -------------------------------------------------
# Upload multiple CSV files
# -------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload result CSV files",
    type="csv",
    accept_multiple_files=True
)

if uploaded_files:

    dfs = []

    for file in uploaded_files:
        df = pd.read_csv(file)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    st.success(f"Loaded {len(df)} predictions from {len(uploaded_files)} file(s)")

    # -------------------------------------------------
    # Metric helper
    # -------------------------------------------------

    def compute_metrics(group):

        y_true = group["ground_truth_valence"]
        y_pred = group["predicted_valence"]

        return pd.Series({
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "Clips": len(group)
        })

    # -------------------------------------------------
    # OVERALL MODEL PERFORMANCE
    # -------------------------------------------------

    st.header("Overall Model Performance")

    overall = df.groupby("model").apply(compute_metrics).reset_index()

    st.dataframe(overall, use_container_width=True)

    fig, ax = plt.subplots()

    sns.barplot(data=overall, x="model", y="Accuracy", ax=ax)

    ax.set_title("Accuracy per Model")

    st.pyplot(fig)

    # -------------------------------------------------
    # AGE GROUP PERFORMANCE
    # -------------------------------------------------

    st.header("Performance by Age Group")

    age_metrics = df.groupby(["model", "age_group"]).apply(compute_metrics).reset_index()

    st.dataframe(age_metrics, use_container_width=True)

    fig, ax = plt.subplots()

    sns.barplot(data=age_metrics, x="age_group", y="Accuracy", hue="model", ax=ax)

    ax.set_title("Accuracy by Age Group")

    st.pyplot(fig)

    # -------------------------------------------------
    # PERSON PERFORMANCE
    # -------------------------------------------------

    st.header("Performance by Person")

    person_metrics = df.groupby(["person", "model"]).apply(compute_metrics).reset_index()

    st.dataframe(person_metrics, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10,5))

    sns.barplot(data=person_metrics, x="person", y="Accuracy", hue="model", ax=ax)

    ax.set_title("Accuracy per Person")

    st.pyplot(fig)

    # -------------------------------------------------
    # MODEL SPEED
    # -------------------------------------------------

    st.header("Model Speed Comparison")

    speed = df.groupby("model")["latency_ms"].mean().reset_index()

    st.dataframe(speed)

    fig, ax = plt.subplots()

    sns.barplot(data=speed, x="model", y="latency_ms", ax=ax)

    ax.set_title("Average Latency per Model (ms)")

    st.pyplot(fig)

    # -------------------------------------------------
    # AGE BIAS ANALYSIS
    # -------------------------------------------------

    st.header("Age Bias Analysis")

    age_accuracy = (
        df.groupby(["model", "age_group"])
        .apply(lambda g: accuracy_score(
            g["ground_truth_valence"],
            g["predicted_valence"]
        ))
        .reset_index(name="accuracy")
    )

    pivot = age_accuracy.pivot(
        index="model",
        columns="age_group",
        values="accuracy"
    ).reset_index()

    pivot["Age Bias Gap"] = abs(pivot["young"] - pivot["old"])

    pivot.rename(columns={
        "young": "Accuracy Young",
        "old": "Accuracy Old"
    }, inplace=True)

    # Bias label
    pivot["Bias Level"] = pivot["Age Bias Gap"].apply(
        lambda x: "Low" if x < 0.05 else "Moderate" if x < 0.15 else "High"
    )

    st.dataframe(pivot, use_container_width=True)

    # Age accuracy chart
    fig, ax = plt.subplots()

    pivot_plot = pivot.melt(
        id_vars="model",
        value_vars=["Accuracy Young", "Accuracy Old"],
        var_name="Age Group",
        value_name="Accuracy"
    )

    sns.barplot(
        data=pivot_plot,
        x="model",
        y="Accuracy",
        hue="Age Group",
        ax=ax
    )

    ax.set_title("Accuracy by Age Group")

    st.pyplot(fig)

    # -------------------------------------------------
    # CONFUSION MATRIX
    # -------------------------------------------------

    st.header("Confusion Matrix")

    selected_model = st.selectbox(
        "Select model",
        df["model"].unique()
    )

    model_df = df[df["model"] == selected_model]

    cm = confusion_matrix(
        model_df["ground_truth_valence"],
        model_df["predicted_valence"],
        labels=["positive", "neutral", "negative"]
    )

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["positive","neutral","negative"],
        yticklabels=["positive","neutral","negative"],
        ax=ax
    )

    ax.set_title(f"Confusion Matrix — {selected_model}")

    st.pyplot(fig)

    # -------------------------------------------------
    # RAW DATA VIEW
    # -------------------------------------------------

    st.header("Combined Raw Data")

    st.dataframe(df, use_container_width=True)