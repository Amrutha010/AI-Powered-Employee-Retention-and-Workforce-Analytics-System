from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.config import DEFAULT_DATA_PATH, DEFAULT_MODEL_DIR, TARGET_COLUMN
from src.data_utils import split_features_target

st.set_page_config(page_title="Employee Attrition Analytics", page_icon="📊", layout="wide")

MODEL_PATH = DEFAULT_MODEL_DIR / "best_model.pkl"
METRICS_PATH = DEFAULT_MODEL_DIR / "metrics.json"
FEATURE_COLUMNS_PATH = DEFAULT_MODEL_DIR / "feature_columns.json"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def ensure_input_schema(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    return df[expected_columns]


def plot_overview_charts(df: pd.DataFrame) -> None:
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x=TARGET_COLUMN, palette="Set2", ax=ax)
        ax.set_title("Attrition Distribution")
        st.pyplot(fig)

    with c2:
        if "OverTime" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x="OverTime", hue=TARGET_COLUMN, palette="Set1", ax=ax)
            ax.set_title("OverTime vs Attrition")
            st.pyplot(fig)
        else:
            st.info("'OverTime' column not found for overtime analysis.")

    c3, c4 = st.columns(2)

    with c3:
        if "JobRole" in df.columns:
            role_attrition = (
                df.groupby("JobRole")[TARGET_COLUMN]
                .apply(lambda x: (x == "Yes").mean())
                .sort_values(ascending=False)
                .reset_index(name="AttritionRate")
            )
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.barplot(data=role_attrition, y="JobRole", x="AttritionRate", palette="crest", ax=ax)
            ax.set_title("Attrition Rate by Job Role")
            st.pyplot(fig)
        else:
            st.info("'JobRole' column not found for role-wise analysis.")

    with c4:
        if "YearsAtCompany" in df.columns:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.boxplot(data=df, x=TARGET_COLUMN, y="YearsAtCompany", palette="pastel", ax=ax)
            ax.set_title("Years At Company by Attrition")
            st.pyplot(fig)
        else:
            st.info("'YearsAtCompany' column not found for tenure analysis.")


st.title("AI-Based Employee Attrition Prediction and Analytics System")
st.caption("Predict attrition risk and explore retention insights from HR data.")

with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload IBM HR Attrition CSV", type=["csv"])
    st.markdown("If no file is uploaded, the app will try to use data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
elif DEFAULT_DATA_PATH.exists():
    data = load_data(str(DEFAULT_DATA_PATH))
else:
    st.error("Dataset not found. Upload a CSV file from the sidebar or place it in data/.")
    st.stop()

if not MODEL_PATH.exists() or not METRICS_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
    st.warning("Model artifacts are missing. Run: python train.py --data data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    st.stop()

model = load_model(str(MODEL_PATH))

with METRICS_PATH.open("r", encoding="utf-8") as f:
    report = json.load(f)
with FEATURE_COLUMNS_PATH.open("r", encoding="utf-8") as f:
    schema = json.load(f)
expected_columns = schema["input_columns"]

st.subheader("Model Performance")
metrics_df = pd.DataFrame(report["metrics"]).T
st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)
st.success(f"Best model: {report['best_model']}")

t1, t2, t3 = st.tabs(["Analytics", "Batch Prediction", "Single Employee Prediction"])

with t1:
    st.subheader("Attrition Trends and Insights")
    st.dataframe(data.head(), use_container_width=True)
    plot_overview_charts(data)

with t2:
    st.subheader("Batch Prediction")
    if TARGET_COLUMN in data.columns:
        X_batch, _ = split_features_target(data)
    else:
        X_batch = data.copy()

    X_batch = ensure_input_schema(X_batch.copy(), expected_columns)
    probs = model.predict_proba(X_batch)[:, 1]
    preds = (probs >= 0.5).astype(int)

    result = data.copy()
    result["Attrition_Probability"] = probs
    result["Predicted_Attrition"] = preds
    result["Risk_Label"] = result["Predicted_Attrition"].map({1: "High Risk", 0: "Low Risk"})

    threshold = st.slider("High-risk threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    high_risk = result[result["Attrition_Probability"] >= threshold].sort_values(
        by="Attrition_Probability", ascending=False
    )

    c1, c2 = st.columns(2)
    c1.metric("Employees analyzed", len(result))
    c2.metric("High-risk employees", len(high_risk))

    st.dataframe(high_risk, use_container_width=True)

with t3:
    st.subheader("Single Employee Prediction")
    template = data.drop(columns=[TARGET_COLUMN], errors="ignore").iloc[0:1].copy()
    single_input = {}

    for col in expected_columns:
        if col in template.columns:
            value = template.iloc[0][col]
            if pd.api.types.is_numeric_dtype(data[col]):
                single_input[col] = st.number_input(col, value=float(value))
            else:
                options = sorted(data[col].dropna().astype(str).unique().tolist())
                default_value = str(value) if str(value) in options else options[0]
                single_input[col] = st.selectbox(col, options=options, index=options.index(default_value))
        else:
            single_input[col] = 0

    if st.button("Predict Attrition"):
        single_df = pd.DataFrame([single_input])
        single_df = ensure_input_schema(single_df, expected_columns)
        prob = float(model.predict_proba(single_df)[:, 1][0])
        pred = int(prob >= 0.5)

        st.metric("Attrition Probability", f"{prob:.2%}")
        st.write("Prediction:", "Likely to Leave" if pred == 1 else "Likely to Stay")
