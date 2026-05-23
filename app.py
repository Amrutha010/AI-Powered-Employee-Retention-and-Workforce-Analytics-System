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
from train import train_from_dataframe

st.set_page_config(page_title="Employee Attrition Analytics", page_icon="📊", layout="wide")

MODEL_PATH = DEFAULT_MODEL_DIR / "best_model.pkl"
METRICS_PATH = DEFAULT_MODEL_DIR / "metrics.json"
FEATURE_COLUMNS_PATH = DEFAULT_MODEL_DIR / "feature_columns.json"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_model(path: str):
    return joblib.load(path)


def load_artifacts() -> tuple[object, dict, list[str]]:
    with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
        report = json.load(metrics_file)
    with FEATURE_COLUMNS_PATH.open("r", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    model = load_model(str(MODEL_PATH))
    return model, report, schema["input_columns"]


def artifacts_available() -> bool:
    return MODEL_PATH.exists() and METRICS_PATH.exists() and FEATURE_COLUMNS_PATH.exists()


def ensure_input_schema(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    aligned = df.copy()
    for col in expected_columns:
        if col not in aligned.columns:
            aligned[col] = 0
    return aligned.reindex(columns=expected_columns)


def extract_feature_importance(model, top_n: int = 10) -> pd.DataFrame:
    if not hasattr(model, "named_steps"):
        return pd.DataFrame(columns=["feature", "importance"])

    classifier = model.named_steps.get("classifier")
    preprocessor = model.named_steps.get("preprocessor")

    if classifier is None or preprocessor is None:
        return pd.DataFrame(columns=["feature", "importance"])

    if hasattr(classifier, "coef_"):
        importance_values = abs(classifier.coef_[0])
    elif hasattr(classifier, "feature_importances_"):
        importance_values = classifier.feature_importances_
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{index}" for index in range(len(importance_values))]

    importance = (
        pd.DataFrame({"feature": feature_names, "importance": importance_values})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return importance


def render_kpis(df: pd.DataFrame, report: dict) -> None:
    attrition_rate = 0.0
    if TARGET_COLUMN in df.columns:
        attrition_rate = float((df[TARGET_COLUMN] == "Yes").mean() * 100)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees", f"{len(df):,}")
    c2.metric("Attrition rate", f"{attrition_rate:.1f}%")
    c3.metric("Best model", report.get("best_model", "Unknown").replace("_", " ").title())
    c4.metric("Evaluation metrics", f"{len(report.get('metrics', {}))}")


def plot_overview_charts(df: pd.DataFrame) -> None:
    c1, c2 = st.columns(2)

    with c1:
        if TARGET_COLUMN in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x=TARGET_COLUMN, palette="Set2", ax=ax)
            ax.set_title("Attrition Distribution")
            st.pyplot(fig)
        else:
            st.info(f"'{TARGET_COLUMN}' column not found for attrition distribution.")

    with c2:
        if "OverTime" in df.columns and TARGET_COLUMN in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x="OverTime", hue=TARGET_COLUMN, palette="Set1", ax=ax)
            ax.set_title("OverTime vs Attrition")
            st.pyplot(fig)
        else:
            st.info("'OverTime' column not found for overtime analysis.")

    c3, c4 = st.columns(2)

    with c3:
        if "JobRole" in df.columns and TARGET_COLUMN in df.columns:
            role_attrition = (
                df.groupby("JobRole")[TARGET_COLUMN]
                .apply(lambda values: (values == "Yes").mean())
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
        if "YearsAtCompany" in df.columns and TARGET_COLUMN in df.columns:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.boxplot(data=df, x=TARGET_COLUMN, y="YearsAtCompany", palette="pastel", ax=ax)
            ax.set_title("Years At Company by Attrition")
            st.pyplot(fig)
        else:
            st.info("'YearsAtCompany' column not found for tenure analysis.")


st.title("AI-Based Employee Attrition Prediction and Analytics System")
st.caption("Predict attrition risk and explore retention insights from HR data.")

if "last_training_summary" in st.session_state:
    summary = st.session_state.pop("last_training_summary")
    st.success(f"Model trained successfully. Best model: {summary['best_model'].replace('_', ' ').title()}")

with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload IBM HR Attrition CSV", type=["csv"])
    st.caption("If no file is uploaded, the app will try to use the dataset in data/.")
    train_now = st.button("Train / refresh model", use_container_width=True)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data_source = "uploaded"
elif DEFAULT_DATA_PATH.exists():
    data = load_data(str(DEFAULT_DATA_PATH))
    data_source = "local"
else:
    data = None
    data_source = "missing"

if data is None:
    st.error("Dataset not found. Upload a CSV file from the sidebar or place the IBM HR dataset in data/.")
    st.stop()

if train_now or not artifacts_available():
    try:
        with st.spinner("Training models and saving artifacts..."):
            summary = train_from_dataframe(data, DEFAULT_MODEL_DIR)
    except Exception as exc:
        st.error(f"Training failed: {exc}")
        st.stop()

    st.session_state["last_training_summary"] = summary
    st.rerun()

model, report, expected_columns = load_artifacts()

st.subheader("Dataset Summary")
render_kpis(data, report)

metrics_df = pd.DataFrame(report["metrics"]).T.sort_values(by="f1_score", ascending=False)
st.dataframe(metrics_df.round(3), use_container_width=True)

performance_cols = st.columns(2)
with performance_cols[0]:
    st.markdown(f"**Best model:** {report['best_model'].replace('_', ' ').title()}")
with performance_cols[1]:
    importance_df = extract_feature_importance(model)
    if not importance_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis", ax=ax)
        ax.set_title("Top Feature Importance")
        st.pyplot(fig)

tab_dashboard, tab_batch, tab_single, tab_models = st.tabs(
    ["Dashboard", "Batch Scoring", "Single Employee", "Model Performance"]
)

with tab_dashboard:
    st.subheader("Attrition Trends and Insights")
    st.write(f"Data source: {data_source}")
    st.dataframe(data.head(), use_container_width=True)
    plot_overview_charts(data)

    if TARGET_COLUMN in data.columns:
        numeric_cols = data.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            correlation = data[numeric_cols].corr(numeric_only=True)
            sns.heatmap(correlation, cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Numeric Feature Correlation")
            st.pyplot(fig)

with tab_batch:
    st.subheader("Batch Prediction")
    if TARGET_COLUMN in data.columns:
        X_batch, _ = split_features_target(data)
    else:
        X_batch = data.copy()

    threshold = st.slider("High-risk threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    X_batch = ensure_input_schema(X_batch, expected_columns)
    probabilities = model.predict_proba(X_batch)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    result = data.copy()
    result["Attrition_Probability"] = probabilities
    result["Predicted_Attrition"] = predictions
    result["Risk_Label"] = result["Predicted_Attrition"].map({1: "High Risk", 0: "Low Risk"})

    high_risk = result[result["Attrition_Probability"] >= threshold].sort_values(
        by="Attrition_Probability", ascending=False
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Employees analyzed", f"{len(result):,}")
    c2.metric("High-risk employees", f"{len(high_risk):,}")
    c3.metric("Average risk", f"{probabilities.mean():.1%}")

    st.dataframe(high_risk, use_container_width=True)
    st.download_button(
        "Download scored employees",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="employee_attrition_scored.csv",
        mime="text/csv",
        use_container_width=True,
    )

with tab_single:
    st.subheader("Single Employee Prediction")
    template = data.drop(columns=[TARGET_COLUMN], errors="ignore").head(1)

    if template.empty:
        st.info("Upload a dataset with at least one row to build the employee input form.")
    else:
        single_input: dict[str, object] = {}
        with st.form("single_employee_form"):
            form_columns = st.columns(2)
            for index, col in enumerate(expected_columns):
                container = form_columns[index % 2]
                with container:
                    if col not in template.columns:
                        single_input[col] = 0
                        st.caption(f"{col}: defaulted because the uploaded data does not contain this field.")
                        continue

                    series = data[col]
                    if pd.api.types.is_numeric_dtype(series):
                        numeric_values = pd.to_numeric(series, errors="coerce").dropna()
                        default_value = float(numeric_values.median()) if not numeric_values.empty else 0.0
                        min_value = float(numeric_values.min()) if not numeric_values.empty else 0.0
                        max_value = float(numeric_values.max()) if not numeric_values.empty else max(default_value + 1.0, 1.0)
                        step = 1.0 if pd.api.types.is_integer_dtype(series) else 0.1
                        if min_value == max_value:
                            max_value = min_value + step
                        single_input[col] = st.number_input(
                            col,
                            value=default_value,
                            min_value=min_value,
                            max_value=max_value,
                            step=step,
                        )
                    else:
                        options = sorted(series.dropna().astype(str).unique().tolist())
                        if not options:
                            options = ["Unknown"]
                        default_value = str(template.iloc[0][col]) if str(template.iloc[0][col]) in options else options[0]
                        single_input[col] = st.selectbox(col, options=options, index=options.index(default_value))

            submitted = st.form_submit_button("Predict attrition risk")

        if submitted:
            single_df = ensure_input_schema(pd.DataFrame([single_input]), expected_columns)
            probability = float(model.predict_proba(single_df)[:, 1][0])
            predicted_class = int(probability >= threshold)

            c1, c2 = st.columns(2)
            c1.metric("Attrition probability", f"{probability:.2%}")
            c2.metric("Prediction", "Likely to Leave" if predicted_class == 1 else "Likely to Stay")

with tab_models:
    st.subheader("Model Performance")
    st.dataframe(metrics_df.round(3), use_container_width=True)

    performance_chart = metrics_df.reset_index(names="model")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=performance_chart, x="model", y="f1_score", palette="Blues_r", ax=ax)
    ax.set_ylabel("F1 score")
    ax.set_xlabel("Model")
    ax.set_title("Model Comparison by F1 Score")
    st.pyplot(fig)

    st.caption("The model with the highest F1 score is saved as the production artifact.")
