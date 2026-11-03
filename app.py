
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

with open(BASE_DIR / "metadata.json", "r") as f:
    metadata = json.load(f)

FEATURES = metadata["feature_columns"]
TARGET = metadata["target_column"]
MODEL_FILES = metadata["model_files"]

st.title("Dry Bean Classification Dashboard")
st.write(
    "Compare five trained machine learning classifiers "
    "on an uploaded test dataset."
)

st.info(
    "Upload labeled test data containing all 16 input features "
    "and the target column. The app uses saved models and "
    "does not retrain them."
)

uploaded_file = st.file_uploader(
    "Upload test data (CSV)",
    type=["csv"]
)

model_name = st.selectbox(
    "Select a classification model",
    list(MODEL_FILES.keys())
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")
        st.write("Rows:", data.shape[0])
        st.write("Columns:", data.shape[1])
        st.dataframe(data.head())

        required_columns = FEATURES + [TARGET]
        missing_columns = [
            col for col in required_columns
            if col not in data.columns
        ]

        if missing_columns:
            st.error(
                "The CSV is missing required columns: "
                + ", ".join(missing_columns)
            )
            st.stop()

        if data.empty:
            st.error("The uploaded CSV is empty.")
            st.stop()

        X_uploaded = data[FEATURES].copy()
        y_true = data[TARGET].astype(str).str.strip()

        if X_uploaded.isnull().any().any():
            st.error("Input features contain missing values.")
            st.stop()

        if y_true.isnull().any():
            st.error("Target contains missing values.")
            st.stop()

        for col in FEATURES:
            X_uploaded[col] = pd.to_numeric(
                X_uploaded[col],
                errors="raise"
            )

        if not all(
            pd.api.types.is_numeric_dtype(X_uploaded[col])
            for col in FEATURES
        ):
            st.error("All input features must be numeric.")
            st.stop()

        model_path = MODEL_DIR / MODEL_FILES[model_name]
        model = joblib.load(model_path)

        unseen_classes = set(y_true.unique()) - set(model.classes_)

        if unseen_classes:
            st.error(
                "The uploaded target contains classes not known "
                "to the selected model: "
                + ", ".join(sorted(unseen_classes))
            )
            st.stop()

        y_pred = model.predict(X_uploaded)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        )
        recall = recall_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        )
        mcc = matthews_corrcoef(y_true, y_pred)

        y_prob = model.predict_proba(X_uploaded)

        try:
            auc = roc_auc_score(
                y_true,
                y_prob,
                labels=model.classes_,
                multi_class="ovr",
                average="weighted"
            )
        except ValueError:
            auc = None

        st.subheader("Evaluation Results")
        st.write("Selected model:", model_name)

        c1, c2, c3 = st.columns(3)

        c1.metric("Accuracy", f"{accuracy:.4f}")
        c2.metric(
            "AUC",
            f"{auc:.4f}" if auc is not None else "Unavailable"
        )
        c3.metric("Precision", f"{precision:.4f}")

        c4, c5, c6 = st.columns(3)

        c4.metric("Recall", f"{recall:.4f}")
        c5.metric("F1 Score", f"{f1:.4f}")
        c6.metric("MCC", f"{mcc:.4f}")

        st.subheader("Classification Report")

        report = classification_report(
            y_true,
            y_pred,
            zero_division=0,
            output_dict=True
        )

        st.dataframe(pd.DataFrame(report).T)

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=model.classes_
        )

        fig, ax = plt.subplots(figsize=(9, 7))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
            ax=ax
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title(f"Confusion Matrix: {model_name}")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Model Comparison")

        comparison_rows = []

        for other_name, filename in MODEL_FILES.items():

            other_model = joblib.load(MODEL_DIR / filename)
            other_pred = other_model.predict(X_uploaded)
            other_prob = other_model.predict_proba(X_uploaded)

            try:
                other_auc = roc_auc_score(
                    y_true,
                    other_prob,
                    labels=other_model.classes_,
                    multi_class="ovr",
                    average="weighted"
                )
            except ValueError:
                other_auc = None

            comparison_rows.append({
                "Model": other_name,
                "Accuracy": accuracy_score(y_true, other_pred),
                "AUC": other_auc,
                "Precision": precision_score(
                    y_true, other_pred,
                    average="weighted",
                    zero_division=0
                ),
                "Recall": recall_score(
                    y_true, other_pred,
                    average="weighted",
                    zero_division=0
                ),
                "F1 Score": f1_score(
                    y_true, other_pred,
                    average="weighted",
                    zero_division=0
                ),
                "MCC": matthews_corrcoef(y_true, other_pred)
            })

        comparison_df = pd.DataFrame(comparison_rows)
        st.dataframe(
            comparison_df.set_index("Model").round(4)
        )

        st.success("Evaluation completed successfully.")

    except Exception as e:
        st.error(f"Could not evaluate the uploaded CSV: {e}")

else:
    st.warning(
        "Upload a labeled test CSV to view evaluation results."
    )
