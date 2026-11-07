"""
Water Quality and Potability Prediction - Web interface (Streamlit)

Run from the terminal (with (venv) active):
    streamlit run app.py

Educational use only. This is not a certified drinking-water safety test.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.joblib"

# Same order as FEATURES in preprocessing.py.
# Ranges come from the min and max values of the dataset (Phase 2).
INPUT_FIELDS = [
    {"label": "pH", "key": "ph", "min": 0.1, "max": 14.0, "example": 7.0},
    {"label": "Hardness (mg/L)", "key": "Hardness", "min": 47.0, "max": 325.0, "example": 196.0},
    {"label": "Solids (ppm)", "key": "Solids", "min": 300.0, "max": 62000.0, "example": 20900.0},
    {"label": "Chloramines (ppm)", "key": "Chloramines", "min": 0.3, "max": 14.0, "example": 7.1},
    {"label": "Sulfate (mg/L)", "key": "Sulfate", "min": 129.0, "max": 482.0, "example": 333.0},
    {"label": "Conductivity (uS/cm)", "key": "Conductivity", "min": 180.0, "max": 760.0, "example": 426.0},
    {"label": "Organic Carbon (ppm)", "key": "Organic_carbon", "min": 2.0, "max": 29.0, "example": 14.3},
    {"label": "Trihalomethanes (ug/L)", "key": "Trihalomethanes", "min": 0.7, "max": 125.0, "example": 66.4},
]


@st.cache_resource
def load_bundle():
    """Load the saved model once and reuse it."""
    return joblib.load(MODEL_PATH)


def validate_inputs(raw_values):
    """Check every field. Returns (clean_values, error_messages)."""
    clean, errors = {}, []
    for field in INPUT_FIELDS:
        text = raw_values[field["key"]].strip()
        if text == "":
            errors.append(f"{field['label']}: please enter a value.")
            continue
        try:
            value = float(text)
        except ValueError:
            errors.append(f"{field['label']}: '{text}' is not a number.")
            continue
        if not (field["min"] <= value <= field["max"]):
            errors.append(
                f"{field['label']}: {value} is outside the expected range "
                f"{field['min']} to {field['max']}."
            )
            continue
        clean[field["key"]] = value
    return clean, errors


def main():
    st.set_page_config(page_title="Water Potability Predictor", page_icon="💧")
    st.title("Water Quality and Potability Prediction")
    st.caption("Supervised machine learning (binary classification) | Educational project")

    st.warning(
        "**Educational use only.** This model predicts a label learned from a dataset. "
        "It does NOT certify whether real water is safe to drink. "
        "Use accredited laboratory testing for any drinking-water decision."
    )

    if not MODEL_PATH.exists():
        st.error("best_model.joblib was not found. Run `python water_quality_prediction.py` first.")
        st.stop()

    bundle = load_bundle()
    st.write(f"**Selected model:** {bundle['model_name']}")

    with st.form("water_form"):
        st.subheader("Enter water quality parameters")
        raw_values = {}
        col1, col2 = st.columns(2)
        for i, field in enumerate(INPUT_FIELDS):
            column = col1 if i % 2 == 0 else col2
            raw_values[field["key"]] = column.text_input(
                field["label"],
                placeholder=f"{field['min']} to {field['max']} (e.g. {field['example']})",
            )
        submitted = st.form_submit_button("Predict")

    if submitted:
        values, errors = validate_inputs(raw_values)
        if errors:
            for message in errors:
                st.error(message)
        else:
            X_new = pd.DataFrame([values], columns=bundle["features"])
            pipeline = bundle["pipeline"]
            label = int(pipeline.predict(X_new)[0])
            prob_potable = pipeline.predict_proba(X_new)[0, 1]

            st.subheader("Prediction")
            if label == 1:
                st.success("Predicted class: **Potable (1)**")
            else:
                st.error("Predicted class: **Not Potable (0)**")

            st.metric("Model probability of Potable", f"{prob_potable * 100:.1f}%")
            st.write(f"Model used: **{bundle['model_name']}**")
            st.caption(
                "This probability is the model's estimate learned from the training data. "
                "It is not a laboratory measurement of safety."
            )

    st.divider()
    st.caption("Educational project. Not for real drinking-water certification.")


if __name__ == "__main__":
    main()