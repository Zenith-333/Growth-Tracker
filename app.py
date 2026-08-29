import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Child Growth Status Checker",
    page_icon="🧒",
    layout="centered",
)

MODEL_DIR = "models"
FEATURE_COLUMNS = [
    "Age_in_Months",
    "Sex",
    "Weight_kg",
    "Height_cm",
    "Vaccinated",
    "4ps_Beneficiary",
]

# Human-readable info + severity styling for each possible predicted label.
# "success" = green, "info" = blue, "warning" = orange, "error" = red.
HFA_INFO = {
    "Normal":            ("success", "Height-for-age is within the normal range."),
    "Tall":              ("info",    "Height-for-age is above the normal range."),
    "Stunted":           ("warning", "Signs of stunting (low height-for-age). Nutritional follow-up recommended."),
    "Severely Stunted":  ("error",   "Severe stunting detected. Prompt referral to a health worker is recommended."),
}

BMI_INFO = {
    "Normal":            ("success", "Weight status is within the normal range."),
    "Overweight":        ("warning", "Weight-for-height is above normal."),
    "Obese":             ("error",   "Weight-for-height indicates obesity. Follow-up recommended."),
    "Wasted":            ("warning", "Signs of wasting (low weight-for-height). Nutritional follow-up recommended."),
    "Severely Wasted":   ("error",   "Severe wasting detected. Prompt referral to a health worker is recommended."),
}


# ----------------------------------------------------------------------
# Load models (cached so this only runs once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    artifacts = {
        "feature_encoders": joblib.load(f"{MODEL_DIR}/feature_encoders.pkl"),
        "hfa_model": joblib.load(f"{MODEL_DIR}/hfa_model.pkl"),
        "hfa_scaler": joblib.load(f"{MODEL_DIR}/hfa_scaler.pkl"),
        "hfa_label_encoder": joblib.load(f"{MODEL_DIR}/hfa_label_encoder.pkl"),
        "bmi_model": joblib.load(f"{MODEL_DIR}/bmi_model.pkl"),
        "bmi_scaler": joblib.load(f"{MODEL_DIR}/bmi_scaler.pkl"),
        "bmi_label_encoder": joblib.load(f"{MODEL_DIR}/bmi_label_encoder.pkl"),
    }
    return artifacts


def build_feature_row(artifacts, age_months, sex, weight_kg, height_cm, vaccinated, ps4_beneficiary):
    """Encode raw form inputs into the exact row shape the scalers/models expect."""
    fe = artifacts["feature_encoders"]
    row = {
        "Age_in_Months": age_months,
        "Sex": fe["sex"].transform([sex])[0],
        "Weight_kg": weight_kg,
        "Height_cm": height_cm,
        "Vaccinated": fe["vaccinated"].transform([vaccinated])[0],
        "4ps_Beneficiary": fe["4ps"].transform([ps4_beneficiary])[0],
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict(artifacts, row_df):
    hfa_scaled = artifacts["hfa_scaler"].transform(row_df)
    hfa_pred = artifacts["hfa_label_encoder"].inverse_transform(
        artifacts["hfa_model"].predict(hfa_scaled)
    )[0]
    hfa_proba = artifacts["hfa_model"].predict_proba(hfa_scaled)[0].max()

    bmi_scaled = artifacts["bmi_scaler"].transform(row_df)
    bmi_pred = artifacts["bmi_label_encoder"].inverse_transform(
        artifacts["bmi_model"].predict(bmi_scaled)
    )[0]
    bmi_proba = artifacts["bmi_model"].predict_proba(bmi_scaled)[0].max()

    return hfa_pred, hfa_proba, bmi_pred, bmi_proba


def show_result(label, confidence, info_map, title):
    style, description = info_map.get(label, ("info", ""))
    alert_fn = {"success": st.success, "info": st.info, "warning": st.warning, "error": st.error}[style]
    st.markdown(f"**{title}: {label}**")
    alert_fn(f"{description}  \nModel confidence: {confidence:.0%}")


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🧒 Child Growth Status Checker")
st.caption(
    "Estimates height-for-age (stunting) and weight-for-height (BMI) status "
    "from basic growth-monitoring data."
)

try:
    artifacts = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Could not find the model files. Make sure the `models/` folder with all "
        f"seven `.pkl` files is in the same repo as `app.py`.\n\nDetails: {e}"
    )
    st.stop()

with st.form("growth_form"):
    col1, col2 = st.columns(2)
    with col1:
        age_months = st.number_input("Age (months)", min_value=0, max_value=71, value=24, step=1)
        weight_kg = st.number_input("Weight (kg)", min_value=1.0, max_value=40.0, value=11.5, step=0.1, format="%.1f")
        sex = st.selectbox("Sex", options=["Male", "Female"])
    with col2:
        height_cm = st.number_input("Height (cm)", min_value=30.0, max_value=130.0, value=85.0, step=0.1, format="%.1f")
        vaccinated = st.selectbox("Fully vaccinated?", options=["yes", "no"])
        ps4_beneficiary = st.selectbox("4Ps beneficiary?", options=["yes", "no"])

    submitted = st.form_submit_button("Check status", use_container_width=True)

if submitted:
    row_df = build_feature_row(artifacts, age_months, sex, weight_kg, height_cm, vaccinated, ps4_beneficiary)
    hfa_pred, hfa_conf, bmi_pred, bmi_conf = predict(artifacts, row_df)

    st.divider()
    st.subheader("Results")
    r1, r2 = st.columns(2)
    with r1:
        show_result(hfa_pred, hfa_conf, HFA_INFO, "Height-for-age")
    with r2:
        show_result(bmi_pred, bmi_conf, BMI_INFO, "Weight-for-height (BMI)")

    st.caption(
        "This tool provides an automated estimate only and is not a substitute for "
        "assessment by a qualified health worker."
    )

st.divider()
with st.expander("About this tool"):
    st.write(
        "This app uses two scikit-learn models trained on child growth-monitoring "
        "data: a logistic regression model for height-for-age (stunting) status, "
        "and a decision tree model for weight-for-height (BMI) status. Inputs are "
        "encoded and scaled with the same preprocessing used during training."
    )
