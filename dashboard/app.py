import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Retail Credit Decision Engine", layout="wide")
st.title("Credit Risk Scoring & Anomaly Detection Portal")
st.markdown("Underwriting decision support tool with SHAP explainability and automated risk flags.")

@st.cache_resource
def load_models():
    model = joblib.load("src/lgbm_credit_model.pkl")
    anomaly = joblib.load("src/anomaly_isolation_forest.pkl")
    features = joblib.load("src/model_features.pkl")
    explainer = joblib.load("src/shap_explainer.pkl")
    return model, anomaly, features, explainer

try:
    model, anomaly_model, feature_cols, explainer = load_models()
except Exception:
    st.warning("Run training scripts first to generate model artifacts before evaluating applicants.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Applicant Financial Details")
    income = st.number_input("Annual Income (INR)", min_value=10000.0, value=500000.0, step=10000.0)
    credit = st.number_input("Loan Amount Requested (INR)", min_value=10000.0, value=1000000.0, step=50000.0)
    annuity = st.number_input("Monthly Annuity / EMI (INR)", min_value=1000.0, value=25000.0, step=1000.0)
    age = st.slider("Applicant Age", 20, 70, 32)
    emp_years = st.slider("Years of Employment", 0, 40, 5)

with col2:
    st.subheader("Bureau & External Risk Ratings")
    ext_1 = st.slider("External Source 1 Score", 0.0, 1.0, 0.5)
    ext_2 = st.slider("External Source 2 Score", 0.0, 1.0, 0.5)
    ext_3 = st.slider("External Source 3 Score", 0.0, 1.0, 0.5)
    is_cash_loan = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])

if st.button("Evaluate Credit Application"):
    c_to_i = credit / income
    d_to_s = annuity / income
    a_to_c = annuity / credit

    input_dict = {
        'AMT_INCOME_TOTAL': income,
        'AMT_CREDIT': credit,
        'AMT_ANNUITY': annuity,
        'credit_to_income_ratio': c_to_i,
        'debt_service_ratio': d_to_s,
        'annuity_to_credit_ratio': a_to_c,
        'age_years': age,
        'employment_years': emp_years,
        'EXT_SOURCE_1': ext_1,
        'EXT_SOURCE_2': ext_2,
        'EXT_SOURCE_3': ext_3,
        'NAME_CONTRACT_TYPE_Revolving loans': 1 if is_cash_loan == "Revolving loans" else 0
    }
    
    input_df = pd.DataFrame([input_dict])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]

    pd_score = model.predict_proba(input_df)[0][1]
    anomaly_input = pd.DataFrame([[income, credit, annuity, c_to_i, d_to_s]], 
                                 columns=['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'credit_to_income_ratio', 'debt_service_ratio'])
    is_anomaly = anomaly_model.predict(anomaly_input)[0] == -1

    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Default Probability (PD)", f"{pd_score * 100:.2f}%")
        if pd_score > 0.40:
            st.error("Decision: High Risk - Decline / Underwriter Review")
        else:
            st.success("Decision: Low/Moderate Risk - Pre-Approval Eligible")
    with res_col2:
        if is_anomaly:
            st.warning("Behavioral Anomaly Detected: Irregular loan-to-income distribution.")
        else:
            st.info("Applicant distribution matches standard retail profile.")

    st.subheader("Underwriter Explanation (SHAP Waterfall)")
    shap_values = explainer(input_df)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=7, show=False)
    st.pyplot(fig)