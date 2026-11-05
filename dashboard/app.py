"""Credit-card early-warning dashboard.

Tab 1 - Account review: pick a held-out test account, edit its 6-month history
        (what-if), see the probability of default, its early-warning band and the
        SHAP reasons behind the score.
Tab 2 - Portfolio view: test-set metrics, band summary, decile/lift table,
        calibration.

Run from the repository root:  python -m streamlit run dashboard/app.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import MODELS_DIR, RAW_CSV, REPORTS_DIR  # noqa: E402
from src.feature_store import build_features  # noqa: E402
from src.model import assign_band  # noqa: E402

st.set_page_config(page_title="Credit Card Early-Warning System", layout="wide")
st.title("Credit Card Delinquency Early-Warning System")
st.caption("Real data: UCI 'Default of Credit Card Clients' (Taiwan, 2005, 30,000 accounts). "
           "Predicts whether an existing cardholder defaults on next month's payment.")

MONTHS = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
STATUS_HELP = "-2 no usage, -1 paid in full, 0 minimum paid, 1-8 = months overdue"


def ensure_trained() -> None:
    """On a fresh deployment (e.g. Streamlit Community Cloud) model files are not
    in git, so run the full pipeline once: validate data -> features -> train."""
    if (MODELS_DIR / "lgbm_model.pkl").exists() and (MODELS_DIR / "metadata.pkl").exists():
        return
    from src import data, feature_store, model as model_step
    with st.spinner("First start: validating data and training the model (about 20-40 seconds)..."):
        data.main()
        feature_store.main()
        model_step.main(fast=True)


@st.cache_resource
def load_artifacts():
    ensure_trained()
    model = joblib.load(MODELS_DIR / "lgbm_model.pkl")
    meta = joblib.load(MODELS_DIR / "metadata.pkl")
    explainer = shap.TreeExplainer(model)
    raw = pd.read_csv(RAW_CSV).set_index("ID")
    return model, meta, explainer, raw


model, meta, explainer, raw = load_artifacts()

features, cut = meta["features"], meta["band_cutoffs"]
BAND_COLOUR = {"RED": "🔴", "AMBER": "🟠", "GREEN": "🟢"}

tab_account, tab_portfolio = st.tabs(["Account review", "Portfolio view"])

with tab_account:
    acc_id = st.selectbox("Held-out test account ID", meta["test_ids"][:500])
    row = raw.loc[acc_id]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Profile")
        limit = st.number_input("Credit limit (NT$)", 10_000, 1_000_000, int(row["LIMIT_BAL"]), step=10_000)
        age = st.number_input("Age", 21, 79, int(row["AGE"]))
        edu_now = int(row["EDUCATION"]) if row["EDUCATION"] in (1, 2, 3) else 4
        edu = st.selectbox("Education", [1, 2, 3, 4], index=edu_now - 1,
                           format_func={1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}.get)
        actual = int(row["default.payment.next.month"])
        st.write(f"Actual outcome next month: **{'DEFAULT' if actual else 'No default'}**")
    with c2:
        st.subheader("Last 6 months (editable for what-if)")
        hist = pd.DataFrame({
            "month": MONTHS,
            "repayment_status": [int(row[c]) for c in PAY_COLS],
            "bill_amount": [int(row[f"BILL_AMT{i}"]) for i in range(1, 7)],
            "amount_paid": [int(row[f"PAY_AMT{i}"]) for i in range(1, 7)],
        })
        hist = st.data_editor(hist, disabled=["month"], hide_index=True, key=f"hist_{acc_id}",
                              column_config={"repayment_status": st.column_config.NumberColumn(
                                  min_value=-2, max_value=8, step=1, help=STATUS_HELP)})

    # Rebuild a raw-format row and push it through the SAME SQL used in training.
    rec = {"ID": acc_id, "LIMIT_BAL": limit, "SEX": row["SEX"], "EDUCATION": edu,
           "MARRIAGE": row["MARRIAGE"], "AGE": age}
    for i, c in enumerate(PAY_COLS):
        rec[c] = int(hist.loc[i, "repayment_status"])
    for i in range(6):
        rec[f"BILL_AMT{i + 1}"] = float(hist.loc[i, "bill_amount"])
        rec[f"PAY_AMT{i + 1}"] = float(hist.loc[i, "amount_paid"])
    X = build_features(pd.DataFrame([rec]))[features]

    pd_score = float(model.predict_proba(X)[0, 1])
    band = assign_band([pd_score], cut)[0]

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Probability of default (next month)", f"{pd_score:.1%}")
    m2.metric("Early-warning band", f"{BAND_COLOUR[band]} {band}")
    m3.metric("Portfolio average PD", f"{meta['train_scores'].mean():.1%}")
    action = {"RED": "Immediate collections contact, freeze limit increases, consider limit reduction.",
              "AMBER": "Add to watch-list: payment reminder / SMS nudge before due date.",
              "GREEN": "No action; business as usual."}[band]
    st.info(f"Suggested action: {action}")

    st.subheader("Why this score? (SHAP, log-odds scale)")
    sv = explainer(X)
    fig = plt.figure()
    shap.plots.waterfall(sv[0], max_display=8, show=False)
    st.pyplot(fig, clear_figure=True)
    with st.expander("Engineered features sent to the model"):
        st.dataframe(X.T.rename(columns={0: "value"}))

with tab_portfolio:
    m = json.loads((REPORTS_DIR / "metrics.json").read_text())
    st.subheader("Hold-out test set performance (6,000 accounts)")
    perf = pd.DataFrame({k: m[k]["test"] for k in ("logistic_regression", "lightgbm")}).T
    perf["cv_auc (5-fold)"] = [f"{m[k]['cv_auc_mean']:.3f} ± {m[k]['cv_auc_std']:.3f}"
                               for k in ("logistic_regression", "lightgbm")]
    st.dataframe(perf)
    st.subheader("Early-warning bands")
    st.dataframe(pd.read_csv(REPORTS_DIR / "early_warning_bands.csv"), hide_index=True)
    st.subheader("Decile / lift table (decile 1 = riskiest)")
    st.dataframe(pd.read_csv(REPORTS_DIR / "decile_table.csv"), hide_index=True)
    st.subheader("Calibration (predicted vs observed default rate)")
    cal = pd.read_csv(REPORTS_DIR / "calibration_table.csv")
    st.line_chart(cal[["predicted_pd", "observed_rate"]])
