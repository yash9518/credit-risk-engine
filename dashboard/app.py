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

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DEPLOY_DIR, RAW_CSV, REPORTS_DIR  # noqa: E402
from src.feature_store import build_features  # noqa: E402
from src.model import assign_band  # noqa: E402
from src.scorecard import load_card, score_points  # noqa: E402
from src.analysis import LGD  # noqa: E402

st.set_page_config(page_title="Credit Card Early-Warning System", layout="wide")
st.title("Credit Card Delinquency Early-Warning System")
st.caption("Real data: UCI 'Default of Credit Card Clients' (Taiwan, 2005, 30,000 accounts). "
           "Predicts whether an existing cardholder defaults on next month's payment.")

MONTHS = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
STATUS_HELP = "-2 no usage, -1 paid in full, 0 minimum paid, 1-8 = months overdue"


@st.cache_resource
def load_artifacts():
    """Load the committed, pre-trained model (no training happens in the app)."""
    booster = lgb.Booster(model_file=str(DEPLOY_DIR / "lgbm_booster.txt"))
    meta = json.loads((DEPLOY_DIR / "deploy_meta.json").read_text())
    explainer = shap.TreeExplainer(booster)
    raw = pd.read_csv(RAW_CSV).set_index("ID")
    card = load_card()
    return booster, meta, explainer, raw, card


model, meta, explainer, raw, card = load_artifacts()

features, cut = meta["features"], meta["band_cutoffs"]
BAND_COLOUR = {"RED": "🔴", "AMBER": "🟠", "GREEN": "🟢"}

tab_account, tab_portfolio, tab_card = st.tabs(["Account review", "Portfolio view", "Scorecard & fairness"])

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

    pd_score = float(model.predict(X)[0])  # booster returns P(default)
    band = assign_band([pd_score], cut)[0]
    points = float(score_points(X, card)[0])
    ead = max(float(hist.loc[0, "bill_amount"]), 0.0)
    exp_loss = pd_score * LGD * ead

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Probability of default (LightGBM)", f"{pd_score:.1%}",
              help=f"Portfolio average: {meta['portfolio_avg_pd']:.1%}")
    m2.metric("Early-warning band", f"{BAND_COLOUR[band]} {band}")
    m3.metric("Scorecard points", f"{points:.0f}",
              help="Traditional WoE scorecard: 600 = 50:1 good:bad odds; +20 points doubles the odds.")
    m4.metric("Expected loss (NT$)", f"{exp_loss:,.0f}",
              help=f"PD x LGD ({LGD:.0%}, assumed) x EAD (latest bill, NT${ead:,.0f})")
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
    st.subheader("Collections prioritisation: who to call with capacity for 10% of accounts?")
    st.dataframe(pd.read_csv(REPORTS_DIR / "collections_prioritisation.csv"), hide_index=True)
    st.caption(f"Expected loss = PD x LGD ({LGD:.0%}, assumed) x EAD (latest statement balance). "
               "Ranking by expected loss contacts fewer defaulters but covers far more money at risk.")

with tab_card:
    sm = json.loads((REPORTS_DIR / "scorecard_metrics.json").read_text())
    st.subheader("Traditional points scorecard (WoE + logistic regression)")
    st.write(f"Test AUC **{sm['test']['roc_auc']:.3f}** (Gini {sm['test']['gini']:.3f}, KS {sm['test']['ks']:.3f}) "
             f"using {sm['n_features']} features, all coefficients with the expected sign. "
             f"Median score: defaulters **{sm['median_score_defaulters']}** vs non-defaulters "
             f"**{sm['median_score_non_defaulters']}**.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Information Value (IV) per feature**")
        st.dataframe(pd.read_csv(REPORTS_DIR / "information_value.csv"), hide_index=True)
        st.caption("Rule of thumb: <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong. "
                   "*Above 0.5 is 'suspiciously strong': check for leakage. Here it is genuine: "
                   "past delinquency is the best predictor of future delinquency.")
    with c2:
        st.markdown("**Features removed and why**")
        st.dataframe(pd.DataFrame(sm["dropped_features"].items(), columns=["feature", "reason"]), hide_index=True)
    st.markdown("**Points table** (bin -> WoE -> points)")
    st.dataframe(pd.read_csv(REPORTS_DIR / "scorecard_points.csv"), hide_index=True, height=350)
    st.subheader("Fairness audit (gender is not a model input)")
    st.dataframe(pd.read_csv(REPORTS_DIR / "fairness_audit.csv"), hide_index=True)
