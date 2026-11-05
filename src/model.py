"""Train and evaluate the default-risk models.

1. Stratified 80/20 split (the data has no usable time column, see README).
2. 5-fold cross-validation on the training set for both models (mean +/- std).
3. Final fit on the full training set, one evaluation on the untouched test set.
4. Banking metrics: ROC-AUC, Gini, KS, PR-AUC, Brier score.
5. Decile (lift) table and RED / AMBER / GREEN early-warning bands.
6. Save model, SHAP explainer and reports.

Run:  python -m src.model
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (AMBER_PCT, DEPLOY_DIR, FEATURES_CSV, ID_COL, MODELS_DIR, RANDOM_STATE, RED_PCT,
                        REPORTS_DIR, TARGET, TEST_SIZE)

# SEX and marital status are deliberately NOT used: they are protected/sensitive
# attributes and using them in a credit decision is a fair-lending risk.
MODEL_FEATURES = [
    "credit_limit", "age", "education_grp",
    "pay_status_latest", "max_dpd_bucket_6m", "months_delinquent_6m", "months_delinquent_3m",
    "months_paid_in_full_6m", "delinquency_trend",
    "utilization_latest", "utilization_avg_6m", "utilization_change_3m", "over_limit_flag",
    "payment_ratio_latest", "payment_ratio_avg_3m", "months_zero_payment_5m", "log_last_payment",
]


def make_lgbm() -> lgb.LGBMClassifier:
    # Shallow trees + strong min_child_samples: 30k rows is small, so we
    # regularise hard. No class re-weighting: a 22% default rate is not extreme,
    # and re-weighting would distort the probabilities (hurting calibration).
    return lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )


def make_logit():
    # Note: education_grp is used as an ordinal number here for simplicity.
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def ks_statistic(y, p) -> float:
    """Kolmogorov-Smirnov: max gap between cumulative good and bad distributions."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {
        "roc_auc": round(auc, 4),
        "gini": round(2 * auc - 1, 4),
        "ks": round(ks_statistic(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
    }


def cross_validate(make_model, X, y) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for tr, va in skf.split(X, y):
        m = make_model().fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
    return {"cv_auc_mean": round(float(np.mean(aucs)), 4), "cv_auc_std": round(float(np.std(aucs)), 4)}


def decile_table(y, p) -> pd.DataFrame:
    """Rank accounts by score, cut into 10 equal groups (decile 1 = riskiest)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": p})
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    t = df.groupby("decile", observed=True).agg(
        accounts=("y", "size"), defaults=("y", "sum"),
        avg_pd=("p", "mean"), min_score=("p", "min"), max_score=("p", "max"))
    t["default_rate"] = t["defaults"] / t["accounts"]
    t["cum_capture"] = t["defaults"].cumsum() / t["defaults"].sum()
    t["lift"] = t["default_rate"] / df["y"].mean()
    return t.round(4).reset_index()


def band_cutoffs(train_scores) -> dict:
    return {
        "red": float(np.quantile(train_scores, 1 - RED_PCT)),
        "amber": float(np.quantile(train_scores, 1 - RED_PCT - AMBER_PCT)),
    }


def assign_band(p, cut) -> np.ndarray:
    p = np.asarray(p)
    return np.where(p >= cut["red"], "RED", np.where(p >= cut["amber"], "AMBER", "GREEN"))


def band_table(y, p, cut) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "band": assign_band(p, cut)})
    t = df.groupby("band").agg(accounts=("y", "size"), defaults=("y", "sum"))
    t["share_of_book"] = t["accounts"] / len(df)
    t["default_rate"] = t["defaults"] / t["accounts"]
    t["share_of_all_defaults"] = t["defaults"] / df["y"].sum()
    return t.loc[["RED", "AMBER", "GREEN"]].round(4).reset_index()


def calibration_table(y, p, bins=10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": p})
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    t = df.groupby("bin", observed=True).agg(predicted_pd=("p", "mean"), observed_rate=("y", "mean"),
                                             accounts=("y", "size"))
    return t.round(4).reset_index(drop=True)


def main() -> None:
    df = pd.read_csv(FEATURES_CSV)
    X, y = df[MODEL_FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
        X, y, df[ID_COL], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"Train {len(X_tr):,} | Test {len(X_te):,} | default rate {y.mean():.2%}")

    results = {}
    fitted = {}
    for name, factory in [("logistic_regression", make_logit), ("lightgbm", make_lgbm)]:
        cv = cross_validate(factory, X_tr, y_tr)
        model = factory().fit(X_tr, y_tr)
        p_te = model.predict_proba(X_te)[:, 1]
        results[name] = {**cv, "test": metrics(y_te, p_te)}
        fitted[name] = model
        print(f"{name:20s} CV AUC {cv['cv_auc_mean']:.4f} +/- {cv['cv_auc_std']:.4f} | test {results[name]['test']}")

    lgbm = fitted["lightgbm"]
    p_tr = lgbm.predict_proba(X_tr)[:, 1]
    p_te = lgbm.predict_proba(X_te)[:, 1]
    results["lightgbm"]["train_auc"] = round(roc_auc_score(y_tr, p_tr), 4)  # overfitting check

    cut = band_cutoffs(p_tr)
    deciles = decile_table(y_te, p_te)
    bands = band_table(y_te, p_te, cut)
    calib = calibration_table(y_te, p_te)
    importance = (pd.DataFrame({"feature": MODEL_FEATURES,
                                "mean_abs_shap": np.abs(shap.TreeExplainer(lgbm)(X_te).values).mean(0)})
                  .sort_values("mean_abs_shap", ascending=False).round(4))

    print("\nDecile table (test set, LightGBM)\n", deciles.to_string(index=False))
    print("\nEarly-warning bands (test set)\n", bands.to_string(index=False))
    print("\nTop drivers (mean |SHAP|)\n", importance.head(8).to_string(index=False))

    # ---- save artefacts ----
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(lgbm, MODELS_DIR / "lgbm_model.pkl")
    joblib.dump(fitted["logistic_regression"], MODELS_DIR / "logit_model.pkl")
    joblib.dump({"features": MODEL_FEATURES, "band_cutoffs": cut,
                 "test_ids": id_te.tolist(), "train_scores": p_tr, "test_scores": p_te},
                MODELS_DIR / "metadata.pkl")

    # Deployment artefacts: LightGBM's native text format loads across library
    # versions (unlike a pickle), so the live dashboard never has to retrain.
    DEPLOY_DIR.mkdir(exist_ok=True)
    lgbm.booster_.save_model(str(DEPLOY_DIR / "lgbm_booster.txt"))
    (DEPLOY_DIR / "deploy_meta.json").write_text(json.dumps({
        "features": MODEL_FEATURES, "band_cutoffs": cut,
        "portfolio_avg_pd": float(p_tr.mean()), "test_ids": [int(i) for i in id_te],
    }))

    results["band_cutoffs"] = {k: round(v, 4) for k, v in cut.items()}
    results["n_train"], results["n_test"] = len(X_tr), len(X_te)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    deciles.to_csv(REPORTS_DIR / "decile_table.csv", index=False)
    bands.to_csv(REPORTS_DIR / "early_warning_bands.csv", index=False)
    calib.to_csv(REPORTS_DIR / "calibration_table.csv", index=False)
    importance.to_csv(REPORTS_DIR / "feature_importance_shap.csv", index=False)
    print(f"\nSaved models to {MODELS_DIR} and reports to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
