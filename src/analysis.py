"""Business and fairness analysis on the hold-out test set.

1. Expected loss  EL = PD x LGD x EAD
     PD  = LightGBM probability of default next month
     EAD = exposure at default, proxied by the latest statement balance (floored at 0)
     LGD = loss given default, ASSUMED 75% (typical for unsecured cards; the data has
           no recovery information, so this is a stated assumption, not an estimate)
   Question answered: if collections can only call 10% of accounts, should they rank
   by PD or by expected loss?

2. Fairness audit. SEX is NOT a model input, but proxies could still create bias,
   so we check calibration and band rates per group (and per age group).

Run:  python -m src.analysis   (after src.model)
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import DEPLOY_DIR, FEATURES_CSV, ID_COL, RAW_CSV, REPORTS_DIR, TARGET
from src.model import assign_band

LGD = 0.75
CAPACITY = 0.10  # share of accounts collections can contact


def load_test_frame() -> pd.DataFrame:
    meta = json.loads((DEPLOY_DIR / "deploy_meta.json").read_text())
    booster = lgb.Booster(model_file=str(DEPLOY_DIR / "lgbm_booster.txt"))
    feats = pd.read_csv(FEATURES_CSV).set_index(ID_COL).loc[meta["test_ids"]]
    raw = pd.read_csv(RAW_CSV).set_index("ID").loc[meta["test_ids"]]
    df = pd.DataFrame({
        "y": feats[TARGET].to_numpy(),
        "pd": booster.predict(feats[meta["features"]]),
        "ead": raw["BILL_AMT1"].clip(lower=0).to_numpy(),
        "sex": raw["SEX"].map({1: "male", 2: "female"}).to_numpy(),
        "age_group": pd.cut(raw["AGE"], [0, 29, 45, 200], labels=["<30", "30-45", ">45"]).astype(str).to_numpy(),
    }, index=feats.index)
    df["band"] = assign_band(df["pd"], meta["band_cutoffs"])
    df["expected_loss"] = df["pd"] * LGD * df["ead"]
    df["actual_loss"] = df["y"] * LGD * df["ead"]
    return df


def prioritisation(df: pd.DataFrame) -> pd.DataFrame:
    k = int(len(df) * CAPACITY)
    rows = []
    for rule, col in [("rank by PD", "pd"), ("rank by expected loss", "expected_loss"), ("random", None)]:
        top = df.sample(k, random_state=0) if col is None else df.nlargest(k, col)
        rows.append({
            "strategy": rule, "accounts_contacted": k,
            "defaulters_caught": int(top["y"].sum()),
            "share_of_defaulters": round(top["y"].sum() / df["y"].sum(), 4),
            "loss_covered_NT$": round(float(top["actual_loss"].sum())),
            "share_of_total_loss": round(top["actual_loss"].sum() / df["actual_loss"].sum(), 4),
        })
    return pd.DataFrame(rows)


def group_audit(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for g, d in df.groupby(col):
        rows.append({
            "group": f"{col}={g}", "accounts": len(d),
            "observed_default_rate": round(d["y"].mean(), 4),
            "mean_predicted_pd": round(d["pd"].mean(), 4),
            "auc": round(roc_auc_score(d["y"], d["pd"]), 4),
            "share_red_band": round((d["band"] == "RED").mean(), 4),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = load_test_frame()
    prio = prioritisation(df)
    fair = pd.concat([group_audit(df, "sex"), group_audit(df, "age_group")], ignore_index=True)
    el = {
        "lgd_assumption": LGD,
        "portfolio_expected_loss_NT$": round(float(df["expected_loss"].sum())),
        "realised_loss_at_same_lgd_NT$": round(float(df["actual_loss"].sum())),
        "expected_by_band_NT$": {b: round(float(df.loc[df["band"] == b, "expected_loss"].sum()))
                                 for b in ["RED", "AMBER", "GREEN"]},
    }
    print(f"Collections capacity {CAPACITY:.0%} of {len(df):,} test accounts\n", prio.to_string(index=False))
    print("\nExpected vs realised loss:", el)
    print("\nFairness audit\n", fair.to_string(index=False))

    prio.to_csv(REPORTS_DIR / "collections_prioritisation.csv", index=False)
    fair.to_csv(REPORTS_DIR / "fairness_audit.csv", index=False)
    (REPORTS_DIR / "expected_loss.json").write_text(json.dumps(el, indent=2))


if __name__ == "__main__":
    main()
