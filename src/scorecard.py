"""Traditional bank-style credit scorecard: WoE binning -> IV selection ->
logistic regression -> points.

Why build this next to LightGBM?  Banks and regulators still rely heavily on
points scorecards because every point can be traced to a bin of one variable.
Comparing the two shows the accuracy-vs-interpretability trade-off.

Definitions (goods = no default, bads = default):
  WoE (bin)   = ln( %goods in bin / %bads in bin )   higher = safer
  IV (feature)= sum over bins of (%goods - %bads) * WoE
                <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, >0.3 strong
  Score       = Offset + Factor * ln(odds of being good)
                Factor = PDO / ln(2), Offset = BASE_SCORE - Factor * ln(BASE_ODDS)
  -> a score of 600 means odds of 50:1 good:bad; every +20 points doubles the odds.

Run:  python -m src.scorecard   (after src.feature_store)
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import DEPLOY_DIR, FEATURES_CSV, RANDOM_STATE, REPORTS_DIR, TARGET, TEST_SIZE
from src.model import MODEL_FEATURES, ks_statistic

N_BINS = 10          # start from deciles; ties in discrete features merge bins automatically
MIN_IV = 0.02        # drop features below this information value
PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50


MAX_CORR = 0.70     # of two WoE features correlated above this, keep the higher-IV one


def make_cuts(x: pd.Series) -> list:
    """Inner quantile cut points learned on TRAIN only (bins are open-ended)."""
    return np.unique(np.quantile(x, np.linspace(0, 1, N_BINS + 1))[1:-1]).tolist()


def bin_index(values, cuts) -> np.ndarray:
    """Bin number for each value; bins are (-inf, c1], (c1, c2], ..., (ck, inf)."""
    return np.searchsorted(np.asarray(cuts, dtype=float), np.asarray(values, dtype=float), side="left")


def woe_table(x: pd.Series, y: pd.Series, cuts: list) -> pd.DataFrame:
    b = pd.cut(x, [-np.inf, *cuts, np.inf], right=True)
    t = pd.DataFrame({"bin": b, "y": y}).groupby("bin", observed=False)["y"].agg(["size", "sum"])
    t.columns = ["accounts", "bads"]
    t["goods"] = t["accounts"] - t["bads"]
    # +0.5 smoothing avoids ln(0) for a bin with no goods or no bads
    pg = (t["goods"] + 0.5) / (t["goods"].sum() + 0.5)
    pb = (t["bads"] + 0.5) / (t["bads"].sum() + 0.5)
    t["bad_rate"] = t["bads"] / t["accounts"]
    t["woe"] = np.log(pg / pb)
    t["iv_part"] = (pg - pb) * t["woe"]
    return t.reset_index()


def to_woe(X: pd.DataFrame, bins: dict) -> pd.DataFrame:
    out = {}
    for f, spec in bins.items():
        out[f] = np.asarray(spec["woe"])[bin_index(X[f], spec["cuts"])]
    return pd.DataFrame(out, index=X.index)


def score_points(X: pd.DataFrame, card: dict) -> np.ndarray:
    """Total scorecard points for each row = intercept points + sum of bin points."""
    total = np.full(len(X), card["base_points"], dtype=float)
    for f, spec in card["features"].items():
        total += np.asarray(spec["points"])[bin_index(X[f], spec["cuts"])]
    return total


def points_to_pd(points) -> np.ndarray:
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    ln_odds_good = (np.asarray(points) - offset) / factor
    return 1 / (1 + np.exp(ln_odds_good))


def main() -> None:
    df = pd.read_csv(FEATURES_CSV)
    X, y = df[MODEL_FEATURES], df[TARGET]
    # identical split to src.model (same seed, same stratification)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                              random_state=RANDOM_STATE, stratify=y)

    # 1) bin + WoE + IV on the training set only
    bins, iv_rows, woe_rows = {}, [], []
    for f in MODEL_FEATURES:
        cuts = make_cuts(X_tr[f])
        t = woe_table(X_tr[f], y_tr, cuts)
        iv = float(t["iv_part"].sum())
        iv_rows.append({"feature": f, "iv": round(iv, 4), "n_bins": len(t)})
        bins[f] = {"cuts": cuts, "woe": t["woe"].tolist(), "table": t}
    iv_df = pd.DataFrame(iv_rows).sort_values("iv", ascending=False)
    iv_df["strength"] = pd.cut(iv_df["iv"], [-1, 0.02, 0.1, 0.3, 0.5, 99],
                               labels=["useless", "weak", "medium", "strong", "very strong*"])
    candidates = iv_df.loc[iv_df["iv"] >= MIN_IV, "feature"].tolist()  # sorted by IV, high first

    # 2) drop multicollinear features: walk down the IV ranking and skip any feature
    #    whose WoE correlates > MAX_CORR with one already kept
    W_all = to_woe(X_tr, {f: bins[f] for f in candidates})
    corr = W_all.corr().abs()
    selected, dropped = [], {}
    for f in candidates:
        clash = [k for k in selected if corr.loc[f, k] > MAX_CORR]
        if clash:
            dropped[f] = f"corr {corr.loc[f, clash[0]]:.2f} with {clash[0]}"
        else:
            selected.append(f)

    # 3) logistic regression on WoE. Every coefficient must be negative (higher WoE =
    #    safer bin = lower PD); otherwise points would be counter-intuitive. Remove the
    #    weakest wrong-signed feature and refit until all signs are correct.
    while True:
        lr = LogisticRegression(max_iter=1000).fit(to_woe(X_tr, {f: bins[f] for f in selected}), y_tr)
        coefs = dict(zip(selected, lr.coef_[0]))
        wrong = [f for f in selected if coefs[f] >= 0]
        if not wrong:
            break
        worst = wrong[-1]  # lowest IV among wrong-signed (selected is IV-ordered)
        dropped[worst] = "wrong coefficient sign"
        selected.remove(worst)

    # 4) convert to points
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    base_points = offset - factor * lr.intercept_[0]  # ln(odds_good) = -(b0 + sum b_i * woe_i)
    card = {"pdo": PDO, "base_score": BASE_SCORE, "base_odds": BASE_ODDS,
            "base_points": round(float(base_points), 2), "features": {}}
    for f in selected:
        t = bins[f]["table"]
        pts = (-coefs[f] * t["woe"] * factor).round(1)
        card["features"][f] = {"cuts": bins[f]["cuts"], "points": pts.tolist()}
        for (_, r), p in zip(t.iterrows(), pts):
            woe_rows.append({"feature": f, "bin": str(r["bin"]), "accounts": int(r["accounts"]),
                             "bad_rate": round(r["bad_rate"], 4), "woe": round(r["woe"], 4), "points": p})

    # 5) evaluate on the untouched test set
    s_te = score_points(X_te, card)
    p_te = points_to_pd(s_te)
    auc = roc_auc_score(y_te, p_te)
    metrics = {
        "selected_features": selected, "dropped_features": dropped, "n_features": len(selected),
        "all_coefficients_negative": bool(all(c < 0 for c in coefs.values())),
        "test": {"roc_auc": round(auc, 4), "gini": round(2 * auc - 1, 4), "ks": round(ks_statistic(y_te, p_te), 4)},
        "score_range_test": [round(float(s_te.min())), round(float(s_te.max()))],
        "median_score_defaulters": round(float(np.median(s_te[y_te.to_numpy() == 1]))),
        "median_score_non_defaulters": round(float(np.median(s_te[y_te.to_numpy() == 0]))),
    }
    print("Information Value\n", iv_df.to_string(index=False))
    print("\nDropped:", dropped)
    print(f"\nScorecard on test: {metrics['test']} | features used: {len(selected)} "
          f"| all coefficients negative: {metrics['all_coefficients_negative']}")
    print(f"Median score: defaulters {metrics['median_score_defaulters']} vs "
          f"non-defaulters {metrics['median_score_non_defaulters']}")

    REPORTS_DIR.mkdir(exist_ok=True)
    DEPLOY_DIR.mkdir(exist_ok=True)
    iv_df.to_csv(REPORTS_DIR / "information_value.csv", index=False)
    pd.DataFrame(woe_rows).to_csv(REPORTS_DIR / "scorecard_points.csv", index=False)
    (REPORTS_DIR / "scorecard_metrics.json").write_text(json.dumps(metrics, indent=2))
    (DEPLOY_DIR / "scorecard.json").write_text(json.dumps(card, indent=1))
    print(f"Saved scorecard to {DEPLOY_DIR / 'scorecard.json'} and reports to {REPORTS_DIR}")


def load_card(path=DEPLOY_DIR / "scorecard.json") -> dict:
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
