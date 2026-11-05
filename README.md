# Credit Card Delinquency Early-Warning System

Predicts which **existing credit-card customers will miss next month's payment**, so a collections team can act before the default happens. The project uses a **real, public dataset** of 30,000 Taiwanese cardholders. It covers SQL feature engineering, a logistic-regression baseline compared against LightGBM, banking-standard validation (AUC, Gini, KS, calibration, lift), RED/AMBER/GREEN early-warning bands, and a Streamlit dashboard with SHAP reason codes.

> This is **behavioural scoring** (monitoring accounts already on the book).
> It is *not* application/underwriting scoring (deciding whether to lend in the first place).

## Results (hold-out test set, 6,000 accounts never seen in training)

| Model | 5-fold CV AUC | Test ROC-AUC | Gini | KS | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.762 ± 0.006 | 0.745 | 0.491 | 0.387 | 0.506 | 0.142 |
| **LightGBM** | **0.788 ± 0.004** | **0.783** | **0.567** | **0.439** | **0.567** | **0.134** |

Base default rate is 22.1%, so a random model has PR-AUC ≈ 0.22.

**Early-warning bands** (cut-offs fixed on the training set, applied to the test set):

| Band | Share of book | Default rate | Share of all defaults |
|---|---|---|---|
| 🔴 RED (top 10%) | 10.3% | 69.6% | 32.3% |
| 🟠 AMBER (next 20%) | 19.8% | 35.5% | 31.7% |
| 🟢 GREEN (rest) | 70.0% | 11.4% | 36.0% |

Working only the RED and AMBER bands (30% of accounts) reaches **64% of next month's defaulters**. The top decile has a **3.2× lift** over the average.
The model is well **calibrated**: in every score decile, predicted and observed default rates are within about 3 percentage points of each other (`reports/calibration_table.csv`).

**Top drivers (mean |SHAP|):** worst delinquency in the last 6 months, number of delinquent months in the last 3, latest repayment status, credit limit, size of last payment, and utilisation.

### Is this realistic?
Yes. Published studies on this dataset typically report AUCs of about 0.77–0.79, and this model lands inside that range. An AUC of 0.99 on this data would indicate leakage, not skill. The gap between train AUC (0.82) and test AUC (0.78) is small, so overfitting is under control.

## Dataset

**Default of Credit Card Clients**: UCI Machine Learning Repository, dataset #350 (CC BY 4.0)
<https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
Yeh, I-C. & Lien, C-H. (2009), *Expert Systems with Applications*, 36(2), 2473–2480.

- 30,000 credit-card accounts from a Taiwanese bank, April–September 2005.
- Each account has 6 months of repayment status, bill amounts and payments, plus credit limit and demographics.
- Target: default on the October 2005 payment, which applies to 6,636 accounts (22.12%).

**Validation** (`python -m src.data`):
- The CSV is committed at `data/raw/UCI_Credit_Card.csv` and pinned by SHA-256 `a0f0ab49…76ca1`. Several independent public mirrors serve byte-identical copies.
- The script asserts the facts published by UCI and the paper: 30,000 rows, 23 predictors, 6,636 defaults, unique IDs, no missing values, and valid code ranges.
- It also reports known quirks:
  - 345 rows have undocumented EDUCATION codes (0/5/6); these are grouped as "other".
  - 54 rows have MARRIAGE = 0.
  - 35 rows are duplicates.
  - 590 accounts have a credit (negative) balance.

## Pipeline

```bash
python -m pip install -r requirements.txt
python -m src.data            # verify checksum + validate dataset
python -m src.feature_store   # SQL feature engineering -> data/processed/features.csv
python -m src.model           # CV, train, evaluate, save models/ and reports/
python -m streamlit run dashboard/app.py
```

| File | Purpose |
|---|---|
| `src/data.py` | Download (if missing), checksum, schema and fact validation, data-quality notes |
| `src/feature_store.py` | 18 behavioural features written in SQL (SQLite); also used by the dashboard, so training and serving use one code path |
| `src/model.py` | Logistic vs LightGBM, 5-fold CV, test metrics, decile/lift, bands, calibration, SHAP importance |
| `dashboard/app.py` | Account review (editable what-if history, PD, band, action, SHAP waterfall) and portfolio view |
| `reports/` | Committed outputs of the last run (metrics, deciles, bands, calibration, SHAP importance) |

### Engineered features (all known at the September month-end, so there is no look-ahead)
- **Delinquency**: latest status, worst delinquency bucket in 6 months, months delinquent (in the last 3 and last 6 months), months paid in full, delinquency trend.
- **Utilisation**: latest and 6-month-average balance/limit, 3-month change in utilisation, over-limit flag.
- **Repayment capacity**: payment ÷ previous bill (latest and 3-month average, capped at 1), months with zero payment, log of last payment.
- **Profile**: credit limit, age, education group.
- **Not used on purpose**: SEX and marital status, because they are sensitive attributes and pose a fair-lending risk.

## Limitations (say these before the interviewer does)
- **Snapshot data**: there is only one observation month, so out-of-time validation and real PSI drift monitoring are not possible. The split is a random stratified one.
- **Single market and year**: Taiwan in 2005, around the time of its card-debt crisis. Default rates and drivers will differ for an Indian portfolio today.
- **Short horizon**: the target is a single missed payment next month, not the regulatory 90+ days-past-due definition of default.
- **Arbitrary band cut-offs**: the 10%/20% cut-offs reflect collections capacity, not an optimised cost function.
