# Credit Card Delinquency Early-Warning System

Predicts which **existing credit-card customers will miss next month's payment**, so a collections team can act before the default happens. The project uses a **real, public dataset** of 30,000 Taiwanese cardholders. It covers SQL feature engineering, a traditional WoE/IV points scorecard alongside LightGBM, banking-standard validation (AUC, Gini, KS, calibration, lift), RED/AMBER/GREEN early-warning bands, expected-loss (PD × LGD × EAD) prioritisation, a fairness audit, automated tests, and a Streamlit dashboard with SHAP reason codes.

[![tests](https://github.com/yash9518/credit-risk-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/yash9518/credit-risk-engine/actions/workflows/ci.yml)

**Live demo:** https://credit-risk-engine-dj6pjefck3bzfmpxfswsz9.streamlit.app/  
_(Free hosting: if the app is asleep, click "Yes, get this app back up!" and wait ~1 minute.)_

> This is **behavioural scoring** (monitoring accounts already on the book).
> It is *not* application/underwriting scoring (deciding whether to lend in the first place).

## Results (hold-out test set, 6,000 accounts never seen in training)

| Model | 5-fold CV AUC | Test ROC-AUC | Gini | KS | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.762 ± 0.006 | 0.745 | 0.491 | 0.387 | 0.506 | 0.142 |
| WoE points scorecard (9 features) | – | 0.759 | 0.519 | 0.395 | – | – |
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

### Traditional points scorecard (`src/scorecard.py`)
The scorecard follows the steps banks use:
1. **Binning:** each feature is split into quantile bins on the training set.
2. **Weight of Evidence:** each bin gets a WoE value, ln(%goods ÷ %bads).
3. **Information Value (IV):** features with IV below 0.02 are dropped.
4. **Correlation filter:** if two features' WoE values correlate above 0.7, the one with the higher IV is kept.
5. **Logistic regression:** fitted on the WoE values, refitting until **every coefficient has the expected sign**.
6. **Points:** the result is scaled so that 600 points means 50:1 good:bad odds, and every +20 points doubles the odds (PDO).

The resulting 9-variable scorecard reaches a test AUC of 0.759, compared with 0.783 for LightGBM. That gap is the price of full transparency. The median score is 508 for defaulters and 544 for non-defaulters. Outputs: `reports/information_value.csv`, `reports/scorecard_points.csv`.

### Expected loss and collections prioritisation (`src/analysis.py`)
Expected loss is EL = PD × LGD × EAD, where:
- **PD** comes from LightGBM;
- **EAD** is the latest statement balance;
- **LGD** is *assumed* to be 75%, typical for unsecured cards, because the data has no recovery information.

Suppose collections can contact only 10% of accounts (600 in the test set):

| Strategy | Defaulters caught | Share of money lost to default covered |
|---|---|---|
| Random | 129 (10%) | 10% |
| Rank by PD | 420 (32%) | 43% |
| **Rank by expected loss** | 301 (23%) | **64%** |

Ranking by expected loss catches fewer defaulters but protects far more money. The portfolio's expected loss (NT$50.1M) closely matches the realised loss (NT$50.4M) at the same LGD, so the PDs are calibrated in money terms too.

### Fairness audit
Gender is not a model input, but proxy features could still introduce bias. On the test set:

| Group | AUC | Mean predicted PD | Observed default rate |
|---|---|---|---|
| Female | 0.785 | 21.7% | 21.3% |
| Male | 0.779 | 22.9% | 23.4% |

Results for the three age groups are in `reports/fairness_audit.csv`; they are similarly consistent.

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
python -m src.model           # CV, train, evaluate, save models/, artifacts/ and reports/
python -m src.scorecard       # WoE/IV points scorecard
python -m src.analysis        # expected loss, collections prioritisation, fairness audit
python -m pytest -q           # 8 automated tests (also run by GitHub Actions on every push)
python -m streamlit run dashboard/app.py
```

| File | Purpose |
|---|---|
| `src/data.py` | Download (if missing), checksum, schema and fact validation, data-quality notes |
| `src/feature_store.py` | 18 behavioural features written in SQL (SQLite); also used by the dashboard, so training and serving use one code path |
| `src/model.py` | Logistic vs LightGBM, 5-fold CV, test metrics, decile/lift, bands, calibration, SHAP importance |
| `src/scorecard.py` | WoE binning, IV, correlation and sign checks, points scaling |
| `src/analysis.py` | Expected loss, collections prioritisation, fairness audit |
| `tests/` | Data checksum, no target leakage, train/serve feature parity, reproducible AUC, scorecard monotonicity |
| `dashboard/app.py` | Account review (what-if history, PD, band, scorecard points, expected loss, SHAP), portfolio view, scorecard & fairness |
| `artifacts/` | Pre-trained LightGBM (native text format) + metadata used by the live app |
| `reports/` | Committed outputs of the last run (metrics, deciles, bands, calibration, SHAP importance) |

### Engineered features (all known at the September month-end, so there is no look-ahead)
- **Delinquency**: latest status, worst delinquency bucket in 6 months, months delinquent (in the last 3 and last 6 months), months paid in full, delinquency trend.
- **Utilisation**: latest and 6-month-average balance/limit, 3-month change in utilisation, over-limit flag.
- **Repayment capacity**: payment ÷ previous bill (latest and 3-month average, capped at 1), months with zero payment, log of last payment.
- **Profile**: credit limit, age, education group.
- **Not used on purpose**: SEX and marital status, because they are sensitive attributes and pose a fair-lending risk.

## Deploy (free, Streamlit Community Cloud)
1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **Create app**, then **Deploy a public app from GitHub**.
3. Set **Repository** to `yash9518/credit-risk-engine`, **Branch** to `main` (or the feature branch) and **Main file path** to `dashboard/app.py`. Optionally set a custom subdomain, e.g. `credit-early-warning`.
4. Click **Deploy**. The app loads the pre-trained model committed in `artifacts/` (LightGBM text format + JSON metadata), so it never trains on the server and starts in seconds. Re-running `python -m src.model` regenerates these files.

## Limitations (say these before the interviewer does)
- **Snapshot data**: there is only one observation month, so out-of-time validation and real PSI drift monitoring are not possible. The split is a random stratified one.
- **Single market and year**: Taiwan in 2005, around the time of its card-debt crisis. Default rates and drivers will differ for an Indian portfolio today.
- **Short horizon**: the target is a single missed payment next month, not the regulatory 90+ days-past-due definition of default.
- **Assumed LGD and simple EAD**: LGD is a fixed 75% assumption, and EAD ignores undrawn limit (no credit conversion factor).
- **One non-monotonic scorecard bin**: customers who "paid in full / had no usage" default slightly more often (15.4%) than those who "paid the minimum" (12.8%). This is a known quirk of this dataset, and the scorecard keeps it rather than forcing monotonicity.
- **Arbitrary band cut-offs**: the 10%/20% cut-offs reflect collections capacity, not an optimised cost function.
