# Interview study guide

## Updated resume entry

**Credit Card Delinquency Early-Warning System** — *Python, SQL, LightGBM, scikit-learn, SHAP, Streamlit*
- Built a behavioural early-warning model on 30,000 real credit-card accounts (UCI Taiwan dataset, 22% default rate), with checksum and schema validation of the data.
- Engineered 18 delinquency, utilisation and repayment-capacity features in SQL from 6 months of statement history.
- LightGBM achieved 0.783 test AUC (Gini 0.57, KS 0.44), beating a logistic baseline at 0.745, with stable 5-fold CV (0.788 ± 0.004) and good calibration.
- Designed RED/AMBER bands that flag 30% of accounts and capture 64% of next-month defaulters (3.2× lift in the top decile); a Streamlit dashboard shows SHAP reason codes.

How this differs from your RAG project: the RAG project is about **origination** (policy rules applied when a loan is approved). This project is **portfolio monitoring** (predicting distress among existing customers), and it is trained on real data.

## 60-second pitch
"Banks lose money when existing card customers roll into delinquency. I took the public UCI dataset of 30,000 Taiwanese cardholders, which has six months of repayment status, bills and payments, and asked one question: who misses next month's payment? I built behavioural features in SQL, such as worst delinquency, utilisation and the share of the previous bill that was actually repaid. I compared an interpretable logistic baseline with LightGBM. LightGBM reached an AUC of 0.78 on a held-out set, which matches the published results for this dataset. I translated the scores into RED and AMBER action bands. Working only those 30% of accounts catches about two thirds of next month's defaulters. SHAP gives every account a reason code, so a collections agent knows why it was flagged."

## Questions you will likely get

**Why is AUC only 0.78? Isn't that low?**
For this dataset, 0.77–0.79 is the published state of the art. Credit behaviour is noisy, and many defaults are driven by events not in the data, such as job loss. An AUC of 0.99 would be a red flag for leakage. (The earlier version of this project hit 0.9975 because the labels were generated from the same features; that is why I moved to real data.)

**What are Gini and KS?**
- Gini = 2·AUC − 1, which gives 0.57 here. Banks quote Gini more often than AUC.
- KS is the maximum gap between the cumulative score distributions of defaulters and non-defaulters, which gives 0.44 here. Above 0.40 is generally considered good for behavioural scorecards.

**Why didn't you oversample or use `scale_pos_weight`?**
A 22% default rate is not severely imbalanced. Re-weighting inflates the predicted probabilities. Because the bands and expected-loss calculations need probabilities that are actually correct, I kept the model calibrated. The calibration table shows predicted and observed rates within a few points in every decile.

**How did you prevent leakage?**
- All features use data up to September only; the target is October.
- The test set is split off first and touched once.
- CV runs only on the training data.
- Band cut-offs come from training scores.
- The dashboard calls the same SQL function as training, and I checked that both produce identical feature values.

**Why did logistic regression lose?**
The relationships are non-linear and involve interactions. For example, high utilisation is worrying only when payments are also falling, and the effect of a delinquency bucket is not linear. To close the gap, the banking-standard route is a WoE-binned logistic scorecard, which keeps it interpretable for regulators.

**Why exclude gender and marital status?**
They are sensitive or protected attributes. Using them creates fair-lending and regulatory risk, and dropping them barely changes AUC because behaviour carries the signal.

**What does PAY_0 = −2 / −1 / 0 mean?**
- −2: no card usage.
- −1: paid in full.
- 0: revolving credit (minimum payment made).
- 1–8: number of months the payment is overdue.

**What would you do in production?**
- Out-of-time validation on later months.
- PSI monitoring of score and feature drift.
- Monthly re-scoring.
- Set band cut-offs by collections capacity and the cost of a missed default versus a wasted call.
- Use a 90+ DPD default definition for IFRS 9 / Basel PD.
- Run champion–challenger tests of the bands.

**What are the limitations?**
- Snapshot data with no time split.
- Taiwan in 2005, during a card-debt crisis.
- A one-month horizon.
- Demographic codes are partly undocumented.

## Numbers to memorise
| | |
|---|---|
| Accounts / default rate | 30,000 / 22.12% (6,636) |
| Split | 24,000 train / 6,000 test, stratified |
| LightGBM test AUC / Gini / KS | 0.783 / 0.567 / 0.439 |
| Logistic test AUC | 0.745 |
| CV AUC | 0.788 ± 0.004 |
| Train AUC (overfit check) | 0.821 |
| RED band | 10% of book, 70% default rate, 32% of defaults |
| RED + AMBER | 30% of book, 64% of defaults |
| Top-decile lift | 3.2× |
| Top 3 drivers | worst delinquency (6m), months delinquent (3m), latest status |
