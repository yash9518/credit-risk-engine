# Interview study guide

## Updated resume entry

**Credit Card Delinquency Early-Warning System** — *Python, SQL, LightGBM, scikit-learn, SHAP, Streamlit*
- Built an early-warning model on 30,000 real credit-card accounts (UCI Taiwan dataset, 22% default rate).
- Engineered 18 delinquency, utilisation and repayment features in SQL; built a WoE/IV points scorecard (AUC 0.76).
- Reached 0.78 test AUC (Gini 0.57, KS 0.44) with LightGBM; flagged 30% of accounts, catching 64% of defaulters.
- Ranked accounts by expected loss (PD×LGD×EAD), covering 64% of losses vs 43% by PD; deployed a Streamlit app.

Swap-in bullets:
- "Ran a fairness audit: AUC 0.78–0.79 and calibration consistent across gender and age groups."
- "Added 8 pytest checks (data checksum, target leakage, train/serve parity) run by GitHub Actions CI."

How this differs from your RAG project: the RAG project is about **origination** (policy rules applied when a loan is approved). This project is **portfolio monitoring** (predicting distress among existing customers), and it is trained on real data.

## 60-second pitch
"Banks lose money when existing card customers roll into delinquency. I took the public UCI dataset of 30,000 Taiwanese cardholders, which has six months of repayment status, bills and payments, and asked one question: who misses next month's payment? I built behavioural features in SQL, such as worst delinquency, utilisation and the share of the previous bill that was actually repaid. I compared an interpretable logistic baseline with LightGBM. LightGBM reached an AUC of 0.78 on a held-out set, which matches the published results for this dataset. I translated the scores into RED and AMBER action bands. Working only those 30% of accounts catches about two thirds of next month's defaulters. SHAP gives every account a reason code, so a collections agent knows why it was flagged. I also built the bank-standard points scorecard to show the interpretability trade-off: it reaches 0.76 against LightGBM's 0.78. Finally, I ranked accounts by expected loss rather than probability alone. With capacity to call 10% of customers, that covers 64% of the money at risk instead of 43%."

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


## Questions on the newer additions

**What are WoE and IV?**
- **WoE** (Weight of Evidence) for a bin = ln(% of all goods in the bin ÷ % of all bads in the bin). A positive WoE means the bin is safer than average; a negative one means riskier. It turns every variable onto one log-odds scale and handles outliers, because extreme values fall into the end bins.
- **IV** (Information Value) = Σ (%goods − %bads) × WoE, a measure of how much a variable separates goods from bads. Rule of thumb: below 0.02 useless, 0.02–0.1 weak, 0.1–0.3 medium, 0.3–0.5 strong. Above 0.5 is "suspicious", so check for leakage.
- My delinquency variables have IV around 0.8. That's genuine here, not leakage: past delinquency is the best predictor of future delinquency.

**Why did you drop features from the scorecard?**
For two reasons:
1. **Multicollinearity.** For example, months-delinquent-in-3-months correlates 0.83 with latest status. Correlated inputs make logistic-regression coefficients unstable, and some even flip sign.
2. **Wrong signs.** A wrong sign would give *more* points for being delinquent, which no credit committee accepts. I removed the weakest wrong-signed variable and refitted until every sign was correct.

The final scorecard has 9 variables.

**How does points scaling work? What is PDO?**
- Score = Offset + Factor × ln(odds good:bad), where Factor = PDO ÷ ln 2.
- I anchored the scale at **600 points = 50:1 odds** with **PDO = 20**, so every +20 points doubles the odds of being good.
- Each bin's points = −coefficient × WoE × Factor.

**The scorecard has lower AUC (0.76) than LightGBM (0.78). Which would a bank use?**
- **Scorecard:** often used for regulatory capital and adverse-action reasons, because every point can be explained and audited.
- **LightGBM:** often used as a challenger, or for internal early warning where SHAP explanations are acceptable.
- The 0.02 AUC gap is the price of full transparency, and showing that trade-off is the point of having both.

**Which non-monotonic pattern did you find?**
Customers whose latest status is "paid in full / no usage" default slightly more often (15.4%) than "minimum payment" customers (12.8%). It's a known quirk of this dataset; one explanation is that inactive cards include customers already in financial trouble elsewhere. I kept the data-driven pattern and documented it rather than hiding it. A bank's model-risk team would ask me to justify it or merge the bins.

**What is expected loss and why rank by it?**
- EL = PD × LGD × EAD: the probability of default × the share of exposure lost if it happens × the exposure at default.
- I used EAD = latest balance and assumed LGD = 75%, because the data has no recovery information.
- Ranking by PD finds the *most* defaulters. Ranking by EL finds the most *money*. With capacity to call 600 accounts, EL ranking catches fewer defaulters (301 vs 420) but covers 64% of the rupee loss instead of 43%.
- In practice banks mix the two: for example, EL for the call queue and PD for limit decisions.

**What is EAD in a real bank?**
EAD = current balance + CCF × undrawn limit. The CCF (credit conversion factor) captures customers drawing down more of their limit before they default. I didn't model it, and I say so in the limitations.

**How did you check fairness?**
- I computed AUC, mean predicted PD, observed default rate and the share of accounts in the RED band for each gender and each age group.
- AUC is 0.78–0.79 everywhere, and predicted PD is within about 1 percentage point of the observed rate for each gender.
- Excluding gender as an input isn't enough on its own ("fairness through unawareness"), because other features can act as proxies. That's why you audit the outcomes.

**What do your tests check?**
1. The dataset's checksum and its published facts.
2. No target leakage: flipping the labels changes no feature.
3. A single dashboard row produces exactly the same features as the training batch.
4. Feature values stay in sensible ranges.
5. Band assignment.
6. The deployed model reproduces the reported AUC.
7. Scorecard points fall with each extra month of delinquency, and 600 points corresponds to 50:1 odds.
8. The scorecard returns valid scores for real rows.

GitHub Actions runs them on every push.

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
| Scorecard | 9 features, test AUC 0.759, 600 pts = 50:1, PDO 20 |
| Median score | defaulters 508, non-defaulters 544 |
| 10% capacity, rank by PD vs EL | 43% vs 64% of loss covered |
| Expected vs realised loss | NT$50.1M vs NT$50.4M (LGD 75% assumed) |
| Fairness AUC | female 0.785, male 0.779 |
| Top 3 drivers | worst delinquency (6m), months delinquent (3m), latest status |
