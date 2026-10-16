# Credit Risk Engine

A Python and Streamlit demonstration of retail credit scoring, anomaly detection, and model explanations. Generate synthetic loan applications, build financial features with SQL, train a LightGBM classifier, and inspect individual applicants in an interactive dashboard.

## Features

- Synthetic dataset of 10,000 loan applications with a default label derived from financial ratios and external scores.
- In-memory SQLite feature engineering for loan-to-income, debt-service, age, and employment features.
- LightGBM classification with an 80/20 stratified train/test split and printed ROC-AUC and average precision metrics.
- Isolation Forest anomaly detection with a 5% contamination setting.
- Applicant scoring dashboard with SHAP waterfall explanations.

## Quick start

Run all commands from the repository root. Use a Python environment compatible with the packages in `requirements.txt`.

```bash
git clone https://github.com/yash9518/credit-risk-engine.git
cd credit-risk-engine
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies, create the data directories, and run the pipeline in order:

```bash
python -m pip install -r requirements.txt
python -c "from pathlib import Path; Path('data/raw').mkdir(parents=True, exist_ok=True); Path('data/processed').mkdir(parents=True, exist_ok=True)"
python -m src.generate_data
python -m src.feature_store
python -m src.model
python -m src.anomaly
python -m streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit. Enter applicant income, loan amount, annuity/EMI, age, employment duration, external scores, and contract type, then select **Evaluate Credit Application**. The dashboard displays the model score, a risk decision using a 0.40 threshold, an anomaly flag, and a SHAP explanation.

## Project layout

| File | Purpose |
| --- | --- |
| `src/generate_data.py` | Generate `data/raw/application_train.csv` |
| `src/feature_store.py` | Transform raw applications into `data/processed/features.csv` |
| `src/model.py` | Train and save the classifier, feature list, and SHAP explainer |
| `src/anomaly.py` | Train and save the Isolation Forest |
| `dashboard/app.py` | Interactive applicant evaluation |
| `requirements.txt` | Python dependencies |

Training creates `src/lgbm_credit_model.pkl`, `src/model_features.pkl`, `src/shap_explainer.pkl`, and `src/anomaly_isolation_forest.pkl`. Data and model artifacts are excluded from version control and must be generated locally.

## Implementation notes

- The bundled generator produces synthetic data and labels the highest-risk 8% of its generated risk scores as defaults. Metrics on this dataset describe the simulation, not performance on real applicants.
- The dashboard labels annuity as monthly EMI but divides it directly by annual income, matching the current feature formula. Reconcile time units before adapting the project to another dataset.
- The displayed default probability is the classifier output; calibration and real-world validation are not implemented.
- If the dashboard asks for model artifacts, run both training commands and launch Streamlit from the repository root.
- Dependency versions are not pinned. Preserve the environment used to train artifacts when loading them later.
