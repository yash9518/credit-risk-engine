import numpy as np
import pandas as pd

def generate_retail_loans(n_records=10000, output_path="data/raw/application_train.csv"):
    np.random.seed(42)
    
    ids = np.arange(100001, 100001 + n_records)
    incomes = np.random.lognormal(mean=12.2, sigma=0.6, size=n_records).round(-3)
    
    # Loan amount requested typically 1x to 4x of annual income
    multiplier = np.random.uniform(1.0, 4.0, size=n_records)
    credits = (incomes * multiplier).round(-3)
    
    # Annuity ~ 5% to 15% of loan credit
    annuities = (credits * np.random.uniform(0.06, 0.14, size=n_records)).round(-2)
    goods_prices = (credits * np.random.uniform(0.9, 1.05, size=n_records)).round(-3)
    
    # Demographics
    contract_types = np.random.choice(["Cash loans", "Revolving loans"], size=n_records, p=[0.9, 0.1])
    genders = np.random.choice(["M", "F"], size=n_records, p=[0.45, 0.55])
    days_birth = -1 * np.random.randint(21 * 365, 65 * 365, size=n_records)
    days_employed = -1 * np.random.randint(1 * 365, 30 * 365, size=n_records)
    
    # External bureau scores (0 to 1)
    ext_1 = np.random.beta(a=5, b=2, size=n_records)
    ext_2 = np.random.beta(a=4, b=2, size=n_records)
    ext_3 = np.random.beta(a=3, b=2, size=n_records)
    
    # Logistic default target correlated with DTI and external scores
    debt_burden = (annuities / np.maximum(incomes, 1))
    logit = (
        2.0 * debt_burden
        - 2.8 * ext_2
        - 2.2 * ext_3
        - 0.05 * (np.abs(days_employed) / 365)
        + 0.5
    )
    prob_default = 1 / (1 + np.exp(-logit))
    # Target: ~8% default rate (industry standard for retail)
    target = (prob_default > np.percentile(prob_default, 92)).astype(int)
    
    df = pd.DataFrame({
        "SK_ID_CURR": ids,
        "TARGET": target,
        "NAME_CONTRACT_TYPE": contract_types,
        "CODE_GENDER": genders,
        "AMT_INCOME_TOTAL": incomes,
        "AMT_CREDIT": credits,
        "AMT_ANNUITY": annuities,
        "AMT_GOODS_PRICE": goods_prices,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "EXT_SOURCE_1": np.round(ext_1, 4),
        "EXT_SOURCE_2": np.round(ext_2, 4),
        "EXT_SOURCE_3": np.round(ext_3, 4),
    })
    
    df.to_csv(output_path, index=False)
    print(f"Generated {n_records} records at {output_path} with default rate: {target.mean():.2%}")

if __name__ == "__main__":
    generate_retail_loans()