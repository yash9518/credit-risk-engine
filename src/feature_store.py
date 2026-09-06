import sqlite3
import pandas as pd
import numpy as np

def load_and_transform_features(csv_path="data/raw/application_train.csv"):
    df = pd.read_csv(csv_path)
    
    cols = [
        'SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER',
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'DAYS_EMPLOYED', 'DAYS_BIRTH', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
    ]
    df = df[[c for c in cols if c in df.columns]]
    
    conn = sqlite3.connect(":memory:")
    df.to_sql("applications", conn, index=False, if_exists="replace")
    
    query = """
    SELECT 
        SK_ID_CURR,
        TARGET,
        NAME_CONTRACT_TYPE,
        AMT_INCOME_TOTAL,
        AMT_CREDIT,
        AMT_ANNUITY,
        ROUND(AMT_CREDIT / NULLIF(AMT_INCOME_TOTAL, 0), 4) AS credit_to_income_ratio,
        ROUND(AMT_ANNUITY / NULLIF(AMT_INCOME_TOTAL, 0), 4) AS debt_service_ratio,
        ROUND(AMT_ANNUITY / NULLIF(AMT_CREDIT, 0), 4) AS annuity_to_credit_ratio,
        ROUND(ABS(DAYS_BIRTH) / 365.25, 1) AS age_years,
        CASE WHEN DAYS_EMPLOYED > 0 THEN NULL ELSE ROUND(ABS(DAYS_EMPLOYED) / 365.25, 1) END AS employment_years,
        EXT_SOURCE_1,
        EXT_SOURCE_2,
        EXT_SOURCE_3
    FROM applications
    """
    
    features_df = pd.read_sql_query(query, conn)
    conn.close()
    return features_df

if __name__ == "__main__":
    data = load_and_transform_features()
    data.to_csv("data/processed/features.csv", index=False)
    print(f"Features saved: {data.shape}")