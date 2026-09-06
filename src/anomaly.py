import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

def train_anomaly_detector():
    df = pd.read_csv("data/processed/features.csv")
    
    anomaly_features = [
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
        'credit_to_income_ratio', 'debt_service_ratio'
    ]
    
    X = df[anomaly_features].fillna(df[anomaly_features].median())
    
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X)
    
    joblib.dump(iso, "src/anomaly_isolation_forest.pkl")
    print("Isolation Forest model trained and saved.")

if __name__ == "__main__":
    train_anomaly_detector()