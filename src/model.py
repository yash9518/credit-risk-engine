import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import shap

def train_credit_model():
    df = pd.read_csv("data/processed/features.csv")
    
    X = df.drop(columns=['SK_ID_CURR', 'TARGET'])
    X = pd.get_dummies(X, drop_first=True)
    y = df['TARGET']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    print(f"Test ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"Test PR-AUC: {average_precision_score(y_test, y_pred_proba):.4f}")
    
    joblib.dump(clf, "src/lgbm_credit_model.pkl")
    joblib.dump(list(X.columns), "src/model_features.pkl")
    
    explainer = shap.TreeExplainer(clf)
    joblib.dump(explainer, "src/shap_explainer.pkl")
    print("Saved model and SHAP artifacts successfully.")

if __name__ == "__main__":
    train_credit_model()