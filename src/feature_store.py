"""Behavioural feature engineering in SQL (SQLite, in-memory).

Timeline in the raw data (column suffix -> month, 2005):
    PAY_0 / BILL_AMT1 / PAY_AMT1 -> September (most recent)
    PAY_2 / BILL_AMT2 / PAY_AMT2 -> August
    ...
    PAY_6 / BILL_AMT6 / PAY_AMT6 -> April (oldest)
Target: did the account default on its October 2005 payment?

PAY_x codes: -2 = no consumption, -1 = paid in full, 0 = revolving (minimum paid),
             1..8 = months the payment is overdue (8 = 8+ months).

Every feature below uses only information available at the end of September,
so there is no leakage from the target month. The same function is used by the
training pipeline and by the dashboard, so training and serving never diverge.

Run:  python -m src.feature_store
"""
import sqlite3

import pandas as pd

from src.config import FEATURES_CSV, ID_COL, RAW_CSV, TARGET

FEATURE_SQL = """
WITH base AS (
    SELECT
        ID AS account_id,
        "default.payment.next.month" AS default_next_month,
        LIMIT_BAL, AGE, SEX, EDUCATION, MARRIAGE,
        PAY_0 AS s1, PAY_2 AS s2, PAY_3 AS s3, PAY_4 AS s4, PAY_5 AS s5, PAY_6 AS s6,
        BILL_AMT1 AS b1, BILL_AMT2 AS b2, BILL_AMT3 AS b3,
        BILL_AMT4 AS b4, BILL_AMT5 AS b5, BILL_AMT6 AS b6,
        PAY_AMT1 AS p1, PAY_AMT2 AS p2, PAY_AMT3 AS p3,
        PAY_AMT4 AS p4, PAY_AMT5 AS p5, PAY_AMT6 AS p6
    FROM accounts
)
SELECT
    account_id,
    default_next_month,

    -- Static profile
    LIMIT_BAL                                              AS credit_limit,
    AGE                                                    AS age,
    CASE WHEN EDUCATION IN (1, 2, 3) THEN EDUCATION ELSE 4 END AS education_grp,  -- 4 = other/unknown
    CASE WHEN MARRIAGE IN (1, 2) THEN MARRIAGE ELSE 3 END      AS marital_grp,    -- 3 = other/unknown

    -- Delinquency behaviour
    s1                                                     AS pay_status_latest,
    MAX(s1, s2, s3, s4, s5, s6, 0)                         AS max_dpd_bucket_6m,
    (s1 >= 1) + (s2 >= 1) + (s3 >= 1) + (s4 >= 1) + (s5 >= 1) + (s6 >= 1)
                                                           AS months_delinquent_6m,
    (s1 >= 1) + (s2 >= 1) + (s3 >= 1)                      AS months_delinquent_3m,
    (s1 = -1) + (s2 = -1) + (s3 = -1) + (s4 = -1) + (s5 = -1) + (s6 = -1)
                                                           AS months_paid_in_full_6m,
    MAX(s1, 0) - MAX(s4, 0)                                AS delinquency_trend,     -- >0 = worsening

    -- Utilisation (balance / limit)
    ROUND(b1 * 1.0 / LIMIT_BAL, 4)                         AS utilization_latest,
    ROUND((b1 + b2 + b3 + b4 + b5 + b6) / 6.0 / LIMIT_BAL, 4) AS utilization_avg_6m,
    ROUND((b1 - b4) * 1.0 / LIMIT_BAL, 4)                  AS utilization_change_3m,
    (b1 > LIMIT_BAL)                                       AS over_limit_flag,

    -- Repayment capacity: payment made in month t versus the bill issued in month t-1
    ROUND(CASE WHEN b2 <= 0 THEN 1.0 ELSE MIN(p1 * 1.0 / b2, 1.0) END, 4) AS payment_ratio_latest,
    ROUND((CASE WHEN b2 <= 0 THEN 1.0 ELSE MIN(p1 * 1.0 / b2, 1.0) END
         + CASE WHEN b3 <= 0 THEN 1.0 ELSE MIN(p2 * 1.0 / b3, 1.0) END
         + CASE WHEN b4 <= 0 THEN 1.0 ELSE MIN(p3 * 1.0 / b4, 1.0) END) / 3.0, 4)
                                                           AS payment_ratio_avg_3m,
    (p1 = 0 AND b2 > 0) + (p2 = 0 AND b3 > 0) + (p3 = 0 AND b4 > 0)
  + (p4 = 0 AND b5 > 0) + (p5 = 0 AND b6 > 0)              AS months_zero_payment_5m,
    ROUND(LOG(1 + MAX(p1, 0)), 4)                          AS log_last_payment
FROM base
"""

def _connect() -> sqlite3.Connection:
    """SQLite on some builds lacks LOG(); register it so the query is portable."""
    import math
    conn = sqlite3.connect(":memory:")
    conn.create_function("LOG", 1, lambda x: math.log(x) if x and x > 0 else 0.0)
    return conn


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw UCI columns -> one row of engineered features per account."""
    raw = raw.copy()
    if "default.payment.next.month" not in raw.columns:  # scoring a new account
        raw["default.payment.next.month"] = -1
    conn = _connect()
    raw.to_sql("accounts", conn, index=False, if_exists="replace")
    feats = pd.read_sql_query(FEATURE_SQL, conn)
    conn.close()
    return feats


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in (ID_COL, TARGET)]


def main() -> None:
    raw = pd.read_csv(RAW_CSV)
    feats = build_features(raw)
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(FEATURES_CSV, index=False)
    print(f"Features saved to {FEATURES_CSV}: {feats.shape[0]:,} rows x {len(feature_columns(feats))} features")


if __name__ == "__main__":
    main()
