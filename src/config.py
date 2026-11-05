"""Shared paths and constants so every script reads/writes the same locations."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "UCI_Credit_Card.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_CSV = PROCESSED_DIR / "features.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TARGET = "default_next_month"
ID_COL = "account_id"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Early-warning bands, defined by score percentile on the TRAINING set:
# top 10% riskiest accounts -> RED, next 20% -> AMBER, remaining 70% -> GREEN.
RED_PCT = 0.10
AMBER_PCT = 0.20
DEPLOY_DIR = ROOT / "artifacts"  # small, version-independent files committed for the live app
