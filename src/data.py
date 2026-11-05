"""Fetch and validate the real dataset.

Dataset: "Default of Credit Card Clients" (Taiwan, Apr-Sep 2005)
  - Original source: UCI Machine Learning Repository, dataset id 350
    https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
  - Paper: Yeh, I-C. & Lien, C-H. (2009). "The comparisons of data mining
    techniques for the predictive accuracy of probability of default of
    credit card clients." Expert Systems with Applications, 36(2), 2473-2480.
  - Licence: CC BY 4.0
  - This repo uses the CSV export of the same data (as published on Kaggle as
    "UCI_Credit_Card.csv"). The file is committed under data/raw/ and pinned by
    SHA-256, so every run trains on byte-identical data.

Run:  python -m src.data
"""
import hashlib
import sys
import urllib.request

import pandas as pd

from src.config import RAW_CSV

EXPECTED_SHA256 = "a0f0ab49d6326671d6cd83be5c88dcf18007025fe9a53ecd699119c871176ca1"

# Independent public mirrors that all serve the identical file (same SHA-256).
MIRRORS = [
    "https://raw.githubusercontent.com/lemire/datasciencebook/main/chapter5/UCI_Credit_Card.csv",
    "https://raw.githubusercontent.com/mrtungleung/credit_default/master/UCI_Credit_Card.csv",
    "https://raw.githubusercontent.com/MachineLP/CodeFun/master/04-machine_learning/01-sklearn_examples/credit_default/UCI_Credit_Card.csv",
]

EXPECTED_COLUMNS = (
    ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
    + ["default.payment.next.month"]
)

# Facts published on the UCI dataset page / in Yeh & Lien (2009).
EXPECTED_ROWS = 30_000
EXPECTED_DEFAULTS = 6_636  # 22.12 %


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> None:
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    for url in MIRRORS:
        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, RAW_CSV)
            if sha256(RAW_CSV) == EXPECTED_SHA256:
                return
            print("  checksum mismatch, trying next mirror")
        except Exception as exc:  # network errors -> try next mirror
            print(f"  failed: {exc}")
    sys.exit("Could not obtain a file matching the expected checksum.")


def validate(df: pd.DataFrame) -> None:
    """Fail loudly if the file is not the dataset we think it is."""
    assert list(df.columns) == EXPECTED_COLUMNS, "unexpected schema"
    assert len(df) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} rows, got {len(df)}"
    assert df["ID"].is_unique, "duplicate account IDs"
    assert df.isna().sum().sum() == 0, "unexpected missing values"
    n_def = int(df["default.payment.next.month"].sum())
    assert n_def == EXPECTED_DEFAULTS, f"expected {EXPECTED_DEFAULTS} defaults, got {n_def}"
    assert df["LIMIT_BAL"].between(10_000, 1_000_000).all()
    assert df["AGE"].between(21, 79).all()
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    assert df[pay_cols].isin(range(-2, 9)).all().all(), "repayment status outside -2..8"


def data_quality_report(df: pd.DataFrame) -> None:
    """Known quirks of this dataset that are worth mentioning in an interview."""
    print("\nData-quality notes")
    print(f"  EDUCATION codes outside documented 1-4: "
          f"{int((~df['EDUCATION'].isin([1, 2, 3, 4])).sum())} rows (grouped into 'other')")
    print(f"  MARRIAGE code 0 (undocumented): {int((df['MARRIAGE'] == 0).sum())} rows (grouped into 'other')")
    print(f"  Exact duplicate rows ignoring ID: {int(df.drop(columns='ID').duplicated().sum())} (kept; negligible)")
    print(f"  Accounts with a negative latest bill (credit balance): {int((df['BILL_AMT1'] < 0).sum())}")


def main() -> None:
    if not RAW_CSV.exists():
        download()
    digest = sha256(RAW_CSV)
    if digest != EXPECTED_SHA256:
        sys.exit(f"Checksum mismatch for {RAW_CSV}: {digest}")
    df = pd.read_csv(RAW_CSV)
    validate(df)
    y = df["default.payment.next.month"]
    print(f"OK  {RAW_CSV.name}: {len(df):,} accounts, {int(y.sum()):,} defaults ({y.mean():.2%}), sha256 verified")
    data_quality_report(df)


if __name__ == "__main__":
    main()
