"""Run with:  python -m pytest -q"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src import data
from src.config import DEPLOY_DIR, RAW_CSV, REPORTS_DIR
from src.feature_store import build_features
from src.model import assign_band
from src.scorecard import load_card, points_to_pd, score_points


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(RAW_CSV)


def test_dataset_is_the_published_one(raw):
    assert data.sha256(RAW_CSV) == data.EXPECTED_SHA256
    data.validate(raw)  # 30,000 rows, 6,636 defaults, schema, ranges


def test_features_do_not_use_the_target(raw):
    """Flipping the label must not change any feature (no target leakage)."""
    a = build_features(raw.head(500))
    flipped = raw.head(500).copy()
    flipped["default.payment.next.month"] = 1 - flipped["default.payment.next.month"]
    b = build_features(flipped)
    cols = [c for c in a.columns if c != "default_next_month"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_single_row_scoring_matches_batch(raw):
    """Training (batch) and dashboard (one row, no label) produce identical features."""
    batch = build_features(raw).set_index("account_id")
    one = build_features(raw.drop(columns="default.payment.next.month").iloc[[123]]).set_index("account_id")
    cols = [c for c in one.columns if c != "default_next_month"]
    np.testing.assert_allclose(one[cols].to_numpy(float), batch.loc[one.index, cols].to_numpy(float))


def test_feature_sanity(raw):
    f = build_features(raw)
    assert f["payment_ratio_latest"].between(0, 1).all()
    assert f["months_delinquent_3m"].le(f["months_delinquent_6m"]).all()
    assert f["max_dpd_bucket_6m"].between(0, 8).all()


def test_band_assignment():
    cut = {"red": 0.6, "amber": 0.25}
    assert list(assign_band([0.9, 0.6, 0.3, 0.25, 0.1], cut)) == ["RED", "RED", "AMBER", "AMBER", "GREEN"]


def test_deployed_model_reproduces_reported_auc(raw):
    from sklearn.metrics import roc_auc_score
    meta = json.loads((DEPLOY_DIR / "deploy_meta.json").read_text())
    booster = lgb.Booster(model_file=str(DEPLOY_DIR / "lgbm_booster.txt"))
    f = build_features(raw).set_index("account_id").loc[meta["test_ids"]]
    auc = roc_auc_score(f["default_next_month"], booster.predict(f[meta["features"]]))
    reported = json.loads((REPORTS_DIR / "metrics.json").read_text())["lightgbm"]["test"]["roc_auc"]
    assert abs(auc - reported) < 1e-3


def test_scorecard_points_are_monotonic_in_delinquency():
    """From 'minimum paid' (0) onwards, every extra month overdue must cost points.
    Note: bin (-inf,-1] (paid in full / no usage) scores slightly BELOW bin 0 because in
    this dataset those customers default a little more often (15.4% vs 12.8%), a known
    quirk documented in INTERVIEW_NOTES.md, so it is excluded from this check."""
    card = load_card()
    spec = card["features"]["pay_status_latest"]
    assert spec["cuts"][:2] == [-1.0, 0.0]
    pts = spec["points"][1:]  # bins: 0, 1, 2, 3+
    assert all(a > b for a, b in zip(pts, pts[1:]))
    assert points_to_pd([600])[0] == pytest.approx(1 / 51, rel=1e-6)  # 600 points = 50:1 odds


def test_score_points_runs_on_features(raw):
    card = load_card()
    s = score_points(build_features(raw.head(50)), card)
    assert s.shape == (50,) and np.isfinite(s).all()
