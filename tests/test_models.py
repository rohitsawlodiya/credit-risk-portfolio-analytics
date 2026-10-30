import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

from src.anomalies import detect_anomalies
from src.config import ARTIFACTS, FEATURES, NUMERIC_FEATURES
from src.model import account_drivers


def test_no_customer_overlap_and_train_only_preprocessing(portfolio):
    groups = {s: set(portfolio.loc[portfolio.split.eq(s), "customer_id"]) for s in ["train", "validation", "test"]}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    bundle = joblib.load(ARTIFACTS / "default_model.joblib")
    imputer = bundle["model"].named_steps["prepare"].named_transformers_["numeric"].named_steps["impute"]
    expected = portfolio.loc[portfolio.split.eq("train"), NUMERIC_FEATURES].median().to_numpy()
    np.testing.assert_allclose(imputer.statistics_, expected)
    assert not set(FEATURES) & {"target_default", "account_id", "customer_id", "gender", "age", "region"}
    assert not any("collection" in f for f in FEATURES)


def test_independent_metrics_and_frozen_validation_selection(portfolio, report):
    test = portfolio[portfolio.split.eq("test")]
    for name, column in [("Logistic regression", "baseline_score"), (report["model"]["champion"], "risk_score")]:
        actual = report["model"]["models"][name]["test"]
        auc = roc_auc_score(test.target_default, test[column])
        fpr, tpr, _ = roc_curve(test.target_default, test[column])
        assert np.isclose(auc, actual["auc"])
        assert np.isclose((tpr-fpr).max(), actual["ks"])
        assert np.isclose(2*auc-1, actual["gini"])
    champion = max(report["model"]["models"], key=lambda n: report["model"]["models"][n]["validation"]["auc"])
    assert report["model"]["champion"] == champion
    assert portfolio.risk_score.between(0, 1).all()


def test_model_reload_and_local_drivers(portfolio):
    bundle = joblib.load(ARTIFACTS / "default_model.joblib")
    rows = portfolio.iloc[[0, 1000, -1]]
    with threadpool_limits(limits=2):
        scores = bundle["model"].predict_proba(rows[FEATURES])[:, 1]
    np.testing.assert_allclose(scores, rows.risk_score)
    drivers = account_drivers(bundle, rows.iloc[[0]])
    assert len(drivers) == len(FEATURES)
    assert np.isfinite(drivers.score_delta_pp).all()


def test_anomalies_do_not_use_target_and_explain_every_flag(portfolio):
    altered = portfolio.copy()
    altered["target_default"] = 1 - altered.target_default
    result, meta, _ = detect_anomalies(altered)
    np.testing.assert_allclose(result.anomaly_score, portfolio.anomaly_score)
    np.testing.assert_array_equal(result.anomaly_flag, portfolio.anomaly_flag)
    assert result.loc[result.anomaly_flag, "anomaly_reason"].str.len().gt(15).all()
    assert 0 < meta["isolation_flagged"] < len(portfolio) * .04
