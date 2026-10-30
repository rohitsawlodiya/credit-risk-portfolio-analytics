"""Customer-disjoint evaluation with preprocessing fitted only on training rows."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from src.config import APPLICATION_FEATURES, CATEGORICAL_FEATURES, FEATURES, SEED


def group_split(df: pd.DataFrame) -> pd.Series:
    dev, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(df, groups=df.customer_id))
    train_pos, val_pos = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED + 1).split(
        df.iloc[dev], groups=df.iloc[dev].customer_id))
    result = pd.Series("test", index=df.index, name="split")
    result.iloc[dev[train_pos]] = "train"
    result.iloc[dev[val_pos]] = "validation"
    groups = {name: set(df.loc[result.eq(name), "customer_id"]) for name in result.unique()}
    assert not (groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"])
    return result


def make_model(kind: str, features: list[str] = FEATURES) -> Pipeline:
    categorical = [c for c in features if c in CATEGORICAL_FEATURES]
    numeric = [c for c in features if c not in categorical]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                              ("scale", StandardScaler())]), numeric),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                               ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], remainder="drop")
    estimator = (LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
                 if kind == "Logistic regression" else
                 HistGradientBoostingClassifier(max_iter=180, learning_rate=.05, max_leaf_nodes=15,
                     min_samples_leaf=40, l2_regularization=5.0, early_stopping=False, random_state=SEED))
    return Pipeline([("prepare", preprocessor), ("classifier", estimator)])


def metrics(y: pd.Series | np.ndarray, score: np.ndarray, threshold: float = .5) -> dict:
    fpr, tpr, _ = roc_curve(y, score)
    auc = roc_auc_score(y, score)
    pred = score >= threshold
    return {"auc": float(auc), "ks": float(np.max(tpr - fpr)), "gini": float(2 * auc - 1),
            "average_precision": float(average_precision_score(y, score)),
            "brier": float(brier_score_loss(y, score)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "threshold": float(threshold), "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def _cluster_auc_interval(df: pd.DataFrame, score: np.ndarray, repeats: int = 250) -> list[float]:
    rng = np.random.default_rng(SEED)
    customers = df.customer_id.unique()
    positions = df.reset_index(drop=True).groupby("customer_id").indices
    y = df.target_default.to_numpy()
    aucs = []
    for _ in range(repeats):
        idx = np.concatenate([positions[c] for c in rng.choice(customers, len(customers), replace=True)])
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], score[idx]))
    return [float(x) for x in np.quantile(aucs, [.025, .975])]


def train_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict, pd.DataFrame, pd.DataFrame]:
    split = group_split(df)
    train, val, test = [df.loc[split.eq(s)] for s in ["train", "validation", "test"]]
    fitted, report, val_scores = {}, {}, {}
    with threadpool_limits(limits=2):
        for name in ["Logistic regression", "Gradient boosting"]:
            pipeline = make_model(name)
            pipeline.fit(train[FEATURES], train.target_default)
            vs = pipeline.predict_proba(val[FEATURES])[:, 1]
            fpr, tpr, thresholds = roc_curve(val.target_default, vs)
            threshold = float(thresholds[np.argmax(tpr - fpr)])
            ts = pipeline.predict_proba(test[FEATURES])[:, 1]
            report[name] = {"validation": metrics(val.target_default, vs, threshold),
                            "test": metrics(test.target_default, ts, threshold)}
            report[name]["test"]["auc_95_ci"] = _cluster_auc_interval(test, ts)
            fitted[name], val_scores[name] = pipeline, vs
        champion = max(report, key=lambda name: report[name]["validation"]["auc"])
        model = fitted[champion]
        permutation = permutation_importance(model, val[FEATURES], val.target_default,
            scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=1)
        importance = pd.DataFrame({"feature": FEATURES, "auc_drop": permutation.importances_mean,
            "std": permutation.importances_std}).sort_values("auc_drop", ascending=False)
        ablation = make_model("Logistic regression", APPLICATION_FEATURES)
        ablation.fit(train[APPLICATION_FEATURES], train.target_default)
        ablation_scores = ablation.predict_proba(test[APPLICATION_FEATURES])[:, 1]
        scored = df.copy()
        scored["risk_score"] = model.predict_proba(df[FEATURES])[:, 1]
        scored["baseline_score"] = fitted["Logistic regression"].predict_proba(df[FEATURES])[:, 1]
    scored["split"] = split
    scored["risk_band"] = pd.cut(scored.risk_score, [-np.inf, .10, .25, np.inf],
        labels=["Low (<10%)", "Watch (10-<25%)", "High (25%+)"], right=False).astype(str)
    threshold = report[champion]["validation"]["threshold"]
    scored["above_review_threshold"] = scored.risk_score.ge(threshold)
    test_result = scored[scored.split.eq("test")].copy()
    test_result["score_bin"] = pd.qcut(test_result.risk_score, 10, duplicates="drop")
    calibration = test_result.groupby("score_bin", observed=True).agg(
        accounts=("account_id", "size"), mean_score=("risk_score", "mean"),
        actual_rate=("target_default", "mean")).reset_index(drop=True)
    refs = {f: (train[f].mode().iloc[0] if f in CATEGORICAL_FEATURES else float(train[f].median())) for f in FEATURES}
    model_report = {
        "champion": champion, "models": report, "selection": "Highest validation ROC-AUC; test never used for selection or threshold setting.",
        "split": {s: {"accounts": int(split.eq(s).sum()), "customers": int(df.loc[split.eq(s), "customer_id"].nunique()),
                      "defaults": int(df.loc[split.eq(s), "target_default"].sum())} for s in ["train", "validation", "test"]},
        "seed": SEED, "features": FEATURES,
        "no_behavior_ablation": metrics(test.target_default, ablation_scores),
        "ablation_note": "Logistic regression without repayment/bureau features; customer profile timestamps are still unknown.",
        "score_threshold": threshold, "risk_band_thresholds": [.10, .25],
        "calibration": calibration.to_dict("records"),
        "limitations": ["Retrospective target classification, not validated forward-looking PD.",
            "Bureau reporting dates, default dates and target performance windows are absent.",
            "Customer holdout prevents shared-borrower leakage but cannot repair temporal leakage.",
            "Excluded all collections, target, IDs, age, gender, marital status, dependents, region and calendar dates.",
            "Bands are illustrative; threshold maximizes validation KS, not expected profit. No automatic decisions.",
            "Scores on training accounts are fitted values; use only the labelled test partition for reported metrics.",
            "AUC intervals are 250 customer-cluster bootstrap draws with a fixed model; they do not capture model-training uncertainty."],
    }
    bundle = {"model": model, "baseline": fitted["Logistic regression"], "features": FEATURES,
              "reference_values": refs, "champion": champion, "threshold": threshold}
    return scored, model_report, bundle, importance, calibration


def account_drivers(bundle: dict, row: pd.DataFrame) -> pd.DataFrame:
    """Local sensitivity, not additive SHAP or causal attribution."""
    model, features = bundle["model"], bundle["features"]
    with threadpool_limits(limits=2):
        original = float(model.predict_proba(row[features])[:, 1][0])
        variants = pd.concat([row[features]] * len(features), ignore_index=True)
        for i, feature in enumerate(features):
            variants.loc[i, feature] = bundle["reference_values"][feature]
        counterfactual = model.predict_proba(variants)[:, 1]
    result = pd.DataFrame({"feature": features, "observed": [str(row.iloc[0][f]) for f in features],
        "reference": [str(bundle["reference_values"][f]) for f in features],
        "score_delta_pp": (original - counterfactual) * 100})
    return result.iloc[np.argsort(-np.abs(result.score_delta_pp))].reset_index(drop=True)
