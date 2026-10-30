"""Run with python -m src.pipeline from the repository root."""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone

import joblib

from src.analytics import build_insights, segment_table, summary
from src.anomalies import detect_anomalies
from src.config import ARTIFACTS, DIMENSIONS, PROCESSED
from src.ingestion import prepare_data
from src.model import train_models


def build() -> dict:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print("1/4 Ingesting and reconciling five source tables", flush=True)
    df, audit, _ = prepare_data()
    source_id = hashlib.sha256("".join(v["sha256"] for v in audit["tables"].values()).encode()).hexdigest()[:16]
    print("2/4 Training baseline and boosting model with customer-disjoint holdouts", flush=True)
    scored, model_report, model_bundle, importance, calibration = train_models(df)
    print("3/4 Detecting peer and history anomalies without target labels", flush=True)
    anomalies, anomaly_report, anomaly_bundle = detect_anomalies(scored)
    scored = scored.merge(anomalies, on="account_id", validate="one_to_one")
    assert len(scored) == len(df) and scored.account_id.is_unique
    print("4/4 Saving reproducible data, model and analytical outputs", flush=True)
    scored.to_csv(PROCESSED / "portfolio.csv", index=False)
    anomalies.to_csv(PROCESSED / "anomalies.csv", index=False)
    importance.to_csv(ARTIFACTS / "feature_importance.csv", index=False)
    calibration.to_csv(ARTIFACTS / "calibration.csv", index=False)
    for dim in DIMENSIONS:
        segment_table(scored, [dim]).to_csv(ARTIFACTS / f"segment_{dim}.csv", index=False)
    segment_table(scored, ["income_band", "employment_type"]).to_csv(ARTIFACTS / "segment_income_employment.csv", index=False)
    joblib.dump(model_bundle, ARTIFACTS / "default_model.joblib", compress=3)
    joblib.dump(anomaly_bundle, ARTIFACTS / "anomaly_model.joblib", compress=3)
    result = {"data_version": source_id, "built_at_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "portfolio": summary(scored), "insights": build_insights(scored),
              "model": model_report, "anomalies": anomaly_report}
    for name, obj in [("data_quality", audit), ("report", result)]:
        (ARTIFACTS / f"{name}.json").write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"portfolio": result["portfolio"], "champion": model_report["champion"],
        "test": {k: v["test"] for k, v in model_report["models"].items()}, "flagged": anomaly_report["flagged"]}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    build()
