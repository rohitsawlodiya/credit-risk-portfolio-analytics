"""Unsupervised peer-relative outliers plus explicit history deterioration rules."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import SEED

ANOMALY_FEATURES = ["repay_dpd_change", "repay_missed_share", "repay_paid_ratio",
                    "bureau_utilisation", "bureau_max_dpd", "bureau_debt_to_income"]
LABELS = {"repay_dpd_change": "DPD change (days)", "repay_missed_share": "missed-payment share",
          "repay_paid_ratio": "paid/due ratio", "bureau_utilisation": "bureau balance/limit",
          "bureau_max_dpd": "bureau maximum DPD", "bureau_debt_to_income": "bureau debt/income"}


def detect_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    # Quantize below measurement precision so CSV round trips cannot turn an
    # effectively constant ratio into a tiny nonzero IQR and a huge residual.
    x = df[ANOMALY_FEATURES].round(10)
    group_keys = [df.loan_type, df.income_band]
    peer = x.groupby(group_keys, observed=True)
    medians = peer.transform("median")
    spreads = peer.transform(lambda s: s.quantile(.75) - s.quantile(.25))
    counts = peer.transform("count")
    # For sparse peers and zero IQRs, use the whole-portfolio distribution.
    global_med = x.median().fillna(0)
    global_iqr = x.quantile(.75) - x.quantile(.25)
    global_iqr = global_iqr.where(global_iqr > 1e-8).fillna(1)
    medians = medians.where(counts >= 30).fillna(global_med)
    spreads = spreads.where((counts >= 30) & spreads.gt(1e-8)).fillna(global_iqr)
    residuals = ((x.fillna(medians) - medians) / spreads).clip(-20, 20).round(8)
    forest = IsolationForest(n_estimators=200, contamination=.03, random_state=SEED, n_jobs=2)
    forest.fit(residuals)
    severity = -forest.score_samples(residuals)
    cutoff = float(-forest.offset_)
    iso_flag = severity > cutoff
    own_flag = df.repay_dpd_change.ge(20) & df.repay_last3_mean_dpd.ge(30)
    limit_flag = df.bureau_utilisation.gt(1)
    flagged = iso_flag | own_flag | limit_flag
    reasons = []
    for i, (_, row) in enumerate(df.iterrows()):
        if not bool(np.asarray(flagged)[i]):
            reasons.append("")
            continue
        parts = []
        if own_flag.iloc[i]:
            parts.append(f"Last 3 months average DPD {row.repay_last3_mean_dpd:.1f}d vs earlier {row.repay_prior_mean_dpd:.1f}d (rise >=20d, latest >=30d)")
        if limit_flag.iloc[i]:
            parts.append(f"Bureau balance/limit {row.bureau_utilisation:.2f} exceeds 1.00")
        if iso_flag[i]:
            top = residuals.iloc[i].abs().nlargest(2).index
            parts.append("Peer outlier: " + "; ".join(f"{LABELS[f]} {x.iloc[i][f]:.3g} vs peer median {medians.iloc[i][f]:.3g}" for f in top))
        reasons.append(". ".join(parts))
    result = pd.DataFrame({"account_id": df.account_id, "anomaly_score": severity,
        "anomaly_percentile": pd.Series(severity).rank(pct=True).to_numpy(),
        "anomaly_flag": np.asarray(flagged), "isolation_flag": iso_flag,
        "history_deterioration_flag": own_flag.to_numpy(), "over_limit_flag": limit_flag.to_numpy(),
        "anomaly_reason": reasons})
    metadata = {"method": "Isolation Forest on robust loan-type/income-band peer residuals plus history and over-limit rules",
        "features": ANOMALY_FEATURES, "contamination": .03, "cutoff": cutoff,
        "flagged": int(flagged.sum()), "isolation_flagged": int(iso_flag.sum()),
        "history_flagged": int(own_flag.sum()), "over_limit_flagged": int(limit_flag.sum()),
        "peer_minimum_nonmissing": 30,
        "note": "3% is a review-capacity assumption, not an observed anomaly prevalence. The full snapshot fits the detector without target labels. No anomaly ground truth is supplied."}
    bundle = {"forest": forest, "features": ANOMALY_FEATURES, "global_median": global_med,
              "global_iqr": global_iqr, "peer_median": x.groupby(group_keys).median(),
              "peer_iqr": x.groupby(group_keys).quantile(.75) - x.groupby(group_keys).quantile(.25),
              "peer_count": x.groupby(group_keys).count()}
    return result, metadata, bundle
