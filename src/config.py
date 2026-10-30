from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
SEED = 42

DIMENSIONS = {
    "loan_type": "Loan type",
    "region": "Region",
    "employment_type": "Employment",
    "income_band": "Annual income (USD)",
    "age_band": "Age",
    "risk_band": "Model score band",
    "origination_year": "Origination year",
}

# Explicit allowlist: identifiers, target, collections and protected attributes
# are never selected implicitly from the joined table.
NUMERIC_FEATURES = [
    "annual_income", "years_employed", "years_at_residence",
    "loan_amount", "interest_rate_pct", "tenure_months", "collateral_flag",
    "loan_to_income", "bureau_lines", "bureau_active_lines", "bureau_written_off",
    "bureau_outstanding", "bureau_utilisation", "bureau_max_dpd",
    "bureau_dpd_missing_share", "bureau_debt_to_income", "no_bureau_history",
    "repay_mean_dpd", "repay_max_dpd", "repay_missed_share", "repay_late_share",
    "repay_paid_ratio", "repay_dpd_change", "repay_last3_mean_dpd",
]
CATEGORICAL_FEATURES = ["employment_type", "loan_type", "disbursal_channel", "housing_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
APPLICATION_FEATURES = [f for f in FEATURES if not f.startswith(("bureau_", "repay_", "no_bureau"))]
