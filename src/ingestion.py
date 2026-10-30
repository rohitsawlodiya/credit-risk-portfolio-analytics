"""Cardinality-checked ingestion. Unknown observations remain unknown."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW

FILES = ["customers", "loan_accounts", "bureau_credit", "repayment_history", "collections_events"]
KEYS = {
    "customers": ["customer_id"], "loan_accounts": ["account_id"],
    "bureau_credit": ["bureau_id"], "repayment_history": ["account_id", "payment_month"],
    "collections_events": ["collection_event_id"],
}
DATES = {
    "customers": ["customer_since"], "loan_accounts": ["origination_date"],
    "bureau_credit": ["date_opened", "date_closed"], "collections_events": ["event_date"],
}


def load_raw(raw_dir: Path = RAW) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(raw_dir / f"{name}.csv") for name in FILES}


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def prepare_data(raw_dir: Path = RAW) -> tuple[pd.DataFrame, dict, dict[str, pd.DataFrame]]:
    tables = load_raw(raw_dir)
    audit = {"tables": {}, "decisions": [], "warnings": {}, "checks": {}}
    for name, df in tables.items():
        audit["tables"][name] = {
            "rows": len(df), "columns": len(df.columns),
            "missing": {c: int(n) for c, n in df.isna().sum().items() if n},
            "duplicate_keys": int(df.duplicated(KEYS[name]).sum()),
            "sha256": hashlib.sha256((raw_dir / f"{name}.csv").read_bytes()).hexdigest(),
        }
        _check(not df[KEYS[name]].isna().any().any(), f"{name}: null primary key")
        _check(not df.duplicated(KEYS[name]).any(), f"{name}: duplicate primary key")
        for col in df.select_dtypes("object"):
            df[col] = df[col].str.strip()
        for col in DATES.get(name, []):
            df[col] = pd.to_datetime(df[col], errors="raise")

    c, l, b, r, e = [tables[n] for n in FILES]
    for child, column, parent, parent_key in [
        (l, "customer_id", c, "customer_id"), (b, "customer_id", c, "customer_id"),
        (r, "account_id", l, "account_id"), (e, "account_id", l, "account_id"),
    ]:
        _check(child[column].isin(parent[parent_key]).all(), f"Orphan foreign key: {column}")
    _check(l.target_default.notna().all() and l.target_default.isin([0, 1]).all(), "Invalid target")
    _check((l.loan_amount > 0).all() and (c.annual_income > 0).all(), "Nonpositive financial denominator")
    _check(c.age.between(18, 100).all(), "Age outside supported range")
    _check((r.emi_due > 0).all(), "Nonpositive installment due")
    for df, cols in [(r, ["emi_paid", "days_past_due", "outstanding_balance"]),
                     (b, ["sanctioned_amount", "outstanding_amount", "days_past_due"]),
                     (e, ["amount_recovered"])]:
        for col in cols:
            _check(df[col].dropna().ge(0).all(), f"Negative value in {col}")
    _check(r.payment_status.isin(["On-Time", "Late", "Missed"]).all(), "Unknown repayment status")
    _check(b.account_status.isin(["Active", "Closed", "Written-Off"]).all(), "Unknown bureau status")
    _check((r.payment_month > 0).all(), "Invalid payment month")
    audit["warnings"]["gender_labels_normalized"] = int(c.gender.isin(["M", "F"]).sum())
    c["gender"] = c.gender.replace({"M": "Male", "F": "Female"})
    # Age is in whole years. Allow a one-year rounding margin rather than
    # declaring a fractional residence tenure just above age impossible.
    for col in ["years_employed", "years_at_residence"]:
        c[f"{col}_raw"] = c[col]
        c[f"{col}_invalid"] = c[col].gt(c.age + 1).astype(int)
        audit["warnings"][f"{col}_exceeds_age_plus_one"] = int(c[f"{col}_invalid"].sum())
        c.loc[c[f"{col}_invalid"].eq(1), col] = np.nan
    for df, cols in [(c, ["years_employed", "num_dependents", "years_at_residence"]),
                     (l, ["interest_rate_pct"]), (b, ["days_past_due"])]:
        for col in cols:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    # Bureau data is a customer-level snapshot, not an application-time history.
    b["is_active"] = b.account_status.eq("Active").astype(int)
    b["is_written_off"] = b.account_status.eq("Written-Off").astype(int)
    ba = b.groupby("customer_id").agg(
        bureau_lines=("bureau_id", "size"), bureau_active_lines=("is_active", "sum"),
        bureau_written_off=("is_written_off", "sum"), bureau_sanctioned=("sanctioned_amount", "sum"),
        bureau_outstanding=("outstanding_amount", "sum"), bureau_max_dpd=("days_past_due", "max"),
        bureau_mean_dpd=("days_past_due", "mean"),
        bureau_dpd_missing_share=("days_past_due_missing", "mean"),
    )
    ba["bureau_utilisation"] = ba.bureau_outstanding.div(ba.bureau_sanctioned.replace(0, np.nan))

    r = r.sort_values(["account_id", "payment_month"]).copy()
    tables["repayment_history"] = r
    r["missed"] = r.payment_status.eq("Missed").astype(int)
    r["late"] = r.payment_status.eq("Late").astype(int)
    ra = r.groupby("account_id").agg(
        repay_months=("payment_month", "size"), repay_mean_dpd=("days_past_due", "mean"),
        repay_max_dpd=("days_past_due", "max"), repay_missed_share=("missed", "mean"),
        repay_late_share=("late", "mean"), repay_due=("emi_due", "sum"),
        repay_paid=("emi_paid", "sum"), latest_balance=("outstanding_balance", "last"),
    )
    ra["repay_paid_ratio"] = ra.repay_paid / ra.repay_due
    # Relative windows work with shorter histories. Require at least six observations
    # for a non-overlapping last-three vs earlier-history comparison.
    last3 = r.groupby("account_id").tail(3).index
    ra["repay_last3_mean_dpd"] = r.loc[last3].groupby("account_id").days_past_due.mean()
    ra["repay_prior_mean_dpd"] = r.drop(last3).groupby("account_id").days_past_due.mean()
    ra["repay_dpd_change"] = (ra.repay_last3_mean_dpd - ra.repay_prior_mean_dpd).where(ra.repay_months >= 6)

    ce = e.merge(l[["account_id", "origination_date"]], on="account_id", validate="many_to_one")
    ce["before_origination"] = ce.event_date.lt(ce.origination_date)
    audit["warnings"]["collections_before_origination"] = int(ce.before_origination.sum())
    ca = ce.groupby("account_id").agg(
        collection_events_raw=("collection_event_id", "size"),
        collection_invalid_dates=("before_origination", "sum"),
        collection_recovered_raw=("amount_recovered", "sum"),
    )
    valid = ce[~ce.before_origination]
    ca = ca.join(valid.groupby("account_id").agg(
        collection_events_valid=("collection_event_id", "size"),
        collection_recovered_valid=("amount_recovered", "sum"),
    )).fillna(0)
    # Preserve all source events. The valid total is separately labelled, never
    # silently backdate an action or use it as a model predictor.
    tables["collections_checked"] = ce
    joined = l.merge(c, on="customer_id", how="left", validate="many_to_one")
    joined = joined.merge(ba, on="customer_id", how="left", validate="many_to_one")
    joined["no_bureau_history"] = joined.bureau_lines.isna().astype(int)
    for col in ["bureau_lines", "bureau_active_lines", "bureau_written_off", "bureau_sanctioned", "bureau_outstanding"]:
        joined[col] = joined[col].fillna(0)
    joined = joined.merge(ra, on="account_id", how="left", validate="one_to_one")
    joined = joined.merge(ca, on="account_id", how="left", validate="one_to_one")
    joined[ca.columns] = joined[ca.columns].fillna(0)
    joined["no_repayment_history"] = joined.repay_months.isna().astype(int)
    joined["loan_to_income"] = joined.loan_amount / joined.annual_income
    joined["bureau_debt_to_income"] = joined.bureau_outstanding / joined.annual_income
    joined["income_band"] = pd.cut(joined.annual_income, [0, 30000, 60000, 100000, np.inf],
        labels=["<30k", "30k-<60k", "60k-<100k", "100k+"], right=False).astype(str)
    joined["age_band"] = pd.cut(joined.age, [18, 30, 45, 60, np.inf],
        labels=["18-29", "30-44", "45-59", "60+"], right=False).astype(str)
    joined["origination_year"] = joined.origination_date.dt.year.astype(str)
    joined["customer_since_after_origination"] = joined.customer_since.gt(joined.origination_date).astype(int)
    # This temporary many-to-many expansion is only a date diagnostic, never the analytical fact table.
    bl = l[["account_id", "customer_id", "origination_date"]].merge(
        b[["customer_id", "date_opened"]], on="customer_id", how="left")
    future = bl.date_opened.gt(bl.origination_date).groupby(bl.account_id).sum()
    joined["bureau_lines_opened_after_origination"] = joined.account_id.map(future).fillna(0).astype(int)
    audit["warnings"].update({
        "accounts_with_later_bureau_lines": int(joined.bureau_lines_opened_after_origination.gt(0).sum()),
        "customer_since_after_origination": int(joined.customer_since_after_origination.sum()),
        "bureau_closure_before_open": int(b.date_closed.lt(b.date_opened).sum()),
        "closed_bureau_with_positive_balance": int(((b.account_status == "Closed") & b.outstanding_amount.gt(0)).sum()),
        "bureau_utilisation_above_one": int(ba.bureau_utilisation.gt(1).sum()),
        "customers_without_bureau": int((~c.customer_id.isin(b.customer_id)).sum()),
        "accounts_without_repayment": int(joined.no_repayment_history.sum()),
    })
    _check(len(joined) == len(l) and joined.account_id.is_unique, "Loan grain was not preserved")
    _check(np.isclose(joined.loan_amount.sum(), l.loan_amount.sum()), "Exposure reconciliation failed")
    _check(joined.target_default.sum() == l.target_default.sum(), "Target reconciliation failed")
    _check(np.isclose(joined.collection_recovered_raw.sum(), e.amount_recovered.sum()), "Recovery reconciliation failed")
    audit["checks"] = {
        "loan_grain_preserved": True, "foreign_keys_valid": True, "original_exposure_reconciles": True,
        "default_count_reconciles": True, "raw_recoveries_reconcile": True,
        "output_rows": len(joined), "output_columns": len(joined.columns),
        "customers": int(joined.customer_id.nunique()),
    }
    audit["decisions"] = [
        "Preserve all raw files byte-for-byte; strip categorical whitespace and parse dates strictly in working copies.",
        "Reject null/duplicate primary keys, orphan foreign keys, invalid target, impossible negative financial values; no silent row dropping.",
        "Normalize M/F gender labels to Male/Female. Gender, age, marital status, dependents and region are excluded from model features; kept for profiling.",
        "Tenure longer than age plus one year is treated as unknown for years_employed/years_at_residence. Keep original values in *_raw and explicit *_invalid flags; a one-year margin accommodates integer-age rounding. Do not invent corrected ages or tenure values.",
        "Retain missing source numeric values and explicit missing flags. Fit median imputation and missing indicators on model training customers only.",
        "Bureau DPD missing is unknown, not zero. No bureau record means zero observed lines/balances, not verified debt-free; retain no-history flag.",
        "Null date_closed is expected for active bureau records; do not impute. Preserve unusual balances and dates for audit.",
        "Aggregate bureau to customer and repayment/collections to account before left-joining to loans; validate cardinality at every join.",
        "Sum installments over months, but use only the final observed balance. Do not sum monthly balances or replicate bureau debt across customers in portfolio totals.",
        "Exclude collections before origination from valid activity/recovery totals, retain raw totals and exception counts. All collections fields are excluded from the default model.",
        "Bureau is undated current-state data. Later-opened lines and current write-offs cannot be treated as origination predictors. Use snapshot monitoring only.",
        "Repayment months are relative to origination, not a common calendar timeline. Origination-year charts are vintages, not observed default trends.",
        "Original loan amount is original exposure. Latest observed repayment balance is a separate snapshot proxy, not common-date regulatory EAD.",
        "No default event date, performance horizon or observation date is supplied. Model evaluation is retrospective and cannot establish prospective default prediction.",
    ]
    return joined.sort_values("account_id").reset_index(drop=True), audit, tables
