"""Metrics use account counts, original USD exposure, and explicit denominators."""
from __future__ import annotations

import numpy as np
import pandas as pd


def summary(df: pd.DataFrame) -> dict:
    n = len(df)
    return {"accounts": n, "customers": int(df.customer_id.nunique()),
            "defaults": int(df.target_default.sum()),
            "default_rate": float(df.target_default.mean()) if n else None,
            "original_exposure": float(df.loan_amount.sum()),
            "latest_observed_balance": float(df.latest_balance.sum(min_count=1)) if n else 0,
            "defaulted_original_exposure": float(df.loc[df.target_default.eq(1), "loan_amount"].sum()),
            "recovery_valid": float(df.collection_recovered_valid.sum())}


def segment_table(df: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    data = df.assign(defaulted_amount=df.loan_amount * df.target_default)
    out = data.groupby(dimensions, observed=True, dropna=False).agg(
        accounts=("account_id", "size"), customers=("customer_id", "nunique"),
        defaults=("target_default", "sum"), original_exposure=("loan_amount", "sum"),
        defaulted_exposure=("defaulted_amount", "sum"),
        latest_observed_balance=("latest_balance", "sum"),
    ).reset_index()
    out["default_rate"] = out.defaults / out.accounts
    out["exposure_share"] = out.original_exposure / data.loan_amount.sum() if len(data) else np.nan
    # Wilson intervals describe account-rate sampling uncertainty only. Accounts
    # within a customer are dependent; do not present these as causal tests.
    z, n, p = 1.96, out.accounts, out.default_rate
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1 + z*z/n)
    out["rate_low"] = center - half
    out["rate_high"] = center + half
    return out


def build_insights(df: pd.DataFrame) -> list[dict]:
    base = summary(df)
    income = segment_table(df, ["income_band"]).set_index("income_band")
    emp = segment_table(df, ["employment_type"]).sort_values("default_rate", ascending=False)
    loan = segment_table(df, ["loan_type"]).sort_values("original_exposure", ascending=False)
    cross = segment_table(df, ["income_band", "employment_type"])
    cross = cross[cross.accounts >= 100].sort_values("default_rate", ascending=False)
    bureau = segment_table(df.assign(bureau_flag=np.where(df.bureau_written_off.gt(0), "Write-off observed", "None observed")), ["bureau_flag"]).set_index("bureau_flag")
    low, high = income.loc["<30k"], income.loc["100k+"]
    e, l, x = emp.iloc[0], loan.iloc[0], cross.iloc[0]
    b1, b0 = bureau.loc["Write-off observed"], bureau.loc["None observed"]
    return [
        {"title": "Income separates observed risk", "finding": f"Below $30k: {low.default_rate:.2%} ({int(low.defaults):,}/{int(low.accounts):,}) defaulted; $100k+: {high.default_rate:.2%} ({int(high.defaults):,}/{int(high.accounts):,}).",
         "so_what": "Prioritize affordability checks and early support for lower-income borrowers; income alone is not an approval rule."},
        {"title": "Employment adds context", "finding": f"{e.employment_type} has the highest employment-group default rate: {e.default_rate:.2%} ({int(e.defaults):,}/{int(e.accounts):,}), versus {base['default_rate']:.2%} overall.",
         "so_what": "Validate income stability and route vulnerable borrowers to manual review rather than assuming a causal effect."},
        {"title": "Exposure concentration differs from account volume", "finding": f"{l.loan_type} represents {l.exposure_share:.2%} of original exposure (${l.original_exposure:,.0f}) and has a {l.default_rate:.2%} default rate.",
         "so_what": "Review concentration limits and collateral valuations. Default counts alone miss the dollar exposure."},
        {"title": "Income and employment together identify a review segment", "finding": f"{x.employment_type}, income {x.income_band}: {x.default_rate:.2%} ({int(x.defaults):,}/{int(x.accounts):,}); highest among cross-segments with at least 100 accounts.",
         "so_what": "Test targeted outreach in this segment and monitor results. This is an exploratory association, not a policy cutoff."},
        {"title": "Bureau write-offs are a snapshot warning", "finding": f"Customers with an observed bureau write-off: {b1.default_rate:.2%} account default rate ({int(b1.defaults):,}/{int(b1.accounts):,}); none observed: {b0.default_rate:.2%} ({int(b0.defaults):,}/{int(b0.accounts):,}).",
         "so_what": "Request dated bureau snapshots before using this as an underwriting feature. Current write-offs may postdate the target."},
    ]
