import numpy as np
import pandas as pd
import pytest

from src import ingestion
from src.config import RAW


def test_all_tables_reconcile_without_join_inflation(portfolio):
    raw = ingestion.load_raw()
    assert len(portfolio) == len(raw["loan_accounts"]) == 14685
    assert portfolio.account_id.is_unique
    assert portfolio.customer_id.nunique() == 10000
    assert portfolio.target_default.sum() == 2364
    assert np.isclose(portfolio.loan_amount.sum(), raw["loan_accounts"].loan_amount.sum())
    assert np.isclose(portfolio.collection_recovered_raw.sum(), raw["collections_events"].amount_recovered.sum())
    final = raw["repayment_history"].sort_values("payment_month").groupby("account_id").tail(1)
    assert np.isclose(portfolio.latest_balance.sum(), final.outstanding_balance.sum())


def test_explicit_missingness_and_normalization(portfolio):
    assert set(portfolio.gender) == {"Male", "Female"}
    assert portfolio.interest_rate_pct.isna().sum() == 294
    assert portfolio.groupby("customer_id").years_employed_raw.first().isna().sum() == 500
    assert portfolio.loc[portfolio.years_employed_invalid.eq(1), "years_employed"].isna().all()
    assert portfolio.loc[portfolio.years_at_residence_invalid.eq(1), "years_at_residence"].isna().all()
    absent = portfolio[portfolio.no_bureau_history.eq(1)]
    assert len(absent) > 0
    assert absent.bureau_lines.eq(0).all()
    assert absent.bureau_max_dpd.isna().all()
    assert absent.bureau_utilisation.isna().all()


def test_collections_quarantine_matches_independent_date_join(portfolio):
    e = pd.read_csv(RAW / "collections_events.csv", parse_dates=["event_date"])
    l = pd.read_csv(RAW / "loan_accounts.csv", parse_dates=["origination_date"])
    e = e.merge(l[["account_id", "origination_date"]], on="account_id", validate="many_to_one")
    invalid = e.event_date < e.origination_date
    assert invalid.sum() == 3875
    assert portfolio.collection_invalid_dates.sum() == invalid.sum()
    assert np.isclose(portfolio.collection_recovered_valid.sum(), e.loc[~invalid, "amount_recovered"].sum())


@pytest.mark.parametrize("kind", ["duplicate", "orphan", "bad_target", "negative"])
def test_bad_inputs_fail_visibly(monkeypatch, kind):
    raw = ingestion.load_raw()
    if kind == "duplicate":
        raw["loan_accounts"] = pd.concat([raw["loan_accounts"], raw["loan_accounts"].iloc[[0]]])
    elif kind == "orphan":
        raw["bureau_credit"].loc[0, "customer_id"] = "NONEXISTENT"
    elif kind == "bad_target":
        raw["loan_accounts"].loc[0, "target_default"] = 2
    else:
        raw["repayment_history"].loc[0, "emi_paid"] = -1
    monkeypatch.setattr(ingestion, "load_raw", lambda _: raw)
    with pytest.raises(ValueError):
        ingestion.prepare_data()
