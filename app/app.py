"""Streamlit entry point. Start from the repository root as documented."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.analytics import segment_table, summary
from src.assistant import AssistantUnavailable, apply_filters, ask_portfolio
from src.config import ARTIFACTS, DIMENSIONS, PROCESSED, RAW
from src.model import account_drivers
from src.reviews import latest_reviews, review_history, save_review

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Credit Portfolio | Risk Monitor", page_icon="📊", layout="wide")

COLORS = ["#087F8C", "#335C81", "#DA9B50", "#A44A3F", "#718A5E"]
px.defaults.color_discrete_sequence = COLORS
px.defaults.template = "plotly_white"


@st.cache_data
def load_data():
    return (pd.read_csv(PROCESSED / "portfolio.csv", dtype={"origination_year": str}),
            json.loads((ARTIFACTS / "report.json").read_text()),
            json.loads((ARTIFACTS / "data_quality.json").read_text()),
            pd.read_csv(RAW / "repayment_history.csv"))


@st.cache_resource
def load_model():
    # Only load the locally built, trusted artifact shipped with this repository.
    return joblib.load(ARTIFACTS / "default_model.joblib")


@st.cache_data
def explain_account(account_id: str, data_version: str):
    all_data, _, _, _ = load_data()
    return account_drivers(load_model(), all_data[all_data.account_id.eq(account_id)])


def secret(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return default


def csv_bytes(df: pd.DataFrame) -> bytes:
    safe = df.copy()
    # Guard analyst-supplied text against spreadsheet formula injection on export.
    for col in safe.select_dtypes("object"):
        safe[col] = safe[col].map(lambda x: "'" + x if isinstance(x, str) and x.startswith(("=", "+", "-", "@")) else x)
    return safe.to_csv(index=False).encode("utf-8")


def chart_style(fig, height=350):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#16283E"), legend_title_text="")
    return fig


def portfolio_view(df, report, repayments):
    st.subheader("Portfolio overview")
    s = summary(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Loan accounts", f"{s['accounts']:,}", help=f"{s['customers']:,} distinct customers in this selection")
    c2.metric("Observed default rate", f"{s['default_rate']:.2%}" if len(df) else "N/A", help=f"{s['defaults']:,} defaulted accounts / selected accounts")
    c3.metric("Original exposure", f"${s['original_exposure']/1e6:,.2f}m", help="Sum of original loan amounts. Not current EAD or loss.")
    c4.metric("Latest observed balance", f"${s['latest_observed_balance']/1e6:,.2f}m", help="Final observed balance per loan. Accounts do not share one observation date.")
    if df.empty:
        st.info("No accounts match the selected filters.")
        return
    a, b = st.columns([1, 1])
    with a:
        dimension = st.selectbox("Compare portfolio by", list(DIMENSIONS), format_func=DIMENSIONS.get)
    with b:
        measure = st.selectbox("Measure", ["default_rate", "original_exposure", "accounts"],
                               format_func={"default_rate": "Observed default rate", "original_exposure": "Original exposure (USD)", "accounts": "Number of accounts"}.get)
    segment = segment_table(df, [dimension]).sort_values(measure, ascending=False)
    a, b = st.columns([1.35, 1])
    with a:
        fig = px.bar(segment, x=dimension, y=measure, hover_data=["accounts", "defaults", "original_exposure"],
                     labels={dimension: DIMENSIONS[dimension], measure: measure.replace("_", " ").title()})
        if measure == "default_rate":
            fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(chart_style(fig), use_container_width=True)
    with b:
        risk = segment_table(df, ["risk_band"])
        fig = px.pie(risk, names="risk_band", values="original_exposure", hole=.65,
                     title="Original exposure by model score band",
                     color="risk_band", color_discrete_map={"Low (<10%)": COLORS[0], "Watch (10-<25%)": COLORS[2], "High (25%+)": COLORS[3]})
        fig.update_traces(textinfo="percent", hovertemplate="%{label}<br>$%{value:,.0f}<extra></extra>")
        st.plotly_chart(chart_style(fig), use_container_width=True)
    st.caption("Score bands are illustrative monitoring thresholds: <10%, 10-<25%, and >=25%. They do not authorize credit decisions.")
    a, b = st.columns(2)
    with a:
        st.markdown("#### Income × employment")
        cross = segment_table(df, ["income_band", "employment_type"])
        rates = cross.pivot(index="employment_type", columns="income_band", values="default_rate").reindex(columns=["<30k", "30k-<60k", "60k-<100k", "100k+"])
        counts = cross.pivot(index="employment_type", columns="income_band", values="accounts").reindex(index=rates.index, columns=rates.columns)
        fig = go.Figure(go.Heatmap(z=rates.values, x=rates.columns, y=rates.index,
            customdata=counts.values, texttemplate="%{z:.1%}", colorscale=[[0, "#E7F3F2"], [1, "#087F8C"]],
            colorbar=dict(tickformat=".0%"), hovertemplate="%{y} / %{x}<br>Default rate %{z:.2%}<br>Accounts %{customdata:,.0f}<extra></extra>"))
        st.plotly_chart(chart_style(fig), use_container_width=True)
        st.caption("Account-weighted rates. Hover for sample size; small segments may be unstable.")
    with b:
        st.markdown("#### Repayment by months since origination")
        r = repayments[repayments.account_id.isin(df.account_id)]
        monthly = r.groupby("payment_month").agg(mean_dpd=("days_past_due", "mean"), accounts=("account_id", "nunique")).reset_index()
        fig = px.line(monthly, x="payment_month", y="mean_dpd", markers=True, hover_data=["accounts"],
            labels={"payment_month": "Month since origination", "mean_dpd": "Average days past due"})
        st.plotly_chart(chart_style(fig), use_container_width=True)
        st.caption("Relative loan age, not a calendar trend. Original vintages can be compared in the dimension selector above.")
    with st.expander("Segment figures and download"):
        st.dataframe(segment, hide_index=True, use_container_width=True)
        st.download_button("Download selected segment figures", csv_bytes(segment), "portfolio_segments.csv", "text/csv")
    with st.expander("Five findings from the full portfolio", expanded=True):
        st.caption("These findings refer to all 14,685 accounts, independently of the filters above.")
        for insight in report["insights"]:
            finding = insight["finding"].replace("$", "\\$")
            st.markdown(f"**{insight['title']}** — {finding}  \n**So what:** {insight['so_what']}")


def account_view(df, report, repayments):
    st.subheader("Account explorer")
    if df.empty:
        st.info("No accounts match the filters.")
        return
    chosen = st.selectbox("Find an account", df.account_id.tolist(), key="account_selector")
    row = df[df.account_id.eq(chosen)].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Retrospective risk score", f"{row.risk_score:.2%}")
    b.metric("Score band", row.risk_band)
    c.metric("Original loan amount", f"${row.loan_amount:,.0f}")
    d.metric("Observed target", "Defaulted" if row.target_default else "Performing")
    st.caption(f"{row.customer_id} · {row.loan_type} · {row.region} · Evaluation partition: {row.split}. "
               "Scores for training accounts are fitted values. The target is shown for review and is excluded from model inputs.")
    a, b = st.columns([1, 1.25])
    with a:
        st.markdown("#### Account context")
        context = {"Employment": row.employment_type, "Annual income (USD)": f"{row.annual_income:,.0f}",
                   "Observed bureau lines": int(row.bureau_lines), "Bureau write-offs": int(row.bureau_written_off),
                   "Repayment months": int(row.repay_months), "Mean DPD": round(row.repay_mean_dpd, 2),
                   "Missed-payment share": f"{row.repay_missed_share:.2%}",
                   "Collections before origination": int(row.collection_invalid_dates)}
        st.dataframe(pd.DataFrame(context.items(), columns=["Measure", "Value"]).astype(str), hide_index=True, use_container_width=True)
        if row.anomaly_flag:
            st.warning(f"Flagged for analyst review: {row.anomaly_reason}")
        else:
            st.info("No anomaly threshold triggered. Unflagged does not establish that an account is safe.")
    with b:
        st.markdown("#### Drivers of this score")
        drivers = explain_account(chosen, report["data_version"]).head(8)
        fig = px.bar(drivers.iloc[::-1], x="score_delta_pp", y="feature", orientation="h",
            color=drivers.iloc[::-1].score_delta_pp.gt(0), color_discrete_map={True: COLORS[3], False: COLORS[0]},
            labels={"score_delta_pp": "Score difference vs reference (percentage points)", "feature": ""})
        fig.update_layout(showlegend=False)
        st.plotly_chart(chart_style(fig, 370), use_container_width=True)
        st.caption("Replace one feature with its training median/mode and rescore. Positive means the observed value raises this score relative to that reference. These sensitivities are not additive or causal.")
        with st.expander("Observed values and references"):
            st.dataframe(drivers, hide_index=True, use_container_width=True)
    history = repayments[repayments.account_id.eq(chosen)].sort_values("payment_month")
    st.markdown("#### Repayment history")
    fig = px.bar(history, x="payment_month", y=["emi_due", "emi_paid"], barmode="group",
        labels={"payment_month": "Month since origination", "value": "Installment (USD)", "variable": ""})
    st.plotly_chart(chart_style(fig, 280), use_container_width=True)
    st.dataframe(history, hide_index=True, use_container_width=True)


def anomaly_view(df, report):
    st.subheader("Anomaly review queue")
    st.markdown("A flag opens an investigation. It does not establish default or fraud.")
    with st.expander("Trigger, errors and human review", expanded=True):
        st.markdown("**Trigger:** Isolation Forest flags approximately the top 3% of unusual peer patterns. Additional rules flag a last-three-month DPD rise of at least 20 days with a latest average of at least 30 days, or bureau balance above sanctioned amount. Peers share loan type and income band.")
        st.markdown("**False positive:** a legitimate seasonal payment pattern or stale bureau record looks unusual. **False negative:** a deteriorating borrower remains close to peers or has missing bureau data. **Human step:** inspect history, verify source records, then confirm, dismiss or request investigation with a note.")
        st.caption("The 3% setting is a review-capacity assumption. No labelled anomalies are provided, so detector precision and recall are unknown.")
    history = latest_reviews(dataset_id=report["data_version"])
    queue = df[df.anomaly_flag].copy()
    if len(history):
        queue = queue.merge(history[["account_id", "decision"]], on="account_id", how="left", validate="one_to_one")
    else:
        queue["decision"] = "Pending"
    queue["decision"] = queue.decision.fillna("Pending")
    a, b, c = st.columns(3)
    a.metric("Flagged in selection", f"{len(queue):,}")
    b.metric("Awaiting first review", f"{queue.decision.eq('Pending').sum():,}")
    c.metric("Reviewed in selection", f"{queue.decision.ne('Pending').sum():,}")
    statuses = st.multiselect("Review status", ["Pending", "Confirmed", "Dismissed", "Needs investigation"], default=["Pending", "Confirmed", "Dismissed", "Needs investigation"])
    queue = queue[queue.decision.isin(statuses)].sort_values("anomaly_score", ascending=False)
    columns = ["account_id", "loan_type", "income_band", "anomaly_percentile", "anomaly_reason", "decision"]
    st.dataframe(queue[columns], hide_index=True, use_container_width=True, column_config={"anomaly_reason": st.column_config.TextColumn("Why flagged", width="large"), "anomaly_percentile": st.column_config.NumberColumn("Outlier percentile", format="%.3f")})
    if len(queue):
        selected = st.selectbox("Account to review", queue.account_id.tolist())
        selected_row = queue[queue.account_id.eq(selected)].iloc[0]
        st.info(selected_row.anomaly_reason)
        with st.form("review_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            analyst = c1.text_input("Analyst name or initials", max_chars=80)
            decision = c2.selectbox("Decision", ["Confirmed", "Dismissed", "Needs investigation"])
            note = st.text_area("Evidence / review note", max_chars=1000)
            submitted = st.form_submit_button("Save review", type="primary")
            if submitted:
                try:
                    save_review(selected, decision, analyst, note, report["data_version"])
                    st.success("Review recorded. The append-only audit log retains previous decisions.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
                except OSError:
                    st.error("The review store is not writable. Ask the operator to check REVIEW_DB_PATH.")
        st.caption("Demo review log: analyst names are self-declared, not authenticated. Local SQLite persists across app restarts on this machine. Hosted durability requires persistent storage.")
    st.download_button("Download flagged accounts", csv_bytes(queue[columns]), "anomaly_queue.csv", "text/csv")
    audit = review_history(dataset_id=report["data_version"])
    # Export only accounts in the user's current portfolio selection.
    audit = audit[audit.account_id.isin(df.account_id)]
    with st.expander("Review audit trail"):
        st.dataframe(audit, hide_index=True, use_container_width=True)
        st.download_button("Download review audit", csv_bytes(audit), "review_audit.csv", "text/csv")


def assistant_view(portfolio, filters, report):
    st.subheader("Ask the Portfolio")
    st.markdown("Ask a question about the selected portfolio. Every numeric answer is calculated from the underlying data.")
    key = secret("OPENAI_API_KEY")
    if not key:
        st.warning("LLM not configured. Set OPENAI_API_KEY in the server's environment or Streamlit secrets. The rest of the dashboard is fully available; no simulated LLM answers are shown.")
    else:
        st.caption("Server-side API key detected. Model access is verified when a question is submitted.")
    with st.expander("How this AI feature works", expanded=True):
        st.markdown("**Trigger:** submit a natural-language question. An LLM creates a restricted query plan; Python validates it, computes exact figures and renders the answer with its source. **False positive:** the LLM selects a valid but unintended metric or filter. **False negative:** it declines a question that could be expressed using a supported measure. **Human step:** inspect the interpreted query, denominator, filters and source before using the result.")
        st.caption("Only the question, allowed field/category names and active filter selections go to OpenAI. Raw rows, account histories, scores and the API key are not placed in the prompt. Do not enter personal or secret information in a question.")
    st.markdown('Examples: “Which loan type has the highest default rate?” · “Compare original exposure by region” · “What is the default rate for unemployed customers below 30k income?”')
    with st.form("portfolio_question"):
        question = st.text_input("Portfolio question", max_chars=1000, placeholder="Which loan type has the highest default rate?")
        submitted = st.form_submit_button("Ask portfolio", disabled=not bool(key), type="primary")
    scope = json.dumps(filters, sort_keys=True)
    if submitted:
        last = st.session_state.get("last_question_at", 0)
        if time.monotonic() - last < 3:
            st.warning("Please wait a moment before submitting another question.")
        else:
            st.session_state["last_question_at"] = time.monotonic()
            # Clear a previous answer before a failed/new request to avoid presenting it as current.
            st.session_state.pop("portfolio_answer", None)
            try:
                with st.spinner("Interpreting the question and checking the portfolio..."):
                    answer = ask_portfolio(question, portfolio, filters, report["data_version"], key, secret("OPENAI_MODEL", "gpt-4.1-mini"))
                st.session_state["portfolio_answer"] = (scope, question, answer)
            except (AssistantUnavailable, ValueError) as exc:
                st.error(str(exc))
    saved = st.session_state.get("portfolio_answer")
    if saved and saved[0] == scope:
        _, original_question, answer = saved
        st.markdown(f"**Question:** {original_question}")
        st.info(answer["answer"].replace("$", "\\$"))
        if answer["rows"]:
            st.dataframe(pd.DataFrame(answer["rows"]), hide_index=True, use_container_width=True)
        with st.expander("Interpreted query and exact provenance", expanded=True):
            st.json({"query": answer["query_plan"], "source": answer["provenance"]})
        feedback = st.radio("Does the interpretation match your question?", ["Not reviewed", "Yes", "No - rephrase"], horizontal=True)
        if feedback == "No - rephrase":
            st.warning("Do not use this answer. Refine the measure, grouping or filters and ask again.")
    elif saved:
        st.info("Filters changed. Submit the question again to calculate an answer for this selection.")


def model_data_view(report, audit):
    st.subheader("Model evaluation and data quality")
    st.info("Evaluation is a retrospective customer holdout. Missing target dates and bureau snapshot timestamps prevent prospective default validation.")
    rows = []
    for name, results in report["model"]["models"].items():
        for split, m in results.items():
            rows.append({"Model": name, "Partition": split, "AUC": m["auc"], "KS": m["ks"], "Gini": m["gini"], "PR-AUC (AP)": m["average_precision"], "Brier": m["brier"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(f"Selected by validation AUC: {report['model']['champion']}. Frozen validation-KS review threshold: {report['model']['score_threshold']:.4f}. The test set is not used to select a winner or a threshold.")
    a, b = st.columns(2)
    with a:
        importance = pd.read_csv(ARTIFACTS / "feature_importance.csv").head(10).iloc[::-1]
        fig = px.bar(importance, x="auc_drop", y="feature", orientation="h", error_x="std", title="Validation permutation importance", labels={"auc_drop": "Mean ROC-AUC decrease", "feature": ""})
        st.plotly_chart(chart_style(fig, 390), use_container_width=True)
        st.caption("Five shuffles per raw feature. Correlated predictors may share importance. These are associations, not causal effects.")
    with b:
        cal = pd.DataFrame(report["model"]["calibration"])
        fig = px.line(cal, x="mean_score", y="actual_rate", markers=True, hover_data=["accounts"], title="Test calibration by score decile", labels={"mean_score": "Mean score", "actual_rate": "Observed default rate"})
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(dash="dash", color="#AAB4C0")))
        fig.update_xaxes(tickformat=".0%", range=[0, max(.5, cal.mean_score.max() * 1.1)])
        fig.update_yaxes(tickformat=".0%", range=[0, max(.5, cal.actual_rate.max() * 1.1)])
        st.plotly_chart(chart_style(fig, 390), use_container_width=True)
    with st.expander("Evaluation detail, uncertainty and model limitations"):
        st.json(report["model"])
    with st.expander("Production monitoring plan", expanded=True):
        st.markdown("**Daily:** schema/key checks, missingness, stale files, score-range errors, API failures and review-store health. **Weekly:** score and feature drift by loan type and segment, alert volumes, analyst confirmation rate, LLM interpretation acceptance and cost. Example investigation triggers: missingness +5 percentage points or PSI >0.20; calibrate these before launch.")
        st.markdown("**After labels mature:** AUC, KS, Gini, precision/recall at the agreed capacity, Brier/calibration, and segment performance on dated cohorts. Investigate an AUC drop >0.05 or material calibration error. Establish a signed-off observation date and 12-month horizon, independent validation, fairness review, versioned approvals and rollback before production. Neither feature makes automatic lending decisions.")
    st.markdown("#### Data-quality findings")
    warnings = pd.DataFrame(audit["warnings"].items(), columns=["Finding", "Count"])
    st.dataframe(warnings, hide_index=True, use_container_width=True)
    with st.expander("Every ingestion decision and reconciliation"):
        for text in audit["decisions"]:
            st.markdown(f"- {text}")
        st.json(audit["checks"])
        st.json(audit["tables"])
    st.download_button("Download data-quality audit", json.dumps(audit, indent=2), "data_quality.json", "application/json")


def main():
    if not (PROCESSED / "portfolio.csv").exists() or not (ARTIFACTS / "report.json").exists():
        st.error("Build the data and models first: python -m src.pipeline")
        st.stop()
    portfolio, report, audit, repayments = load_data()
    review_path = secret("REVIEW_DB_PATH")
    if review_path:
        os.environ["REVIEW_DB_PATH"] = review_path
    st.markdown("<style>.block-container{padding-top:2rem;padding-bottom:2rem} [data-testid='stMetricValue']{font-size:1.9rem} h1{letter-spacing:-.035em} [data-testid='stSidebar']{border-right:1px solid #dce5ee}</style>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("## Credit portfolio")
        st.caption("Synthetic lending portfolio · USD")
        st.markdown("### Filter the portfolio")
        filters = {}
        for field in ["loan_type", "region", "income_band", "employment_type"]:
            options = sorted(portfolio[field].unique().tolist())
            filters[field] = st.multiselect(DIMENSIONS[field], options, default=options, key=f"filter_{field}")
        st.caption("Filters apply to portfolio, account, anomaly and Q&A views. Empty selection means no matching accounts. Model evaluation always uses its fixed holdout.")
        st.divider()
        st.caption(f"Source version: {report['data_version']}\n\n{len(portfolio):,} accounts · {portfolio.customer_id.nunique():,} customers")
    selected = apply_filters(portfolio, filters)
    st.title("Credit risk monitor")
    st.caption("Explore portfolio exposure, investigate unusual repayment patterns, and review account risk.")
    st.warning("Snapshot analysis: target-default dates and a prediction horizon are not supplied. Scores support retrospective review, not forward-looking lending decisions.", icon="⚠️")
    page = st.radio("View", ["Portfolio", "Account explorer", "Anomaly review", "Ask the Portfolio", "Model & data"], horizontal=True, label_visibility="collapsed")
    if page == "Portfolio":
        portfolio_view(selected, report, repayments)
    elif page == "Account explorer":
        account_view(selected, report, repayments)
    elif page == "Anomaly review":
        anomaly_view(selected, report)
    elif page == "Ask the Portfolio":
        assistant_view(portfolio, filters, report)
    else:
        model_data_view(report, audit)


main()
